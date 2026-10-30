#!/usr/bin/env python3
"""Gap detection job for shelf scans: two baselines on the ORIGINAL keyframes.

    python -u gaps_runner.py --scan-dir out/scans/<id> --job-id <job_id>

  A "dino"   Grounding DINO with gap prompts on every keyframe; SAM tightens each box
             to its mask. Open-vocabulary, no training: no published evidence it works
             for empty shelf space, which is why it is scored against the user's labels.
  B "depth"  the per-keyframe depth maps of the conversion (Depth Anything V2 Small fitted
             to the SfM points): pixels more than --recess-m behind the local shelf front
             (a low percentile of the depth in a wide window: the product faces) form
             connected regions -> boxes. Depth exists even where the mesh has holes.

Every 2D box is placed on the shelf front in 3D exactly like a user label
(shelf_labels.lift_box). Boxes of one method that overlap on the shelf front across
keyframes are merged into one gap; a gap needs >= --min-views keyframes.

Results: <scan>/results/gaps.json, 3D in the EXPORTED usd frame (before the scene's
calibration scale / flip), so the server can map them to the current scene frame and
score them against the labels on every read.

Stdout protocol (same as pipeline_runner.py):
    PROGRESS <0..1> <stage> <message>    RESULT <path>    ERROR <message>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PROMPTS = ["empty shelf space", "gap on shelf", "missing product on shelf"]


def say(s: str) -> None:
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def distort_px(u, v, f, cx, cy, k1):
    """Undistorted pinhole pixel -> original (SIMPLE_RADIAL) pixel."""
    x, y = (u - cx) / f, (v - cy) / f
    r2 = x * x + y * y
    return f * x * (1 + k1 * r2) + cx, f * y * (1 + k1 * r2) + cy


# --------------------------------------------------------------------------- baseline A
def detect_dino(images, names, a, progress):
    import detect_2d as d2
    from PIL import Image
    det = d2.GroundingDinoDetector(PROMPTS, box_threshold=a.box_threshold, text_threshold=a.text_threshold,
                                   device="auto", prompt_group_size=1)
    seg = d2.BoxSegmenter(device="auto") if a.sam else None
    out = {}
    for i, (name, path) in enumerate(zip(names, images)):
        im = Image.open(path).convert("RGB")
        (dets,) = det.detect_batch([im])
        dets, _ = d2.postprocess(dets, (im.width, im.height), nms_iou=0.5,
                                 max_box_area_frac=a.max_box_frac, min_box_px=a.min_box_px)
        boxes, scores, _ = dets
        if seg is not None and len(boxes):
            masks = seg.segment(im, boxes)
            for j, m in enumerate(masks):
                ys, xs = np.nonzero(m)
                if len(xs) < 0.2 * (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1]):
                    continue                       # degenerate mask: keep the detector box
                tight = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
                # only tighten (never grow), so a mask that bleeds onto products cannot widen the gap
                boxes[j] = [max(boxes[j, 0], tight[0]), max(boxes[j, 1], tight[1]),
                            min(boxes[j, 2], tight[2]), min(boxes[j, 3], tight[3])]
        out[name] = [{"box_px": [round(float(c), 1) for c in b], "score": round(float(s), 4)}
                     for b, s in zip(boxes, scores)]
        progress((i + 1) / len(names), f"keyframe {i + 1}/{len(names)}: {len(out[name])} boxes")
    return out


# --------------------------------------------------------------------------- baseline B
def detect_depth(work: Path, cams: dict, m_per_unit: float, a, progress):
    from scipy import ndimage as ndi
    dcam = json.loads((work / "dense" / "cameras.json").read_text())
    K = np.asarray(dcam["K"]); fh, cxh, cyh = K[0, 0], K[0, 2], K[1, 2]
    c = cams["camera"]
    sx = c["width"] / dcam["width"]                 # half-res undistorted -> full-res pinhole
    names = [k["name"] for k in cams["keyframes"]]
    out = {}
    for i, name in enumerate(names):
        p = work / "dense" / "depth" / f"{name}.npy"
        if not p.is_file():
            out[name] = []
            continue
        z = np.load(p).astype(np.float32) * m_per_unit          # metres (scene scale)
        valid = np.isfinite(z) & (z > 0)
        zf = np.where(valid, z, np.nan)
        H, W = z.shape
        # local shelf front: the product faces, i.e. a low percentile of the depth around.
        # Computed on a 4x downsampled map (a sliding percentile over a ~120x135 px window at
        # full 540x960 is ~1e10 operations per keyframe) and upsampled: the front is smooth.
        ds = 4
        small = np.where(valid, z, np.nanmax(zf))[::ds, ::ds]
        win = (max(3, int(a.front_win * H * 0.5 / ds)), max(3, int(a.front_win * W / ds)))
        front_s = ndi.percentile_filter(small, a.front_pct, size=win)
        front = ndi.zoom(front_s, (H / front_s.shape[0], W / front_s.shape[1]), order=1)[:H, :W]
        if front.shape != z.shape:
            front = np.pad(front, ((0, H - front.shape[0]), (0, W - front.shape[1])), mode="edge")
        recess = np.where(valid, z - front, 0)
        mask = ndi.binary_opening(recess > a.recess_m, np.ones((5, 5), bool))
        mask = ndi.binary_closing(mask, np.ones((7, 7), bool))
        lab, n = ndi.label(mask)
        dets = []
        for sl, k in zip(ndi.find_objects(lab), range(1, n + 1)):
            if sl is None:
                continue
            region = lab[sl] == k
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            zmed = float(np.median(front[sl][region]))
            w_m, h_m = (x1 - x0) * zmed / fh, (y1 - y0) * zmed / fh
            if w_m < a.min_gap_m or h_m < a.min_gap_m or w_m > 1.5 or h_m > 1.0:
                continue
            if region.mean() < 0.35:                 # thin / diagonal blobs are not slots
                continue
            # half-res undistorted box -> full-res original (distorted) pixels
            us = np.array([x0, x1, x0, x1], float) * sx; vs = np.array([y0, y0, y1, y1], float) * sx
            ud, vd = distort_px(us, vs, fh * sx, cxh * sx, cyh * sx, c["k1"])
            box = [float(np.clip(ud.min(), 0, c["width"])), float(np.clip(vd.min(), 0, c["height"])),
                   float(np.clip(ud.max(), 0, c["width"])), float(np.clip(vd.max(), 0, c["height"]))]
            dets.append({"box_px": [round(v, 1) for v in box],
                         "score": round(float(np.clip(np.median(recess[sl][region]) / 0.4, 0, 1)), 4),
                         "recess_cm": round(float(np.median(recess[sl][region])) * 100, 1)})
        out[name] = dets
        progress((i + 1) / len(names), f"keyframe {i + 1}/{len(names)}: {len(dets)} regions")
    return out


# --------------------------------------------------------------------------- 3D + merge
def front_rect(g):
    """(x0, x1, y0, y1) of a lifted box on the shelf front (usd frame: x along aisle, y up)."""
    return g["bmin"][0], g["bmax"][0], g["bmin"][1], g["bmax"][1]


def rect_iou(a, b):
    ix = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ix * iy
    ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / ua if ua > 0 else 0.0


def merge(dets: list[dict], iou: float, min_views: int) -> list[dict]:
    """Union-find over detections whose shelf-front rectangles overlap (IoU >= iou)."""
    n = len(dets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    rects = [front_rect(d) for d in dets]
    for i in range(n):
        for j in range(i + 1, n):
            if rect_iou(rects[i], rects[j]) >= iou:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    gaps = []
    for members in groups.values():
        views = sorted({dets[i]["keyframe"] for i in members})
        if len(views) < min_views:
            continue
        bmin = np.median([dets[i]["bmin"] for i in members], 0)
        bmax = np.median([dets[i]["bmax"] for i in members], 0)
        gaps.append({"bmin": bmin.round(4).tolist(), "bmax": bmax.round(4).tolist(),
                     "score": round(float(np.mean([dets[i]["score"] for i in members])), 4),
                     "num_views": len(views), "num_detections": len(members),
                     "detections": [{"keyframe": dets[i]["keyframe"], "box_px": dets[i]["box_px"],
                                     "score": dets[i]["score"]} for i in members]})
    gaps.sort(key=lambda g: (g["bmin"][1], g["bmin"][0]))
    for k, g in enumerate(gaps, 1):
        g["id"] = f"gap_{k:03d}"
    return gaps


def lift_all(sc, cams, W, per_kf: dict) -> list[dict]:
    import shelf_labels as sl
    by_name = {k["name"]: k for k in cams["keyframes"]}
    Winv = np.linalg.inv(W)
    out = []
    for name, dets in per_kf.items():
        for d in dets:
            try:
                r = sl.lift_box(sc, cams, by_name[name], d["box_px"], W)
            except Exception:  # noqa: BLE001 - boxes that see no shelf are dropped
                continue
            # scene -> usd frame (W is scale * optional flip: map both corners, re-sort)
            P = np.array([r["bbox"]["min"], r["bbox"]["max"]]) @ Winv.T
            out.append({"keyframe": name, "box_px": d["box_px"], "score": d["score"],
                        "bmin": P.min(0).tolist(), "bmax": P.max(0).tolist()})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-dir", required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--box-threshold", type=float, default=0.25)
    ap.add_argument("--text-threshold", type=float, default=0.2)
    ap.add_argument("--max-box-frac", type=float, default=0.2, help="drop DINO boxes larger than this image fraction")
    ap.add_argument("--min-box-px", type=float, default=20)
    ap.add_argument("--sam", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--recess-m", type=float, default=0.10, help="depth baseline: min depth behind the shelf front")
    ap.add_argument("--front-pct", type=float, default=20, help="percentile of the local depth = shelf front")
    ap.add_argument("--front-win", type=float, default=0.25, help="window for the local front, fraction of width")
    ap.add_argument("--min-gap-m", type=float, default=0.04)
    ap.add_argument("--merge-iou", type=float, default=0.3)
    ap.add_argument("--min-views", type=int, default=2)
    a = ap.parse_args(argv)
    import shelf_labels as sl

    sdir = Path(a.scan_dir).resolve()
    rec = json.loads((sdir / "scan.json").read_text())
    cams = sl.load_cameras(sdir)
    if cams is None:
        say("ERROR this scan has no keyframe cameras (convert it again)")
        return 2
    work = sdir / "convert" / "work"
    W = sl.scene_transform(rec)
    unit = float(np.linalg.norm(W[:, 0]))                      # calibration factor
    m_per_unit = float(cams["T_sfm_to_usd"]["s"]) * unit if "T_sfm_to_usd" in cams else None
    names = [k["name"] for k in cams["keyframes"]]
    images = [sdir / sl.IMAGES_REL / f"{n}.jpg" for n in names]
    t0 = time.time()

    def prog(p0, p1, stage):
        return lambda f, msg: say(f"PROGRESS {p0 + (p1 - p0) * f:.4f} {stage} {msg}")

    say("PROGRESS 0.01 dino loading Grounding DINO + SAM")
    per_dino = detect_dino(images, names, a, prog(0.02, 0.62, "dino"))
    t_dino = time.time() - t0
    if m_per_unit is None:
        say("ERROR cameras_usd.json has no SfM scale (re-export with 07_cameras.py)")
        return 2
    t1 = time.time()
    per_depth = detect_depth(work, cams, m_per_unit, a, prog(0.62, 0.82, "depth"))
    t_depth = time.time() - t1
    say("PROGRESS 0.83 lift placing detections on the shelf front")
    sc = sl._scene(sdir / "scene.usda")
    methods = {}
    for key, per_kf, desc in (("dino", per_dino, "Grounding DINO + SAM, prompts: " + ", ".join(PROMPTS)),
                              ("depth", per_depth, f"depth > {a.recess_m * 100:.0f} cm behind the local shelf front")):
        lifted = lift_all(sc, cams, W, per_kf)
        gaps = merge(lifted, a.merge_iou, a.min_views)
        methods[key] = {"description": desc, "gaps": gaps,
                        "detections_2d": sum(len(v) for v in per_kf.values()), "detections_placed": len(lifted)}
        say(f"PROGRESS {0.83 + (0.16 if key == 'depth' else 0.08):.4f} lift {key}: {len(gaps)} gaps "
            f"from {len(lifted)} placed detections")
    doc = {"schema": 1, "scan_id": sdir.name, "job_id": a.job_id, "frame": "usd",
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": {"total": round(time.time() - t0, 1), "dino": round(t_dino, 1), "depth": round(t_depth, 1)},
           "num_keyframes": len(names), "params": {k: v for k, v in vars(a).items() if k not in ("scan_dir", "job_id")},
           "methods": methods}
    out = sdir / "results" / "gaps.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():   # keep results of other tools (e.g. the pretrained "yolo" gap model, 08_yolo_gaps.py)
        try:
            prev = json.loads(out.read_text()).get("methods", {})
            for k, v in prev.items():
                if k not in methods:
                    doc["methods"][k] = v
        except (OSError, json.JSONDecodeError):
            pass
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(out)
    say(f"RESULT {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
