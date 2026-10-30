#!/usr/bin/env python3
"""Step 8: blank shelf spaces on the 3D model, found by a pretrained gap detector.

Model: akul-29/Retail-Shelf-Gap-Detection_Model (Hugging Face): YOLOv8s fine-tuned on
"Shelf Images for Planograms" (2,095 images), one class "void". The repo states no
licence and Ultralytics YOLOv8 is AGPL-3.0: fine for this on-prem evaluation, check
before any commercial use.

For one shelf scan (an open_vocab_3d scan dir):
  1. render the textured 3D model face-on with the app's own ray caster
     (open_vocab_3d/render_views_lite.py): pinhole cameras looking along -Z (shelf scans
     have +Z toward the aisle), in 2 rows over the shelf height and a row of positions
     along the aisle with ~50 % overlap, like shelf photos;
  2. run the model on every render;
  3. place each box on the shelf FRONT in 3D using the render's exact depth: the front
     depth is a low percentile of the depth in a ring around the box (the products next
     to the gap); the box corners are back-projected at that depth;
  4. merge boxes of overlapping renders that cover the same place on the shelf front.

Writes <scan>/results/yolo_gaps/ (renders, overlays, detections.json) and adds the
method "yolo" to <scan>/results/gaps.json (exported usd frame), which the app shows.
Note: mesh holes (dark shelf backs that were not reconstructed) render as the grey
background and may be reported as voids.

usage: 08_yolo_gaps.py SCAN_DIR [--conf 0.25] [--dist 1.2] [--weights ...]
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
APP = HERE.parent.parent / "open_vocab_3d"
sys.path.insert(0, str(APP))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
import render_views_lite as rvl   # noqa: E402  (read-only use of the app's renderer)
import shelf_labels as sl         # noqa: E402

W_PX, H_PX, HFOV = 1280, 960, 60.0     # 4:3 landscape "photo"


def camera_poses(bounds, dist, overlap=0.5):
    """Camera centres (scene frame) looking along -Z at the facade; returns list of (C, K)."""
    f = (W_PX / 2) / np.tan(np.radians(HFOV / 2))
    K = np.array([[f, 0, W_PX / 2], [0, f, H_PX / 2], [0, 0, 1.0]])
    lo, hi = np.array(bounds["min"]), np.array(bounds["max"])
    z_front = hi[2]
    cover_w = W_PX / f * dist; cover_h = H_PX / f * dist        # metres seen at the front plane
    nrow = max(1, int(np.ceil((hi[1] - lo[1]) / (cover_h * 0.85))))
    ys = np.linspace(lo[1] + cover_h / 2, hi[1] - cover_h / 2, nrow) if nrow > 1 else [(lo[1] + hi[1]) / 2]
    step = cover_w * (1 - overlap)
    ncol = max(1, int(np.ceil((hi[0] - lo[0] - cover_w) / step)) + 1)
    xs = np.linspace(lo[0] + cover_w / 2, hi[0] - cover_w / 2, ncol) if ncol > 1 else [(lo[0] + hi[0]) / 2]
    return [(np.array([x, y, z_front + dist]), K) for y in ys for x in xs]


def c2w(C):
    """OpenCV camera (x right, y down, z forward) looking along -Z world, +Y world up."""
    T = np.eye(4)
    T[:3, :3] = np.stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]], 1)   # columns: cam axes in world
    T[:3, 3] = C
    return T


def lift(box, depth, K, T, ring=0.35):
    """Box on the shelf front in the scene frame, from the render's z-depth."""
    x0, y0, x1, y1 = box
    H, W = depth.shape
    mx, my = ring * (x1 - x0), ring * (y1 - y0)
    X0, Y0 = int(max(0, x0 - mx)), int(max(0, y0 - my)); X1, Y1 = int(min(W, x1 + mx)), int(min(H, y1 + my))
    win = depth[Y0:Y1, X0:X1].copy()
    inner = np.zeros_like(win, bool)
    inner[int(y0) - Y0:int(y1) - Y0, int(x0) - X0:int(x1) - X0] = True
    ring_d = win[~inner & (win > 0)]
    in_d = win[inner & (win > 0)]
    if ring_d.size >= 20:
        z_front, src = float(np.percentile(ring_d, 20)), "ring"
    elif in_d.size:
        z_front, src = float(np.percentile(in_d, 10)), "inside"
    else:
        return None
    behind = in_d - z_front
    behind = behind[behind > 0.01]
    depth_m = float(np.clip(np.median(behind), 0.02, 1.5)) if behind.size > 0.1 * inner.sum() else sl.DEFAULT_DEPTH_M
    us = np.array([x0, x1, x0, x1]); vs = np.array([y0, y0, y1, y1])
    Pc = np.stack([(us - K[0, 2]) / K[0, 0] * z_front, (vs - K[1, 2]) / K[1, 1] * z_front, np.full(4, z_front)], 1)
    Pw = Pc @ T[:3, :3].T + T[:3, 3]
    zf_world = float(Pw[0, 2])
    lo = np.array([Pw[:, 0].min(), Pw[:, 1].min(), zf_world - depth_m])
    hi = np.array([Pw[:, 0].max(), Pw[:, 1].max(), zf_world])
    return lo, hi, src, "mesh" if behind.size > 0.1 * inner.sum() else "default"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--weights", default=str(HERE.parent / "models" / "gap_yolov8" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dist", type=float, default=1.2, help="camera distance from the shelf front (m)")
    ap.add_argument("--merge-iou", type=float, default=0.3)
    a = ap.parse_args()
    t0 = time.time()
    sdir = Path(a.scan_dir).resolve()
    rec = json.loads((sdir / "scan.json").read_text())
    out = sdir / "results" / "yolo_gaps"; out.mkdir(parents=True, exist_ok=True)
    from pxr import Usd
    meshes = rvl.load_usd_meshes(Usd.Stage.Open(str(sdir / "scene.usda")))
    rr = rvl.RayRenderer(meshes, cull_backfaces=False)
    poses = camera_poses(rec["scene_bounds"], a.dist)
    from ultralytics import YOLO
    model = YOLO(a.weights)
    dets, lifted = [], []
    for i, (C, K) in enumerate(poses):
        T = c2w(C)
        rgb, depth = rr.render(K, T, W_PX, H_PX, supersample=2)
        name = f"view_{i:03d}"
        Image.fromarray(rgb).save(out / f"{name}.jpg", quality=92)
        res = model.predict(rgb[..., ::-1].copy(), conf=a.conf, verbose=False)[0]   # ultralytics expects BGR arrays
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        scores = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
        ov = Image.fromarray(rgb); d = ImageDraw.Draw(ov)
        for b, s in zip(boxes, scores):
            d.rectangle(b.tolist(), outline=(255, 140, 0), width=4)
            d.text((b[0] + 4, b[1] + 2), f"void {s:.2f}", fill=(255, 140, 0))
            dets.append({"view": name, "box_px": [round(float(v), 1) for v in b], "score": round(float(s), 4)})
            L = lift(b, depth, K, T)
            if L is not None:
                lo, hi, src, dsrc = L
                lifted.append({"keyframe": name, "box_px": [round(float(v), 1) for v in b], "score": round(float(s), 4),
                               "bmin": lo.tolist(), "bmax": hi.tolist(), "front_source": src, "depth_source": dsrc})
        ov.save(out / f"{name}_voids.jpg", quality=88)
        print(f"{name}: {len(boxes)} voids", flush=True)
    # merge overlapping renders; every render is a different view, so one view is enough
    sys.path.insert(0, str(APP))
    import gaps_runner as gr
    gaps = gr.merge(lifted, a.merge_iou, min_views=1)
    Winv = np.linalg.inv(sl.scene_transform(rec))                 # scene -> exported usd frame
    for g in gaps:
        P = np.array([g["bmin"], g["bmax"]]) @ Winv.T
        g["bmin"], g["bmax"] = P.min(0).round(4).tolist(), P.max(0).round(4).tolist()
    (out / "detections.json").write_text(json.dumps({"views": len(poses), "detections": dets,
                                                      "camera": {"width": W_PX, "height": H_PX, "hfov_deg": HFOV,
                                                                 "distance_m": a.dist}}, indent=1))
    gpath = sdir / "results" / "gaps.json"
    doc = json.loads(gpath.read_text()) if gpath.is_file() else {"schema": 1, "scan_id": sdir.name, "frame": "usd",
                                                                    "methods": {}}
    doc.setdefault("methods", {})["yolo"] = {
        "description": "pretrained YOLOv8 shelf-gap model (akul-29/Retail-Shelf-Gap-Detection_Model, class 'void') "
                       f"on {len(poses)} face-on renders of the 3D model, conf >= {a.conf}",
        "gaps": gaps, "detections_2d": len(dets), "detections_placed": len(lifted),
        "views": len(poses), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 1)}
    doc.setdefault("created_at", doc["methods"]["yolo"]["created_at"])
    doc.setdefault("num_keyframes", len(poses))
    tmp = gpath.with_suffix(".tmp"); tmp.write_text(json.dumps(doc, indent=1)); tmp.replace(gpath)
    print(json.dumps({"scan": sdir.name, "views": len(poses), "detections": len(dets), "placed": len(lifted),
                      "gaps": len(gaps), "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
