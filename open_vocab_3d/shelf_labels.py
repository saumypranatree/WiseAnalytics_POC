"""Empty-slot labels on shelf scans: 2D boxes on keyframes -> 3D boxes on the shelf front.

Source of truth is 2D: <scan>/labels/empty_slots.json holds
    {"labels": [{"id": "lab_001", "keyframe": "00004", "box_px": [x0, y0, x1, y1],
                 "kind": "empty" | "pushed_back", "note": "", "created_at": ...}]}
in pixels of the ORIGINAL keyframe image (convert/work/sfm_sr/images/<keyframe>.jpg).
The 3D box is recomputed on every read (lift_labels), so it follows later scale and
flip changes of the scan.

Lifting one box:
  * cameras: convert/work/cameras_usd.json (07_cameras.py), in the frame of the
    exported scan.usdz ("usd frame": +Y up, +Z from the shelves toward the aisle).
  * usd -> scene: scene.usda applies the calibration scale and the optional 180 deg
    flip about X:  p_scene = Flip @ (scale * p_usd).
  * rays through pixels are cast against the scene mesh (Open3D). An empty slot often
    shows no mesh at all (dark shelf backs are not reconstructed), so the box is placed
    on the shelf FRONT plane. Its depth comes from the products around the box: a ring of
    rays just outside it; the front is a high percentile of their depth toward the aisle.
  * the box corners are intersected with that plane: width and height are measured on
    the shelf front, in scene metres (calibrated or approximate, as the scan is).
  * depth: from the mesh hits inside the box when there are any, otherwise DEFAULT_DEPTH_M
    (the field "depth_source" says which).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

KINDS = ("empty", "pushed_back")
LABELS_REL = Path("labels") / "empty_slots.json"
CAMERAS_REL = Path("convert") / "work" / "cameras_usd.json"
IMAGES_REL = Path("convert") / "work" / "sfm_sr" / "images"
DEFAULT_DEPTH_M = 0.30    # shelf depth used when nothing behind the slot was reconstructed
RING = 0.35               # ring around the box, as a fraction of the box size
GRID = 24                 # rays per box side

_cache_lock = threading.Lock()
_scene_cache: dict[str, tuple[float, object]] = {}


# --------------------------------------------------------------------------- frames
def scene_transform(rec: dict) -> np.ndarray:
    """3x3 linear map usd frame -> scene frame (uniform scale, then optional X-180 flip)."""
    sc = rec.get("scale") if isinstance(rec.get("scale"), dict) else {}
    try:
        f = float(sc.get("factor", 1.0))
    except (TypeError, ValueError):
        f = 1.0
    f = f if np.isfinite(f) and f > 0 else 1.0
    flip = bool((rec.get("orientation") or {}).get("flip_applied"))
    return (np.diag([1.0, -1.0, -1.0]) if flip else np.eye(3)) * f


def undistort(xd: np.ndarray, yd: np.ndarray, k1: float, iters: int = 10):
    """Inverse of SIMPLE_RADIAL: normalized distorted coords -> undistorted."""
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        d = 1.0 + k1 * r2
        x, y = xd / d, yd / d
    return x, y


def pixel_rays(cams: dict, kf: dict, u: np.ndarray, v: np.ndarray, W: np.ndarray):
    """Scene-frame ray origin (3,) and unit directions (n, 3) for original-image pixels."""
    c = cams["camera"]
    x, y = undistort((u - c["cx"]) / c["f"], (v - c["cy"]) / c["f"], c["k1"])
    d_cam = np.stack([x, y, np.ones_like(x)], 1)
    R_cu = np.asarray(kf["R_cam_from_usd"])
    d = (d_cam @ R_cu) @ W.T                      # cam -> usd (R^T), then usd -> scene
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return W @ np.asarray(kf["C_usd"], float), d


# --------------------------------------------------------------------------- mesh
def _scene(scene_usda: Path):
    """Cached Open3D raycasting scene of scene.usda (in scene coordinates)."""
    key = str(scene_usda)
    mt = scene_usda.stat().st_mtime
    with _cache_lock:
        hit = _scene_cache.get(key)
        if hit and hit[0] == mt:
            return hit[1]
    import open3d as o3d
    from pxr import Usd
    import render_views_lite as rvl
    meshes = rvl.load_usd_meshes(Usd.Stage.Open(str(scene_usda)))
    sc = o3d.t.geometry.RaycastingScene()
    for m in meshes:
        sc.add_triangles(m.verts.astype(np.float32), m.tris.astype(np.uint32))
    with _cache_lock:
        _scene_cache[key] = (mt, sc)
    return sc


def _cast(sc, origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    import open3d as o3d
    rays = np.concatenate([np.broadcast_to(origin, dirs.shape), dirs], 1).astype(np.float32)
    t = sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return np.where(np.isfinite(t), t, np.nan)


# --------------------------------------------------------------------------- lifting
def lift_box(sc, cams: dict, kf: dict, box_px, W: np.ndarray) -> dict:
    """3D box of one label on the shelf front (scene frame)."""
    x0, y0, x1, y1 = map(float, box_px)
    axis_a, up, front = W @ [1.0, 0, 0], W @ [0, 1.0, 0], W @ [0, 0, 1.0]
    unit = float(np.linalg.norm(axis_a))          # scene units per usd unit (= scale factor)
    axis_a, up, front = axis_a / unit, up / unit, front / unit

    # ring of rays just outside the box -> depth of the products around the slot
    mx, my = RING * (x1 - x0), RING * (y1 - y0)
    gu, gv = np.meshgrid(np.linspace(x0 - mx, x1 + mx, GRID + 8), np.linspace(y0 - my, y1 + my, GRID + 8))
    inside = (gu >= x0) & (gu <= x1) & (gv >= y0) & (gv <= y1)
    origin, d_ring = pixel_rays(cams, kf, gu[~inside], gv[~inside], W)
    t_ring = _cast(sc, origin, d_ring)
    ok = np.isfinite(t_ring)
    if ok.sum() >= 8:
        z_front = float(np.percentile((origin + d_ring[ok] * t_ring[ok, None]) @ front, 80))
        front_source = "ring"
    else:
        # nothing around the box: fall back to the front of whatever is inside it
        z_front = None
        front_source = "inside"

    # rays inside the box -> what is behind the slot (often nothing: holes at dark backs)
    iu, iv = np.meshgrid(np.linspace(x0, x1, GRID), np.linspace(y0, y1, GRID))
    _, d_in = pixel_rays(cams, kf, iu.ravel(), iv.ravel(), W)
    t_in = _cast(sc, origin, d_in)
    hin = np.isfinite(t_in)
    depths_in = (origin + d_in[hin] * t_in[hin, None]) @ front if hin.any() else np.array([])
    if z_front is None:
        if not depths_in.size:
            raise ValueError("the box does not cover any part of the scan")
        z_front = float(np.percentile(depths_in, 90))

    # corners and edge midpoints of the box, intersected with the front plane
    eu = np.array([x0, (x0 + x1) / 2, x1, x0, x1, x0, (x0 + x1) / 2, x1])
    ev = np.array([y0, y0, y0, (y0 + y1) / 2, (y0 + y1) / 2, y1, y1, y1])
    _, d_c = pixel_rays(cams, kf, eu, ev, W)
    denom = d_c @ front
    if np.any(np.abs(denom) < 1e-6):
        raise ValueError("the view is parallel to the shelf front")
    tc = (z_front - origin @ front) / denom
    if np.any(tc <= 0):
        raise ValueError("the shelf front is behind the camera for this box")
    P = origin + d_c * tc[:, None]
    a, h = P @ axis_a, P @ up

    behind = z_front - depths_in
    behind = behind[behind > 0.01]                     # scene units are metres (scaled scan)
    if behind.size >= 0.1 * GRID * GRID:
        depth = float(np.clip(np.percentile(behind, 50), 0.02, 1.5))
        depth_source = "mesh"
    else:
        depth = DEFAULT_DEPTH_M
        depth_source = "default"
    # box in the (aisle, up, front) basis, back to scene coordinates
    lo_b = np.array([a.min(), h.min(), z_front - depth])
    hi_b = np.array([a.max(), h.max(), z_front])
    B = np.stack([axis_a, up, front], 1)               # columns: basis vectors
    corners = np.array([[i, j, k] for i in (lo_b[0], hi_b[0]) for j in (lo_b[1], hi_b[1])
                        for k in (lo_b[2], hi_b[2])]) @ B.T
    bmin, bmax = corners.min(0), corners.max(0)
    return {"bbox": {"min": [round(float(c), 4) for c in bmin], "max": [round(float(c), 4) for c in bmax],
                     "size": [round(float(c), 4) for c in bmax - bmin]},
            "center": [round(float(c), 4) for c in (bmin + bmax) / 2],
            "width_cm": round(float(hi_b[0] - lo_b[0]) * 100, 1),
            "height_cm": round(float(hi_b[1] - lo_b[1]) * 100, 1),
            "depth_cm": round(depth * 100, 1),
            "front_source": front_source, "depth_source": depth_source,
            "ring_hits": int(ok.sum()), "inside_hits": int(hin.sum())}


# --------------------------------------------------------------------------- files
def load_cameras(scan_dir: Path) -> dict | None:
    p = Path(scan_dir) / CAMERAS_REL
    return json.loads(p.read_text()) if p.is_file() else None


def read_labels(scan_dir: Path) -> list[dict]:
    p = Path(scan_dir) / LABELS_REL
    if not p.is_file():
        return []
    doc = json.loads(p.read_text())
    return [l for l in doc.get("labels", []) if isinstance(l, dict)]


def lift_labels(scan_dir: Path, rec: dict, labels: list[dict]) -> list[dict]:
    """Labels with their current 3D box; a label that cannot be lifted keeps an `error`."""
    scan_dir = Path(scan_dir)
    cams = load_cameras(scan_dir)
    out = []
    if not labels:
        return out
    by_name = {k["name"]: k for k in (cams or {}).get("keyframes", [])}
    sc = _scene(scan_dir / "scene.usda") if cams else None
    W = scene_transform(rec)
    for l in labels:
        item = dict(l)
        kf = by_name.get(l.get("keyframe"))
        if cams is None or kf is None:
            item["error"] = "no camera for this keyframe"
        else:
            try:
                item.update(lift_box(sc, cams, kf, l["box_px"], W))
            except Exception as e:  # noqa: BLE001
                item["error"] = f"{type(e).__name__}: {e}"
        out.append(item)
    return out


# --------------------------------------------------------------------------- gaps vs labels
def front_rect(bmin, bmax) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of a box on the shelf front: x along the aisle, y up (scene frame)."""
    return float(bmin[0]), float(bmax[0]), float(bmin[1]), float(bmax[1])


def rect_iou(a, b) -> float:
    ix = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ix * iy
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def gaps_in_scene(doc: dict, rec: dict) -> dict:
    """gaps.json (usd frame) -> current scene frame, with sizes in cm."""
    W = scene_transform(rec)
    out = {}
    for key, m in (doc.get("methods") or {}).items():
        gaps = []
        for g in m.get("gaps", []):
            P = np.array([g["bmin"], g["bmax"]], float) @ W.T
            lo, hi = P.min(0), P.max(0)
            gaps.append({**{k: v for k, v in g.items() if k not in ("bmin", "bmax")},
                         "bbox": {"min": lo.round(4).tolist(), "max": hi.round(4).tolist(),
                                  "size": (hi - lo).round(4).tolist()},
                         "center": ((lo + hi) / 2).round(4).tolist(),
                         "width_cm": round(float(hi[0] - lo[0]) * 100, 1),
                         "height_cm": round(float(hi[1] - lo[1]) * 100, 1)})
        out[key] = {**{k: v for k, v in m.items() if k != "gaps"}, "gaps": gaps}
    return out


def unique_slots(labels: list[dict], iou: float = 0.3) -> list[dict]:
    """Labels of the same physical slot drawn on several keyframes -> one slot each."""
    labs = [l for l in labels if l.get("bbox")]
    parent = list(range(len(labs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    R = [front_rect(l["bbox"]["min"], l["bbox"]["max"]) for l in labs]
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            if rect_iou(R[i], R[j]) >= iou:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(labs)):
        groups.setdefault(find(i), []).append(i)
    slots = []
    for members in groups.values():
        lo = np.median([labs[i]["bbox"]["min"] for i in members], 0)
        hi = np.median([labs[i]["bbox"]["max"] for i in members], 0)
        kinds = sorted({labs[i].get("kind", "empty") for i in members})
        slots.append({"label_ids": [labs[i]["id"] for i in members], "kinds": kinds,
                      "rect": front_rect(lo, hi), "width_cm": round(float(hi[0] - lo[0]) * 100, 1),
                      "height_cm": round(float(hi[1] - lo[1]) * 100, 1)})
    return slots


def score_method(slots: list[dict], gaps: list[dict], iou: float = 0.25) -> dict:
    """Greedy one-to-one matching on the shelf front (IoU >= iou)."""
    pairs = []
    for i, s in enumerate(slots):
        for j, g in enumerate(gaps):
            v = rect_iou(s["rect"], front_rect(g["bbox"]["min"], g["bbox"]["max"]))
            if v >= iou:
                pairs.append((v, i, j))
    used_s, used_g, matches = set(), set(), []
    for v, i, j in sorted(pairs, reverse=True):
        if i in used_s or j in used_g:
            continue
        used_s.add(i); used_g.add(j)
        matches.append({"label_ids": slots[i]["label_ids"], "gap_id": gaps[j]["id"], "iou": round(v, 3),
                        "width_err_cm": round(gaps[j]["width_cm"] - slots[i]["width_cm"], 1),
                        "height_err_cm": round(gaps[j]["height_cm"] - slots[i]["height_cm"], 1)})
    n_s, n_g, n_m = len(slots), len(gaps), len(matches)
    werr = [abs(m["width_err_cm"]) for m in matches]
    return {"labelled_slots": n_s, "gaps": n_g, "found": n_m,
            "missed": n_s - n_m, "false_gaps": n_g - n_m,
            "recall": round(n_m / n_s, 3) if n_s else None,
            "precision": round(n_m / n_g, 3) if n_g else None,
            "mean_abs_width_err_cm": round(float(np.mean(werr)), 1) if werr else None,
            "matches": matches}
