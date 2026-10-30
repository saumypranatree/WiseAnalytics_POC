"""Stage 3 - lift 2D detections into 3D, fuse across views, write scene_objects.json.

CV environment (numpy + scikit-learn + scipy), no GPU needed:
    python scan_perception/lift_and_cluster.py --views out/views --out out/scene_objects.json

Per detection: select its pixels (SAM mask, or the box with a depth-band filter
that drops background), back-project them with the view's K and extrinsics.
Per class: voxel-downsample all lifted points, DBSCAN them, and keep a cluster
only if detections from at least --min-views different views mainly landed in it.
Lengths on the command line are metres; outputs are in stage units.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import aabb_iou, pixels_to_world  # noqa: E402
from views import DETECTIONS, load_manifest, write_json  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--views", required=True)
    p.add_argument("--detections", help=f"Defaults to <views>/{DETECTIONS}")
    p.add_argument("--out", default="scene_objects.json")
    p.add_argument("--scene-id", help="Defaults to the manifest's scene_id")
    p.add_argument("--min-score", type=float, default=0.0, help="Ignore detections below this score")
    p.add_argument("--max-depth", type=float, default=10.0, help="m; far pixels are noisy and get dropped")
    p.add_argument("--max-points-per-det", type=int, default=4000)
    p.add_argument("--box-shrink", type=float, default=0.1, help="Box mode: trim this fraction off each side")
    p.add_argument("--depth-band", type=float, default=0.15,
                   help="m; box mode keeps pixels within band + half the object's size of its centre depth")
    p.add_argument("--mask-erode", type=int, default=2, help="px; erode SAM masks to drop edge bleed")
    p.add_argument("--voxel", type=float, default=0.02, help="m; downsample before DBSCAN")
    p.add_argument("--eps", type=float, default=0.06, help="m; DBSCAN neighbourhood radius")
    p.add_argument("--min-samples", type=int, default=6, help="DBSCAN core-point threshold (in voxels)")
    p.add_argument("--min-views", type=int, default=2, help="Views that must agree on an object")
    p.add_argument("--trim-percentile", type=float, default=1.0, help="Per-axis outlier trim for the AABB")
    p.add_argument("--cross-class-iou", type=float, default=0.5,
                   help="Different-class objects overlapping more than this keep only the more confident one; 1 disables")
    p.add_argument("--debug-ply", help="Write clustered points, coloured per object, to this .ply")
    return p.parse_args(argv)


# ---------------------------------------------------------------------- lifting

def load_masks(views_dir, view_id, height):
    path = Path(views_dir) / "masks" / f"{view_id}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return np.unpackbits(data["masks"], axis=-1, count=int(data["width"])).astype(bool)


def select_pixels(det, depth, valid, K, mask, args, mpu):
    """(v, u) indices of the pixels that belong to the detected object."""
    H, W = depth.shape
    if mask is not None:
        if args.mask_erode > 0:
            mask = binary_erosion(mask, iterations=args.mask_erode)
        return np.nonzero(mask & valid)

    x0, y0, x1, y1 = det["box_xyxy"]
    dx, dy = (x1 - x0) * args.box_shrink, (y1 - y0) * args.box_shrink
    x0, x1 = int(max(0, x0 + dx)), int(min(W, x1 - dx))
    y0, y1 = int(max(0, y0 + dy)), int(min(H, y1 - dy))
    if x1 <= x0 or y1 <= y0:
        return np.array([], int), np.array([], int)
    sub, sub_valid = depth[y0:y1, x0:x1], valid[y0:y1, x0:x1]

    # Most of the box centre is the object, so its median depth is the object's depth.
    cy0, cy1 = int(0.35 * sub.shape[0]), max(int(0.65 * sub.shape[0]), int(0.35 * sub.shape[0]) + 1)
    cx0, cx1 = int(0.35 * sub.shape[1]), max(int(0.65 * sub.shape[1]), int(0.35 * sub.shape[1]) + 1)
    centre = sub[cy0:cy1, cx0:cx1][sub_valid[cy0:cy1, cx0:cx1]]
    if centre.size == 0:
        return np.array([], int), np.array([], int)
    d_obj = float(np.median(centre))
    obj_size = max(x1 - x0, y1 - y0) * d_obj / K[0, 0]  # rough metric size of the object
    tol = args.depth_band / mpu + 0.5 * obj_size
    keep = sub_valid & (np.abs(sub - d_obj) < tol)
    vv, uu = np.nonzero(keep)
    return vv + y0, uu + x0


def lift_all(views_dir, views, detections, args, mpu):
    """Returns per-class point arrays and a table describing each detection."""
    rng = np.random.default_rng(0)
    points = defaultdict(list)   # class -> [(n_i, 3)]
    owners = defaultdict(list)   # class -> [(n_i,)] detection index per point
    det_table = []               # index -> {view_id, label, score}

    for view in views:
        dets = [d for d in detections["views"].get(view.view_id, []) if d["score"] >= args.min_score]
        if not dets:
            continue
        depth = view.load_depth()
        valid = np.isfinite(depth) & (depth > 0) & (depth < args.max_depth / mpu)
        masks = load_masks(views_dir, view.view_id, depth.shape[0]) if detections.get("masks") else None

        for det in dets:
            mask = masks[det["mask_index"]] if masks is not None and "mask_index" in det else None
            vv, uu = select_pixels(det, depth, valid, view.K, mask, args, mpu)
            if len(vv) < 10:
                continue
            if len(vv) > args.max_points_per_det:
                pick = rng.choice(len(vv), args.max_points_per_det, replace=False)
                vv, uu = vv[pick], uu[pick]
            pts = pixels_to_world(uu, vv, depth[vv, uu], view.K, view.world_to_camera)
            idx = len(det_table)
            det_table.append({"view_id": view.view_id, "label": det["label"], "score": float(det["score"])})
            points[det["label"]].append(pts)
            owners[det["label"]].append(np.full(len(pts), idx))

    return ({c: np.concatenate(v) for c, v in points.items()},
            {c: np.concatenate(v) for c, v in owners.items()},
            det_table)


# ------------------------------------------------------------------- clustering

def dbscan_voxelized(pts, voxel, eps, min_samples):
    """DBSCAN on one representative per voxel; every point inherits its voxel's
    label. Gives uniform density (one camera close to an object would otherwise
    dominate) and keeps DBSCAN fast on millions of points."""
    keys = np.floor(pts / voxel).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts[first])
    return labels[inverse.reshape(-1)]


def objects_for_class(name, pts, owner, det_table, args, mpu):
    labels = dbscan_voxelized(pts, args.voxel / mpu, args.eps / mpu, args.min_samples)

    # Assign each detection to the cluster holding most of its points. Clusters
    # nobody "owns" are background bleed or fragments, and get dropped.
    supporters = defaultdict(list)
    for det_idx in np.unique(owner):
        lab = labels[owner == det_idx]
        lab = lab[lab >= 0]
        if lab.size:
            supporters[np.bincount(lab).argmax()].append(det_idx)

    objects = []
    for cluster_id, dets in supporters.items():
        n_views = len({det_table[d]["view_id"] for d in dets})
        if n_views < args.min_views:
            continue
        cpts = pts[labels == cluster_id]
        p = args.trim_percentile
        lo, hi = np.percentile(cpts, [p, 100 - p], axis=0)
        inliers = cpts[np.all((cpts >= lo) & (cpts <= hi), axis=1)]
        objects.append({
            "class_name": name,
            "confidence": float(np.mean([det_table[d]["score"] for d in dets])),
            "centroid": inliers.mean(axis=0),
            "min": lo, "max": hi,
            "n_views": n_views,
            "points": inliers,
        })
    return objects


def suppress_cross_class(objects, iou_thresh):
    """The same physical object can be found as e.g. 'monitor' and 'tv'."""
    if iou_thresh >= 1.0:
        return objects
    kept = []
    for obj in sorted(objects, key=lambda o: -o["confidence"]):
        if all(o["class_name"] == obj["class_name"] or
               aabb_iou(o["min"], o["max"], obj["min"], obj["max"]) <= iou_thresh for o in kept):
            kept.append(obj)
    return kept


# ------------------------------------------------------------------------ output

def to_schema(scene_id, objects):
    r = lambda a: [round(float(x), 4) for x in a]
    return {
        "scene_id": scene_id,
        "objects": [{
            "class_name": o["class_name"],
            "confidence": round(o["confidence"], 4),
            "position_world": dict(zip("xyz", r(o["centroid"]))),
            "bounding_box_3d": {"min": r(o["min"]), "max": r(o["max"])},
        } for o in sorted(objects, key=lambda o: (o["class_name"], -o["confidence"]))],
    }


def write_ply(path, objects):
    rng = np.random.default_rng(1)
    rows = []
    for o in objects:
        color = rng.integers(40, 255, 3)
        rows.append(np.hstack([o["points"], np.tile(color, (len(o["points"]), 1))]))
    data = np.vstack(rows) if rows else np.zeros((0, 6))
    header = ("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(data))
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.5f %.5f %.5f %d %d %d")


def main(argv=None):
    args = parse_args(argv)
    views_dir = Path(args.views)
    manifest, views = load_manifest(views_dir)
    mpu = float(manifest.get("meters_per_unit", 1.0))
    detections = json.loads(Path(args.detections or views_dir / DETECTIONS).read_text())

    points, owners, det_table = lift_all(views_dir, views, detections, args, mpu)
    print(f"[lift] lifted {len(det_table)} detections: " +
          ", ".join(f"{c}={len(p)} pts" for c, p in points.items()))

    objects = []
    for name in points:
        found = objects_for_class(name, points[name], owners[name], det_table, args, mpu)
        print(f"[lift] {name}: {len(found)} object(s)")
        objects += found
    objects = suppress_cross_class(objects, args.cross_class_iou)

    out = to_schema(args.scene_id or manifest.get("scene_id", views_dir.name), objects)
    write_json(args.out, out)
    for o in objects:
        print(f"[lift]   {o['class_name']:<16} conf {o['confidence']:.2f}  views {o['n_views']:<3} "
              f"centre {np.round(o['centroid'], 3)}  size {np.round(o['max'] - o['min'], 3)}")
    print(f"[lift] {len(objects)} objects -> {args.out}")
    if args.debug_ply:
        write_ply(args.debug_ply, objects)
        print(f"[lift] point clusters -> {args.debug_ply}")
    return out


if __name__ == "__main__":
    main()
