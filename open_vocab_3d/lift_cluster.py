#!/usr/bin/env python3
"""Stage 3: lift 2D open-vocabulary detections into 3D objects.

Inputs   (see CONTRACT.md):  <renders>/meta.json + depth/*.npy
                             <detections>/detections.json (+ masks/*.png)
Output:                      scene_objects.json (world frame, meters)

Pipeline
--------
1. Per detection, select pixels: eroded SAM mask if present, else a shrunken
   bbox. Then keep only depths inside a robust band around the median depth
   (removes background and floor seen through the box or bleeding masks).
2. Back-project the selected pixels with the pinhole model. Transform them to
   world space and tag each point with its detection index.
3. Per class: voxel-downsample, run statistical outlier removal, then DBSCAN.
   One cluster = one physical instance.
4. A detection *supports* a cluster if at least --support-frac of its lifted
   points fall in that cluster. num_views = distinct frames among supporters.
5. Filter clusters on min points, min views and max extent.
6. confidence = mean(score of supporting detections)
                [* min(1, supporting_views / visible_views) with --visibility-weighting]
   AABB = per-axis percentile clip of the cluster points. Centroid = mean.
7. Optional cross-class NMS on 3D AABB IoU.

Dependencies: numpy, scikit-learn, pillow. open3d is optional and is only used
for statistical outlier removal (numpy/sklearn fallback otherwise). Voxelization
is always numpy, because we need the point->voxel inverse map to carry
detection ids through clustering.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

try:  # optional accelerator; wheels often lag new Python releases
    import open3d as o3d  # type: ignore
except Exception:  # pragma: no cover - depends on environment
    o3d = None

log = logging.getLogger("lift_cluster")

# MAD -> standard deviation for a Gaussian.
_MAD_TO_SIGMA = 1.4826


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def backproject(depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None,
                max_depth: float = np.inf) -> np.ndarray:
    """Back-project masked depth pixels to camera-frame points (OpenCV frame).

    Pinhole model with pixel centers at integer (u, v), u = column, v = row:
        u = fx * x / z + cx   =>   x = (u - cx) * z / fx
        v = fy * y / z + cy   =>   y = (v - cy) * z / fy
    `depth` is z-depth (distance to the image plane), so z = depth[v, u] directly.
    If depth were ray length r, we would first need z = r / ||((u-cx)/fx, (v-cy)/fy, 1)||.
    Pixels with z <= 0 (no hit), non-finite z, or z > max_depth are skipped.
    Returns an (N, 3) float64 array in the camera frame (+X right, +Y down, +Z forward).
    """
    depth = np.asarray(depth)
    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    if mask is not None:
        valid &= mask.astype(bool)
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def cam_to_world(points: np.ndarray, T_cam_to_world: np.ndarray) -> np.ndarray:
    """Apply a 4x4 column-vector transform: p_world = R @ p_cam + t."""
    T = np.asarray(T_cam_to_world, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def project_points(points_world: np.ndarray, K: np.ndarray, T_world_to_cam: np.ndarray):
    """World points -> (u, v, z_cam). u, v are float pixels (centers at integers)."""
    T = np.asarray(T_world_to_cam, dtype=np.float64)
    pc = points_world @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return u, v, z


# --------------------------------------------------------------------------- #
# Pixel selection
# --------------------------------------------------------------------------- #
def erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Binary erosion with a 3x3 cross. Pixels outside the image count as background."""
    m = mask.astype(bool).copy()
    for _ in range(max(0, int(iterations))):
        p = np.pad(m, 1, constant_values=False)
        m = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m


def bbox_to_mask(bbox_xyxy, shape, shrink: float = 0.0) -> np.ndarray:
    """Boolean mask of pixels whose *centers* lie inside the bbox after shrinking
    it by `shrink` x (width or height) on each side."""
    H, W = shape
    x0, y0, x1, y1 = map(float, bbox_xyxy)
    dw, dh = (x1 - x0) * shrink, (y1 - y0) * shrink
    x0, x1, y0, y1 = x0 + dw, x1 - dw, y0 + dh, y1 - dh
    u0, u1 = max(0, int(np.ceil(x0))), min(W - 1, int(np.floor(x1)))
    v0, v1 = max(0, int(np.ceil(y0))), min(H - 1, int(np.floor(y1)))
    m = np.zeros(shape, dtype=bool)
    if u1 >= u0 and v1 >= v0:
        m[v0:v1 + 1, u0:u1 + 1] = True
    return m


def depth_band_filter(depth: np.ndarray, sel: np.ndarray, k: float, floor: float) -> np.ndarray:
    """Keep selected pixels whose depth is within max(k * sigma_MAD, floor) of the
    median depth inside the selection. The median is the dominant surface, which is
    the object for a reasonably tight box or mask."""
    z = depth[sel]
    if z.size == 0:
        return sel
    med = np.median(z)
    sigma = _MAD_TO_SIGMA * np.median(np.abs(z - med))
    band = max(k * sigma, floor)
    return sel & (np.abs(depth - med) < band)


def select_pixels(depth: np.ndarray, bbox_xyxy, mask: np.ndarray | None, *,
                  bbox_shrink: float, mask_erode: int, stride: int,
                  band_k: float, band_floor: float, max_depth: float) -> np.ndarray:
    """Pick the pixels of one detection that most likely belong to the object."""
    H, W = depth.shape
    if mask is not None:
        # Masks bleed at their edges, so erode them. Also clip to the raw bbox as a sanity check.
        sel = erode_mask(mask, mask_erode) & bbox_to_mask(bbox_xyxy, (H, W), 0.0)
    else:
        # A bbox contains background, so shrink it toward the object.
        sel = bbox_to_mask(bbox_xyxy, (H, W), bbox_shrink)
    sel &= np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    sel = depth_band_filter(depth, sel, band_k, band_floor)
    if stride > 1:
        grid = np.zeros((H, W), dtype=bool)
        grid[::stride, ::stride] = True
        sel &= grid
    return sel


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    frame_id: str
    K: np.ndarray
    T_cam_to_world: np.ndarray
    T_world_to_cam: np.ndarray
    depth_path: str
    _depth: np.ndarray | None = None

    @property
    def depth(self) -> np.ndarray:
        if self._depth is None:
            d = np.load(self.depth_path).astype(np.float32)
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)  # contract: no hit = 0
            d[d < 0] = 0.0
            self._depth = d
        return self._depth


@dataclass
class Detection:
    idx: int
    det_id: str
    frame_id: str
    class_name: str
    score: float
    bbox_xyxy: list
    mask_path: str | None
    n_points: int = 0
    centroid: np.ndarray | None = None


def load_renders(renders_dir: str):
    with open(os.path.join(renders_dir, "meta.json")) as f:
        meta = json.load(f)
    frames = {}
    for fr in meta["frames"]:
        T_c2w = np.asarray(fr["T_cam_to_world"], dtype=np.float64)
        T_w2c = (np.asarray(fr["T_world_to_cam"], dtype=np.float64)
                 if fr.get("T_world_to_cam") is not None else np.linalg.inv(T_c2w))
        frames[str(fr["frame_id"])] = Frame(
            frame_id=str(fr["frame_id"]),
            K=np.asarray(fr["K"], dtype=np.float64),
            T_cam_to_world=T_c2w, T_world_to_cam=T_w2c,
            depth_path=os.path.join(renders_dir, fr["depth"]))
    return meta, frames


def load_detections(det_dir: str):
    with open(os.path.join(det_dir, "detections.json")) as f:
        data = json.load(f)
    dets = []
    for fr in data.get("frames", []):
        for d in fr.get("detections", []):
            mp = d.get("mask")
            dets.append(Detection(
                idx=len(dets), det_id=str(d["det_id"]), frame_id=str(fr["frame_id"]),
                class_name=str(d["class_name"]), score=float(d["score"]),
                bbox_xyxy=list(d["bbox_xyxy"]),
                mask_path=os.path.join(det_dir, mp) if mp else None))
    return data, dets


def _load_mask(path: str, shape) -> np.ndarray:
    img = Image.open(path).convert("L")
    if (img.height, img.width) != tuple(shape):
        img = img.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(img) > 127


# --------------------------------------------------------------------------- #
# Lifting
# --------------------------------------------------------------------------- #
def lift_detections(frames: dict, dets: list, p) -> dict:
    """Return {class_name: (points Nx3 world, det_idx N)} over all views."""
    per_class: dict[str, tuple[list, list]] = {}
    for d in dets:
        fr = frames.get(d.frame_id)
        if fr is None:
            log.warning("detection %s references unknown frame %s; skipped", d.det_id, d.frame_id)
            continue
        depth = fr.depth
        mask = None
        if d.mask_path and os.path.exists(d.mask_path):
            mask = _load_mask(d.mask_path, depth.shape)
        sel = select_pixels(depth, d.bbox_xyxy, mask, bbox_shrink=p.bbox_shrink,
                            mask_erode=p.mask_erode, stride=p.pixel_stride,
                            band_k=p.depth_band_k, band_floor=p.depth_band_floor,
                            max_depth=p.max_depth)
        if sel.sum() < p.min_det_pixels:
            continue
        pts = cam_to_world(backproject(depth, fr.K, sel, p.max_depth), fr.T_cam_to_world)
        limit = class_extent_limit(p, d.class_name)
        if limit is not None and len(pts):
            lo, hi = np.percentile(pts, [5, 95], axis=0)
            if (hi - lo).max() > limit:   # "whole structure" detection: wrong scale for the class
                log.debug("drop detection %s (%s): 3D extent %.2f m > %.2f",
                          d.det_id, d.class_name, (hi - lo).max(), limit)
                p._det_extent_drops[d.class_name] = p._det_extent_drops.get(d.class_name, 0) + 1
                continue
        d.n_points = len(pts)
        d.centroid = np.median(pts, axis=0) if len(pts) else None
        P, I = per_class.setdefault(d.class_name, ([], []))
        P.append(pts)
        I.append(np.full(len(pts), d.idx, dtype=np.int64))
    return {c: (np.concatenate(P), np.concatenate(I)) for c, (P, I) in per_class.items()}


# --------------------------------------------------------------------------- #
# Point cloud filtering + clustering
# --------------------------------------------------------------------------- #
def voxel_downsample(points: np.ndarray, voxel: float):
    """Returns (voxel centroids Vx3, inverse N->V, counts V). Centroid = mean of
    member points, so precision is not quantized to the voxel grid."""
    if voxel <= 0:
        return points.copy(), np.arange(len(points)), np.ones(len(points), dtype=np.int64)
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    cent = np.stack([np.bincount(inverse, weights=points[:, a], minlength=len(counts))
                     for a in range(3)], axis=1) / counts[:, None]
    return cent, inverse, counts


def statistical_outlier_mask(points: np.ndarray, nb_neighbors: int = 20,
                             std_ratio: float = 2.0) -> np.ndarray:
    """Keep points whose mean k-NN distance <= global mean + std_ratio * std."""
    n = len(points)
    if n <= nb_neighbors or nb_neighbors <= 0:
        return np.ones(n, dtype=bool)
    if o3d is not None:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        _, ind = pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        keep = np.zeros(n, dtype=bool)
        keep[np.asarray(ind, dtype=np.int64)] = True
        return keep
    dist, _ = NearestNeighbors(n_neighbors=nb_neighbors + 1).fit(points).kneighbors(points)
    md = dist[:, 1:].mean(axis=1)  # column 0 is the point itself
    return md <= md.mean() + std_ratio * md.std()


def parse_class_floats(spec) -> dict:
    """'{"window": 2.5}' (JSON), 'window=2.5,decorative plate=0.5', or a path to a
    .json/.csv file with those contents -> {class_name: float}."""
    if not spec:
        return {}
    if isinstance(spec, dict):
        return {str(k): float(v) for k, v in spec.items()}
    text = str(spec)
    if os.path.isfile(text):
        with open(text) as f:
            text = f.read()
    text = text.strip()
    if text.startswith("{"):
        return {str(k): float(v) for k, v in json.loads(text).items()}
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line and ":" not in line:          # CSV row: "class,value"
            k, v = line.rsplit(",", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:                           # header row
                pass
            continue
        for item in line.replace(";", ",").split(","):
            if item.strip():
                k, v = item.rsplit("=", 1) if "=" in item else item.rsplit(":", 1)
                out[k.strip()] = float(v)
    return out


def class_extent_limit(p, class_name: str):
    """Per-class max AABB side (meters) from --max-extent-per-class, or None."""
    lim = getattr(p, "_extent_limits", None)
    if lim is None:
        lim = p._extent_limits = parse_class_floats(getattr(p, "max_extent_per_class", None))
    return lim.get(class_name)


def cluster_detections(det_idx: np.ndarray, dets: list, p) -> np.ndarray:
    """Instance grouping for small, possibly touching objects (e.g. plates hung edge to
    edge on a wall, whose surface points form one connected blob): DBSCAN over the 3D
    centroids of the *detections* (eps = --det-eps), then every point inherits its
    detection's group. Returns a label per point."""
    ids = np.unique(det_idx)
    cents = np.stack([dets[i].centroid for i in ids])
    lab = DBSCAN(eps=p.det_eps, min_samples=1).fit_predict(cents)
    lut = dict(zip(ids.tolist(), lab.tolist()))
    return np.fromiter((lut[i] for i in det_idx), dtype=np.int64, count=len(det_idx))


def cluster_points(points: np.ndarray, p) -> np.ndarray:
    """Voxelize -> SOR -> DBSCAN on voxel centroids. Returns a label per *original*
    point (-1 = noise or outlier)."""
    cent, inverse, _ = voxel_downsample(points, p.voxel)
    keep = statistical_outlier_mask(cent, p.sor_neighbors, p.sor_std) if p.sor else np.ones(len(cent), bool)
    vlabels = np.full(len(cent), -1, dtype=np.int64)
    if keep.sum() >= p.min_samples:
        vlabels[keep] = DBSCAN(eps=p.eps, min_samples=p.min_samples).fit_predict(cent[keep])
    return vlabels[inverse]


@dataclass
class Cluster:
    class_name: str
    points: np.ndarray
    supporters: list = field(default_factory=list)  # Detection objects
    confidence: float = 0.0
    centroid: np.ndarray | None = None
    aabb_min: np.ndarray | None = None
    aabb_max: np.ndarray | None = None

    @property
    def num_views(self) -> int:
        return len({d.frame_id for d in self.supporters})


def robust_aabb(points: np.ndarray, pct: float):
    """Per-axis [pct, 100-pct] percentile box, which resists stray points."""
    if pct <= 0:
        return points.min(0), points.max(0)
    return np.percentile(points, pct, axis=0), np.percentile(points, 100 - pct, axis=0)


def resolve_clusters(class_name, points, det_idx, labels, dets, p) -> list[Cluster]:
    out = []
    for lab in np.unique(labels):
        if lab < 0:
            continue
        inside = labels == lab
        c = Cluster(class_name, points[inside])
        # A detection supports this cluster if >= support_frac of ITS lifted points land here.
        ids, cnt = np.unique(det_idx[inside], return_counts=True)
        c.supporters = [dets[i] for i, n in zip(ids, cnt)
                        if dets[i].n_points and n / dets[i].n_points >= p.support_frac]
        c.aabb_min, c.aabb_max = robust_aabb(c.points, p.aabb_percentile)
        c.centroid = c.points.mean(axis=0)
        out.append(c)
    return out


def visible_frames(c: Cluster, frames: dict, tol_extra: float = 0.05) -> set:
    """Frames in which the cluster centroid projects inside the image and is not
    occluded. The centroid lies inside the object, behind its visible surface by up to
    half the box diagonal, so we accept z_cam - depth <= half_diag + tol_extra."""
    half_diag = 0.5 * float(np.linalg.norm(c.aabb_max - c.aabb_min))
    vis = set()
    for fid, fr in frames.items():
        u, v, z = project_points(c.centroid[None], fr.K, fr.T_world_to_cam)
        if not (z[0] > 0 and np.isfinite(u[0]) and np.isfinite(v[0])):
            continue
        ui, vi = int(round(u[0])), int(round(v[0]))
        H, W = fr.depth.shape
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        d = fr.depth[vi, ui]
        if d <= 0 or z[0] - d <= half_diag + tol_extra:  # no hit behind = see-through, call visible
            vis.add(fid)
    return vis


def aabb_iou(amin, amax, bmin, bmax) -> float:
    inter = np.clip(np.minimum(amax, bmax) - np.maximum(amin, bmin), 0, None).prod()
    va, vb = np.prod(amax - amin), np.prod(bmax - bmin)
    union = va + vb - inter
    return float(inter / union) if union > 0 else 0.0


def cross_class_nms(clusters: list[Cluster], thr: float) -> list[Cluster]:
    """Greedy: across *different* classes, drop the lower-confidence cluster when IoU > thr."""
    order = sorted(clusters, key=lambda c: -c.confidence)
    kept: list[Cluster] = []
    for c in order:
        if all(k.class_name == c.class_name or
               aabb_iou(c.aabb_min, c.aabb_max, k.aabb_min, k.aabb_max) <= thr for k in kept):
            kept.append(c)
    return kept


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(p) -> tuple[dict, list[Cluster]]:
    meta, frames = load_renders(p.renders)
    _, dets = load_detections(p.detections)
    cls_min = parse_class_floats(getattr(p, "min_score_per_class", None))
    if p.min_score > 0 or cls_min:
        dets = [d for d in dets if d.score >= cls_min.get(d.class_name, p.min_score)]
        for i, d in enumerate(dets):
            d.idx = i
    log.info("%d frames, %d detections", len(frames), len(dets))

    p._det_extent_drops = {}
    min_views_cls = {k: int(v) for k, v in
                     parse_class_floats(getattr(p, "min_views_per_class", None)).items()}
    det_cluster_classes = set(_split_list(getattr(p, "det_cluster_classes", None)))
    lifted = lift_detections(frames, dets, p)
    for cls, n in sorted(p._det_extent_drops.items()):
        log.info("%s: dropped %d detections larger than %.2f m in 3D",
                 cls, n, class_extent_limit(p, cls))
    clusters: list[Cluster] = []
    for cls, (pts, didx) in sorted(lifted.items()):
        if cls in det_cluster_classes:
            labels = cluster_detections(didx, dets, p)
        else:
            labels = cluster_points(pts, p)
        raw = resolve_clusters(cls, pts, didx, labels, dets, p)
        for c in raw:
            ext = c.aabb_max - c.aabb_min
            why = None
            if len(c.points) < p.min_points:
                why = f"{len(c.points)} pts < {p.min_points}"
            elif c.num_views < min_views_cls.get(cls, p.min_views):
                why = f"{c.num_views} views < {min_views_cls.get(cls, p.min_views)}"
            elif ext.max() > p.max_extent:
                why = f"extent {ext.max():.2f} m > {p.max_extent}"
            elif (class_extent_limit(p, cls) is not None
                  and ext.max() > class_extent_limit(p, cls)):
                why = f"extent {ext.max():.2f} m > {class_extent_limit(p, cls)} (class limit)"
            if why:
                log.info("drop %s cluster @ %s: %s", cls, np.round(c.centroid, 2), why)
                continue
            # confidence = mean score of supporting detections
            c.confidence = float(np.mean([d.score for d in c.supporters]))
            if p.visibility_weighting:
                sup = {d.frame_id for d in c.supporters}
                vis = visible_frames(c, frames) | sup
                c.confidence *= len(sup) / len(vis)
            clusters.append(c)
        log.info("%s: %d points -> %d clusters, %d kept", cls, len(pts), len(raw),
                 sum(c.class_name == cls for c in clusters))

    if p.cross_class_nms:
        clusters = cross_class_nms(clusters, p.nms_iou)
    clusters.sort(key=lambda c: (c.class_name, -c.confidence))

    def r(x):
        return round(float(x), 4)

    objects = []
    for c in clusters:
        o = {
            "class_name": c.class_name,
            "confidence": r(c.confidence),
            "position_world": {"x": r(c.centroid[0]), "y": r(c.centroid[1]), "z": r(c.centroid[2])},
            "bounding_box_3d": {"min": [r(x) for x in c.aabb_min], "max": [r(x) for x in c.aabb_max]},
        }
        if p.extended:
            o["num_views"] = c.num_views
            o["num_points"] = int(len(c.points))
            o["source_detections"] = sorted(d.det_id for d in c.supporters)
        objects.append(o)
    result = {"scene_id": p.scene_id or meta.get("scene_id", "scene"), "objects": objects}
    return result, clusters


def _split_list(spec) -> list:
    if not spec:
        return []
    if isinstance(spec, (list, tuple)):
        return [str(x).strip() for x in spec if str(x).strip()]
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def write_ply(path: str, clusters: list[Cluster]) -> None:
    """ASCII PLY with one color per cluster (golden-ratio hue spread)."""
    import colorsys
    rows = []
    for i, c in enumerate(clusters):
        rgb = np.array(colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.8, 0.95)) * 255
        col = " ".join(str(int(x)) for x in rgb)
        rows.extend(f"{x:.4f} {y:.4f} {z:.4f} {col}" for x, y, z in c.points)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {len(rows)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        f.write("\n".join(rows) + ("\n" if rows else ""))


def build_parser() -> argparse.ArgumentParser:
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a.add_argument("--renders", required=True, help="stage-1 output dir (meta.json)")
    a.add_argument("--detections", required=True, help="stage-2 output dir (detections.json)")
    a.add_argument("--out", default="./out/scene_objects.json")
    a.add_argument("--scene-id", default=None, help="default: meta.json scene_id")
    g = a.add_argument_group("pixel selection")
    g.add_argument("--bbox-shrink", type=float, default=0.1, help="fraction shrunk per bbox side (no mask)")
    g.add_argument("--mask-erode", type=int, default=2, help="mask erosion, pixels")
    g.add_argument("--pixel-stride", type=int, default=2)
    g.add_argument("--depth-band-k", type=float, default=2.5, help="band = max(k*1.4826*MAD, floor)")
    g.add_argument("--depth-band-floor", type=float, default=0.15, help="meters")
    g.add_argument("--max-depth", type=float, default=20.0, help="meters")
    g.add_argument("--min-score", type=float, default=0.0, help="ignore detections below this score")
    g.add_argument("--min-score-per-class", default=None,
                   help="per-class override of --min-score, same formats as "
                        "--max-extent-per-class (e.g. 'window=0.4')")
    g.add_argument("--min-det-pixels", type=int, default=10)
    g = a.add_argument_group("clustering")
    g.add_argument("--voxel", type=float, default=0.02, help="voxel size, meters (0 = off)")
    g.add_argument("--no-sor", dest="sor", action="store_false", help="disable statistical outlier removal")
    g.add_argument("--sor-neighbors", type=int, default=20)
    g.add_argument("--sor-std", type=float, default=2.0)
    g.add_argument("--eps", type=float, default=0.06, help="DBSCAN radius, meters")
    g.add_argument("--min-samples", type=int, default=8, help="DBSCAN min samples (voxels)")
    g.add_argument("--support-frac", type=float, default=0.25,
                   help="a detection supports a cluster if >= this fraction of its points fall in it")
    g = a.add_argument_group("filtering / output")
    g.add_argument("--min-points", type=int, default=50)
    g.add_argument("--min-views", type=int, default=2)
    g.add_argument("--min-views-per-class", default=None,
                   help="per-class override of --min-views, same formats as "
                        "--max-extent-per-class (e.g. 'table=6,bench seat=6')")
    g.add_argument("--max-extent", type=float, default=4.0, help="max AABB side, meters")
    g.add_argument("--max-extent-per-class", default=None,
                   help="per-class max AABB side, meters: JSON '{\"window\": 2.5}', "
                        "'window=2.5,decorative plate=0.5', or a .json/.csv file. Applied to "
                        "clusters AND to single detections (5-95%% extent of their lifted "
                        "points), which removes whole-structure boxes before clustering")
    g.add_argument("--det-cluster-classes", default=None,
                   help="comma-separated classes whose instances are grouped by DBSCAN over "
                        "per-detection 3D centroids instead of over points (separates small "
                        "touching objects, e.g. plates hung edge to edge)")
    g.add_argument("--det-eps", type=float, default=0.08,
                   help="DBSCAN radius (meters) for --det-cluster-classes")
    g.add_argument("--aabb-percentile", type=float, default=1.0)
    g.add_argument("--visibility-weighting", action="store_true",
                   help="scale confidence by supporting_views / visible_views")
    g.add_argument("--cross-class-nms", action="store_true")
    g.add_argument("--nms-iou", type=float, default=0.5)
    g.add_argument("--extended", action="store_true", help="add num_views, num_points, source_detections")
    g.add_argument("--save-ply", default=None, help="write colored per-cluster point cloud")
    g.add_argument("-q", "--quiet", action="store_true")
    return a


def main(argv=None) -> int:
    p = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if p.quiet else logging.INFO,
                        format="[lift_cluster] %(message)s")
    result, clusters = run(p)
    os.makedirs(os.path.dirname(os.path.abspath(p.out)), exist_ok=True)
    with open(p.out, "w") as f:
        json.dump(result, f, indent=2)
    if p.save_ply:
        write_ply(p.save_ply, clusters)
    log.info("wrote %d objects -> %s", len(result["objects"]), p.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
