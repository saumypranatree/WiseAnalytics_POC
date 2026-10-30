#!/usr/bin/env python3
"""Synthetic end-to-end test for lift_cluster.py (no Isaac Sim, no GPU).

Scene (Y-up, meters): a floor, four walls, one "chair" box, and two "cup" boxes.
Eight OpenCV-convention cameras on a ring render analytic z-depth by ray-box
intersection. Detections come from projecting the true box corners. Some use exact
masks and some use bbox mode. One spurious single-view "cup" is added on a wall.
Run with pytest, or directly: python tests/test_lift_cluster.py
"""
import json
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lift_cluster as lc  # noqa: E402

W, H = 640, 480
FX = FY = 440.0
CX, CY = (W - 1) / 2, (H - 1) / 2
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])

# (class, min, max). Index = object id in the render's id buffer.
OBJECTS = [
    ("chair", (-0.25, 0.0, -0.25), (0.25, 0.9, 0.25)),
    ("cup", (0.80, 0.0, 0.50), (0.88, 0.10, 0.58)),
    ("cup", (-0.90, 0.0, -0.60), (-0.82, 0.10, -0.52)),
]
ROOM = [  # background geometry: floor + 4 walls
    ((-3.5, -0.05, -3.5), (3.5, 0.0, 3.5)),
    ((-3.6, 0.0, -3.5), (-3.5, 2.5, 3.5)), ((3.5, 0.0, -3.5), (3.6, 2.5, 3.5)),
    ((-3.5, 0.0, -3.6), (3.5, 2.5, -3.5)), ((-3.5, 0.0, 3.5), (3.5, 2.5, 3.6)),
]


def look_at_cv(eye, target, up=(0, 1, 0)):
    """OpenCV camera-to-world: columns are cam +X (right), +Y (down), +Z (forward) in world."""
    eye, target, up = map(np.asarray, (eye, target, up))
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)  # points down in a Y-up world
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def render(T_c2w):
    """Analytic z-depth + object-id buffer. The ray direction for pixel (u, v) is
    R @ ((u-cx)/fx, (v-cy)/fy, 1), whose camera-z component is 1, so the slab
    parameter t equals z-depth directly. Also returns each object's unoccluded pixel count."""
    v, u = np.mgrid[0:H, 0:W]
    dc = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u, float)], -1).reshape(-1, 3)
    d = dc @ T_c2w[:3, :3].T
    o = T_c2w[:3, 3]
    depth = np.full(len(d), np.inf)
    ids = np.full(len(d), -1)
    alone = []
    boxes = [(b0, b1) for _, b0, b1 in OBJECTS] + ROOM
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        for i, (b0, b1) in enumerate(boxes):
            t1, t2 = (np.asarray(b0) - o) * inv, (np.asarray(b1) - o) * inv
            tmin = np.nanmax(np.minimum(t1, t2), 1)
            tmax = np.nanmin(np.maximum(t1, t2), 1)
            hit = (tmax >= tmin) & (tmin > 1e-6)
            if i < len(OBJECTS):
                alone.append(int(hit.sum()))
            closer = hit & (tmin < depth)
            depth[closer], ids[closer] = tmin[closer], i
    depth[~np.isfinite(depth)] = 0.0
    return depth.reshape(H, W).astype(np.float32), ids.reshape(H, W), alone


def corners(b0, b1):
    return np.array([[x, y, z] for x in (b0[0], b1[0]) for y in (b0[1], b1[1]) for z in (b0[2], b1[2])])


def make_scene(root, n_cams=8):
    rd, dd = os.path.join(root, "renders"), os.path.join(root, "detections")
    for sub in ("renders/rgb", "renders/depth", "detections/masks"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    rng = np.random.default_rng(0)
    frames, det_frames = [], []
    for k in range(n_cams):
        a = 2 * np.pi * k / n_cams + 0.2
        T_c2w = look_at_cv((2.5 * np.cos(a), 1.4, 2.5 * np.sin(a)), (0, 0.3, 0))
        T_w2c = np.linalg.inv(T_c2w)
        depth, ids, alone = render(T_c2w)
        fid = f"{k:04d}"
        np.save(os.path.join(rd, "depth", fid + ".npy"), depth)
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(os.path.join(rd, "rgb", fid + ".png"))
        frames.append({"frame_id": fid, "rgb": f"rgb/{fid}.png", "depth": f"depth/{fid}.npy",
                       "K": K.tolist(), "T_world_to_cam": T_w2c.tolist(), "T_cam_to_world": T_c2w.tolist()})
        dets = []
        for i, (cls, b0, b1) in enumerate(OBJECTS):
            vis = int((ids == i).sum())
            if alone[i] == 0 or vis < 30 or vis / alone[i] < 0.7:
                continue  # out of view or heavily occluded: a detector would miss or crop it
            u, v, z = lc.project_points(corners(b0, b1), K, T_w2c)
            assert (z > 0).all()
            bbox = [max(0, u.min()), max(0, v.min()), min(W - 1, u.max()), min(H - 1, v.max())]
            did = f"{fid}_{len(dets):02d}"
            mask = None
            if (k + i) % 2 == 0:  # half exact masks, half bbox-only
                mask = f"masks/{did}.png"
                Image.fromarray(((ids == i) * 255).astype(np.uint8)).save(os.path.join(dd, mask))
            dets.append({"det_id": did, "class_name": cls, "score": round(float(rng.uniform(0.5, 0.8)), 3),
                         "bbox_xyxy": [float(x) for x in bbox], "mask": mask})
        if k == 3:  # spurious single-view false positive on background
            fp = [40.0, 30.0, 110.0, 90.0]
            assert (ids[30:91, 40:111] >= len(OBJECTS)).all(), "FP box must not touch real objects"
            dets.append({"det_id": f"{fid}_{len(dets):02d}", "class_name": "cup", "score": 0.9,
                         "bbox_xyxy": fp, "mask": None})
        det_frames.append({"frame_id": fid, "detections": dets})
    meta = {"scene_id": "synthetic", "scene_usd": "none", "up_axis": "Y", "meters_per_unit": 1.0,
            "image_width": W, "image_height": H, "depth_type": "distance_to_image_plane",
            "camera_convention": "opencv",
            "scene_bounds_world": {"min": [-3.6, -0.05, -3.6], "max": [3.6, 2.5, 3.6]}, "frames": frames}
    with open(os.path.join(rd, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(dd, "detections.json"), "w") as f:
        json.dump({"detector": "synthetic", "segmenter": None, "prompts": ["chair", "cup"],
                   "box_threshold": 0.3, "text_threshold": 0.25, "frames": det_frames}, f)
    return rd, dd


# --------------------------------------------------------------------------- #
def test_backproject_roundtrip():
    T_c2w = look_at_cv((2.0, 1.5, -1.0), (0.0, 0.5, 0.0))
    T_w2c = np.linalg.inv(T_c2w)
    u0, v0, z0 = 100, 50, 2.0
    # Expected world point, computed independently of backproject().
    p_cam = np.array([(u0 - CX) * z0 / FX, (v0 - CY) * z0 / FY, z0, 1.0])
    p_world = (T_c2w @ p_cam)[:3]
    depth = np.zeros((H, W), np.float32)
    depth[v0, u0] = z0
    depth[10, 10] = -1.0   # invalid: must be skipped
    depth[20, 20] = 50.0   # beyond max_depth: must be skipped
    pc = lc.backproject(depth, K, None, max_depth=20.0)
    assert pc.shape == (1, 3)
    pw = lc.cam_to_world(pc, T_c2w)
    assert np.allclose(pw[0], p_world, atol=1e-5)
    u, v, z = lc.project_points(pw, K, T_w2c)
    assert np.allclose([u[0], v[0], z[0]], [u0, v0, z0], atol=1e-4)
    # The camera looks at the target: the target projects to the principal point.
    u, v, _ = lc.project_points(np.array([[0.0, 0.5, 0.0]]), K, T_w2c)
    assert np.allclose([u[0], v[0]], [CX, CY], atol=1e-6)
    # Y-up world point above the target must land above the center (smaller v).
    _, v, _ = lc.project_points(np.array([[0.0, 1.0, 0.0]]), K, T_w2c)
    assert v[0] < CY


def test_filters_numpy_fallback():
    """Voxel inverse map and the sklearn SOR path (open3d disabled)."""
    rng = np.random.default_rng(1)
    pts = np.vstack([rng.uniform(0, 0.2, (500, 3)), [[5.0, 5.0, 5.0]]])  # one far outlier
    cent, inv, cnt = lc.voxel_downsample(pts, 0.05)
    assert cnt.sum() == len(pts) and inv.shape == (len(pts),)
    assert np.allclose(cent[inv[-1]], [5, 5, 5])
    saved, lc.o3d = lc.o3d, None
    try:
        keep = lc.statistical_outlier_mask(pts, 10, 2.0)
    finally:
        lc.o3d = saved
    assert not keep[-1] and keep[:-1].mean() > 0.95
    m = lc.erode_mask(np.ones((5, 5), bool), 1)
    assert m.sum() == 9 and m[2, 2]


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        rd, dd = make_scene(tmp)
        out = os.path.join(tmp, "scene_objects.json")
        ply = os.path.join(tmp, "clusters.ply")
        assert lc.main(["--renders", rd, "--detections", dd, "--out", out, "--save-ply", ply, "-q"]) == 0
        res = json.load(open(out))
        assert res["scene_id"] == "synthetic"
        for o in res["objects"]:  # exact contract schema without --extended
            assert set(o) == {"class_name", "confidence", "position_world", "bounding_box_3d"}
        assert os.path.getsize(ply) > 0

        # Extended run with visibility weighting, to inspect support and check the extra fields.
        out2 = os.path.join(tmp, "ext.json")
        lc.main(["--renders", rd, "--detections", dd, "--out", out2, "--extended",
                 "--visibility-weighting", "--cross-class-nms", "-q"])
        ext = json.load(open(out2))

    for name, r in (("default", res), ("extended", ext)):
        objs = r["objects"]
        by_cls = {}
        for o in objs:
            by_cls.setdefault(o["class_name"], []).append(o)
        counts = {c: len(v) for c, v in by_cls.items()}
        assert counts == {"chair": 1, "cup": 2}, f"{name}: got {counts}"  # FP rejected
        assert [o["class_name"] for o in objs] == sorted(o["class_name"] for o in objs)
        for cls, b0, b1 in OBJECTS:
            gc = (np.add(b0, b1)) / 2
            o = min(by_cls[cls], key=lambda o: np.linalg.norm(
                [o["position_world"][k] - gc[i] for i, k in enumerate("xyz")]))
            err = max(np.abs(np.subtract(o["bounding_box_3d"]["min"], b0)).max(),
                      np.abs(np.subtract(o["bounding_box_3d"]["max"], b1)).max())
            print(f"  [{name}] {cls:5s} conf={o['confidence']:.3f} max AABB err={err * 100:.1f} cm"
                  + (f" views={o['num_views']} pts={o['num_points']}" if "num_views" in o else ""))
            assert err < 0.04, f"{name} {cls}: AABB error {err:.3f} m"
            assert 0 < o["confidence"] <= 1
    for o in ext["objects"]:
        assert o["num_views"] >= 2 and o["num_points"] > 0 and o["source_detections"]
        assert all("_" in s for s in o["source_detections"])


if __name__ == "__main__":
    for t in (test_backproject_roundtrip, test_filters_numpy_fallback, test_end_to_end):
        t()
        print(f"PASS {t.__name__}")
    print("all tests passed")
