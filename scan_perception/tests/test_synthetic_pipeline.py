"""End-to-end check of the lift/cluster stage without Isaac Sim or a detector.

Ray-casts a Y-up room containing known boxes into the exact on-disk format that
render_views.py writes (manifest, rgb, depth), fakes detections from the
object-id buffer (what a perfect detector would output), runs
lift_and_cluster.main, and compares the recovered AABBs with the truth.

    python -m pytest scan_perception/tests -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lift_and_cluster  # noqa: E402
from geometry import (USD_TO_CV, intrinsics_from_usd, invert_rigid, look_at_usd,  # noqa: E402
                      pixels_to_world, project)

W, H = 320, 240
UP = np.array([0.0, 1.0, 0.0])
ROOM = (np.array([-2.0, 0.0, -2.0]), np.array([2.0, 2.6, 2.0]))
OBJECTS = {  # class -> (min, max); y is up
    "chair": (np.array([0.3, 0.0, 0.2]), np.array([0.8, 0.9, 0.7])),
    "table": (np.array([-1.0, 0.0, -0.6]), np.array([-0.2, 0.75, 0.0])),
    "coffee cup": (np.array([-0.65, 0.75, -0.35]), np.array([-0.57, 0.85, -0.27])),
}


def ray_box(origin, dirs, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - origin) / dirs, (hi - origin) / dirs
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    return t_near, t_far, (t_near <= t_far) & (t_far > 0)


def render(K, w2c):
    """z-depth and object-id buffers (0 = room shell, k = k-th object)."""
    c2w = invert_rigid(w2c)
    vv, uu = np.mgrid[0:H, 0:W]
    # Rays with camera-frame z = 1, so the hit parameter t *is* the z-depth.
    dirs_cam = np.stack([(uu.ravel() + 0.5 - K[0, 2]) / K[0, 0],
                         (vv.ravel() + 0.5 - K[1, 2]) / K[1, 1],
                         np.ones(W * H)], axis=1)
    dirs = dirs_cam @ c2w[:3, :3].T
    origin = c2w[:3, 3]
    _, depth, _ = ray_box(origin, dirs, *ROOM)  # inside the room: exit point
    ids = np.zeros(W * H, dtype=int)
    for k, (lo, hi) in enumerate(OBJECTS.values(), start=1):
        t_near, _, hit = ray_box(origin, dirs, lo, hi)
        closer = hit & (t_near > 0) & (t_near < depth)
        depth[closer], ids[closer] = t_near[closer], k
    return depth.reshape(H, W).astype(np.float32), ids.reshape(H, W)


def make_dataset(root, with_masks):
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    (root / "masks").mkdir()
    K = intrinsics_from_usd(14.0, 20.955, W, H)
    names = list(OBJECTS)
    manifest = {"scene_id": "synthetic", "up_axis": "Y", "meters_per_unit": 1.0, "width": W, "height": H,
                "views": []}
    detections = {"masks": with_masks, "views": {}}

    for i, az in enumerate(np.linspace(0, 2 * np.pi, 12, endpoint=False)):
        eye = np.array([1.6 * np.cos(az), 1.4, 1.6 * np.sin(az)])
        c2w_cv = look_at_usd(eye, np.array([0.0, 0.5, 0.0]), UP) @ USD_TO_CV
        w2c = invert_rigid(c2w_cv)
        depth, ids = render(K, w2c)
        vid = f"view_{i:04d}"
        Image.fromarray((ids * 60 % 255).astype(np.uint8)).convert("RGB").save(root / "rgb" / f"{vid}.png")
        np.save(root / "depth" / f"{vid}.npy", depth)
        manifest["views"].append({"view_id": vid, "rgb": f"rgb/{vid}.png", "depth": f"depth/{vid}.npy",
                                  "K": K.tolist(), "world_to_camera": w2c.tolist()})

        dets, masks = [], []
        for k, name in enumerate(names, start=1):
            m = ids == k
            if m.sum() < 30:
                continue
            ys, xs = np.nonzero(m)
            dets.append({"label": name, "score": 0.8, "box_xyxy": [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                         "mask_index": len(masks)})
            masks.append(m)
        if i == 0:  # a hallucinated chair on the wall in one view only: must be rejected
            dets.append({"label": "chair", "score": 0.9, "box_xyxy": [5, 5, 40, 40], "mask_index": len(masks)})
            fp = np.zeros((H, W), bool)
            fp[5:40, 5:40] = True
            masks.append(fp)
        if with_masks and masks:
            np.savez_compressed(root / "masks" / f"{vid}.npz", masks=np.packbits(np.stack(masks), axis=-1), width=W)
        else:
            for d in dets:
                d.pop("mask_index")
        detections["views"][vid] = [dict(d, box_xyxy=[float(c) for c in d["box_xyxy"]]) for d in dets]

    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "detections.json").write_text(json.dumps(detections))


def test_lift_round_trip():
    K = intrinsics_from_usd(14.0, 20.955, W, H)
    w2c = invert_rigid(look_at_usd([1.0, 1.5, 2.0], [0.0, 0.3, 0.0], UP) @ USD_TO_CV)
    pts = np.random.default_rng(0).uniform(-0.5, 0.5, (100, 3))
    u, v, z = project(pts, K, w2c)
    assert np.allclose(pixels_to_world(u, v, z, K, w2c), pts, atol=1e-9)


@pytest.mark.parametrize("with_masks,tol", [(True, 0.03), (False, 0.10)])  # 1% AABB trim costs ~2.5 cm on the chair bottom
def test_end_to_end(tmp_path, with_masks, tol):
    make_dataset(tmp_path, with_masks)
    out = lift_and_cluster.main(["--views", str(tmp_path), "--out", str(tmp_path / "scene_objects.json"),
                                 "--mask-erode", "0"])
    assert json.loads((tmp_path / "scene_objects.json").read_text()) == out

    found = {}
    for o in out["objects"]:
        assert o["class_name"] not in found, f"duplicate {o['class_name']}: {out['objects']}"
        found[o["class_name"]] = o
    assert set(found) == set(OBJECTS)  # the one-view false chair was dropped
    for name, (lo, hi) in OBJECTS.items():
        box = found[name]["bounding_box_3d"]
        assert np.allclose(box["min"], lo, atol=tol), (name, box, lo)
        assert np.allclose(box["max"], hi, atol=tol), (name, box, hi)
