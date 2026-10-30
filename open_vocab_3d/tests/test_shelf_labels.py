"""shelf_labels: a 2D box on a keyframe -> 3D box on the shelf front.

Synthetic shelf in the usd frame (+Y up, +Z toward the aisle): a front face at z = 0
with a 40 x 40 cm hole (the empty slot) at x in [-0.2, 0.2], y in [0.8, 1.2], and a
back panel 30 cm behind it. A camera with radial distortion looks at it; the label box
is the projection of the hole. The lift must recover the slot's size, also after a
calibration scale and a flip written into scene.usda by scan_prep.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_shelf_labels.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("open3d")
pytest.importorskip("pxr")
import scan_prep  # noqa: E402
import shelf_labels as sl  # noqa: E402

CAM = {"model": "SIMPLE_RADIAL", "width": 1080, "height": 1920, "f": 1400.0, "cx": 540.0, "cy": 960.0, "k1": -0.04}
C_USD = np.array([0.1, 1.05, 1.4])
R_CU = np.diag([1.0, -1.0, -1.0])        # camera x right, y down, z forward = looking along -Z


def rects():
    """Front face with a hole + back panel, as (x0, x1, y0, y1, z) rectangles."""
    return [(-1.0, -0.2, 0.0, 2.0, 0.0), (0.2, 1.0, 0.0, 2.0, 0.0),        # left / right of the hole
            (-0.2, 0.2, 0.0, 0.8, 0.0), (-0.2, 0.2, 1.2, 2.0, 0.0),        # below / above
            (-0.3, 0.3, 0.7, 1.3, -0.3)]                                  # shelf back behind the slot


def write_scan(d: Path, flip=False, scale=1.0):
    from pxr import Usd, UsdGeom, Vt, Gf
    (d / "extracted").mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(d / "extracted" / "scan.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    pts, idx = [], []
    for x0, x1, y0, y1, z in rects():
        b = len(pts)
        pts += [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        idx += [b, b + 1, b + 2, b + 3]
    m = UsdGeom.Mesh.Define(stage, "/World/Shelf")
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([4] * (len(pts) // 4)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    m.CreateDisplayColorPrimvar().Set(Vt.Vec3fArray([Gf.Vec3f(0.5, 0.5, 0.5)]))
    stage.GetRootLayer().Save()
    (d / "scene.usda").write_text(scan_prep.wrapper_text("extracted/scan.usda", 1.0, "Y", flip, scale=scale))
    (d / "convert" / "work").mkdir(parents=True)
    kf = {"name": "00012", "frame": 12, "time_s": 0.4, "R_cam_from_usd": R_CU.tolist(), "C_usd": C_USD.tolist()}
    (d / sl.CAMERAS_REL).write_text(json.dumps({"camera": CAM, "keyframes": [kf]}))
    return {"scale": {"factor": scale}, "orientation": {"flip_applied": flip}}


def project(P):
    Xc = (np.asarray(P, float) - C_USD) @ R_CU.T
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    r2 = x * x + y * y
    return CAM["f"] * x * (1 + CAM["k1"] * r2) + CAM["cx"], CAM["f"] * y * (1 + CAM["k1"] * r2) + CAM["cy"]


def hole_box_px():
    u, v = project([(-0.2, 0.8, 0), (0.2, 0.8, 0), (0.2, 1.2, 0), (-0.2, 1.2, 0)])
    return [float(u.min()), float(v.min()), float(u.max()), float(v.max())]


def test_undistort_inverts_distortion():
    x = np.array([0.0, 0.2, -0.3]); y = np.array([0.0, 0.25, 0.1])
    r2 = x * x + y * y
    xu, yu = sl.undistort(x * (1 - 0.04 * r2), y * (1 - 0.04 * r2), -0.04)
    np.testing.assert_allclose(xu, x, atol=1e-9); np.testing.assert_allclose(yu, y, atol=1e-9)


@pytest.mark.parametrize("flip,scale", [(False, 1.0), (False, 2.0), (True, 1.0), (True, 0.5)])
def test_lift_hole_on_shelf_front(tmp_path, flip, scale):
    rec = write_scan(tmp_path, flip, scale)
    lab = {"id": "lab_001", "keyframe": "00012", "box_px": hole_box_px(), "kind": "empty"}
    (out,) = sl.lift_labels(tmp_path, rec, [lab])
    assert "error" not in out, out
    assert out["front_source"] == "ring" and out["depth_source"] == "mesh"
    assert out["width_cm"] == pytest.approx(40 * scale, rel=0.03)
    assert out["height_cm"] == pytest.approx(40 * scale, rel=0.03)
    assert out["depth_cm"] == pytest.approx(30 * scale, rel=0.05)
    c = np.array(out["center"])
    want = np.array([0.0, 1.0, -0.15]) * scale
    if flip:
        want = want * [1, -1, -1]
    np.testing.assert_allclose(c, want, atol=0.02 * scale)


def test_missing_camera_and_box_off_scan(tmp_path):
    rec = write_scan(tmp_path)
    cams = json.loads((tmp_path / sl.CAMERAS_REL).read_text())
    away = {"name": "00020", "frame": 20, "time_s": 0.67, "R_cam_from_usd": np.eye(3).tolist(),  # looks along +Z
            "C_usd": C_USD.tolist()}
    cams["keyframes"].append(away)
    (tmp_path / sl.CAMERAS_REL).write_text(json.dumps(cams))
    out = sl.lift_labels(tmp_path, rec, [{"id": "lab_001", "keyframe": "99999", "box_px": [0, 0, 10, 10]},
                                         {"id": "lab_002", "keyframe": "00020", "box_px": [400, 800, 600, 1000]},
                                         {"id": "lab_003", "keyframe": "00012", "box_px": hole_box_px()}])
    assert out[0]["error"] == "no camera for this keyframe"
    assert "does not cover any part of the scan" in out[1]["error"]   # camera looks away from the shelf
    assert "error" not in out[2]                                        # one bad label does not break the rest
