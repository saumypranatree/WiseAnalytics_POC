"""Tests for scan_prep.py: safe .usdz extraction, validation, the orientation
heuristic (synthetic Z-up rooms + the real room_scan.usdz), wrapper and set_flip.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_scan_prep.py -q
"""
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import scan_prep  # noqa: E402
from scan_prep import ScanPrepError, ScanTooLargeError  # noqa: E402

REAL_USDZ = PROJ.parent.parent / "room_scan.usdz"


# --------------------------------------------------------------------------- #
# synthetic scenes
# --------------------------------------------------------------------------- #
def _quad(c, u, v, normal):
    c, u, v = (np.asarray(x, float) for x in (c, u, v))
    if np.dot(np.cross(u, v), normal) < 0:
        u, v = v, u
    return [c - u - v, c + u - v, c + u + v, c - u + v]


def room_quads_zup():
    """Z-up box room 4 x 3 x 2.5 m: floor, 4 walls (inward), a partial ceiling, a
    table top (0.75 m) and a bench seat (0.45 m), all facing into the room."""
    Z = np.array([0, 0, 1.0])
    q = [
        _quad([0, 0, 0], [2, 0, 0], [0, 1.5, 0], Z),                      # floor 12 m^2
        _quad([0, 0, 2.5], [0.8, 0, 0], [0, 0.8, 0], -Z),                 # ceiling patch 2.6 m^2
        _quad([0, -1.5, 1.25], [2, 0, 0], [0, 0, 1.25], [0, 1, 0]),       # walls
        _quad([0, 1.5, 1.25], [2, 0, 0], [0, 0, 1.25], [0, -1, 0]),
        _quad([-2, 0, 1.25], [0, 1.5, 0], [0, 0, 1.25], [1, 0, 0]),
        _quad([2, 0, 1.25], [0, 1.5, 0], [0, 0, 1.25], [-1, 0, 0]),
        _quad([0.5, 0.3, 0.75], [0.5, 0, 0], [0, 0.3, 0], Z),             # table top
        _quad([0.5, -0.6, 0.45], [0.6, 0, 0], [0, 0.2, 0], Z),            # bench seat
        _quad([0.5, 0.0, 0.3], [0.5, 0, 0], [0, 0, 0.3], [0, -1, 0]),     # table front
    ]
    return q


def write_mesh_stage(path, quads, up="Z", mpu=1.0, rotate_x_180=False,
                     default_translate=None, default_prim=True):
    from pxr import Gf, Sdf, Usd, UsdGeom
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up == "Z" else UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, mpu)
    root = UsdGeom.Xform.Define(stage, "/Room")
    if default_translate is not None:
        root.AddTranslateOp().Set(Gf.Vec3d(*default_translate))
    if default_prim:
        stage.SetDefaultPrim(root.GetPrim())
    pts, counts, idx = [], [], []
    for qd in quads:
        for p in qd:
            p = np.asarray(p, float) / mpu
            if rotate_x_180:
                p = np.array([p[0], -p[1], -p[2]])
            idx.append(len(pts))
            pts.append(Gf.Vec3f(*map(float, p)))
        counts.append(4)
    m = UsdGeom.Mesh.Define(stage, "/Room/geo")
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    stage.GetRootLayer().Save()
    return path


def make_usdz(tmp_path, name="room.usdz", **kw):
    """Plain stored zip with the root layer first (what the USDZ spec requires)."""
    layer = write_mesh_stage(tmp_path / "room_src.usda", room_quads_zup(), **kw)
    out = tmp_path / name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        z.write(layer, "room.usda")
    return out


def make_zip(path, entries):
    """entries: [(name, bytes, extra)] where extra may be {'symlink': True}."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name, data, *extra in entries:
            zi = zipfile.ZipInfo(name)
            if extra and extra[0].get("symlink"):
                zi.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(zi, data)
    return path


# --------------------------------------------------------------------------- #
# safe extraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["../evil.usda", "/abs/evil.usda", "a/../../evil.usda",
                                 "..\\evil.usda", "C:/evil.usda", "C:evil.usda"])
def test_extract_rejects_path_escape(tmp_path, bad):
    z = make_zip(tmp_path / "x.usdz", [("ok.usda", b"#usda 1.0\n"), (bad, b"x")])
    with pytest.raises(ScanPrepError):
        scan_prep.safe_extract_usdz(z, tmp_path / "out")
    assert not (tmp_path / "evil.usda").exists()
    assert not (tmp_path.parent / "evil.usda").exists()


def test_extract_rejects_symlink(tmp_path):
    z = make_zip(tmp_path / "x.usdz", [("ok.usda", b"#usda 1.0\n"),
                                       ("link", b"/etc/passwd", {"symlink": True})])
    with pytest.raises(ScanPrepError, match="symlink"):
        scan_prep.safe_extract_usdz(z, tmp_path / "out")


def test_extract_rejects_too_many_entries(tmp_path):
    entries = [("a.usda", b"#usda 1.0\n")] + [(f"t/{i}.txt", b"") for i in range(2000)]
    z = make_zip(tmp_path / "x.usdz", entries)
    with pytest.raises(ScanPrepError, match="entries"):
        scan_prep.safe_extract_usdz(z, tmp_path / "out")


def test_extract_rejects_too_large(tmp_path):
    z = make_zip(tmp_path / "x.usdz", [("a.usda", b"#usda 1.0\n" + b" " * 5000)])
    with pytest.raises(ScanPrepError, match="uncompressed"):
        scan_prep.safe_extract_usdz(z, tmp_path / "out", max_uncompressed=1000)


def test_extract_root_layer_is_first_usd_entry(tmp_path):
    z = make_zip(tmp_path / "x.usdz", [("textures/t.png", b"png"), ("sub/b.usdc", b"x"),
                                       ("a.usda", b"#usda 1.0\n")])
    assert scan_prep.safe_extract_usdz(z, tmp_path / "out") == "sub/b.usdc"
    assert (tmp_path / "out" / "textures" / "t.png").read_bytes() == b"png"


def test_extract_rejects_no_layer_and_not_zip(tmp_path):
    z = make_zip(tmp_path / "x.usdz", [("t.png", b"png")])
    with pytest.raises(ScanPrepError, match="no .usd"):
        scan_prep.safe_extract_usdz(z, tmp_path / "out")
    junk = tmp_path / "junk.usdz"
    junk.write_bytes(b"not a zip at all")
    with pytest.raises(ScanPrepError, match="zip"):
        scan_prep.safe_extract_usdz(junk, tmp_path / "out2")


# --------------------------------------------------------------------------- #
# prepare_scan validation
# --------------------------------------------------------------------------- #
def _no_scan_dirs(root):
    return not root.exists() or not any(root.iterdir())


def test_prepare_rejects_bad_uploads(tmp_path):
    root = tmp_path / "scans"
    good = make_usdz(tmp_path)
    with pytest.raises(ScanPrepError, match=".usdz"):
        scan_prep.prepare_scan(good, root, "room.usda")
    with pytest.raises(ScanTooLargeError):
        scan_prep.prepare_scan(good, root, "room.usdz", max_upload_bytes=100)
    nomesh = tmp_path / "empty.usda"
    nomesh.write_text('#usda 1.0\n(\n defaultPrim = "W"\n)\ndef Xform "W" {}\n')
    z = tmp_path / "nomesh.usdz"
    with zipfile.ZipFile(z, "w") as zz:
        zz.write(nomesh, "empty.usda")
    with pytest.raises(ScanPrepError, match="no mesh"):
        scan_prep.prepare_scan(z, root, "nomesh.usdz")
    broken = make_zip(tmp_path / "broken.usdz", [("b.usda", b"#usda 1.0\ndef Xform \"W\" {")])
    with pytest.raises(ScanPrepError):
        scan_prep.prepare_scan(broken, root, "broken.usdz")
    slip = make_zip(tmp_path / "slip.usdz", [("a.usda", b"#usda 1.0\n"), ("../x", b"x")])
    with pytest.raises(ScanPrepError):
        scan_prep.prepare_scan(slip, root, "slip.usdz")
    assert _no_scan_dirs(root), list(root.iterdir())


def test_sanitize_filename():
    assert scan_prep.sanitize_filename("../../etc/passwd.usdz") == "passwd.usdz"
    assert scan_prep.sanitize_filename("C:\\x\\my scan (1).usdz") == "my scan _1_.usdz"
    assert scan_prep.sanitize_filename("") == "upload.usdz"


# --------------------------------------------------------------------------- #
# orientation: synthetic Z-up rooms
# --------------------------------------------------------------------------- #
def test_orientation_synthetic_upright_zup(tmp_path):
    p = write_mesh_stage(tmp_path / "up.usda", room_quads_zup())
    r = scan_prep.detect_orientation(p)
    ev = r["evidence"]
    assert r["flip_recommended"] is False, ev
    assert r["confidence"] == "high"
    assert ev["dominant_surface_rel_height"] < 0.05
    assert ev["dominant_surface_area_m2"] == pytest.approx(12.0, abs=0.01)
    assert ev["dominant_surface_mean_normal_up"] > 0.99
    assert ev["votes"]["support"] == -1.0       # table + seat face up


def test_orientation_synthetic_flipped_zup(tmp_path):
    p = write_mesh_stage(tmp_path / "down.usda", room_quads_zup(), rotate_x_180=True)
    r = scan_prep.detect_orientation(p)
    assert r["flip_recommended"] is True, r["evidence"]
    assert r["confidence"] == "high"
    assert r["evidence"]["dominant_surface_rel_height"] > 0.95


def test_prepare_synthetic_flipped_then_upright(tmp_path):
    """Upside-down Z-up room in centimetres, default prim with its own translate:
    auto flip; the wrapper keeps the source transform and units; the flipped
    wrapper is detected upright; set_flip(False) brings the recommendation back."""
    root = tmp_path / "scans"
    z = make_usdz(tmp_path, rotate_x_180=True, mpu=0.01, default_translate=(1000, 0, 0))
    rec = scan_prep.prepare_scan(z, root, "flipped room.usdz")
    sd = root / rec["scan_id"]
    assert rec["up_axis"] == "Z" and rec["meters_per_unit"] == 0.01
    o = rec["orientation"]
    assert o["flip_recommended"] and o["flip_applied"] and o["source"] == "auto"
    assert rec["reference_has_xform"] is True
    txt = (sd / "scene.usda").read_text()
    assert "rotateXYZ = (180, 0, 0)" in txt and 'upAxis = "Z"' in txt
    assert "metersPerUnit = 0.01" in txt and "@extracted/room.usda@" in txt
    # source translate (10 m in x) survives; after the flip z is up again: floor at 0
    b = rec["scene_bounds"]
    assert b["min"][0] == pytest.approx(8.0, abs=1e-3) and b["max"][0] == pytest.approx(12.0, abs=1e-3)
    assert b["min"][2] == pytest.approx(0.0, abs=1e-3) and b["max"][2] == pytest.approx(2.5, abs=1e-3)
    assert scan_prep.detect_orientation(sd / "scene.usda")["flip_recommended"] is False
    assert json.loads((sd / "scan.json").read_text()) == rec
    assert (sd / "source" / "flipped room.usdz").read_bytes() == z.read_bytes()

    (sd / "viewer").mkdir()
    (sd / "viewer" / "model.glb").write_bytes(b"glb")
    rec2 = scan_prep.set_flip(sd, False, "manual")
    assert not (sd / "viewer" / "model.glb").exists()
    assert "rotateXYZ" not in (sd / "scene.usda").read_text()
    assert rec2["orientation"]["flip_applied"] is False
    assert rec2["orientation"]["source"] == "manual"
    assert rec2["orientation"]["flip_recommended"] is True       # from the source
    assert rec2["scene_bounds"]["max"][2] == pytest.approx(0.0, abs=1e-3)
    assert scan_prep.detect_orientation(sd / "scene.usda")["flip_recommended"] is True


def test_prepare_synthetic_upright_no_flip(tmp_path):
    z = make_usdz(tmp_path)
    rec = scan_prep.prepare_scan(z, tmp_path / "scans", "room.usdz")
    assert rec["orientation"]["flip_applied"] is False
    txt = (tmp_path / "scans" / rec["scan_id"] / "scene.usda").read_text()
    assert "rotateXYZ" not in txt
    assert rec["scan_id"].startswith("scan_") and len(rec["scan_id"]) == len("scan_20261007_141500_ab12")


def test_wrapper_without_default_prim(tmp_path):
    z = make_usdz(tmp_path, default_prim=False)
    rec = scan_prep.prepare_scan(z, tmp_path / "scans", "room.usdz")
    assert rec["reference_prim"] == "/Room"
    sd = tmp_path / "scans" / rec["scan_id"]
    assert "@extracted/room.usda@</Room>" in (sd / "scene.usda").read_text()
    assert rec["scene_bounds"]["max"][2] == pytest.approx(2.5, abs=1e-3)


# --------------------------------------------------------------------------- #
# the real scan
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REAL_USDZ.is_file(), reason="room_scan.usdz not available")
def test_real_room_scan_flip(tmp_path):
    rec = scan_prep.prepare_scan(REAL_USDZ, tmp_path / "scans", "room_scan.usdz")
    o = rec["orientation"]
    ev = o["evidence"]
    assert o["flip_recommended"] is True and o["flip_applied"] is True
    assert o["confidence"] == "high"
    assert ev["dominant_surface_rel_height"] > 0.9                # floor at the top
    assert 4.5 < ev["dominant_surface_area_m2"] < 6.5             # ~5.5 m^2 floor
    assert ev["dominant_surface_mean_normal_up"] < -0.9
    sd = tmp_path / "scans" / rec["scan_id"]
    assert rec["root_layer"] == "extracted/room_scan.usda"
    assert (sd / "extracted" / "textures" / "chunk_C_texture.jpeg").is_file()
    r2 = scan_prep.detect_orientation(sd / "scene.usda")
    assert r2["flip_recommended"] is False and r2["confidence"] == "high"
    # the upright frame: floor near the bottom, ~3.1 m tall
    b = rec["scene_bounds"]
    assert b["max"][1] - b["min"][1] == pytest.approx(3.12, abs=0.05)
    assert scan_prep.set_flip(sd, False, "manual")["orientation"]["flip_applied"] is False
    assert scan_prep.detect_orientation(sd / "scene.usda")["flip_recommended"] is True


def test_set_scale_and_flip_compose(tmp_path):
    """Shelf-scan calibration: a uniform scale in scene.usda that survives flips, and vice versa."""
    usdz = make_usdz(tmp_path)
    rec = scan_prep.prepare_scan(usdz, tmp_path / "scans", "room.usdz", auto_orient=False)
    d = tmp_path / "scans" / rec["scan_id"]
    b0 = rec["scene_bounds"]
    size0 = np.subtract(b0["max"], b0["min"])
    rec = scan_prep.set_scale(d, {"status": "calibrated", "method": "measured", "factor": 2.5})
    assert "xformOp:scale = (2.5, 2.5, 2.5)" in (d / "scene.usda").read_text()
    np.testing.assert_allclose(np.subtract(rec["scene_bounds"]["max"], rec["scene_bounds"]["min"]), size0 * 2.5, rtol=1e-6)
    assert rec["scale"]["factor"] == 2.5 and rec["orientation"]["flip_applied"] is False
    rec = scan_prep.set_flip(d, True, "manual")                      # flip keeps the scale
    txt = (d / "scene.usda").read_text()
    assert "xformOp:scale" in txt and "xformOp:rotateXYZ = (180, 0, 0)" in txt
    np.testing.assert_allclose(np.subtract(rec["scene_bounds"]["max"], rec["scene_bounds"]["min"]), size0 * 2.5, rtol=1e-6)
    rec = scan_prep.set_scale(d, {"status": "approximate", "factor": 1.0})   # reset keeps the flip
    txt = (d / "scene.usda").read_text()
    assert "xformOp:scale" not in txt and "xformOp:rotateXYZ" in txt
    np.testing.assert_allclose(np.subtract(rec["scene_bounds"]["max"], rec["scene_bounds"]["min"]), size0, rtol=1e-6)
    with pytest.raises(ValueError):
        scan_prep.set_scale(d, {"factor": -1})
