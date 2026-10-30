"""Tests for usd_to_gltf.py: export room_scan.usda and re-parse the GLB with an
independent minimal reader (struct + json only, no usd_to_gltf internals).

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_usd_to_gltf.py -v
  or: python tests/test_usd_to_gltf.py
"""

import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
USD = os.path.abspath(os.path.join(PROJ, "..", "room_scan.usda"))
META = os.path.join(PROJ, "out", "renders", "meta.json")

COMP = {5126: ("<f4", 4), 5125: ("<u4", 4), 5123: ("<u2", 2), 5121: ("u1", 1)}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


# ---------------------------------------------------------------- independent reader
def read_glb(path):
    raw = open(path, "rb").read()
    magic, version, length = struct.unpack_from("<III", raw, 0)
    assert magic == 0x46546C67, "bad magic"
    assert version == 2
    assert length == len(raw), "header length != file size"
    off, chunks = 12, []
    while off < len(raw):
        clen, ctype = struct.unpack_from("<II", raw, off)
        assert off % 4 == 0 and clen % 4 == 0, "chunk not 4-byte aligned"
        chunks.append((ctype, raw[off + 8: off + 8 + clen]))
        off += 8 + clen
    assert off == len(raw)
    assert chunks[0][0] == 0x4E4F534A, "first chunk must be JSON"
    assert chunks[1][0] == 0x004E4942, "second chunk must be BIN"
    gltf = json.loads(chunks[0][1].decode("utf-8"))
    return gltf, chunks[1][1]


def view_bytes(gltf, bin_, vi):
    v = gltf["bufferViews"][vi]
    o = v.get("byteOffset", 0)
    assert o + v["byteLength"] <= len(bin_)
    return bin_[o: o + v["byteLength"]]


def accessor_array(gltf, bin_, ai):
    a = gltf["accessors"][ai]
    dt, size = COMP[a["componentType"]]
    n = NCOMP[a["type"]]
    v = gltf["bufferViews"][a["bufferView"]]
    assert "byteStride" not in v or v["byteStride"] == size * n
    data = view_bytes(gltf, bin_, a["bufferView"])
    start = a.get("byteOffset", 0)
    assert (v.get("byteOffset", 0) + start) % size == 0, "accessor misaligned"
    assert start + a["count"] * size * n <= len(data)
    return np.frombuffer(data, dtype=dt, count=a["count"] * n, offset=start).reshape(a["count"], n)


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def exported():
    if not os.path.isfile(USD):
        pytest.skip(f"{USD} not found")
    import usd_to_gltf
    tmp = tempfile.mkdtemp(prefix="glbtest_")
    out = os.path.join(tmp, "model.glb")
    info = usd_to_gltf.export_glb(USD, out)
    gltf, bin_ = read_glb(out)
    yield info, out, gltf, bin_
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def usd_bounds():
    """Bounds straight from pxr (independent of load_usd_meshes): every Mesh point
    transformed to world, times metersPerUnit."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(USD)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    pts = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            M = np.array(m.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))  # row-vector
            P = np.asarray(m.GetPointsAttr().Get(), np.float64)
            pts.append(P @ M[:3, :3] + M[3, :3])
    allp = np.concatenate(pts) * mpu
    return allp.min(0), allp.max(0), str(UsdGeom.GetStageUpAxis(stage)), mpu


# ---------------------------------------------------------------- tests
def test_info_dict(exported, usd_bounds):
    info, *_ = exported
    bmin, bmax, up, mpu = usd_bounds
    assert info["scene_id"] == "room_scan"
    assert info["usd_path"] == USD
    assert info["up_axis"] == up and info["meters_per_unit"] == mpu
    assert info["num_meshes"] >= 1 and info["num_triangles"] > 0
    np.testing.assert_allclose(info["bounds"]["min"], bmin, atol=1e-4)
    np.testing.assert_allclose(info["bounds"]["max"], bmax, atol=1e-4)


def test_structure(exported):
    info, path, gltf, bin_ = exported
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["buffers"][0]["byteLength"] <= len(bin_) < gltf["buffers"][0]["byteLength"] + 4
    assert "uri" not in gltf["buffers"][0]
    for v in gltf["bufferViews"]:
        assert v.get("byteOffset", 0) % 4 == 0
        assert v.get("byteOffset", 0) + v["byteLength"] <= gltf["buffers"][0]["byteLength"]
    assert "KHR_materials_unlit" in gltf["extensionsUsed"]
    assert len(gltf["meshes"]) == len(gltf["nodes"]) == info["num_meshes"]
    assert sorted(gltf["scenes"][gltf["scene"]]["nodes"]) == list(range(len(gltf["nodes"])))
    for node, mesh in zip(gltf["nodes"], gltf["meshes"]):
        assert node["name"].startswith("/") and node["name"] == mesh["name"]
        assert "matrix" not in node and "rotation" not in node    # no baked up-axis rotation


def test_accessors_and_counts(exported):
    info, _, gltf, bin_ = exported
    total_tris = 0
    for mesh in gltf["meshes"]:
        for prim in mesh["primitives"]:
            assert prim.get("mode", 4) == 4
            attrs = prim["attributes"]
            n = gltf["accessors"][attrs["POSITION"]]["count"]
            if "indices" in prim:
                total_tris += gltf["accessors"][prim["indices"]]["count"] // 3
            else:
                assert n % 3 == 0
                total_tris += n // 3
            for name, ai in attrs.items():
                acc = gltf["accessors"][ai]
                assert acc["count"] == n, name
                assert gltf["bufferViews"][acc["bufferView"]].get("target") == 34962
                arr = accessor_array(gltf, bin_, ai)
                assert np.all(np.isfinite(arr)), name
            pos = accessor_array(gltf, bin_, attrs["POSITION"])
            acc = gltf["accessors"][attrs["POSITION"]]
            assert acc["type"] == "VEC3" and acc["componentType"] == 5126
            np.testing.assert_array_equal(np.float32(acc["min"]), pos.min(0))
            np.testing.assert_array_equal(np.float32(acc["max"]), pos.max(0))
            if "NORMAL" in attrs:
                nrm = accessor_array(gltf, bin_, attrs["NORMAL"])
                np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-4)
    assert total_tris == info["num_triangles"]


def test_position_bounds_match_usd(exported, usd_bounds):
    _, _, gltf, _ = exported
    bmin, bmax, *_ = usd_bounds
    mins, maxs = [], []
    for mesh in gltf["meshes"]:
        for prim in mesh["primitives"]:
            acc = gltf["accessors"][prim["attributes"]["POSITION"]]
            mins.append(acc["min"])
            maxs.append(acc["max"])
    np.testing.assert_allclose(np.min(mins, 0), bmin, atol=1e-4)
    np.testing.assert_allclose(np.max(maxs, 0), bmax, atol=1e-4)


def test_bounds_match_render_meta(exported):
    """The boxes come from renders in meta.json's frame: proves one shared frame."""
    if not os.path.isfile(META):
        pytest.skip("out/renders/meta.json not present")
    meta = json.load(open(META))
    if os.path.abspath(meta.get("scene_usd", USD)) != USD:
        pytest.skip("meta.json is for a different USD")
    info, _, gltf, _ = exported
    sb = meta["scene_bounds_world"]
    np.testing.assert_allclose(info["bounds"]["min"], sb["min"], atol=1e-4)
    np.testing.assert_allclose(info["bounds"]["max"], sb["max"], atol=1e-4)
    assert info["up_axis"] == meta["up_axis"]
    assert info["meters_per_unit"] == meta["meters_per_unit"]


def test_uvs_and_materials(exported):
    _, _, gltf, bin_ = exported
    for mesh in gltf["meshes"]:
        for prim in mesh["primitives"]:
            mat = gltf["materials"][prim["material"]]
            assert mat["doubleSided"] is True
            assert "KHR_materials_unlit" in mat["extensions"]
            pbr = mat["pbrMetallicRoughness"]
            if "TEXCOORD_0" in prim["attributes"]:
                uv = accessor_array(gltf, bin_, prim["attributes"]["TEXCOORD_0"])
                assert uv.shape[1] == 2
                assert uv.min() > -10 and uv.max() < 10, (uv.min(), uv.max())
                tex = gltf["textures"][pbr["baseColorTexture"]["index"]]
                s = gltf["samplers"][tex["sampler"]]
                assert s["wrapS"] == s["wrapT"] == 10497
                assert s["magFilter"] == 9729
            else:
                assert len(pbr["baseColorFactor"]) == 4


def test_images_decode(exported):
    from PIL import Image
    _, _, gltf, bin_ = exported
    assert gltf.get("images"), "expected embedded textures for room_scan"
    for im in gltf["images"]:
        assert im["mimeType"] in ("image/jpeg", "image/png")
        data = view_bytes(gltf, bin_, im["bufferView"])
        assert "target" not in gltf["bufferViews"][im["bufferView"]]
        img = Image.open(io.BytesIO(data))
        assert img.format == {"image/jpeg": "JPEG", "image/png": "PNG"}[im["mimeType"]]
        img.load()
        assert img.size[0] > 0 and img.size[1] > 0


def test_uv_flip_matches_texture_lookup(exported):
    """v_gltf == 1 - v_usd against load_usd_meshes' faceVarying UVs."""
    from pxr import Usd
    from render_views_lite import load_usd_meshes
    _, _, gltf, bin_ = exported
    meshes = {m.path: m for m in load_usd_meshes(Usd.Stage.Open(USD))}
    for mesh in gltf["meshes"]:
        m = meshes[mesh["name"]]
        prim = mesh["primitives"][0]
        if m.corner_uv is None:
            continue
        uv = accessor_array(gltf, bin_, prim["attributes"]["TEXCOORD_0"])
        ref = m.corner_uv.reshape(-1, 2)
        np.testing.assert_allclose(uv[:, 0], ref[:, 0], atol=1e-6)
        np.testing.assert_allclose(uv[:, 1], 1.0 - ref[:, 1], atol=1e-6)


def test_official_validator(exported):
    """Optional: Khronos glTF-Validator via npx (skipped if node/network unavailable)."""
    if os.environ.get("SKIP_GLTF_VALIDATOR") or not shutil.which("npx"):
        pytest.skip("npx not available")
    _, path, _, _ = exported
    script = ("const v=require('gltf-validator');const fs=require('fs');"
              f"v.validateBytes(new Uint8Array(fs.readFileSync({json.dumps(path)})))"
              ".then(r=>{console.log(JSON.stringify(r.issues));})"
              ".catch(e=>{console.error(e);process.exit(2);});")
    tmpdir = tempfile.mkdtemp(prefix="gltfval_")
    try:
        r = subprocess.run(["npm", "install", "--silent", "--no-audit", "--no-fund",
                            "gltf-validator"], cwd=tmpdir, capture_output=True, text=True,
                           timeout=120)
        if r.returncode != 0:
            pytest.skip(f"cannot install gltf-validator: {r.stderr[-200:]}")
        r = subprocess.run(["node", "-e", script], cwd=tmpdir, capture_output=True,
                           text=True, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.skip("gltf-validator timed out")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    assert r.returncode == 0, r.stderr
    issues = json.loads(r.stdout.strip().splitlines()[-1])
    print("gltf-validator:", {k: issues[k] for k in
                              ("numErrors", "numWarnings", "numInfos", "numHints")})
    assert issues["numErrors"] == 0, json.dumps(issues["messages"][:10], indent=1)


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    sys.exit(pytest.main([__file__, "-v", "-s"]))
