"""USD mesh scan -> binary glTF (GLB) for the browser viewer.

Contract (app/API.md, "Coordinates" and GET /api/scene/model.glb):
  * vertex positions are in the stage's NATIVE world frame, in METERS
    (stage units x metersPerUnit). No up-axis rotation is baked in; the viewer
    rotates its "world" group by -90 deg about X when up_axis == "Z".
  * one glTF mesh + node per USD mesh prim, named after the prim path
  * unwelded vertices (one per triangle corner) so faceVarying UVs survive
  * original texture bytes embedded (image/jpeg or image/png), KHR_materials_unlit,
    doubleSided, REPEAT wrap; untextured meshes use baseColorFactor.

Geometry comes from render_views_lite.load_usd_meshes, i.e. exactly the triangles
the renderer ray-casts, so the GLB and out/renders/meta.json share one frame.

Dependencies: numpy, pillow, pxr (usd-core). The GLB writer is hand-rolled.

Usage:
    python usd_to_gltf.py --usd ../room_scan.usda --out out/viewer/model.glb
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:            # render_views_lite lives next to this file
    sys.path.insert(0, _HERE)

GLB_MAGIC = 0x46546C67          # "glTF"
CHUNK_JSON = 0x4E4F534A         # "JSON"
CHUNK_BIN = 0x004E4942          # "BIN\0"
ARRAY_BUFFER = 34962
FLOAT, UNSIGNED_INT = 5126, 5125
LINEAR, LINEAR_MIPMAP_LINEAR, REPEAT = 9729, 9987, 10497


def _log(msg: str):
    print(f"[usd_to_gltf] {msg}", file=sys.stderr, flush=True)


def _pad4(n: int) -> int:
    return (n + 3) & ~3


class _BinBuilder:
    """Accumulates the BIN chunk; every bufferView starts on a 4-byte boundary."""

    def __init__(self):
        self.parts: list[bytes] = []
        self.size = 0
        self.buffer_views: list[dict] = []
        self.accessors: list[dict] = []

    def add_view(self, data: bytes, target: int | None = None) -> int:
        pad = _pad4(self.size) - self.size
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        view = {"buffer": 0, "byteOffset": self.size, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.parts.append(data)
        self.size += len(data)
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def add_accessor(self, arr: np.ndarray, type_: str, with_minmax: bool = False,
                     target: int | None = ARRAY_BUFFER) -> int:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        view = self.add_view(arr.tobytes(), target)
        acc = {"bufferView": view, "byteOffset": 0, "componentType": FLOAT,
               "count": int(arr.shape[0]), "type": type_}
        if with_minmax:
            # computed from the exact float32 values written
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def blob(self) -> bytes:
        data = b"".join(self.parts)
        return data + b"\0" * (_pad4(len(data)) - len(data))


def _image_bytes(path: str, texture: np.ndarray | None) -> tuple[bytes, str]:
    """Original file bytes when the browser can decode them; otherwise re-encode as PNG."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:3] == b"\xff\xd8\xff":
        return raw, "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return raw, "image/png"
    from PIL import Image
    if texture is not None:
        img = Image.fromarray(np.clip(texture * 255.0 + 0.5, 0, 255).astype(np.uint8))
    else:
        img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    _log(f"re-encoded {os.path.basename(path)} as PNG")
    return buf.getvalue(), "image/png"


def _flat_normals(p: np.ndarray) -> np.ndarray:
    """p: (T, 3, 3) corner positions -> (T*3, 3) unit face normals (degenerates get +Y)."""
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]).astype(np.float64)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    bad = ln[:, 0] < 1e-20
    n = np.where(bad[:, None], np.array([0.0, 1.0, 0.0]), n / np.where(bad, 1.0, ln[:, 0])[:, None])
    return np.repeat(n.astype(np.float32), 3, axis=0)


def build_glb(meshes, mpu: float, extras: dict | None = None) -> bytes:
    """meshes: list of render_views_lite.MeshData (world-space, stage units)."""
    b = _BinBuilder()
    gl_meshes, nodes, materials, images, textures = [], [], [], [], []
    image_index: dict[str, int] = {}

    for m in meshes:
        if len(m.tris) == 0:
            continue
        corners = m.verts[m.tris.astype(np.int64)].astype(np.float64) * mpu   # (T, 3, 3) meters
        pos = corners.reshape(-1, 3).astype(np.float32)
        attrs = {"POSITION": b.add_accessor(pos, "VEC3", with_minmax=True),
                 "NORMAL": b.add_accessor(_flat_normals(corners), "VEC3")}

        mat = {"name": m.path, "doubleSided": True,
               "extensions": {"KHR_materials_unlit": {}},
               "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}}
        tex_path = getattr(m, "texture_path", None)
        if m.corner_uv is not None and tex_path:
            uv = m.corner_uv.reshape(-1, 2).astype(np.float32).copy()
            uv[:, 1] = 1.0 - uv[:, 1]                 # USD origin bottom-left -> glTF top-left
            attrs["TEXCOORD_0"] = b.add_accessor(uv, "VEC2")
            if tex_path not in image_index:
                data, mime = _image_bytes(tex_path, m.texture)
                images.append({"name": os.path.basename(tex_path),
                               "bufferView": b.add_view(data), "mimeType": mime})
                textures.append({"sampler": 0, "source": len(images) - 1})
                image_index[tex_path] = len(textures) - 1
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": image_index[tex_path],
                                                               "texCoord": 0}
        else:
            c = np.clip(np.asarray(m.base_color, np.float64).reshape(-1)[:3], 0.0, 1.0)
            mat["pbrMetallicRoughness"]["baseColorFactor"] = [float(c[0]), float(c[1]),
                                                              float(c[2]), 1.0]
        materials.append(mat)
        gl_meshes.append({"name": m.path, "primitives": [
            {"attributes": attrs, "material": len(materials) - 1, "mode": 4}]})
        nodes.append({"name": m.path, "mesh": len(gl_meshes) - 1})

    if not gl_meshes:
        raise ValueError("no triangles to export")

    gltf = {
        "asset": {"version": "2.0", "generator": "open_vocab_3d/usd_to_gltf.py",
                  **({"extras": extras} if extras else {})},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"name": "world", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gl_meshes,
        "materials": materials,
        "accessors": b.accessors,
        "bufferViews": b.buffer_views,
    }
    if images:
        gltf["images"] = images
        gltf["textures"] = textures
        gltf["samplers"] = [{"magFilter": LINEAR, "minFilter": LINEAR_MIPMAP_LINEAR,
                             "wrapS": REPEAT, "wrapT": REPEAT}]
    bin_blob = b.blob()
    gltf["buffers"] = [{"byteLength": len(bin_blob)}]

    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (_pad4(len(js)) - len(js))      # JSON chunk padded with spaces
    total = 12 + 8 + len(js) + 8 + len(bin_blob)
    return b"".join([struct.pack("<III", GLB_MAGIC, 2, total),
                     struct.pack("<II", len(js), CHUNK_JSON), js,
                     struct.pack("<II", len(bin_blob), CHUNK_BIN), bin_blob])


def export_glb(usd_path: str, out_path: str) -> dict:
    """Convert a USD stage to a GLB at out_path. Returns scene metadata (bounds in meters)."""
    from pxr import Usd, UsdGeom
    from render_views_lite import load_usd_meshes

    usd_path = os.path.abspath(usd_path)
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"cannot open {usd_path}")
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    meshes = [m for m in load_usd_meshes(stage) if len(m.tris)]
    if not meshes:
        raise RuntimeError(f"no renderable meshes in {usd_path}")

    # Same reduction as render_views_lite (meta.json scene_bounds_world).
    allv = np.concatenate([m.verts for m in meshes])
    bmin = (allv.min(0).astype(float) * mpu).tolist()
    bmax = (allv.max(0).astype(float) * mpu).tolist()
    info = {
        "scene_id": os.path.splitext(os.path.basename(usd_path))[0],
        "usd_path": usd_path,
        "up_axis": up_axis,
        "meters_per_unit": mpu,
        "bounds": {"min": bmin, "max": bmax},
        "num_meshes": len(meshes),
        "num_triangles": int(sum(len(m.tris) for m in meshes)),
    }
    glb = build_glb(meshes, mpu, extras={"up_axis": up_axis, "meters_per_unit": mpu,
                                         "units": "meters", "source": usd_path})

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(glb)
    os.replace(tmp, out_path)                     # atomic: the server may be serving it
    _log(f"wrote {out_path} ({len(glb) / 1e6:.2f} MB, {info['num_meshes']} meshes, "
         f"{info['num_triangles']} tris)")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Convert a USD mesh scan to GLB for the web viewer.")
    p.add_argument("--usd", required=True, help="input .usd/.usda/.usdc")
    p.add_argument("--out", default="out/viewer/model.glb", help="output .glb path")
    args = p.parse_args(argv)
    info = export_glb(args.usd, args.out)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
