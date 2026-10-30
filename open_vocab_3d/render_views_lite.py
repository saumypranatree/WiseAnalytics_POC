"""Stage 1 (lite): render RGB + depth + cameras from a USD mesh scan WITHOUT Isaac Sim.

Drop-in replacement for render_views.py. It writes the same out/renders/
layout (see CONTRACT.md), so detect_2d.py and lift_cluster.py run unchanged.

How it renders: per-pixel CPU ray casting with Open3D's RaycastingScene
(Embree). Each ray gives the hit triangle and barycentrics, which yield
  * depth  - exact z-depth (ray dirs are K^-1 [u, v, 1], so t_hit == z), and
  * RGB    - the triangle's UVs interpolated and the bound UsdUVTexture sampled
             bilinearly (unlit: photogrammetry textures already carry lighting).
RGB and depth come from the same rays, so they are pixel-aligned by construction.

Supported: UsdGeom.Mesh (any polygons, fan-triangulated), world transforms,
`st` primvars with faceVarying / vertex / varying interpolation (indexed or
not), UsdPreviewSurface diffuseColor from a UsdUVTexture or a constant, and
displayColor as a fallback. Not supported: lighting, transparency, subdivision,
instancing-only content, Gaussian splats (convert to a mesh first).

Usage (plain Python, no GPU / display needed):
    pip install usd-core open3d numpy pillow
    python render_views_lite.py --usd ../room_scan.usda --out out/renders \
        --mode interior --num-views 48 --verify
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

# Camera math + layouts + verification are shared with the Isaac Sim renderer.
# render_views.py imports only numpy at module level, so this is safe.
from render_views import (AXIS_VEC, _parse_floats, _save_png, _write_json_atomic,
                          dome_poses, focal_from_hfov, hfov_from_focal,
                          interior_poses, interior_positions, intrinsics_from_camera,
                          rigid_inverse, scale_translation, usd_to_col, verify_renders)


def _log(msg: str):
    print(f"[render_lite] {msg}", flush=True)


# ----------------------------------------------------------------------------
# USD -> triangle soup with per-corner UVs and a texture per mesh
# ----------------------------------------------------------------------------

class MeshData:
    """World-space triangles of one USD mesh plus what is needed to shade them."""

    def __init__(self, path, verts, tris, corner_uv, texture, base_color, texture_path=None):
        self.path = path
        self.verts = verts            # (V, 3) float32, stage units, world space
        self.tris = tris              # (T, 3) uint32 vertex indices
        self.corner_uv = corner_uv    # (T, 3, 2) float32 UV per triangle corner, or None
        self.texture = texture        # (h, w, 3) float32 in [0, 1], or None
        self.base_color = base_color  # (3,) float32, used when there is no texture
        self.texture_path = texture_path  # source image file of `texture`, or None


def _triangulate(counts, indices):
    """Fan-triangulate polygons. Returns (T, 3) face-vertex *slots* (positions in
    faceVertexIndices), so faceVarying primvars can be gathered with the same array."""
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    if np.all(counts == 3):                       # fast path: already triangles
        return np.arange(len(indices), dtype=np.int64).reshape(-1, 3)
    slots = []
    for s, n in zip(starts, counts):
        for k in range(1, n - 1):
            slots.append((s, s + k, s + k + 1))
    return np.asarray(slots, dtype=np.int64).reshape(-1, 3)


def _find_texture_and_color(prim, stage_dir):
    """Follow the bound material to a diffuse texture path / constant color."""
    from pxr import UsdShade, Sdf

    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not mat:
        return None, None, "st"
    surface = mat.ComputeSurfaceSource()[0]
    if not surface:
        return None, None, "st"
    diffuse = surface.GetInput("diffuseColor")
    if not diffuse:
        return None, None, "st"
    if diffuse.HasConnectedSource():
        src, _, _ = diffuse.GetConnectedSource()
        tex_shader = UsdShade.Shader(src.GetPrim())
        file_in = tex_shader.GetInput("file")
        st_name = "st"
        st_in = tex_shader.GetInput("st")
        if st_in and st_in.HasConnectedSource():
            reader = UsdShade.Shader(st_in.GetConnectedSource()[0].GetPrim())
            vn = reader.GetInput("varname")
            if vn and vn.Get():
                st_name = str(vn.Get())
        if file_in:
            asset = file_in.Get()
            path = asset.resolvedPath if isinstance(asset, Sdf.AssetPath) else None
            if not path and asset:                # unresolved: try relative to the layer
                raw = asset.path if isinstance(asset, Sdf.AssetPath) else str(asset)
                cand = os.path.normpath(os.path.join(stage_dir, raw))
                path = cand if os.path.isfile(cand) else None
            return path, None, st_name
        return None, None, st_name
    val = diffuse.Get()
    return None, (np.asarray(val, np.float32) if val is not None else None), "st"


def _load_texture(path, cache):
    if path not in cache:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        cache[path] = np.asarray(img, dtype=np.float32) / 255.0
        _log(f"texture {os.path.basename(path)} {img.size[0]}x{img.size[1]}")
    return cache[path]


def load_usd_meshes(stage):
    from pxr import Usd, UsdGeom

    stage_dir = os.path.dirname(stage.GetRootLayer().realPath or ".")
    tex_cache: dict = {}
    meshes = []
    it = iter(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    for prim in it:
        img = UsdGeom.Imageable(prim)
        if img and img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            it.PruneChildren()
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if img.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not fvi:
            continue
        fvi = np.asarray(fvi, dtype=np.int64)
        # local -> world (USD row-vector matrix -> column-vector)
        M = usd_to_col(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        P = np.asarray(pts, dtype=np.float64)
        verts = (P @ M[:3, :3].T + M[:3, 3]).astype(np.float32)

        slots = _triangulate(counts, fvi)            # (T, 3) face-vertex slots
        tris = fvi[slots].astype(np.uint32)          # (T, 3) point indices
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            slots, tris = slots[:, ::-1], tris[:, ::-1]  # keep normals consistent for culling

        tex_path, const_color, st_name = _find_texture_and_color(prim, stage_dir)
        corner_uv = None
        pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar(st_name)
        if pv and pv.HasValue():
            st = np.asarray(pv.ComputeFlattened(), dtype=np.float32)   # resolves indices
            interp = pv.GetInterpolation()
            if interp == UsdGeom.Tokens.faceVarying and len(st) == len(fvi):
                corner_uv = st[slots]
            elif interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying) and len(st) == len(P):
                corner_uv = st[tris.astype(np.int64)]
            else:
                _log(f"WARNING {prim.GetPath()}: unsupported st interpolation {interp}")

        texture = _load_texture(tex_path, tex_cache) if (tex_path and corner_uv is not None) else None
        if const_color is None:
            dc = mesh.GetDisplayColorPrimvar().Get()
            const_color = np.asarray(dc[0], np.float32) if dc else np.full(3, 0.7, np.float32)
        meshes.append(MeshData(str(prim.GetPath()), verts, tris, corner_uv, texture, const_color,
                               texture_path=tex_path if texture is not None else None))
        _log(f"mesh {prim.GetPath()}: {len(verts)} verts, {len(tris)} tris, "
             f"texture={'yes' if texture is not None else 'no'}")
    return meshes


# ----------------------------------------------------------------------------
# Ray casting renderer
# ----------------------------------------------------------------------------

def _bilinear(tex, uv):
    """Sample (h, w, 3) texture at UVs (N, 2) with repeat wrap. USD UV origin is
    bottom-left, image row 0 is the top, hence the 1 - v flip."""
    h, w, _ = tex.shape
    u = np.mod(uv[:, 0], 1.0) * w - 0.5
    v = np.mod(1.0 - uv[:, 1], 1.0) * h - 0.5
    x0, y0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    x0m, x1m = np.mod(x0, w), np.mod(x0 + 1, w)
    y0m, y1m = np.mod(y0, h), np.mod(y0 + 1, h)
    return ((tex[y0m, x0m] * (1 - fx) + tex[y0m, x1m] * fx) * (1 - fy)
            + (tex[y1m, x0m] * (1 - fx) + tex[y1m, x1m] * fx) * fy)


class RayRenderer:
    def __init__(self, meshes, cull_backfaces=False):
        import open3d as o3d
        self.o3c = o3d.core
        self.scene = o3d.t.geometry.RaycastingScene()
        self.meshes = {}
        for m in meshes:
            gid = self.scene.add_triangles(m.verts, m.tris)
            self.meshes[gid] = m
        self.cull_backfaces = cull_backfaces
        self.miss_id = self.scene.INVALID_ID

    def _cast(self, origins, dirs):
        rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
        r = self.scene.cast_rays(self.o3c.Tensor(rays))
        return (r["t_hit"].numpy(), r["geometry_ids"].numpy(), r["primitive_ids"].numpy(),
                r["primitive_uvs"].numpy(), r["primitive_normals"].numpy())

    def cast(self, origin, dirs, max_layers=4):
        """Cast rays from one origin. dirs must have unit z in the camera frame so
        t_hit is z-depth. With cull_backfaces, rays that hit a triangle from behind
        are continued past it (lets a dome camera see into a room through walls)."""
        n = len(dirs)
        org = np.broadcast_to(origin.astype(np.float32), (n, 3)).copy()
        t_tot = np.zeros(n, np.float32)
        t, gid, pid, buv, nrm = self._cast(org, dirs)
        if self.cull_backfaces:
            todo = np.arange(n)
            for _ in range(max_layers):
                hit = gid[todo] != self.miss_id
                back = hit & (np.einsum("ij,ij->i", nrm[todo], dirs[todo]) > 0)
                idx = todo[back]
                if len(idx) == 0:
                    break
                step = t[idx] + 1e-4                         # nudge past the back face
                t_tot[idx] += step
                org[idx] += dirs[idx] * step[:, None]
                t2, g2, p2, b2, n2 = self._cast(org[idx], dirs[idx])
                t[idx], gid[idx], pid[idx], buv[idx], nrm[idx] = t2, g2, p2, b2, n2
                todo = idx
        return t + t_tot, gid, pid, buv

    def shade(self, gid, pid, buv, background):
        rgb = np.empty((len(gid), 3), np.float32)
        rgb[:] = background
        for g, m in self.meshes.items():
            sel = np.nonzero(gid == g)[0]
            if len(sel) == 0:
                continue
            if m.texture is None:
                rgb[sel] = m.base_color
                continue
            # Open3D barycentrics: p = (1 - a - b) v0 + a v1 + b v2
            a, b = buv[sel, 0:1], buv[sel, 1:2]
            cuv = m.corner_uv[pid[sel]]                  # (n, 3, 2)
            uv = (1 - a - b) * cuv[:, 0] + a * cuv[:, 1] + b * cuv[:, 2]
            rgb[sel] = _bilinear(m.texture, uv)
        return rgb

    def render(self, K, T_c2w_stage, W, H, supersample=1, background=(0.5, 0.5, 0.5)):
        """Returns (rgb uint8 HxWx3, z-depth float32 HxW in stage units, 0 = miss)."""
        R, origin = T_c2w_stage[:3, :3], T_c2w_stage[:3, 3]
        Kinv = np.linalg.inv(K)
        bg = np.asarray(background, np.float32)

        def rays_for(du, dv):
            v, u = np.mgrid[0:H, 0:W].astype(np.float64)
            pix = np.stack([u.ravel() + du, v.ravel() + dv, np.ones(H * W)], axis=0)
            d_cam = Kinv @ pix                            # z component == 1
            return (R @ d_cam).T.astype(np.float32)

        # Depth (and the s == 1 color) from exact pixel-center rays (contract convention).
        t, gid, pid, buv = self.cast(origin, rays_for(0.0, 0.0))
        hit = gid != self.miss_id
        depth = np.where(hit & np.isfinite(t), t, 0.0).astype(np.float32).reshape(H, W)

        if supersample <= 1:
            rgb = self.shade(gid, pid, buv, bg)
        else:                                             # color-only anti-aliasing
            s = supersample
            offs = (np.arange(s) + 0.5) / s - 0.5
            acc = np.zeros((H * W, 3), np.float32)
            for dv in offs:
                for du in offs:
                    _, g2, p2, b2 = self.cast(origin, rays_for(du, dv))
                    acc += self.shade(g2, p2, b2, bg)
            rgb = acc / (s * s)
        rgb8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(H, W, 3)
        return rgb8, depth


# ----------------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--usd", required=True)
    p.add_argument("--out", default="./out/renders")
    p.add_argument("--scene-id", default=None, help="defaults to the USD file stem")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mode", choices=["dome", "interior", "both", "focus"], default="dome",
                   help="both = dome poses followed by interior poses (each with its own "
                        "default lens and back-face culling); focus = only the "
                        "--focus-domes views (no wide dome / interior poses)")
    p.add_argument("--num-views", type=int, default=36)
    p.add_argument("--interior-num-views", type=int, default=None,
                   help="mode both: number of interior views (default: --num-views)")
    p.add_argument("--rings", default="15,35,60", help="dome elevation rings, degrees")
    p.add_argument("--radius-scale", type=float, default=1.0)
    p.add_argument("--focus-domes", default=None,
                   help="extra close-up domes appended to any mode, ';'-separated "
                        "'x,y,z,radius,num_views' in world meters (dome lens, back-face "
                        "culling on), e.g. a dome around a table for small objects")
    p.add_argument("--focus-rings", default="25,45,65",
                   help="elevation rings (degrees) for --focus-domes")
    lens = p.add_mutually_exclusive_group()
    lens.add_argument("--focal-length-mm", type=float, default=None)
    lens.add_argument("--hfov", type=float, default=None,
                      help="horizontal FOV in degrees (default: 55 dome, 90 interior; "
                           "applies to both halves in mode both)")
    p.add_argument("--h-aperture-mm", type=float, default=20.955)
    p.add_argument("--principal-point", choices=["exact", "half"], default="exact")
    p.add_argument("--eye-height", type=float, default=1.5, help="interior: meters above bbox floor")
    p.add_argument("--pitches", default="-30,-5,20", help="interior pitch rows (use --pitches=...)")
    p.add_argument("--interior-positions", type=int, default=3)
    p.add_argument("--interior-offset", type=float, default=0.35)
    p.add_argument("--cull-backfaces", action=argparse.BooleanOptionalAction, default=None,
                   help="see through back faces (default: on for dome, off for interior)")
    p.add_argument("--supersample", type=int, default=2, help="color anti-aliasing factor")
    p.add_argument("--max-empty-frac", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from pxr import Usd, UsdGeom

    W, H = args.width, args.height
    usd_path = os.path.abspath(args.usd)
    out_dir = os.path.abspath(args.out)
    os.makedirs(os.path.join(out_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        _log(f"ERROR: cannot open {usd_path}")
        return 1
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    up, up_idx = AXIS_VEC[up_axis], "XYZ".index(up_axis)
    _log(f"upAxis={up_axis} metersPerUnit={mpu}")

    meshes = load_usd_meshes(stage)
    if not meshes:
        _log("ERROR: no renderable meshes found")
        return 1
    allv = np.concatenate([m.verts for m in meshes])
    bmin_s, bmax_s = allv.min(0).astype(float), allv.max(0).astype(float)
    diag_s = float(np.linalg.norm(bmax_s - bmin_s))
    center_s = 0.5 * (bmin_s + bmax_s)
    _log(f"bounds (m) min={bmin_s * mpu} max={bmax_s * mpu}")

    h_ap = args.h_aperture_mm

    def lens_for(mode):
        f = (args.focal_length_mm if args.focal_length_mm is not None else
             focal_from_hfov(args.hfov or (90.0 if mode == "interior" else 55.0), h_ap))
        return f, intrinsics_from_camera(f, h_ap, W, H, args.principal_point)

    def cull_for(mode):
        return args.cull_backfaces if args.cull_backfaces is not None else (mode == "dome")

    # views: list of (pose_cv_stage, K, cull_backfaces, sub_mode)
    views = []
    if args.mode == "focus" and not args.focus_domes:
        _log("ERROR: --mode focus needs --focus-domes")
        return 1
    sub_modes = (["dome", "interior"] if args.mode == "both" else
                 [] if args.mode == "focus" else [args.mode])
    for sm in sub_modes:
        focal, K_sm = lens_for(sm)
        hfov = hfov_from_focal(focal, h_ap)
        vfov = math.degrees(2 * math.atan(0.5 * h_ap * H / W / focal))
        if sm == "dome":
            r_fit = 0.5 * diag_s / math.sin(math.radians(min(hfov, vfov)) / 2.0)
            poses = dome_poses(center_s, args.radius_scale * r_fit, _parse_floats(args.rings),
                               args.num_views, up, rng)
        else:
            n_int = (args.interior_num_views if (args.mode == "both" and
                                                 args.interior_num_views is not None)
                     else args.num_views)
            positions = interior_positions(bmin_s, bmax_s, up_idx, args.eye_height / mpu,
                                           args.interior_positions, args.interior_offset)
            poses = interior_poses(positions, _parse_floats(args.pitches), n_int, up, rng)
        views += [(T, K_sm, cull_for(sm), sm) for T in poses]
        _log(f"{len(poses)} poses ({sm}), hfov={hfov:.1f}, fx={K_sm[0, 0]:.1f}, "
             f"cull_backfaces={cull_for(sm)}, supersample={args.supersample}")
    if args.focus_domes:
        focal, K_f = lens_for("dome")
        for spec in str(args.focus_domes).split(";"):
            if not spec.strip():
                continue
            vals = [float(v) for v in spec.split(",")]
            if len(vals) != 5:
                raise SystemExit(f"--focus-domes entry needs x,y,z,radius,num_views: {spec!r}")
            c_s, r_s = np.asarray(vals[:3]) / mpu, vals[3] / mpu
            poses = dome_poses(c_s, r_s, _parse_floats(args.focus_rings), int(vals[4]), up, rng)
            views += [(T, K_f, cull_for("dome"), "focus")
                      for T in poses]
            _log(f"{len(poses)} poses (focus dome at {vals[:3]} r={vals[3]} m)")
    first_mode = sub_modes[0] if sub_modes else "dome"   # focus views use the dome lens
    hfov = hfov_from_focal(lens_for(first_mode)[0], h_ap)
    cull = cull_for(first_mode)

    renderer = RayRenderer(meshes, cull_backfaces=cull)
    meta = {
        "scene_id": args.scene_id or os.path.splitext(os.path.basename(usd_path))[0],
        "scene_usd": usd_path,
        "up_axis": up_axis,
        "meters_per_unit": mpu,
        "image_width": W,
        "image_height": H,
        "depth_type": "distance_to_image_plane",
        "camera_convention": "opencv",
        "scene_bounds_world": {"min": (bmin_s * mpu).tolist(), "max": (bmax_s * mpu).tolist()},
        "frames": [],
        "renderer": "open3d-raycast",
        "render_mode": args.mode,
        "hfov_deg": hfov,
        "skipped_views": [],
    }
    meta_path = os.path.join(out_dir, "meta.json")

    t0 = time.time()
    for vi, (T_cv_stage, K, view_cull, sub_mode) in enumerate(views):
        renderer.cull_backfaces = view_cull
        rgb8, depth_s = renderer.render(K, T_cv_stage, W, H, args.supersample)
        depth_m = (depth_s * mpu).astype(np.float32)
        empty = float((depth_m <= 0).mean())
        if empty > args.max_empty_frac:
            _log(f"view {vi}: {100 * empty:.1f}% empty depth, skipping")
            meta["skipped_views"].append({"view_index": vi, "reason": f"empty {empty:.3f}"})
            continue
        fid = f"{len(meta['frames']):04d}"
        _save_png(os.path.join(out_dir, "rgb", f"{fid}.png"), rgb8)
        np.save(os.path.join(out_dir, "depth", f"{fid}.npy"), depth_m)
        T_c2w = scale_translation(T_cv_stage, mpu)
        meta["frames"].append({
            "frame_id": fid,
            "rgb": f"rgb/{fid}.png",
            "depth": f"depth/{fid}.npy",
            "K": K.tolist(),
            "T_world_to_cam": rigid_inverse(T_c2w).tolist(),
            "T_cam_to_world": T_c2w.tolist(),
            "view_index": vi,
            **({"view_mode": sub_mode} if (args.mode == "both" or args.focus_domes) else {}),
        })
        _write_json_atomic(meta_path, meta)
        _log(f"view {vi} -> frame {fid} ({100 * (1 - empty):.0f}% depth hits)")

    _write_json_atomic(meta_path, meta)
    _log(f"wrote {len(meta['frames'])} frames ({len(meta['skipped_views'])} skipped) "
         f"in {time.time() - t0:.1f}s")
    if not meta["frames"]:
        return 1
    if args.verify and not verify_renders(out_dir):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
