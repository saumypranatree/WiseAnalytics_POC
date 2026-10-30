#!/usr/bin/env python3
"""Step 6: gravity-align (+Y up), apply a provisional metric scale, and write a
UV-textured .usdz (UsdGeom.Mesh + UsdPreviewSurface + UsdUVTexture, faceVarying st).

Frame: +Y = estimated gravity up; +Z = horizontal shelf-facade normal (points
from the shelves toward the aisle / camera); X = Y x Z. Origin: x,z at the
centre of the scan footprint, y = 0 at the lowest scanned geometry (0.5th pct).
Scale (PROVISIONAL): the scanned vertical extent (0.5..99.5 pct) is set to
--height-m metres (default 1.9 m; the pan covers roughly floor-to-top-shelf).

usage: 06_export_usdz.py WORK_DIR OUT_USDZ [--height-m 1.9]
"""
import argparse, json, os, shutil, tempfile
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work"); ap.add_argument("out")
    ap.add_argument("--height-m", type=float, default=1.9)
    a = ap.parse_args()
    from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt, UsdUtils
    tex_dir = os.path.join(a.work, "tex")
    d = np.load(os.path.join(tex_dir, "textured.npz"))
    V, F, st, tri_atlas = d["V"], d["F"], d["st"], d["tri_atlas"]
    tinfo = json.load(open(os.path.join(tex_dir, "texture_info.json")))
    g = np.load(os.path.join(a.work, "mesh", "gravity_up_sfm.npy"))
    nf = np.load(os.path.join(a.work, "mesh", "facade_normal_sfm.npy"))
    y = g / np.linalg.norm(g)
    z = nf - y * (nf @ y); z /= np.linalg.norm(z)
    x = np.cross(y, z)
    R = np.stack([x, y, z])                                   # rows: new axes in SfM coords
    assert np.linalg.det(R) > 0.999
    P = V @ R.T
    h = P[:, 1]
    lo, hi = np.percentile(h, 0.5), np.percentile(h, 99.5)
    s = a.height_m / (hi - lo)
    P = P * s
    P[:, 1] -= lo * s
    c = 0.5 * (np.percentile(P, 0.5, 0) + np.percentile(P, 99.5, 0))
    P[:, 0] -= c[0]; P[:, 2] -= c[2]
    tri_atlas = np.where(tri_atlas < 0, 0, tri_atlas)          # degenerate faces xatlas skipped

    work = tempfile.mkdtemp(prefix="usdz_", dir=os.path.dirname(os.path.abspath(a.out)))
    stem = os.path.splitext(os.path.basename(a.out))[0]
    os.makedirs(os.path.join(work, "textures"))
    root = os.path.join(work, f"{stem}.usdc")
    stage = Usd.Stage.CreateNew(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().SetCustomDataByKey("video_to_3d", {
        "source": "phone video -> pycolmap SfM + Depth Anything V2 Small + TSDF + xatlas",
        "scale_note": f"PROVISIONAL scale: vertical extent set to {a.height_m} m",
        "sfm_to_world_scale": float(s)})
    for ai, tx in enumerate(tinfo["textures"]):
        sel = np.nonzero(tri_atlas == ai)[0]
        if len(sel) == 0:
            continue
        used, inv = np.unique(F[sel].ravel(), return_inverse=True)
        pts = P[used].astype(np.float32)
        fvi = inv.reshape(-1, 3).astype(np.int32)
        shutil.copy(os.path.join(tex_dir, tx["file"]), os.path.join(work, "textures", tx["file"]))
        mp = f"/World/Scan_{ai}"
        mesh = UsdGeom.Mesh.Define(stage, mp)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(sel), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(fvi.ravel()))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*pts.min(0).tolist()), Gf.Vec3f(*pts.max(0).tolist())]))
        stpv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                       UsdGeom.Tokens.faceVarying)
        stpv.Set(Vt.Vec2fArray.FromNumpy(st[sel].reshape(-1, 2).astype(np.float32)))
        mesh.CreateDisplayColorPrimvar().Set(Vt.Vec3fArray([Gf.Vec3f(0.5, 0.5, 0.5)]))
        mat = UsdShade.Material.Define(stage, f"/World/Looks/Mat_{ai}")
        sh = UsdShade.Shader.Define(stage, f"/World/Looks/Mat_{ai}/PreviewSurface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        rd = UsdShade.Shader.Define(stage, f"/World/Looks/Mat_{ai}/stReader")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        tex = UsdShade.Shader.Define(stage, f"/World/Looks/Mat_{ai}/diffuseTexture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"textures/{tx['file']}")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if os.path.exists(a.out):
        os.remove(a.out)
    ok = UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(root), a.out)
    shutil.rmtree(work)
    bmin, bmax = P.min(0), P.max(0)
    info = {"usdz": os.path.abspath(a.out), "ok": bool(ok), "bytes": os.path.getsize(a.out),
            "scale_m_per_sfm_unit": float(s), "height_m_assumed": a.height_m,
            "bounds_m": {"min": bmin.round(3).tolist(), "max": bmax.round(3).tolist()},
            "triangles": int(len(F)), "R_sfm_to_world_rows": R.tolist(),
            # full similarity: x_usd = s * R @ x_sfm - offset  (07_cameras.py uses it)
            "T_sfm_to_usd": {"s": float(s), "R": R.tolist(),
                             "offset": [float(c[0]), float(lo * s), float(c[2])]}}
    json.dump(info, open(os.path.join(a.work, "export_info.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
