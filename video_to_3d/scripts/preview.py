#!/usr/bin/env python3
"""Debug preview: raycast a PLY (vertex colours) from SfM camera(s) and from an
oblique side view; writes a side-by-side PNG with the real frame.
usage: preview.py WORK_DIR MESH.ply OUT.png [--frames 00000,00128] [--oblique 35]
"""
import argparse, json, os
import numpy as np, cv2, open3d as o3d


def render(mesh, K, T_cw, W, H, shade=False):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    v, u = np.mgrid[0:H, 0:W]
    pix = np.stack([u.ravel() + .5, v.ravel() + .5, np.ones(H * W)], 0)
    R, t = T_cw[:3, :3], T_cw[:3, 3]
    d = (R.T @ (np.linalg.inv(K) @ pix)).T
    o = np.broadcast_to(-R.T @ t, d.shape)
    r = sc.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))
    pid = r["primitive_ids"].numpy().astype(np.int64); hit = r["t_hit"].numpy() < np.inf
    uv = r["primitive_uvs"].numpy()
    tri = np.asarray(mesh.triangles); col = np.asarray(mesh.vertex_colors)
    out = np.full((H * W, 3), 0.5)
    p = pid[hit]; a, b = uv[hit, 0:1], uv[hit, 1:2]
    out[hit] = (1 - a - b) * col[tri[p, 0]] + a * col[tri[p, 1]] + b * col[tri[p, 2]]
    if shade:
        n = r["primitive_normals"].numpy()[hit]
        out[hit] *= (0.35 + 0.65 * np.abs(np.einsum("ij,ij->i", n, d[hit] / np.linalg.norm(d[hit], axis=1, keepdims=True))))[:, None]
    return (np.clip(out, 0, 1).reshape(H, W, 3) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work"); ap.add_argument("mesh"); ap.add_argument("out")
    ap.add_argument("--frames", default="00000,00128,00257")
    ap.add_argument("--oblique", type=float, default=40.0)
    a = ap.parse_args()
    cams = json.load(open(os.path.join(a.work, "dense", "cameras.json")))
    K = np.array(cams["K"]); W, H = cams["width"] // 2, cams["height"] // 2
    K2 = K.copy(); K2[:2] *= 0.5
    mesh = o3d.io.read_triangle_mesh(a.mesh)
    fr = {f["name"]: f for f in cams["frames"]}
    rows = []
    for n in a.frames.split(","):
        n = min(fr, key=lambda k: abs(int(k) - int(n)))
        T = np.array(fr[n]["T_cam_from_world"])
        real = cv2.resize(cv2.imread(os.path.join(a.work, "dense", "rgb", n + ".jpg")), (W, H))[..., ::-1]
        ren = render(mesh, K2, T, W, H)
        # oblique: rotate camera about its own vertical axis around the look-at point
        dm = fr[n]["depth_median"]
        R, t = T[:3, :3], T[:3, 3]; C = -R.T @ t; P = C + R[2] * dm
        ang = np.radians(a.oblique); Ry = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
        R2 = Ry @ R; C2 = P - R2[2] * dm * 1.3
        T2 = np.eye(4); T2[:3, :3] = R2; T2[:3, 3] = -R2 @ C2
        obl = render(mesh, K2, T2, W, H, shade=True)
        rows.append(np.hstack([real, ren, obl]))
    cv2.imwrite(a.out, np.vstack(rows)[..., ::-1])


if __name__ == "__main__":
    main()
