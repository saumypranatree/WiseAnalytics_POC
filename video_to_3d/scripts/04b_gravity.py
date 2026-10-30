#!/usr/bin/env python3
"""Step 4b: estimate gravity (up) and the shelf-facade normal from the fused mesh.

facade normal n: area-weighted mean of the triangle normals facing the cameras
(iterated mode, |cos| > 0.8). Up g: the direction inside the facade plane that
maximises the area of horizontal triangles (|n.g| > 0.97: shelf tops/undersides,
product tops), searched +-45 deg around the camera up vector projected into
that plane (0.25 deg steps); sign from the camera up vector.

usage: 04b_gravity.py WORK_DIR
Writes WORK_DIR/mesh/{gravity_up_sfm.npy, facade_normal_sfm.npy, gravity_info.json}
"""
import json, os, sys
import numpy as np, open3d as o3d


def main():
    work = sys.argv[1]
    m = o3d.io.read_triangle_mesh(os.path.join(work, "mesh", "fused.ply"))
    m.compute_triangle_normals()
    V = np.asarray(m.vertices); T = np.asarray(m.triangles); N = np.asarray(m.triangle_normals)
    A = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    cams = json.load(open(os.path.join(work, "dense", "cameras.json")))
    Rs = np.array([np.array(f["T_cam_from_world"])[:3, :3] for f in cams["frames"]])
    ts = np.array([np.array(f["T_cam_from_world"])[:3, 3] for f in cams["frames"]])
    C = np.einsum("nji,nj->ni", Rs, -ts)
    camup = -Rs[:, 1].mean(0); camup /= np.linalg.norm(camup)
    fwd = Rs[:, 2].mean(0); fwd /= np.linalg.norm(fwd)
    nf = -fwd
    for _ in range(5):
        sel = N @ nf > 0.8
        nf = (N[sel] * A[sel, None]).sum(0); nf /= np.linalg.norm(nf)
    e1 = camup - nf * (camup @ nf); e1 /= np.linalg.norm(e1); e2 = np.cross(nf, e1)
    ths = np.radians(np.arange(-45, 45.01, 0.25))
    sc = [A[np.abs(N @ (np.cos(t) * e1 + np.sin(t) * e2)) > 0.97].sum() for t in ths]
    th = ths[int(np.argmax(sc))]
    g = np.cos(th) * e1 + np.sin(th) * e2
    cc = C - C.mean(0); motion = np.linalg.eigh(cc.T @ cc)[1][:, -1]
    info = {"facade_normal": nf.tolist(), "up": g.tolist(),
            "facade_area_share": float(A[N @ nf > 0.9].sum() / A.sum()),
            "up_offset_from_projected_camera_up_deg": float(np.degrees(th)),
            "angle_up_vs_mean_camera_up_deg": float(np.degrees(np.arccos(np.clip(g @ camup, -1, 1)))),
            "pan_direction_deviation_from_horizontal_deg": float(np.degrees(np.arcsin(abs(g @ motion)))),
            "horizontal_area_share": float(max(sc) / A.sum())}
    np.save(os.path.join(work, "mesh", "gravity_up_sfm.npy"), g)
    np.save(os.path.join(work, "mesh", "facade_normal_sfm.npy"), nf)
    json.dump(info, open(os.path.join(work, "mesh", "gravity_info.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
