#!/usr/bin/env python3
"""Step 7: export every registered keyframe's camera in the exported USD frame.

Labelling and gap detection work on the original keyframes and need their cameras in
the scan's coordinates. x_usd = s * R @ x_sfm - offset is the similarity applied by
06_export_usdz.py (read from export_info.json, or recomputed exactly as 06 does for
scans exported before it was recorded).

Per keyframe: R_cam_from_usd (3x3) and C_usd (camera centre). A pixel (u, v) of the
ORIGINAL (distorted) keyframe maps to the USD-frame ray direction
    R_cam_from_usd.T @ normalize(undistort((u - cx) / f, (v - cy) / f), 1)
with the SIMPLE_RADIAL k1 model. Self-check: COLMAP's own 3D points, moved to the
USD frame and projected with the exported cameras, must land on their recorded
2D observations (mean error printed and stored).

usage: 07_cameras.py WORK_DIR [--sfm sfm_sr]
Writes WORK_DIR/cameras_usd.json
"""
import argparse, json, os
import numpy as np
import pycolmap


def sfm_to_usd(work):
    info = json.load(open(os.path.join(work, "export_info.json")))
    if "T_sfm_to_usd" in info:
        T = info["T_sfm_to_usd"]
        return float(T["s"]), np.array(T["R"]), np.array(T["offset"])
    # recompute exactly as 06_export_usdz.py does
    d = np.load(os.path.join(work, "tex", "textured.npz"))
    V = d["V"]
    g = np.load(os.path.join(work, "mesh", "gravity_up_sfm.npy"))
    nf = np.load(os.path.join(work, "mesh", "facade_normal_sfm.npy"))
    y = g / np.linalg.norm(g)
    z = nf - y * (nf @ y); z /= np.linalg.norm(z)
    R = np.stack([np.cross(y, z), y, z])
    P = V @ R.T
    lo, hi = np.percentile(P[:, 1], 0.5), np.percentile(P[:, 1], 99.5)
    s = info["height_m_assumed"] / (hi - lo)
    P = P * s
    P[:, 1] -= lo * s
    c = 0.5 * (np.percentile(P, 0.5, 0) + np.percentile(P, 99.5, 0))
    return float(s), R, np.array([c[0], lo * s, c[2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work"); ap.add_argument("--sfm", default="sfm_sr")
    a = ap.parse_args()
    s, R, off = sfm_to_usd(a.work)
    rec = pycolmap.Reconstruction(os.path.join(a.work, a.sfm, "sparse", "0"))
    cam = next(iter(rec.cameras.values()))
    if cam.model_name != "SIMPLE_RADIAL":
        raise SystemExit(f"unsupported camera model {cam.model_name}")
    f, cx, cy, k1 = map(float, cam.params)
    finfo = json.load(open(os.path.join(a.work, "frames.json")))
    fps = float(finfo["fps"])
    kfs, errs = [], []
    for iid in sorted(rec.images, key=lambda i: rec.images[i].name):
        im = rec.images[iid]
        M = im.cam_from_world().matrix()                    # 3x4, sfm frame
        Rcw, tcw = M[:, :3], M[:, 3]
        R_cu = Rcw @ R.T                                    # cam <- usd (rotation part)
        C_sfm = -Rcw.T @ tcw
        C_usd = s * R @ C_sfm - off
        name = os.path.splitext(im.name)[0]
        kfs.append({"name": name, "frame": int(name), "time_s": round(int(name) / fps, 4),
                    "R_cam_from_usd": R_cu.tolist(), "C_usd": C_usd.tolist()})
        # self-check on this image's observed 3D points
        pts = [(p2.xy, rec.points3D[p2.point3D_id].xyz) for p2 in im.points2D if p2.has_point3D()]
        if pts:
            xy = np.array([p[0] for p in pts]); X = np.array([p[1] for p in pts])
            Xu = X @ (s * R).T - off                        # sfm -> usd
            Xc = (Xu - C_usd) @ R_cu.T                      # usd -> cam (sfm-scaled metric is irrelevant)
            x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
            r2 = x * x + y * y
            u = f * x * (1 + k1 * r2) + cx; v = f * y * (1 + k1 * r2) + cy
            errs.append(np.hypot(u - xy[:, 0], v - xy[:, 1]))
    err = float(np.mean(np.concatenate(errs))) if errs else None
    out = {"camera": {"model": "SIMPLE_RADIAL", "width": int(cam.width), "height": int(cam.height),
                      "f": f, "cx": cx, "cy": cy, "k1": k1},
           "fps": fps, "num_frames": int(finfo["num_frames"]),
           "T_sfm_to_usd": {"s": s, "R": R.tolist(), "offset": off.tolist()},
           "reprojection_check_px": err, "keyframes": kfs}
    json.dump(out, open(os.path.join(a.work, "cameras_usd.json"), "w"))
    print(json.dumps({"keyframes": len(kfs), "reprojection_check_px": err}))


if __name__ == "__main__":
    main()
