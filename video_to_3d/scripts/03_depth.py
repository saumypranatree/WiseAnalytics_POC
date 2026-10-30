#!/usr/bin/env python3
"""Step 3: dense per-frame depth = Depth Anything V2 Small (relative inverse depth)
fitted per frame to the SfM sparse depths (1/z = a*d + b, robust IRLS).

Also undistorts images to a pinhole camera (same K) at --scale resolution.
usage: 03_depth.py WORK_DIR [--sfm sfm_sr] [--scale 0.5]
Writes WORK_DIR/dense/{rgb/*.jpg, depth/*.npy, cameras.json}
"""
import argparse, json, os, time
import numpy as np, cv2, torch, pycolmap
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def robust_fit(d, invz, iters=10):
    A = np.stack([d, np.ones_like(d)], 1)
    w = np.ones_like(d)
    for _ in range(iters):
        sw = np.sqrt(w)
        x, *_ = np.linalg.lstsq(A * sw[:, None], invz * sw, rcond=None)
        r = A @ x - invz
        s = 1.4826 * np.median(np.abs(r)) + 1e-12
        w = 1.0 / np.maximum(1.0, np.abs(r) / (2.0 * s))       # Huber
    return x, r, s


def correction_field(uv, logr, W, H, gx=16, gy=28, sigma=1.5):
    """Smooth multiplicative depth correction from sparse log-ratios (robust per-cell
    median, then normalized Gaussian fill/smoothing), upsampled to H x W."""
    cx = np.clip((uv[:, 0] / W * gx).astype(int), 0, gx - 1)
    cy = np.clip((uv[:, 1] / H * gy).astype(int), 0, gy - 1)
    val = np.zeros((gy, gx), np.float32); wt = np.zeros((gy, gx), np.float32)
    cell = cy * gx + cx
    for c in np.unique(cell):
        m = cell == c
        if m.sum() >= 3:
            val.flat[c] = np.median(logr[m]); wt.flat[c] = min(m.sum(), 20)
    k = int(3 * sigma) * 2 + 1
    num = cv2.GaussianBlur(val * wt, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
    den = cv2.GaussianBlur(wt, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
    g = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 0.0)
    # far from any data, fall back to no correction
    g = g * np.clip(den / 2.0, 0, 1)
    return cv2.resize(g.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work"); ap.add_argument("--sfm", default="sfm_sr")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--model-dir", default=None, help="local model path (offline)")
    a = ap.parse_args()
    t0 = time.time()
    rec = pycolmap.Reconstruction(os.path.join(a.work, a.sfm, "sparse", "0"))
    cam = next(iter(rec.cameras.values()))
    f, cx, cy, k1 = cam.params
    W0, H0 = cam.width, cam.height
    K0 = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float64)
    W, H = int(round(W0 * a.scale)), int(round(H0 * a.scale))
    K = K0.copy(); K[:2] *= a.scale
    mapx, mapy = cv2.initUndistortRectifyMap(K0, np.array([k1, 0, 0, 0]), None, K, (W, H), cv2.CV_32FC1)

    dev = "cuda"
    src = a.model_dir or MODEL
    proc = AutoImageProcessor.from_pretrained(src)
    model = AutoModelForDepthEstimation.from_pretrained(src).to(dev).eval().half()
    out = os.path.join(a.work, "dense")
    os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out, "depth"), exist_ok=True)
    frames = []
    img_dir = os.path.join(a.work, a.sfm, "images")
    for iid in sorted(rec.images, key=lambda i: rec.images[i].name):
        im = rec.images[iid]
        bgr = cv2.imread(os.path.join(img_dir, im.name))
        und = cv2.remap(bgr, mapx, mapy, cv2.INTER_AREA if a.scale < 1 else cv2.INTER_LINEAR)
        rgb = und[..., ::-1].copy()
        with torch.no_grad():
            inp = proc(images=rgb, return_tensors="pt").to(dev)
            pred = model(pixel_values=inp["pixel_values"].half()).predicted_depth  # (1,h,w)
            disp = torch.nn.functional.interpolate(pred[:, None].float(), size=(H, W),
                                                   mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        # sparse SfM depths in this view
        T = im.cam_from_world().matrix()            # 3x4
        uv, z = [], []
        for p2 in im.points2D:
            if p2.has_point3D():
                X = rec.points3D[p2.point3D_id].xyz
                Xc = T[:, :3] @ X + T[:, 3]
                if Xc[2] <= 0:
                    continue
                q = K @ (Xc / Xc[2])
                uv.append(q[:2]); z.append(Xc[2])
        uv, z = np.asarray(uv), np.asarray(z)
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        uv, z = uv[inb], z[inb]
        dsp = cv2.remap(disp, uv[:, 0].astype(np.float32)[None], uv[:, 1].astype(np.float32)[None],
                        cv2.INTER_LINEAR)[0]
        (sa, sb), r, s = robust_fit(dsp, 1.0 / z)
        invz = sa * disp + sb
        valid = invz > (1.0 / (z.max() * 3.0))
        depth = np.where(valid, 1.0 / np.maximum(invz, 1e-9), 0.0).astype(np.float32)
        zfit = 1.0 / np.maximum(sa * dsp + sb, 1e-9)
        rel0 = np.abs(zfit - z) / z
        ok = rel0 < 0.3
        logr = np.clip(np.log(z / zfit), -0.3, 0.3)
        corr = correction_field(uv[ok], logr[ok], W, H)
        depth = np.where(depth > 0, depth * np.exp(corr), 0.0).astype(np.float32)
        zc = zfit * np.exp(cv2.remap(corr, uv[:, 0].astype(np.float32)[None],
                                     uv[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0])
        rel = np.abs(zc - z) / z
        stem = os.path.splitext(im.name)[0]
        cv2.imwrite(os.path.join(out, "rgb", stem + ".jpg"), und, [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.save(os.path.join(out, "depth", stem + ".npy"), depth)
        T44 = np.eye(4); T44[:3] = T
        frames.append({"name": stem, "T_cam_from_world": T44.tolist(), "n_sparse": int(len(z)),
                       "fit_a": float(sa), "fit_b": float(sb),
                       "sparse_rel_err_median_affine": float(np.median(rel0)),
                       "sparse_rel_err_median": float(np.median(rel)),
                       "sparse_rel_err_p90": float(np.percentile(rel, 90)),
                       "depth_median": float(np.median(depth[depth > 0]))})
        print(f"{stem}: n={len(z)} affine med={np.median(rel0):.3f} corrected med={np.median(rel):.3f} p90={np.percentile(rel, 90):.3f}",
              flush=True)
    json.dump({"K": K.tolist(), "width": W, "height": H, "model": MODEL, "frames": frames,
               "seconds": round(time.time() - t0, 1)},
              open(os.path.join(out, "cameras.json"), "w"), indent=1)
    print("seconds", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
