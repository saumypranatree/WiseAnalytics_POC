#!/usr/bin/env python3
"""Step 1: decode the iPhone HEVC (HLG HDR, rotated) video, tone-map to SDR sRGB,
apply the display rotation, write every frame (JPEG), a blur score per frame,
a selected keyframe list, and a browser-playable H.264 .mp4.

usage: 01_extract.py VIDEO OUT_DIR [--stem IMG_5198] [--step 2] [--max-side 1920]
"""
import argparse, json, os, sys, time
import numpy as np
import av, cv2

M2020_TO_709 = np.array([[1.6605, -0.5876, -0.0728],
                         [-0.1246, 1.1329, -0.0083],
                         [-0.0182, -0.1006, 1.1187]], np.float32)


def hlg_inv_oetf(e):
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    e = np.clip(e, 0, 1)
    return np.where(e <= 0.5, (e * e) / 3.0, (np.exp((e - c) / a) + b) / 12.0)


def to_srgb8(frame, is_hlg, exposure):
    """VideoFrame -> HxWx3 uint8 sRGB (unrotated)."""
    if not is_hlg:
        return frame.to_ndarray(format="rgb24")
    # 16-bit RGB with the BT.2020 non-constant-luminance matrix (still HLG-encoded)
    rgb = frame.reformat(format="rgb48le", src_colorspace="BT2020",
                         dst_colorspace="BT2020", src_color_range="MPEG",
                         dst_color_range="JPEG").to_ndarray().astype(np.float32) / 65535.0
    lin = hlg_inv_oetf(rgb)                                  # scene-linear, [0, 1]
    # HLG OOTF (system gamma 1.2 at 1000 nits) on luminance
    y = (0.2627 * lin[..., 0] + 0.6780 * lin[..., 1] + 0.0593 * lin[..., 2])[..., None]
    lin = lin * np.power(np.maximum(y, 1e-6), 0.2)
    lin = lin @ M2020_TO_709.T
    lin = np.clip(lin * exposure, 0, None)
    lin = lin / (1.0 + lin / 4.0) * 1.25                    # soft shoulder (white ~ 4)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(np.clip(lin, 0, 1), 1 / 2.4) - 0.055)
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def rotate(img, rot):
    # PyAV/FFmpeg display-matrix angle: -90 => rotate 90 deg clockwise for display
    r = int(round(rot)) % 360
    if r == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if r == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if r == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out_dir")
    ap.add_argument("--stem", default=None)
    ap.add_argument("--exposure", type=float, default=4.0)
    ap.add_argument("--keyframes", default="auto",
                    help="target keyframe count, or 'auto' = 5 per second of video, clamped to 100..300 "
                         "(IMG_5200, 47 s: 100 keyframes split the SfM into 2 pieces of 30+49 frames; "
                         "250 gave one 129-frame piece covering the whole 5.6 m facade)")
    ap.add_argument("--mp4", default=None, help="H.264 output path")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.join(a.out_dir, "frames"), exist_ok=True)
    c = av.open(a.video)
    vs = c.streams.video[0]
    cc = vs.codec_context
    is_hlg = (cc.color_trc == 18)
    fps = float(vs.average_rate)
    enc = None
    if a.mp4:
        os.makedirs(os.path.dirname(os.path.abspath(a.mp4)), exist_ok=True)
        oc = av.open(a.mp4, "w", options={"movflags": "+faststart"})
        enc = None
    scores, names = [], []
    for i, fr in enumerate(c.decode(vs)):
        img = rotate(to_srgb8(fr, is_hlg, a.exposure), fr.rotation or 0)
        name = f"{i:05d}.jpg"
        cv2.imwrite(os.path.join(a.out_dir, "frames", name), img[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        g = cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_RGB2GRAY)
        scores.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        names.append(name)
        if a.mp4:
            if enc is None:
                enc = oc.add_stream("libx264", rate=vs.average_rate)
                enc.width, enc.height = img.shape[1], img.shape[0]
                enc.pix_fmt = "yuv420p"
                enc.options = {"crf": "20", "preset": "medium", "profile": "high"}
                enc.codec_context.color_primaries = 1; enc.codec_context.color_trc = 1
                enc.codec_context.colorspace = 1
            vf = av.VideoFrame.from_ndarray(img, format="rgb24")
            vf = vf.reformat(format="yuv420p", dst_colorspace="ITU709", dst_color_range="MPEG")
            for p in enc.encode(vf):
                oc.mux(p)
    if a.mp4:
        for p in enc.encode():
            oc.mux(p)
        oc.close()
    n = len(names)
    # keyframes: split into `keyframes` equal windows, keep the sharpest of each
    if str(a.keyframes) == "auto":
        k = int(np.clip(round(5 * n / max(fps, 1e-6)), 100, 300))
    else:
        k = int(a.keyframes)
    k = min(k, n)
    edges = np.linspace(0, n, k + 1).astype(int)
    sel = []
    s = np.asarray(scores)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            sel.append(int(lo + np.argmax(s[lo:hi])))
    info = {"video": a.video, "num_frames": n, "fps": fps, "hlg_tonemapped": is_hlg,
            "exposure": a.exposure, "frame_size_wh": [int(img.shape[1]), int(img.shape[0])],
            "blur_scores": scores, "keyframes": [names[i] for i in sel],
            "blur_median": float(np.median(s)), "keyframe_blur_median": float(np.median(s[sel])),
            "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(a.out_dir, "frames.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k_: v for k_, v in info.items() if k_ not in ("blur_scores", "keyframes")}))
    print("keyframes:", len(sel))


if __name__ == "__main__":
    main()
