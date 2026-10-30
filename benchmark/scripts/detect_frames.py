"""Stage A: run the three void detectors on the ORIGINAL video keyframes of a shelf scan (2D only).

    benchmark/.venv/bin/python scripts/detect_frames.py <scan_dir> [--conf 0.25]

Models (all single class "void"):
  frames_rfdetr   RF-DETR Small fine-tuned here on the public Kaggle void dataset   (benchmark only: CC BY-NC-SA data)
  frames_yolo     YOLOv8s fine-tuned here on the same data                           (benchmark only)
  frames_yolo_dl  akul-29/Retail-Shelf-Gap-Detection_Model, as downloaded

Frames: <scan>/convert/work/sfm_sr/images/<keyframe>.jpg, the sharp frames cut from the video, for every
keyframe that has a solved camera (convert/work/cameras_usd.json). Box coordinates are in pixels of that
image, exactly like the labels drawn in the app, so stage B (open_vocab_3d/frames_gaps.py) can place them
on the 3D model with the same camera maths.

Tall portrait frames (1080x1920) would be squeezed to ~360 px wide by a 640 px model, while the training
photos are 640x640 squares. So each frame is cut into overlapping SQUARE tiles (side = frame width, ~25 %
overlap), each tile goes through the model, and the boxes are mapped back and merged with NMS.

Writes <scan>/results/frame_detections/detections.json and overlay images of the frames that have
detections (<model>/<keyframe>.jpg, 540 px wide).
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms

B = Path(__file__).resolve().parent.parent
ROOT = B.parent
MODELS = {
    "frames_rfdetr": ("RF-DETR Small, fine-tuned on the public void dataset", B / "runs/rfdetr_small_void/checkpoint_best_total.pth"),
    "frames_yolo": ("YOLOv8s, fine-tuned on the public void dataset", B / "runs/yolov8s_void/weights/best.pt"),
    "frames_yolo_dl": ("downloaded YOLOv8 (akul-29/Retail-Shelf-Gap-Detection_Model)", ROOT / "video_to_3d/models/gap_yolov8/best.pt"),
}
COLORS = {"frames_rfdetr": (167, 139, 250), "frames_yolo": (253, 224, 71), "frames_yolo_dl": (248, 113, 113)}


def tiles(w: int, h: int, overlap: float = 0.25):
    """Overlapping square tiles (x0, y0, x1, y1) covering a w x h frame; side = the shorter frame side."""
    side = min(w, h)
    if w == side and h == side:
        return [(0, 0, w, h)]
    long_ = max(w, h)
    n = int(np.ceil((long_ - side) / (side * (1 - overlap)))) + 1
    starts = np.linspace(0, long_ - side, n).round().astype(int)
    return [(0, int(s), w, int(s) + side) if h > w else (int(s), 0, int(s) + side, h) for s in starts]


def load_detectors():
    det = {}
    from ultralytics import YOLO
    for key in ("frames_yolo", "frames_yolo_dl"):
        m = YOLO(str(MODELS[key][1]))
        det[key] = lambda im, m=m, c=0.001: _yolo(m, im, c)
    from rfdetr import RFDETRSmall
    r = RFDETRSmall(pretrain_weights=str(MODELS["frames_rfdetr"][1]))
    det["frames_rfdetr"] = lambda im, r=r, c=0.001: _rfdetr(r, im, c)
    return det


def _yolo(m, im, conf):
    r = m.predict(im, conf=conf, imgsz=640, verbose=False, max_det=300)[0]
    return r.boxes.xyxy.cpu().numpy().reshape(-1, 4), r.boxes.conf.cpu().numpy()


def _rfdetr(m, im, conf):
    d = m.predict(im, threshold=conf)
    return np.asarray(d.xyxy, float).reshape(-1, 4), np.asarray(d.confidence, float).reshape(-1)


def detect_frame(predict, im: Image.Image, conf: float, tile_list):
    """All tiles through the model, boxes mapped back to frame pixels, merged with NMS."""
    boxes, scores = [], []
    for (x0, y0, x1, y1) in tile_list:
        b, s = predict(im.crop((x0, y0, x1, y1)))
        keep = s >= conf
        if keep.any():
            boxes.append(b[keep] + np.array([x0, y0, x0, y0]))
            scores.append(s[keep])
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0)
    b = np.concatenate(boxes); s = np.concatenate(scores)
    k = nms(torch.tensor(b, dtype=torch.float32), torch.tensor(s, dtype=torch.float32), 0.5).numpy()
    return b[k], s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--overlap", type=float, default=0.25)
    a = ap.parse_args()
    sdir = Path(a.scan_dir).resolve()
    cams = json.loads((sdir / "convert/work/cameras_usd.json").read_text())
    names = [k["name"] for k in cams["keyframes"]]
    W, H = cams["camera"]["width"], cams["camera"]["height"]
    tl = tiles(W, H, a.overlap)
    out = sdir / "results" / "frame_detections"
    out.mkdir(parents=True, exist_ok=True)
    detectors = load_detectors()
    result = {"scan": sdir.name, "frame_size": [W, H], "tiles": tl, "conf": a.conf, "keyframes": len(names),
              "models": {}, "seconds": {}}
    for key, predict in detectors.items():
        t0 = time.time()
        (out / key).mkdir(exist_ok=True)
        per = {}
        for n in names:
            im = Image.open(sdir / "convert/work/sfm_sr/images" / f"{n}.jpg").convert("RGB")
            b, s = detect_frame(predict, im, a.conf, tl)
            per[n] = [{"box_px": [round(float(v), 1) for v in bb], "score": round(float(ss), 4)} for bb, ss in zip(b, s)]
            if len(b):
                ov = im.copy(); d = ImageDraw.Draw(ov)
                for bb, ss in zip(b, s):
                    d.rectangle([float(v) for v in bb], outline=COLORS[key], width=6)
                    d.text((float(bb[0]) + 6, float(bb[1]) + 4), f"{ss:.2f}", fill=COLORS[key])
                ov.resize((540, int(540 * H / W))).save(out / key / f"{n}.jpg", quality=80)
        result["models"][key] = per
        result["seconds"][key] = round(time.time() - t0, 1)
        n_det = sum(len(v) for v in per.values())
        n_fr = sum(1 for v in per.values() if v)
        print(f"{sdir.name} {key}: {n_det} boxes on {n_fr}/{len(names)} frames in {result['seconds'][key]} s", flush=True)
    (out / "detections.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
