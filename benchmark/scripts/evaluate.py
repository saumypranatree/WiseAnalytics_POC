"""Benchmark: void (empty shelf space) detectors, one evaluator for all.

Test set: data/split/test (20 held-out, un-augmented supermarket photos, 85 void boxes) of the Kaggle
'Retail Shelf Void Detection Dataset' (CC BY-NC-SA 4.0; benchmark use only).
Models (all single-class; every prediction counts as 'void'):
  yolov8s_downloaded   akul-29/Retail-Shelf-Gap-Detection_Model (as used in the shelf-scan pipeline)
  yolov8s_trained      YOLOv8s fine-tuned here on the clean train split, 50 epochs
  rfdetr_small_trained RF-DETR Small fine-tuned here on the same split, 50 epochs
Metrics: COCO mAP@50 and mAP@50:95 (pycocotools), precision / recall at conf >= 0.25 and IoU >= 0.5,
a 95 % bootstrap interval on mAP@50 (images resampled), and GPU latency per image.
Also runs every model on the face-on renders of the 4 store aisles (no ground truth there yet) and
saves overlays.
"""
import contextlib, io, json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

B = Path(__file__).resolve().parent.parent
TEST = B / "data" / "split" / "test"
OUT = B / "results"; OUT.mkdir(exist_ok=True)
AISLES = Path("/home/saumy/pranatree_work/room_scan/open_vocab_3d/out/scans")
EVAL_CONF, MATCH_IOU = 0.25, 0.5


def load_models():
    models = {}
    from ultralytics import YOLO
    dl = Path("/home/saumy/pranatree_work/room_scan/video_to_3d/models/gap_yolov8/best.pt")
    tr = B / "runs" / "yolov8s_void" / "weights" / "best.pt"
    for name, p in (("yolov8s_downloaded", dl), ("yolov8s_trained", tr)):
        if p.is_file():
            m = YOLO(str(p))
            models[name] = lambda im, m=m: _yolo(m, im)
    # checkpoint_best_total.pth = RF-DETR's recommended final weights (best of regular and EMA)
    ck = [p for p in [B / "runs" / "rfdetr_small_void" / "checkpoint_best_total.pth"] if p.is_file()]
    if ck:
        from rfdetr import RFDETRSmall
        m = RFDETRSmall(pretrain_weights=str(ck[0]))
        models["rfdetr_small_trained"] = lambda im, m=m: _rfdetr(m, im)
        print("rfdetr checkpoint:", ck[0].name)
    return models


def _yolo(m, im):
    r = m.predict(im, conf=0.001, imgsz=640, verbose=False, max_det=300)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()


def _rfdetr(m, im):
    d = m.predict(im, threshold=0.001)
    return np.asarray(d.xyxy, float).reshape(-1, 4), np.asarray(d.confidence, float).reshape(-1)


def iou(a, b):
    x0, y0 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x1, y1 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter)


def coco_map(gt, preds, img_ids=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        G = COCO(); G.dataset = gt; G.createIndex()
        if not preds:
            return 0.0, 0.0
        D = G.loadRes(preds)
        E = COCOeval(G, D, "bbox")
        if img_ids is not None:
            E.params.imgIds = img_ids
        E.evaluate(); E.accumulate(); E.summarize()
    return float(E.stats[1]), float(E.stats[0])


def main():
    gt = json.loads((TEST / "_annotations.coco.json").read_text())
    by_img = {}
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
    models = load_models()
    report = {"test_set": {"images": len(gt["images"]), "boxes": len(gt["annotations"]),
                           "source": "Kaggle aminefaris/retail-shelf-void-detection-dataset (CC BY-NC-SA 4.0)"},
              "models": {}}
    import torch
    rng = np.random.default_rng(0)
    for name, predict in models.items():
        preds, lat, tp, fp, fn = [], [], 0, 0, 0
        ims = [(g, Image.open(TEST / g["file_name"]).convert("RGB")) for g in gt["images"]]
        for _ in range(3):                                         # warm-up
            predict(ims[0][1])
        for g, im in ims:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.time()
            boxes, scores = predict(im)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lat.append((time.time() - t) * 1000)
            for b, s in zip(boxes, scores):
                preds.append({"image_id": g["id"], "category_id": 1, "score": float(s),
                              "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
            G = np.array(by_img.get(g["id"], []), float).reshape(-1, 4)
            keep = boxes[scores >= EVAL_CONF]
            used = np.zeros(len(G), bool)
            for b in keep[np.argsort(-scores[scores >= EVAL_CONF])]:
                if len(G):
                    v = iou(b, G); v[used] = 0
                    j = int(np.argmax(v))
                    if v[j] >= MATCH_IOU:
                        used[j] = True; tp += 1; continue
                fp += 1
            fn += int((~used).sum())
        m50, m5095 = coco_map(gt, preds)
        ids = [g["id"] for g in gt["images"]]
        boots = []
        for _ in range(300):                                       # bootstrap over test images
            samp = rng.choice(ids, len(ids), replace=True)
            # duplicate-aware resample: rebuild a GT/prediction set with renamed image ids
            gt2 = {"images": [], "annotations": [], "categories": gt["categories"]}; p2 = []
            for k, iid in enumerate(samp):
                gt2["images"].append({"id": k, "width": 640, "height": 640, "file_name": str(k)})
                for a in gt["annotations"]:
                    if a["image_id"] == iid:
                        gt2["annotations"].append({**a, "id": len(gt2["annotations"]), "image_id": k})
                p2 += [{**p, "image_id": k} for p in preds if p["image_id"] == iid]
            boots.append(coco_map(gt2, p2)[0])
        report["models"][name] = {
            "mAP50": round(m50, 3), "mAP50_95": round(m5095, 3),
            "mAP50_95ci": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
            f"precision@{EVAL_CONF}": round(tp / max(1, tp + fp), 3), f"recall@{EVAL_CONF}": round(tp / max(1, tp + fn), 3),
            "tp_fp_fn": [tp, fp, fn], "latency_ms_median": round(float(np.median(lat)), 1)}
        print(name, report["models"][name], flush=True)
    # store aisles: the face-on renders of the 3D models (no ground truth yet)
    aisles = {}
    for sdir in sorted(AISLES.glob("scan_20261015_083401_*")):
        views = sorted(p for p in (sdir / "results" / "yolo_gaps").glob("view_*.jpg") if "_voids" not in p.name)
        if not views:
            continue
        name_src = json.loads((sdir / "scan.json").read_text()).get("source_file")
        outd = OUT / "aisles" / sdir.name; outd.mkdir(parents=True, exist_ok=True)
        counts = {}
        for mname, predict in models.items():
            n = 0
            for v in views:
                im = Image.open(v).convert("RGB")
                boxes, scores = predict(im)
                keep = scores >= EVAL_CONF
                n += int(keep.sum())
                if keep.any():
                    d = ImageDraw.Draw(im)
                    for b, s in zip(boxes[keep], scores[keep]):
                        d.rectangle([float(c) for c in b], outline=(255, 140, 0), width=4)
                        d.text((float(b[0]) + 4, float(b[1]) + 2), f"{s:.2f}", fill=(255, 140, 0))
                    im.save(outd / f"{mname}__{v.stem}.jpg", quality=85)
            counts[mname] = n
        aisles[name_src] = {"views": len(views), "detections_at_conf_0.25": counts}
        print(name_src, aisles[name_src], flush=True)
    report["store_aisles_no_ground_truth"] = aisles
    (OUT / "benchmark.json").write_text(json.dumps(report, indent=1))
    print("WROTE", OUT / "benchmark.json")


if __name__ == "__main__":
    main()
