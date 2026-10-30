"""Stage 2 - open-vocabulary 2D detection (+ optional SAM masks) on rendered views.

Runs in the CV environment (torch + transformers), not Isaac Sim:
    python scan_perception/detect_2d.py --views out/views --prompts chair monitor "coffee cup" --sam

Writes <views>/detections.json and, with --sam, <views>/masks/<view_id>.npz.
Models load from the Hugging Face hub through `transformers`, so Grounding DINO
needs no custom CUDA build.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import batched_nms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from views import DETECTIONS, load_manifest, write_json  # noqa: E402

DEFAULT_MODELS = {
    "grounding-dino": "IDEA-Research/grounding-dino-base",
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "owlvit": "google/owlvit-base-patch32",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--views", required=True)
    p.add_argument("--prompts", nargs="+", help="Target classes, e.g. chair monitor 'coffee cup'")
    p.add_argument("--prompts-file", help="One prompt per line (alternative to --prompts)")
    p.add_argument("--detector", choices=list(DEFAULT_MODELS), default="grounding-dino")
    p.add_argument("--model-id", help="Override the Hugging Face model id")
    p.add_argument("--box-threshold", type=float, default=0.35)
    p.add_argument("--text-threshold", type=float, default=0.25, help="Grounding DINO only")
    p.add_argument("--prompts-per-query", type=int, default=8,
                   help="Grounding DINO: split long prompt lists so the text stays under its 256-token limit")
    p.add_argument("--nms-iou", type=float, default=0.5, help="Per-class 2D NMS")
    p.add_argument("--max-box-frac", type=float, default=0.8,
                   help="Drop boxes covering more than this fraction of the image (usually walls/floor)")
    p.add_argument("--sam", action="store_true", help="Refine boxes into masks with SAM")
    p.add_argument("--sam-model", default="facebook/sam-vit-base")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overlays", action="store_true", help="Save <views>/overlays/*.jpg for inspection")
    args = p.parse_args(argv)
    if args.prompts_file:
        args.prompts = [l.strip() for l in Path(args.prompts_file).read_text().splitlines() if l.strip()]
    if not args.prompts:
        p.error("give --prompts or --prompts-file")
    args.prompts = [s.lower().strip() for s in args.prompts]
    return args


# ------------------------------------------------------------------- detectors

def match_prompt(phrase, prompts):
    """Map a Grounding DINO output phrase back to one of our prompts.

    Its phrases are spans of the text query, so they can be partial ("coffee"),
    or merge neighbours ("chair monitor"). Pick the prompt with most shared words.
    """
    phrase = phrase.lower().strip()
    if phrase in prompts:
        return phrase
    words = set(re.findall(r"[a-z0-9]+", phrase))
    best, best_overlap = None, 0
    for p in prompts:
        overlap = len(words & set(re.findall(r"[a-z0-9]+", p)))
        if overlap > best_overlap:
            best, best_overlap = p, overlap
    return best


class GroundingDino:
    def __init__(self, model_id, device, box_threshold, text_threshold, prompts_per_query):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.device, self.box_t, self.text_t, self.chunk = device, box_threshold, text_threshold, prompts_per_query

    @torch.no_grad()
    def __call__(self, image, prompts):
        boxes, scores, labels = [], [], []
        for i in range(0, len(prompts), self.chunk):
            group = prompts[i:i + self.chunk]
            text = ". ".join(group) + "."
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            kwargs = dict(text_threshold=self.text_t, target_sizes=[image.size[::-1]])
            try:  # transformers >= 4.47 renamed box_threshold -> threshold
                res = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=self.box_t, **kwargs)[0]
            except TypeError:
                res = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=self.box_t, **kwargs)[0]
            phrases = res.get("text_labels", res["labels"])
            for box, score, phrase in zip(res["boxes"], res["scores"], phrases):
                label = match_prompt(phrase, group)
                if label is not None:
                    boxes.append(box.tolist()); scores.append(float(score)); labels.append(label)
        return boxes, scores, labels


class OwlDetector:
    def __init__(self, model_id, device, box_threshold, v2):
        from transformers import (Owlv2ForObjectDetection, Owlv2Processor,
                                  OwlViTForObjectDetection, OwlViTProcessor)
        proc_cls, model_cls = (Owlv2Processor, Owlv2ForObjectDetection) if v2 else (OwlViTProcessor, OwlViTForObjectDetection)
        self.processor = proc_cls.from_pretrained(model_id)
        self.model = model_cls.from_pretrained(model_id).to(device).eval()
        self.device, self.box_t, self.v2 = device, box_threshold, v2

    @torch.no_grad()
    def __call__(self, image, prompts):
        inputs = self.processor(text=[[f"a photo of a {p}" for p in prompts]], images=image,
                                return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # OWLv2 pads the image to a square, so boxes are relative to the padded size.
        side = max(image.size)
        target = [(side, side)] if self.v2 else [image.size[::-1]]
        res = self.processor.post_process_object_detection(outputs, threshold=self.box_t, target_sizes=target)[0]
        w, h = image.size
        boxes = [[min(max(c, 0), lim) for c, lim in zip(b.tolist(), (w, h, w, h))] for b in res["boxes"]]
        return boxes, [float(s) for s in res["scores"]], [prompts[int(i)] for i in res["labels"]]


class SamRefiner:
    def __init__(self, model_id, device):
        from transformers import SamModel, SamProcessor
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, image, boxes):
        """(N, H, W) bool masks, one per box (the highest-IoU of SAM's 3 proposals)."""
        if not boxes:
            return np.zeros((0, image.size[1], image.size[0]), dtype=bool)
        inputs = self.processor(image, input_boxes=[boxes], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
        best = outputs.iou_scores[0].argmax(-1).cpu()  # (N,)
        return masks[torch.arange(len(boxes)), best].numpy().astype(bool)


# ------------------------------------------------------------------------ main

def filter_boxes(boxes, scores, labels, prompts, image_size, nms_iou, max_box_frac):
    if not boxes:
        return [], [], []
    b = torch.tensor(boxes, dtype=torch.float32)
    s = torch.tensor(scores, dtype=torch.float32)
    cls = torch.tensor([prompts.index(l) for l in labels])
    w, h = image_size
    area_frac = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])) / (w * h)
    keep_mask = area_frac <= max_box_frac
    idx = torch.nonzero(keep_mask).flatten()
    keep = idx[batched_nms(b[idx], s[idx], cls[idx], nms_iou)].tolist()
    return [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep]


def save_overlay(image, dets, path):
    from PIL import ImageDraw
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for d in dets:
        draw.rectangle(d["box_xyxy"], outline=(255, 40, 40), width=3)
        draw.text((d["box_xyxy"][0] + 4, d["box_xyxy"][1] + 2), f'{d["label"]} {d["score"]:.2f}', fill=(255, 255, 0))
    img.save(path, quality=85)


def main(argv=None):
    args = parse_args(argv)
    views_dir = Path(args.views)
    _, views = load_manifest(views_dir)
    model_id = args.model_id or DEFAULT_MODELS[args.detector]
    print(f"[detect_2d] {args.detector} ({model_id}) on {args.device}; prompts: {args.prompts}")

    if args.detector == "grounding-dino":
        detector = GroundingDino(model_id, args.device, args.box_threshold, args.text_threshold, args.prompts_per_query)
    else:
        detector = OwlDetector(model_id, args.device, args.box_threshold, v2=args.detector == "owlv2")
    sam = SamRefiner(args.sam_model, args.device) if args.sam else None
    if sam:
        (views_dir / "masks").mkdir(exist_ok=True)
    if args.overlays:
        (views_dir / "overlays").mkdir(exist_ok=True)

    result = {"detector": args.detector, "model_id": model_id, "prompts": args.prompts,
              "box_threshold": args.box_threshold, "masks": bool(sam), "views": {}}
    total = 0
    for k, view in enumerate(views):
        image = view.load_rgb()
        boxes, scores, labels = detector(image, args.prompts)
        boxes, scores, labels = filter_boxes(boxes, scores, labels, args.prompts, image.size,
                                             args.nms_iou, args.max_box_frac)
        dets = [{"label": l, "score": round(s, 4), "box_xyxy": [round(c, 1) for c in b]}
                for b, s, l in zip(boxes, scores, labels)]
        if sam and dets:
            masks = sam(image, boxes)
            np.savez_compressed(views_dir / "masks" / f"{view.view_id}.npz", masks=np.packbits(masks, axis=-1),
                                width=image.size[0])
            for i, d in enumerate(dets):
                d["mask_index"] = i
        result["views"][view.view_id] = dets
        total += len(dets)
        if args.overlays:
            save_overlay(image, dets, views_dir / "overlays" / f"{view.view_id}.jpg")
        if k % 10 == 0:
            print(f"[detect_2d] {k + 1}/{len(views)} views, {total} detections so far")

    write_json(views_dir / DETECTIONS, result)
    print(f"[detect_2d] {total} detections over {len(views)} views -> {views_dir / DETECTIONS}")


if __name__ == "__main__":
    main()
