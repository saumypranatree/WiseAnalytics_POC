#!/usr/bin/env python3
"""Stage 2: open-vocabulary 2D detection (+ optional SAM masks) on rendered views.

Reads Stage 1 output (``<renders>/meta.json`` + ``rgb/*.png``) and writes Stage 2
output as specified in CONTRACT.md:

    <out>/detections.json
    <out>/masks/<frame>_<k>.png     (uint8 0/255, only with --sam)
    <out>/vis/<frame>.jpg           (only with --vis)

Backends (Hugging Face transformers, imported lazily inside the classes so the
post-processing logic can be used / tested without torch or transformers):
    * Grounding DINO  (--detector grounding-dino, default)
    * OWLv2 / OWL-ViT (--detector owlv2)
    * SAM box-prompted segmentation (--sam)

Example:
    python detect_2d.py --renders out/renders --out out/detections \\
        --prompts "chair,monitor,coffee cup" --sam --vis
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import re
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

log = logging.getLogger("detect_2d")

DEFAULT_MODELS = {
    "grounding-dino": "IDEA-Research/grounding-dino-base",
    "owlv2": "google/owlv2-base-patch16-ensemble",
}
# OWLv2 sigmoid scores are much lower than Grounding DINO's; 0.3 would miss most objects.
DEFAULT_BOX_THRESHOLD = {"grounding-dino": 0.3, "owlv2": 0.1}
DEFAULT_SAM_MODEL = "facebook/sam-vit-base"

# (boxes_xyxy float32 Nx4, scores float32 N, class_names list[str] -- each one of the prompts)
Detections = Tuple[np.ndarray, np.ndarray, List[str]]


def empty_detections() -> Detections:
    return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []


# --------------------------------------------------------------------------- #
# Label mapping: free-text detector phrase -> exactly one user prompt
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    """Lowercase, drop wordpiece markers / punctuation, collapse whitespace."""
    text = text.lower().replace("##", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def _tokens(text: str) -> List[str]:
    # crude singularisation so "chairs" matches "chair"
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
            for t in text.split()]


class LabelMapper:
    """Maps a detector's phrase label back to one of the user prompts (verbatim).

    Order: exact match -> substring / token-overlap score -> drop (returns None).
    Ties (e.g. merged phrase "chair monitor") go to the prompt that appears
    earliest in the phrase, then to the longer (more specific) prompt.
    """

    def __init__(self, prompts: Sequence[str]):
        self.prompts = list(prompts)
        self._norm = [_norm(p) for p in self.prompts]
        self.dropped: Counter = Counter()
        self.fuzzy: Counter = Counter()
        self._cache: dict = {}

    def __call__(self, label: str) -> Optional[str]:
        if label not in self._cache:
            self._cache[label] = self._map(label)
        out = self._cache[label]
        if out is None:
            self.dropped[label] += 1
        elif _norm(label) != _norm(out):
            self.fuzzy[(label, out)] += 1
        return out

    def _map(self, label: str) -> Optional[str]:
        lab = _norm(label)
        if not lab:
            return None
        for p, pn in zip(self.prompts, self._norm):
            if lab == pn:
                return p
        lab_toks = _tokens(lab)
        lab_set = set(lab_toks)
        best, best_key = None, None
        for p, pn in zip(self.prompts, self._norm):
            p_set = set(_tokens(pn))
            inter = lab_set & p_set
            if inter:   # whole-token overlap ("coffee" / "coffee cup", "chair monitor")
                score = len(inter) / len(lab_set | p_set)
            elif min(len(pn), len(lab)) >= 3 and (pn in lab or lab in pn):
                # sub-word containment ("armchair" / "chair"); ignore 1-2 char fragments
                score = min(len(pn), len(lab)) / max(len(pn), len(lab))
            else:
                score = 0.0
            if score <= 0.0:
                continue
            # earliest position of this prompt's first matching token in the phrase
            pos = min((i for i, t in enumerate(lab_toks) if t in p_set or t in pn or pn in t),
                      default=len(lab_toks))
            key = (score, -pos, len(pn))
            if best_key is None or key > best_key:
                best, best_key = p, key
        return best


# --------------------------------------------------------------------------- #
# Box post-processing
# --------------------------------------------------------------------------- #
def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    x0, y0, x1, y1 = boxes.T
    areas = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.clip(np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]), 0, None)
        h = np.clip(np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]), 0, None)
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def per_class_nms(boxes: np.ndarray, scores: np.ndarray, names: List[str],
                  iou_thr: float) -> np.ndarray:
    """Indices kept after NMS applied independently per class (sorted by score desc)."""
    if len(boxes) == 0:
        return np.zeros((0,), np.int64)
    try:
        import torch
        from torchvision.ops import batched_nms
        cls = {n: i for i, n in enumerate(dict.fromkeys(names))}
        idx = torch.tensor([cls[n] for n in names])
        keep = batched_nms(torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(),
                           idx, iou_thr).numpy().astype(np.int64)
    except ImportError:
        keep_list = []
        for n in dict.fromkeys(names):
            sel = np.array([i for i, m in enumerate(names) if m == n])
            keep_list.extend(sel[_nms_numpy(boxes[sel], scores[sel], iou_thr)])
        keep = np.asarray(keep_list, np.int64)
    return keep[np.argsort(-scores[keep], kind="stable")]


def postprocess(dets: Detections, image_wh: Tuple[int, int], *, nms_iou: float,
                max_box_area_frac: float, min_box_px: float) -> Tuple[Detections, Counter]:
    """Clip, drop giant/tiny boxes, per-class NMS. Returns (dets, drop-reason counts)."""
    boxes, scores, names = dets
    w, h = image_wh
    stats: Counter = Counter()
    if len(boxes) == 0:
        return empty_detections(), stats
    boxes = boxes.astype(np.float32).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    too_big = bw * bh > max_box_area_frac * w * h  # Grounding DINO loves "whole scene" boxes
    too_small = (bw < min_box_px) | (bh < min_box_px)
    stats["too_big"] = int(too_big.sum())
    stats["too_small"] = int((too_small & ~too_big).sum())
    ok = np.flatnonzero(~(too_big | too_small))
    boxes, scores, names = boxes[ok], scores[ok].astype(np.float32), [names[i] for i in ok]
    keep = per_class_nms(boxes, scores, names, nms_iou)
    stats["nms"] = len(ok) - len(keep)
    return (boxes[keep], scores[keep], [names[i] for i in keep]), stats


# --------------------------------------------------------------------------- #
# Model helpers (torch / transformers imported lazily)
# --------------------------------------------------------------------------- #
def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _to_device(inputs, device, dtype):
    """Move a BatchEncoding/dict to device; cast floating tensors to the model dtype."""
    out = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(dtype)
        out[k] = v
    return out


def _call_compat(fn, *args, **kwargs):
    """Call ``fn`` renaming/dropping kwargs its signature does not accept.

    transformers renamed Grounding DINO's ``box_threshold`` -> ``threshold`` (~v4.51);
    callers pass ``threshold=`` and it is renamed to ``box_threshold`` on old versions.
    """
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    if "threshold" in kwargs and "threshold" not in params and "box_threshold" in params:
        kwargs["box_threshold"] = kwargs.pop("threshold")
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


class _TorchModule:
    def _setup_device(self, device: str, fp16: bool):
        import torch
        self.device = resolve_device(device)
        if fp16 and not self.device.startswith("cuda"):
            log.warning("--fp16 ignored on %s", self.device)
            fp16 = False
        self.dtype = torch.float16 if fp16 else torch.float32


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
class Detector:
    """Interface: detect objects for a batch of RGB PIL images.

    ``detect_batch`` returns, per image, (boxes_xyxy float32 Nx4 in original pixel
    coords, scores float32 N, class_names) where every class name is one of
    ``self.prompts`` verbatim. Subclasses only need to produce raw labels and
    call ``_finalize`` (phrase strings or integer prompt indices).
    """

    model_id: str = "unknown"

    def __init__(self, prompts: Sequence[str]):
        self.prompts = list(prompts)
        self.mapper = LabelMapper(self.prompts)

    def detect_batch(self, images: List[Image.Image]) -> List[Detections]:
        raise NotImplementedError

    def _finalize(self, boxes, scores, labels, mapper=None) -> Detections:
        """Map raw labels -> prompts and drop unmappable ones."""
        mapper = mapper or self.mapper
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        scores = np.asarray(scores, np.float32).reshape(-1)
        keep, names = [], []
        for i, lab in enumerate(labels):
            if isinstance(lab, (int, np.integer)):          # index into prompts (OWL)
                name = self.prompts[int(lab)] if 0 <= int(lab) < len(self.prompts) else None
            else:
                name = mapper(str(lab))
            if name is not None:
                keep.append(i)
                names.append(name)
        return boxes[keep], scores[keep], names


class GroundingDinoDetector(Detector, _TorchModule):
    def __init__(self, prompts, model_id=DEFAULT_MODELS["grounding-dino"], box_threshold=0.3,
                 text_threshold=0.25, device="auto", fp16=False, prompt_group_size=0):
        super().__init__(prompts)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self._setup_device(device, fp16)
        self.model_id = model_id
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(self.device, self.dtype).eval()
        # "chair. monitor. coffee cup." -- lowercase, each phrase terminated by "."
        self.caption = " ".join(_norm(p) + "." for p in self.prompts)
        # Optional: split long prompt lists into several shorter captions, one forward pass
        # each. Long captions spread the text-token scores, so phrases fall below
        # text_threshold and get dropped as empty labels. Each group gets its own LabelMapper,
        # so a phrase can only map to a prompt of the group that produced it.
        g = int(prompt_group_size or 0)
        groups = ([self.prompts[i:i + g] for i in range(0, len(self.prompts), g)]
                  if 0 < g < len(self.prompts) else [self.prompts])
        self.groups = [(" ".join(_norm(p) + "." for p in grp),
                        self.mapper if len(groups) == 1 else _SharedStatsMapper(grp, self.mapper))
                       for grp in groups]
        for cap, _ in self.groups:
            log.info("Grounding DINO caption: %r", cap)

    def detect_batch(self, images):
        if len(self.groups) == 1:
            return self._detect_caption(images, self.caption, self.mapper)
        per_group = [self._detect_caption(images, cap, mapper) for cap, mapper in self.groups]
        out = []
        for i in range(len(images)):
            parts = [pg[i] for pg in per_group]
            out.append((np.concatenate([b for b, _, _ in parts]).reshape(-1, 4),
                        np.concatenate([sc for _, sc, _ in parts]),
                        [n for _, _, ns in parts for n in ns]))
        return out

    def _detect_caption(self, images, caption, mapper):
        import torch
        inputs = self.processor(images=images, text=[caption] * len(images),
                                return_tensors="pt", padding=True)
        with torch.inference_mode():
            outputs = self.model(**_to_device(inputs, self.device, self.dtype))
            outputs.logits = outputs.logits.float()
            outputs.pred_boxes = outputs.pred_boxes.float()
            results = _call_compat(
                self.processor.post_process_grounded_object_detection, outputs,
                input_ids=inputs["input_ids"].to(self.device),
                threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=[(im.height, im.width) for im in images])
        out = []
        for r in results:
            # new transformers: phrases in "text_labels" ("labels" is a deprecated alias that
            # may become int ids); old transformers: phrases in "labels".
            labels = r["text_labels"] if "text_labels" in r else r.get("labels", [])
            labels = [l if isinstance(l, str) else "" for l in labels]  # ids are not prompt idx
            out.append(self._finalize(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), labels,
                                      mapper=mapper))
        return out


class _SharedStatsMapper(LabelMapper):
    """LabelMapper over a subset of prompts that reports its fuzzy/dropped counts into the
    detector's main mapper, so the run summary stays complete."""

    def __init__(self, prompts, parent: LabelMapper):
        super().__init__(prompts)
        self.dropped, self.fuzzy = parent.dropped, parent.fuzzy


class Owlv2Detector(Detector, _TorchModule):
    def __init__(self, prompts, model_id=DEFAULT_MODELS["owlv2"], box_threshold=0.1,
                 device="auto", fp16=False):
        super().__init__(prompts)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self._setup_device(device, fp16)
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(self.device, self.dtype).eval()
        # OWLv2 pads to a square (bottom/right) before resizing, so its normalized boxes are
        # relative to the padded square; OWL-ViT v1 resizes without padding.
        self.pads_to_square = "owlv2" in getattr(self.model.config, "model_type", "").lower()
        self.queries = [_norm(p) for p in self.prompts]

    def _target_size(self, im):
        # A square (s, s) target is correct on every transformers version: old ones scale by
        # (w, h) of the target, newer ones already scale both axes by max(h, w).
        if self.pads_to_square:
            s = max(im.width, im.height)
            return (s, s)
        return (im.height, im.width)

    def detect_batch(self, images):
        import torch
        inputs = self.processor(text=[self.queries] * len(images), images=images,
                                return_tensors="pt", padding=True)
        with torch.inference_mode():
            outputs = self.model(**_to_device(inputs, self.device, self.dtype))
            outputs.logits = outputs.logits.float()
            outputs.pred_boxes = outputs.pred_boxes.float()
            post = (getattr(self.processor, "post_process_grounded_object_detection", None)
                    or self.processor.post_process_object_detection)
            results = _call_compat(post, outputs, threshold=self.box_threshold,
                                   target_sizes=torch.tensor([self._target_size(im) for im in images]))
        out = []
        for r in results:
            labels = [int(l) for l in r["labels"].cpu().numpy()]   # query indices
            # boxes reaching into the padded region are clipped later in postprocess()
            out.append(self._finalize(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), labels))
        return out


# --------------------------------------------------------------------------- #
# SAM
# --------------------------------------------------------------------------- #
class BoxSegmenter(_TorchModule):
    """SAM prompted with detection boxes; returns one bool HxW mask per box."""

    def __init__(self, model_id=DEFAULT_SAM_MODEL, device="auto", fp16=False, max_boxes=64):
        from transformers import SamModel, SamProcessor
        self._setup_device(device, fp16)
        self.model_id = model_id
        self.max_boxes = max_boxes
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device, self.dtype).eval()

    def segment(self, image: Image.Image, boxes: np.ndarray) -> np.ndarray:
        import torch
        n = len(boxes)
        masks = np.zeros((n, image.height, image.width), bool)
        if n == 0:
            return masks
        with torch.inference_mode():
            # image embedding computed once, reused for every box chunk
            img_inputs = self.processor(images=image, return_tensors="pt")
            emb = self.model.get_image_embeddings(
                img_inputs["pixel_values"].to(self.device, self.dtype))
            for s in range(0, n, self.max_boxes):
                chunk = boxes[s:s + self.max_boxes].tolist()
                inp = self.processor(images=image, input_boxes=[chunk], return_tensors="pt")
                out = self.model(image_embeddings=emb,
                                 input_boxes=inp["input_boxes"].to(self.device, self.dtype),
                                 multimask_output=True)
                m = self.processor.image_processor.post_process_masks(
                    out.pred_masks.float().cpu(), img_inputs["original_sizes"].cpu(),
                    img_inputs["reshaped_input_sizes"].cpu())[0]        # (B, 3, H, W) bool
                best = out.iou_scores[0].float().argmax(-1).cpu()        # (B,) best of 3
                masks[s:s + len(chunk)] = m[torch.arange(len(chunk)), best].numpy().astype(bool)
        return masks


# --------------------------------------------------------------------------- #
# Visualization (PIL only, no cv2 dependency)
# --------------------------------------------------------------------------- #
def _color(name: str) -> Tuple[int, int, int]:
    rng = np.random.default_rng(zlib.crc32(name.encode()))   # stable across runs
    return tuple(int(c) for c in rng.integers(60, 256, 3))


def draw_overlay(image: Image.Image, dets: Detections,
                 masks: Optional[np.ndarray]) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    boxes, scores, names = dets
    if masks is not None:
        for m, n in zip(masks, names):
            arr[m] = 0.55 * arr[m] + 0.45 * np.array(_color(n), np.float32)
    vis = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(vis)
    for b, s, n in zip(boxes, scores, names):
        c = _color(n)
        draw.rectangle(b.tolist(), outline=c, width=2)
        txt = f"{n} {s:.2f}"
        tw, th = draw.textbbox((0, 0), txt)[2:]
        y = max(0.0, float(b[1]) - th - 2)
        draw.rectangle([b[0], y, b[0] + tw + 4, y + th + 2], fill=c)
        draw.text((b[0] + 2, y + 1), txt, fill=(0, 0, 0))
    return vis


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(renders: Path, out_dir: Path, detector: Detector,
        segmenter: Optional[BoxSegmenter], *, box_threshold: float,
        text_threshold: Optional[float], batch_size: int = 1, nms_iou: float = 0.5,
        max_box_area_frac: float = 0.6, min_box_px: float = 8.0, vis: bool = False) -> dict:
    """Run detection over all frames in ``renders/meta.json``; write detections.json."""
    renders, out_dir = Path(renders), Path(out_dir)
    meta = json.loads((renders / "meta.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    if segmenter is not None:
        (out_dir / "masks").mkdir(exist_ok=True)
    if vis:
        (out_dir / "vis").mkdir(exist_ok=True)
    exp_wh = (meta.get("image_width"), meta.get("image_height"))
    batch_size = max(1, batch_size)

    # Frame list in meta order, skipping entries whose rgb file is missing.
    frames, skipped = [], 0
    for f in meta.get("frames", []):
        p = renders / f.get("rgb", "")
        if not f.get("rgb") or not p.is_file():
            log.warning("frame %s: missing rgb %s, skipped", f.get("frame_id"), p)
            skipped += 1
            continue
        frames.append((str(f["frame_id"]), p))

    out_frames, class_counts, drop_stats = [], Counter(), Counter()
    for b in range(0, len(frames), batch_size):
        batch = []
        for fid, p in frames[b:b + batch_size]:
            try:
                im = Image.open(p).convert("RGB")
            except Exception as e:  # corrupt / truncated PNG
                log.warning("frame %s: cannot read %s (%s), skipped", fid, p, e)
                skipped += 1
                continue
            if None not in exp_wh and im.size != exp_wh:
                log.warning("frame %s: size %s != meta %s", fid, im.size, exp_wh)
            batch.append((fid, im))
        if not batch:
            continue
        raw = detector.detect_batch([im for _, im in batch])

        for (fid, im), dets in zip(batch, raw):
            dets, st = postprocess(dets, im.size, nms_iou=nms_iou,
                                   max_box_area_frac=max_box_area_frac, min_box_px=min_box_px)
            drop_stats.update(st)
            boxes, scores, names = dets
            masks = segmenter.segment(im, boxes) if segmenter is not None else None
            records = []
            for k, (bx, sc, nm) in enumerate(zip(boxes, scores, names)):
                det_id = f"{fid}_{k:02d}"
                mask_rel = None
                if masks is not None:
                    mask_rel = f"masks/{det_id}.png"
                    Image.fromarray(masks[k].astype(np.uint8) * 255).save(out_dir / mask_rel)
                records.append({
                    "det_id": det_id,
                    "class_name": nm,
                    "score": round(float(sc), 4),
                    "bbox_xyxy": [round(float(v), 2) for v in bx],
                    "mask": mask_rel,
                })
                class_counts[nm] += 1
            out_frames.append({"frame_id": fid, "detections": records})
            if vis:
                draw_overlay(im, dets, masks).save(out_dir / "vis" / f"{fid}.jpg", quality=90)
            log.info("frame %s: %d detections", fid, len(records))

    result = {
        "detector": detector.model_id,
        "segmenter": segmenter.model_id if segmenter is not None else None,
        "prompts": detector.prompts,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "frames": out_frames,
    }
    (out_dir / "detections.json").write_text(json.dumps(result, indent=2))

    # ---- summary ----
    print(f"\nProcessed {len(out_frames)} frames ({skipped} skipped) -> "
          f"{out_dir / 'detections.json'}")
    print(f"Dropped boxes: too_big={drop_stats['too_big']} "
          f"too_small={drop_stats['too_small']} nms={drop_stats['nms']}")
    if detector.mapper.fuzzy:
        print("Fuzzy label mappings: " + ", ".join(
            f"{a!r}->{b!r} x{c}" for (a, b), c in detector.mapper.fuzzy.most_common()))
    if detector.mapper.dropped:
        print("Unmappable labels dropped: " + ", ".join(
            f"{a!r} x{c}" for a, c in detector.mapper.dropped.most_common()))
    print("Per-class detections:")
    for p in detector.prompts:
        print(f"  {p:<24s} {class_counts[p]}")
    return result


def parse_prompts(prompts: Optional[str], prompts_file: Optional[str]) -> List[str]:
    """Comma-separated string and/or file (one per line or comma-separated, '#' comments)."""
    items: List[str] = []
    if prompts_file:
        for line in Path(prompts_file).read_text().splitlines():
            items += line.split("#", 1)[0].split(",")
    if prompts:
        items += prompts.split(",")
    # strip, drop empties / trailing dots, dedupe case-insensitively, keep order
    seen, out = set(), []
    for p in (i.strip().rstrip(".").strip() for i in items):
        if p and _norm(p) and _norm(p) not in seen:
            seen.add(_norm(p))
            out.append(p)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 2: open-vocabulary 2D detection (+SAM).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--renders", required=True, type=Path, help="Stage 1 dir with meta.json")
    ap.add_argument("--out", default=Path("./out/detections"), type=Path)
    ap.add_argument("--prompts", help='comma-separated, e.g. "chair,monitor,coffee cup"')
    ap.add_argument("--prompts-file", help="text file: one prompt per line (or comma-separated)")
    ap.add_argument("--detector", choices=sorted(DEFAULT_MODELS), default="grounding-dino")
    ap.add_argument("--model-id", help="override HF model id for the detector")
    ap.add_argument("--box-threshold", type=float, default=None,
                    help="default 0.3 for grounding-dino, 0.1 for owlv2")
    ap.add_argument("--text-threshold", type=float, default=0.25,
                    help="Grounding DINO phrase threshold (unused by owlv2)")
    ap.add_argument("--prompt-group-size", type=int, default=0,
                    help="Grounding DINO: split the prompts into captions of at most this many "
                         "phrases, one forward pass each (0 = all prompts in one caption)")
    ap.add_argument("--nms-iou", type=float, default=0.5, help="per-class NMS IoU")
    ap.add_argument("--max-box-area-frac", type=float, default=0.6,
                    help="drop boxes covering more than this fraction of the image")
    ap.add_argument("--min-box-px", type=float, default=8.0,
                    help="drop boxes whose width or height is below this")
    ap.add_argument("--sam", action=argparse.BooleanOptionalAction, default=False,
                    help="segment each box with SAM and write masks/")
    ap.add_argument("--sam-model", default=DEFAULT_SAM_MODEL)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--vis", action="store_true", help="write vis/<frame>.jpg overlays")
    ap.add_argument("--fp16", action="store_true", help="half precision (CUDA only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    prompts = parse_prompts(args.prompts, args.prompts_file)
    if not prompts:
        log.error("no prompts given (use --prompts or --prompts-file)")
        return 2
    if not (args.renders / "meta.json").is_file():
        log.error("%s/meta.json not found", args.renders)
        return 2
    box_thr = (args.box_threshold if args.box_threshold is not None
               else DEFAULT_BOX_THRESHOLD[args.detector])
    model_id = args.model_id or DEFAULT_MODELS[args.detector]
    log.info("detector=%s model=%s device=%s prompts=%s", args.detector, model_id,
             resolve_device(args.device), prompts)

    if args.detector == "grounding-dino":
        detector: Detector = GroundingDinoDetector(
            prompts, model_id, box_threshold=box_thr, text_threshold=args.text_threshold,
            device=args.device, fp16=args.fp16, prompt_group_size=args.prompt_group_size)
        text_thr = args.text_threshold
    else:
        detector = Owlv2Detector(prompts, model_id, box_threshold=box_thr,
                                 device=args.device, fp16=args.fp16)
        text_thr = None
    segmenter = (BoxSegmenter(args.sam_model, device=args.device, fp16=args.fp16)
                 if args.sam else None)

    run(args.renders, args.out, detector, segmenter, box_threshold=box_thr,
        text_threshold=text_thr, batch_size=args.batch_size, nms_iou=args.nms_iou,
        max_box_area_frac=args.max_box_area_frac, min_box_px=args.min_box_px, vis=args.vis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
