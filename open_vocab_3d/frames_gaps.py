#!/usr/bin/env python3
"""Stage B: place the 2D empty-space boxes found on the VIDEO keyframes into the 3D scan.

    .venv/bin/python frames_gaps.py <scan_dir> [--min-views 2] [--merge-iou 0.3]

Input: <scan>/results/frame_detections/detections.json, written by stage A
(benchmark/scripts/detect_frames.py), which ran the detectors on the original keyframes.
Each box is placed on the shelf front exactly like a label drawn in the app
(shelf_labels.lift_box: keyframe camera -> rays -> shelf-front plane, front depth from the
products around the box); boxes of one model that land on the same spot in several keyframes
are merged into one gap (gaps_runner.merge); a gap must be seen in >= --min-views keyframes.

Output: one method per model added to <scan>/results/gaps.json (exported usd frame, mapped to the
current scale / flip by the server on every read): frames_rfdetr, frames_yolo, frames_yolo_dl.
Methods from other tools in that file are kept.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gaps_runner as gr  # noqa: E402
import shelf_labels as sl  # noqa: E402

DESCRIPTIONS = {
    "frames_rfdetr": "RF-DETR Small fine-tuned on the public Kaggle void dataset (benchmark only: non-commercial data)",
    "frames_yolo": "YOLOv8s fine-tuned on the public Kaggle void dataset (benchmark only: non-commercial data)",
    "frames_yolo_dl": "downloaded YOLOv8 (akul-29/Retail-Shelf-Gap-Detection_Model)",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir")
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--merge-iou", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=None, help="drop boxes below this score (default: keep stage A's)")
    a = ap.parse_args()
    t0 = time.time()
    sdir = Path(a.scan_dir).resolve()
    rec = json.loads((sdir / "scan.json").read_text())
    cams = sl.load_cameras(sdir)
    det = json.loads((sdir / "results" / "frame_detections" / "detections.json").read_text())
    W = sl.scene_transform(rec)
    sc = sl._scene(sdir / "scene.usda")
    gpath = sdir / "results" / "gaps.json"
    doc = json.loads(gpath.read_text()) if gpath.is_file() else {
        "schema": 1, "scan_id": sdir.name, "frame": "usd", "methods": {}}
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary = {}
    for key, per_kf in det["models"].items():
        if a.conf is not None:
            per_kf = {k: [d for d in v if d["score"] >= a.conf] for k, v in per_kf.items()}
        n2d = sum(len(v) for v in per_kf.values())
        lifted = gr.lift_all(sc, cams, W, per_kf)
        gaps = gr.merge(lifted, a.merge_iou, a.min_views)
        conf = a.conf if a.conf is not None else det["conf"]
        doc.setdefault("methods", {})[key] = {
            "description": f"{DESCRIPTIONS.get(key, key)}, run on the original video keyframes "
                           f"(square tiles), confidence >= {conf}, boxes placed on the shelf front and merged "
                           f"across keyframes; kept when seen in >= {a.min_views} keyframes",
            "gaps": gaps, "detections_2d": n2d, "detections_placed": len(lifted),
            "keyframes": det["keyframes"], "created_at": stamp,
            "seconds": det["seconds"].get(key)}
        summary[key] = {"boxes_2d": n2d, "placed_3d": len(lifted), "gaps": len(gaps)}
    doc.setdefault("created_at", stamp)
    doc.setdefault("num_keyframes", det["keyframes"])
    tmp = gpath.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(gpath)
    print(json.dumps({"scan": sdir.name, "min_views": a.min_views, "models": summary,
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
