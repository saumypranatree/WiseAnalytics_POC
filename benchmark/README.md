# benchmark: empty-shelf (void) detectors

Compares two detectors for **void** (empty shelf space) in retail shelf photos:

- **YOLOv8s** (Ultralytics, `yolov8s_void/`)
- **RF-DETR Small** (`rfdetr_small_void/`)

Both start from COCO-pretrained weights, train for 50 epochs with seed 0 on the same clean split, and are evaluated on the same held-out test set. Results are in `results/benchmark.json`.

## Setup

```bash
cd benchmark
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Dataset (not in the repository)

The dataset is the Kaggle "Retail Shelf Void Detection Dataset", licensed **CC BY-NC-SA 4.0**. It is for benchmark use only and is not redistributed here.

1. Download it from Kaggle and extract it. The folder must contain `train/`, `valid/` and `test/`, each with `images/` and `labels/` (YOLO format). The extracted copy is expected under `data/raw/extracted/`.
2. Build the clean split (grouped by source photo, so augmented copies do not leak into the test set):

```bash
.venv/bin/python scripts/make_split.py data/raw/extracted/<dataset-folder> data/split
```

This writes `data/split/{train,valid,test}/` in YOLO format and a COCO `_annotations.coco.json` for RF-DETR, plus `data/split/data.yaml` and `split_stats.json`. The `data.yaml` `path:` line is the output folder. Make sure it matches where you keep the split.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/make_split.py` | Build the leakage-free train/valid/test split |
| `scripts/train_both.py` | Train YOLOv8s and RF-DETR Small on the split |
| `scripts/evaluate.py` | Evaluate all detectors on the test set (COCO metrics) |
| `scripts/detect_frames.py` | Run the detectors on the keyframes of a shelf video scan (2D only) |

```bash
.venv/bin/python scripts/train_both.py
.venv/bin/python scripts/evaluate.py
```

Trained weights are in `runs/` and are stored with Git LFS. Re-run training to regenerate them.
