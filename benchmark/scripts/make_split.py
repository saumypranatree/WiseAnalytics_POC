"""Clean split of the Kaggle 'Retail Shelf Void Detection Dataset' (CC BY-NC-SA 4.0, benchmark use only).

Roboflow names augmented copies after their source photo (<name>_jpg.rf.<hash>). To rule out any
train/test leakage, photos are grouped by that source name:
  test  = original valid + test photos whose source name never appears in train (un-augmented)
  train = train photos + test photos whose source name does appear in train
  valid = ~15 % of the train source groups (whole groups), for model selection only
Writes data/split/<train|valid|test>/{images,labels} (YOLO) + _annotations.coco.json (COCO, for RF-DETR)
and data/split/data.yaml (Ultralytics). Single class 'void'.
"""
import json, os, random, re, shutil, sys
from PIL import Image

SRC = sys.argv[1]; OUT = sys.argv[2]
key = lambda f: re.sub(r"\.rf\.[0-9a-f]+", "", os.path.splitext(f)[0])
files = {s: sorted(os.listdir(f"{SRC}/{s}/images")) for s in ("train", "valid", "test")}
train_keys = {key(f) for f in files["train"]}
test = [("valid", f) for f in files["valid"]] + [("test", f) for f in files["test"] if key(f) not in train_keys]
pool = [("train", f) for f in files["train"]] + [("test", f) for f in files["test"] if key(f) in train_keys]
groups = sorted({key(f) for _, f in pool})
random.Random(0).shuffle(groups)
val_keys = set(groups[: max(1, round(0.15 * len(groups)))])
split = {"test": test, "valid": [p for p in pool if key(p[1]) in val_keys],
         "train": [p for p in pool if key(p[1]) not in val_keys]}
assert not ({key(f) for _, f in split["test"]} & {key(f) for _, f in split["train"] + split["valid"]})
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
stats = {}
for name, items in split.items():
    os.makedirs(f"{OUT}/{name}/images"); os.makedirs(f"{OUT}/{name}/labels")
    coco = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "voids", "supercategory": "none"},
                                                            {"id": 1, "name": "void", "supercategory": "voids"}]}
    nb = 0
    for i, (src_split, f) in enumerate(items):
        shutil.copy(f"{SRC}/{src_split}/images/{f}", f"{OUT}/{name}/images/{f}")
        lab = os.path.splitext(f)[0] + ".txt"
        shutil.copy(f"{SRC}/{src_split}/labels/{lab}", f"{OUT}/{name}/labels/{lab}")
        W, H = Image.open(f"{SRC}/{src_split}/images/{f}").size
        coco["images"].append({"id": i, "file_name": f"images/{f}", "width": W, "height": H})  # relative to the json (RF-DETR layout)
        for line in open(f"{SRC}/{src_split}/labels/{lab}"):
            p = line.split()
            if len(p) != 5:
                continue
            cx, cy, w, h = (float(v) for v in p[1:])
            coco["annotations"].append({"id": len(coco["annotations"]), "image_id": i, "category_id": 1,
                                        "bbox": [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H],
                                        "area": w * W * h * H, "iscrowd": 0})
            nb += 1
    json.dump(coco, open(f"{OUT}/{name}/_annotations.coco.json", "w"))
    stats[name] = {"images": len(items), "source_photos": len({key(f) for _, f in items}), "boxes": nb}
open(f"{OUT}/data.yaml", "w").write(f"path: {OUT}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['void']\n")
json.dump(stats, open(f"{OUT}/split_stats.json", "w"), indent=1)
print(json.dumps(stats))
