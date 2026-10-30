"""Train YOLOv8s and RF-DETR Small on the same clean split (benchmark only; dataset is CC BY-NC-SA 4.0).
Both start from their framework's COCO-pretrained weights and train 50 epochs, seed 0."""
import json, sys, time
from pathlib import Path
B = Path(__file__).resolve().parent.parent
SPLIT = B / "data" / "split"
RUNS = B / "runs"; RUNS.mkdir(exist_ok=True)
which = sys.argv[1:] or ["yolo", "rfdetr"]
times = {}
if "yolo" in which:
    from ultralytics import YOLO
    t = time.time()
    m = YOLO("yolov8s.pt")
    m.train(data=str(SPLIT / "data.yaml"), epochs=50, imgsz=640, batch=16, seed=0, deterministic=True,
            project=str(RUNS), name="yolov8s_void", exist_ok=True, workers=4, verbose=False, plots=False)
    times["yolov8s_train_s"] = round(time.time() - t, 1)
    print("YOLO DONE", times, flush=True)
if "rfdetr" in which:
    from rfdetr import RFDETRSmall
    t = time.time()
    m = RFDETRSmall()
    m.train(dataset_dir=str(SPLIT), epochs=50, batch_size=4, grad_accum_steps=4, lr=1e-4, seed=0,
            output_dir=str(RUNS / "rfdetr_small_void"), num_workers=4, tensorboard=False, run_test=False)
    times["rfdetr_small_train_s"] = round(time.time() - t, 1)
    print("RFDETR DONE", times, flush=True)
(RUNS / f"train_times_{'_'.join(which)}.json").write_text(json.dumps(times, indent=1))
