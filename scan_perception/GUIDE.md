# scan_perception: architecture and offline run guide

## Part 1: Architecture, step by step

The pipeline turns an unsegmented 3D scan (`room_scan.usda`) into a list of named objects, each with a 3D position and box. Nothing in it calls Claude or any online AI service. It has three stages, and each one hands its results to the next as files on disk:

```
room_scan.usda
   │  Stage 1  render_views.py      (Isaac Sim, NVIDIA RTX GPU)
   ▼
out/views/   rgb/*.png  depth/*.npy  manifest.json (camera K + pose per view)
   │  Stage 2  detect_2d.py         (PyTorch, GPU recommended)
   ▼
out/views/detections.json  (+ masks/*.npz)
   │  Stage 3  lift_and_cluster.py  (numpy + scikit-learn, CPU)
   ▼
out/scene_objects.json   (+ objects.ply for visual checking)
```

Because the stages only share files, each runs in its own Python environment: Isaac Sim's Python never needs PyTorch, and the detector never needs Isaac Sim. You can also re-run one stage without re-running the earlier ones.

### Stage 1: Render the scan from many virtual cameras (`render_views.py`)

1. **Open the scan.** Isaac Sim loads the USD file and reads its units (`metersPerUnit`, which is 1 here) and its up axis (Y here).
2. **Measure the room.** The script reads every mesh vertex. It takes the floor height as the 2nd-percentile height, and the room's footprint (the area it covers on the floor) from the 5th to 95th percentiles. Percentiles are used so that a few stray scan points don't distort the measurements.
3. **Place the cameras.**
   - `interior` mode (the default) puts a 3×3 grid of positions inside the room at 1.4 m eye height. At each position the camera turns through 8 directions at 2 downward tilts (−10° and −40°), giving 144 views.
   - `dome` mode puts rings of cameras outside the scene, all looking at its centre. That suits open scenes; a closed room seen from outside only shows its walls.
4. **Add a temporary camera rig.** One camera, plus lights, is added on the session layer, a temporary layer Isaac Sim keeps in memory. Your USD file is never edited.
5. **Capture each view.** For every camera pose, the script moves the camera, renders, and saves three things:
   - **Colour image.** By default this is the unlit texture colour (`albedo`), because a scan's textures already contain the real lighting.
   - **Depth map.** For each pixel, this stores the depth along the camera's viewing axis (`distance_to_image_plane`). A pixel with no surface behind it is stored as infinity.
   - **Camera parameters.** The intrinsics `K` (focal length and image centre in pixels) and the extrinsics (the world-to-camera matrix). These are saved in the OpenCV convention.
6. **Verify each pose.** The pose the script saved is compared with what Isaac Sim reports it actually rendered. If they disagree, the view is rendered again, and a warning is printed if they still disagree.

### Stage 2: Find the objects in each image (`detect_2d.py`)

This stage runs open-weight vision models on your own GPU. You download their weights once; after that they run offline.

1. **Read the prompts.** You give it text prompts such as `chair monitor "coffee cup"`.
2. **Detect boxes** with one of three detectors:
   - **Grounding DINO (default).** An image encoder (Swin Transformer) and a text encoder (BERT) exchange information through cross-attention layers. A DETR-style decoder then outputs boxes, each linked to a phrase from your prompt. Grounding DINO's text input is limited to 256 tokens, so long prompt lists are sent in groups of 8.
   - **OWLv2 or OWL-ViT.** These are based on CLIP: each image region is scored against each prompt's text embedding.
3. **Map phrases to prompts.** Grounding DINO can return part of a prompt ("coffee") or several prompts merged ("chair monitor"). The script maps each phrase back to the prompt it shares the most words with.
4. **Filter the boxes.** Duplicate boxes of the same class are removed with 2D NMS (non-maximum suppression) at an overlap (IoU) of 0.5. Boxes covering more than 80% of the image are dropped, because those are almost always walls or floor.
5. **Turn boxes into masks (optional, `--sam`).** Each box is given to SAM (Segment Anything, ViT-B), which returns three candidate masks. The script keeps the one SAM scores highest, giving the exact pixels of the object.
6. **Save the results.** Detections go to `detections.json`, and masks are bit-packed into `masks/view_XXXX.npz`. With `--overlays`, the script also saves images with the boxes drawn on, for you to check by eye.

### Stage 3: Lift to 3D, merge views, compute boxes (`lift_and_cluster.py`)

This stage is plain geometry and classic clustering, with no neural network.

1. **Choose each detection's pixels.**
   - **With a SAM mask:** the mask, shrunk by 2 px at the edges to drop pixels that bleed onto the background.
   - **With only a box:** the box shrunk by 10% on each side. Pixels are kept only if their depth is close to the median depth at the box's centre, within 0.15 m plus half the object's estimated size. This removes the wall or floor visible behind the object.
2. **Back-project each pixel into 3D.**
   - Pixel to camera frame: `x = (u+½−cx)·d/fx`, `y = (v+½−cy)·d/fy`, `z = d`.
   - Camera frame to world: `P_world = Rᵀ·(P_cam − t)`, where `[R | t]` is the saved world-to-camera matrix.
3. **Pool the points by class.** All "chair" points from all views go into one cloud, all "monitor" points into another, and so on.
4. **Voxel-downsample.** Each cloud is reduced to one point per 2 cm cube. This keeps the point density even, so a camera close to an object doesn't outweigh the others, and it keeps the next step fast.
5. **Cluster with DBSCAN** (eps = 6 cm, min_samples = 6). Points that are densely connected become one cluster, which normally means one physical object. Isolated points are labelled as noise.
6. **Require multi-view agreement.** Each detection is assigned to the cluster holding most of its points. A cluster is kept only if detections from at least 2 different views were assigned to it. This drops one-off false detections and stray fragments.
7. **Compute the box and position.**
   - **AABB (axis-aligned bounding box):** the 1st to 99th percentile of the points along each axis, so a few stray points don't stretch it.
   - **`position_world`:** the average of the points inside that box.
   - **`confidence`:** the average score of the detections assigned to the cluster.
8. **Resolve class conflicts.** If two objects of different classes overlap by more than 0.5 in 3D (for example "monitor" and "tv"), only the more confident one is kept.
9. **Write the output.** Results go to `scene_objects.json` in the required format. `--debug-ply` also writes the clustered points, coloured per object, so you can open them alongside the scan.

### Shared files

| File | Contents |
|---|---|
| `geometry.py` | Look-at camera poses, the USD to OpenCV axis flip, back-projection and projection, 3D IoU. Needs only numpy. |
| `views.py` | Reads `manifest.json` and writes JSON files safely (the old file is replaced only once the new one is complete). |
| `tests/test_synthetic_pipeline.py` | Ray-casts a fake room with known boxes and checks that Stage 3 recovers them. |

---

## Part 2: Running everything yourself, offline

### What you need

| | Stage 1 (render) | Stage 2 (detect) | Stage 3 (cluster) |
|---|---|---|---|
| GPU | NVIDIA RTX (required) | NVIDIA GPU recommended; CPU works but is slow | none |
| Software | Isaac Sim 4.x or 5.x | Python 3.10–3.12, PyTorch, transformers | Python 3.10+, numpy, scipy, scikit-learn |
| Internet | only to install Isaac Sim | only for the one-time download of model weights | none |

Notes for this machine:
- **Python version.** Your default `python3` is miniconda's Python 3.14, which PyTorch may not support yet. Create the CV environment with Python 3.11.
- **ROS Jazzy conflict.** ROS Jazzy puts its packages on `PYTHONPATH`, and this breaks pytest and can break other imports. Run the CV steps with `env -u PYTHONPATH ...`, or don't source ROS in that terminal.

### Step 0: Open a terminal in the project

```bash
cd /home/saumy/pranatree_work/room_scan
```

### Step 1: Install Isaac Sim (one time)

Use either method:

- **Workstation download.** Download Isaac Sim from NVIDIA's site and unzip it to `~/isaacsim`. It comes with its own Python launcher, `~/isaacsim/python.sh`.
- **pip.** Install into a Python 3.11 environment:
  ```bash
  conda create -n isaac python=3.11 -y && conda activate isaac
  pip install "isaacsim[all,extscache]" --extra-index-url https://pypi.nvidia.com
  ```
  Check NVIDIA's install page for the exact version to pin.

Check that it starts. The first launch compiles shaders and can take 5–10 minutes:

```bash
~/isaacsim/python.sh -c "from isaacsim import SimulationApp; a = SimulationApp({'headless': True}); a.close(); print('ok')"
```

### Step 2: Set up the CV environment (one time)

```bash
conda create -n scanperc python=3.11 -y
conda activate scanperc
# Install the PyTorch build that matches your CUDA version (see pytorch.org), for example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r scan_perception/requirements-cv.txt
```

Check that the 3D maths works. This needs no GPU and no Isaac Sim:

```bash
env -u PYTHONPATH python -m pytest scan_perception/tests -q     # expect: 3 passed
```

### Step 3: Download the model weights once, then work offline

```bash
env -u PYTHONPATH python - <<'EOF'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
for m in ["IDEA-Research/grounding-dino-base"]:
    AutoProcessor.from_pretrained(m); AutoModelForZeroShotObjectDetection.from_pretrained(m)
SamProcessor.from_pretrained("facebook/sam-vit-base"); SamModel.from_pretrained("facebook/sam-vit-base")
print("cached in ~/.cache/huggingface")
EOF
export HF_HUB_OFFLINE=1     # from now on, transformers never goes online
```

The weights are about 1.3 GB in total. To move them to a machine without internet, copy `~/.cache/huggingface` across.

### Step 4: Render the views

```bash
~/isaacsim/python.sh scan_perception/render_views.py --usd room_scan.usda --out out/views
```

Check:
- You should get 144 files each in `out/views/rgb/` and `out/views/depth/`.
- Open a few of the PNGs. If they are black, rerun with `--color-source lit`. If they are noisy, raise `--rt-subframes 16`.
- The log prints `NN% pixels hit geometry` every 10 views. Very low numbers mean the cameras are outside the room or inside a wall; adjust `--grid-margin` or `--eye-height`.

### Step 5: Detect objects in 2D

```bash
conda activate scanperc
env -u PYTHONPATH HF_HUB_OFFLINE=1 python scan_perception/detect_2d.py \
    --views out/views --prompts chair table monitor "coffee cup" --sam --overlays
```

Open a few images in `out/views/overlays/`:
- If you see too many false boxes, raise `--box-threshold` (for example to 0.45).
- If real objects are missing, lower it (for example to 0.25).

### Step 6: Lift, cluster and export

```bash
env -u PYTHONPATH python scan_perception/lift_and_cluster.py \
    --views out/views --out out/scene_objects.json --debug-ply out/objects.ply
```

The log prints one line per object with its view count, centre and size. To check the result visually, open `out/objects.ply` together with the scan in MeshLab or CloudCompare (`sudo apt install meshlab`).

Tuning:

| Problem | Fix |
|---|---|
| Two neighbouring objects come out as one | lower `--eps`, e.g. 0.04 |
| One object comes out as two | raise `--eps`, or render more views |
| A real object is missing | `--min-views 1` |
| Boxes include floor or wall | use `--sam` in Step 5 |

Stage 3 runs in seconds, so you can retune it as often as you like without re-rendering or re-detecting.

### Or run all three steps at once

```bash
conda activate scanperc
unset PYTHONPATH; export HF_HUB_OFFLINE=1
ISAAC_PY=~/isaacsim/python.sh CV_PY=python \
  scan_perception/run_pipeline.sh room_scan.usda out chair table monitor "coffee cup"
```

### Running without any neural model at all

Only Stage 2 uses neural networks; Stages 1 and 3 are rendering and geometry. To avoid models completely, skip Stage 2 and write `out/views/detections.json` yourself. You can draw the boxes with any annotation tool, for example LabelImg or CVAT, on a handful of the `rgb/*.png` images:

```json
{
  "masks": false,
  "views": {
    "view_0003": [{"label": "chair", "score": 1.0, "box_xyxy": [412, 300, 655, 690]}],
    "view_0011": [{"label": "chair", "score": 1.0, "box_xyxy": [120, 280, 390, 700]}]
  }
}
```

Then run Step 6 as usual. Label each object in at least 2 views, or pass `--min-views 1`.
