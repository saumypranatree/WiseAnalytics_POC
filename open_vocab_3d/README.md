# open_vocab_3d — open-vocabulary 3D object localization on USD scans

Finds named objects ("chair", "coffee cup", ...) in an **unsegmented** OpenUSD
scan and writes their 3D world positions and axis-aligned bounding boxes to
`scene_objects.json`.

```
room_scan.usda
   │  render_views.py   (Isaac Sim python: Replicator RGB + depth + K + extrinsics)
   ▼
out/renders/
   │  detect_2d.py      (Grounding DINO / OWLv2 boxes  →  optional SAM masks)
   ▼
out/detections/
   │  lift_cluster.py   (back-project → world → per-class DBSCAN → AABB)
   ▼
out/scene_objects.json
```

The stages communicate only through files (see [CONTRACT.md](CONTRACT.md)). That
keeps Isaac Sim's bundled Python separate from the torch/transformers env, and
any stage can be re-run alone. For example, re-cluster with new parameters
without re-rendering.

---

## 1. Mathematical overview: pixel (u, v, depth) → world (x, y, z)

**Intrinsics.** Camera intrinsics `K` map camera-space points to pixels:

```
        ┌ fx  0  cx ┐            fx = f_mm / h_aperture_mm · W
    K = │ 0  fy  cy │            fy = fx  (square pixels)
        └ 0   0   1 ┘            cx = (W−1)/2,  cy = (H−1)/2   (pixel centers at integers)
```

Forward projection: `z · [u, v, 1]ᵀ = K · [x_c, y_c, z_c]ᵀ`.

**Step 1: back-project into camera space (invert K).** With
`d = depth(u, v)` as the z-depth (`distance_to_image_plane`, *not* ray length):

```
    p_c = d · K⁻¹ · [u, v, 1]ᵀ

    x_c = (u − cx) · d / fx
    y_c = (v − cy) · d / fy
    z_c = d
```

If a renderer gives ray length `r` instead, first convert:
`d = r / ‖K⁻¹[u,v,1]ᵀ‖`.

**Step 2: camera → world (rigid transform).** The extrinsics are stored as the
world-to-camera matrix `T_wc = [R | t]`. Its inverse is the camera pose:

```
    T_cw = T_wc⁻¹ = ┌ Rᵀ   −Rᵀt ┐
                    └ 0      1  ┘

    [x, y, z, 1]ᵀ = T_cw · [x_c, y_c, z_c, 1]ᵀ      i.e.  p_w = Rᵀ (p_c − t)
```

**Combined:**

```
    p_w = Rᵀ · ( d · K⁻¹ · [u, v, 1]ᵀ − t )
```

Over a whole mask this is one vectorized operation: stack N pixels into a
3×N matrix, multiply by `K⁻¹`, scale each column by its depth, then apply one
4×4 multiply.

**Convention trap: USD vs OpenCV cameras.** USD/Isaac cameras look down **−Z**
with **+Y up**. The formulas above assume OpenCV (**+Z forward, +Y down**). The
renderer therefore flips the axes once:
`T_wc(opencv) = diag(1, −1, −1, 1) · T_wc(usd)`. Replicator's
`cameraViewTransform` is also stored row-major in USD's row-vector convention,
so it must be **transposed** before use. Most "boxes appear mirrored or
behind the camera" bugs come from one of these two steps. `render_views.py
--verify` checks for them by re-projecting depth and confirming that the points
land inside the scene bounds.

**Units / axes.** The world frame is the scan's native USD frame, scaled by
`metersPerUnit`. **Orientation caveat:** `room_scan.usda` declares Y-up, but its
content is upside down (the floor sits at +Y). Use the wrapper
`scenes/room_scan_upright.usda`, which references the original with a 180°
rotation about X (a rotation, not a mirror). With the wrapper, "height" is the
+y coordinate in the output JSON.

---

## 2. Multi-view fusion

1. **Pixel selection.** If a SAM mask exists, use the eroded mask. Otherwise
   use a shrunken bbox. In both cases, keep only depths within a robust band
   around the median depth. This removes the wall/floor behind the object,
   which would otherwise smear the box toward the background.
2. **Lift** every selected pixel to world space and tag it with its
   detection id.
3. **Aggregate per class** across views, then voxel-downsample and remove
   outliers.
4. **DBSCAN per class.** One cluster = one physical instance, so two cups
   produce two clusters. Noise points (label −1) are dropped.
5. **Filter.** Discard clusters that too few distinct views support. A false
   detection rarely recurs at a geometrically consistent 3D location across
   views, so this filter rejects most hallucinations.
6. **Box + score.** Compute a percentile-clipped AABB and the centroid.
   Confidence is the mean score of the supporting detections.

---

## 3. Running headlessly

### Prerequisites
* **Isaac Sim 4.5 / 5.x** (workstation install or pip `isaacsim`), with an RTX GPU
  and driver.
* A separate Python env for stages 2–3:
  ```bash
  python3 -m venv ~/venvs/scanperc && source ~/venvs/scanperc/bin/activate
  pip install -r open_vocab_3d/requirements.txt
  ```
  If `open3d` has no wheel for your Python version, skip it. The code falls back
  to numpy/sklearn.

### One command
```bash
cd open_vocab_3d
ISAAC_SIM_DIR=~/isaacsim PY=~/venvs/scanperc/bin/python \
PROMPTS="chair,table,monitor,coffee cup" MODE=interior NUM_VIEWS=48 \
./run_pipeline.sh scenes/room_scan_upright.usda ./out
```

### Without Isaac Sim (lite renderer)
`render_views_lite.py` writes the same `out/renders/` layout using CPU ray
casting (Open3D + `usd-core`), with no GPU or display needed. Depth is exact
from the mesh. RGB is the bound UsdPreviewSurface texture without lighting,
which is ideal for photogrammetry scans whose lighting is baked in. In dome
mode it skips back faces by default, so the cameras can see into a room
through its outer walls. Unsupported: lighting, transparency, Gaussian splats.
```bash
RENDERER=lite MODE=dome NUM_VIEWS=30 ./run_pipeline.sh scenes/room_scan_upright.usda ./out
# or stage 1 alone:
python render_views_lite.py --usd scenes/room_scan_upright.usda --out out/renders \
    --mode dome --num-views 30 --rings=15,35,55 --radius-scale 0.6 --verify
```

### Stage by stage
```bash
# 1) Render (Isaac Sim's python; headless by default, no display needed)
~/isaacsim/python.sh render_views.py --usd scenes/room_scan_upright.usda --out out/renders \
    --mode interior --num-views 48 --verify

# 2) Detect (GPU recommended; downloads HF weights on first run)
python detect_2d.py --renders out/renders --out out/detections \
    --prompts "chair,table,monitor,coffee cup" --sam --vis

# 3) Lift + cluster (CPU, seconds)
python lift_cluster.py --renders out/renders --detections out/detections \
    --out out/scene_objects.json --extended --save-ply out/clusters.ply
```

On a remote server or in Docker, use the `nvcr.io/nvidia/isaac-sim` container
and run `./python.sh` inside it, mounting the scan and `out/` directories.
Rendering is headless by default, so no X server is needed.

### Tests (no Isaac Sim / GPU needed)
```bash
python tests/test_lift_cluster.py      # synthetic scene → asserts AABBs within a few cm
# or: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/
#     (the env var avoids ROS's launch_testing pytest plugin crashing on startup
#      when /opt/ros/* is on PYTHONPATH)
```

### Tuning cheatsheet
| Symptom | Knob |
|---|---|
| Boxes stretch into walls/floor | use `--sam` in stage 2; lower the depth-band width in stage 3 |
| One object split into several | raise `--eps` |
| Two adjacent objects merged | lower `--eps`; use SAM masks |
| Spurious objects | raise `--min-views`, `--box-threshold` |
| Missing small objects | more views, `--mode interior`, lower `--box-threshold` |
| Room scan renders show only walls' backsides | `--mode interior` |

### Gaussian splats
The renderer needs geometry that the RTX depth annotator can see. Convert
splats to a mesh first (e.g. SuGaR / 2DGS / Poisson on splat centers), or use
an Isaac Sim build with native splat rendering, then check that depth is
populated with `--verify`.

---

## 4. App (v2): upload a scan, get JSON

A local web app. You upload a `.usdz` scan; the server fixes its orientation,
runs the full two-pass pipeline, and writes a JSON object list. The page shows
the scan in 3D with the boxes, so you can review it and send it on. Contract:
[app/API_V2.md](app/API_V2.md).

```bash
python3.12 -m venv .venv && .venv/bin/pip install --no-cache-dir -r requirements-app.txt
.venv/bin/python app/server.py --out out --port 8000
# open http://127.0.0.1:8000
```

* **Add scan:** use the button, or drag a `.usdz` (max 500 MB) onto the page.
  Each upload becomes a scan in the left-hand list (`out/scans/<scan_id>/`).
* **Orientation:** detected automatically (upside-down scans are rotated 180°;
  the client file is never modified). **Flip upside down** overrides it and
  marks any existing results as stale.
* **Pipeline:** starts automatically with the 13 default prompts, which are
  editable. Pass 1 uses 30 wide views; pass 2 uses 30 close-up views around
  the furniture found in pass 1 (chosen automatically). Measured on
  `room_scan.usdz`: about 8 minutes on the RTX 5060, 20 objects. One job
  runs at a time; further uploads wait in a queue.
* **Raw JSON:** `out/scans/<id>/results/scene_objects_raw.json`, copied to
  `out/backend_outbox/<id>_raw.json`.
* **Review:** accept or reject each object and rename classes, then click
  **Send to backend**. This writes `scene_objects_reviewed.json` and
  `out/backend_outbox/<id>_reviewed.json`.
* **JSON schema:** `object_id`, `class_name`, `position` (centroid, meters,
  upright Y-up frame), `bounding_box` (min/max/size), `confidence` and
  `num_views`, plus scan metadata. See `export_json.py` and `app/API_V2.md`.
* **Source video:** the **Video** button in the top bar opens a floating
  player with the capture footage the scan was made from, so you can compare
  it with the 3D model. Click **Attach video** in the panel, or drop an `.mp4`/`.webm`
  onto the page while a scan is selected. The video is stored as
  `out/scans/<id>/source/video.mp4`. It is display-only: the pipeline never reads
  it. Drag the panel's bottom-right corner to resize it. Use H.264 `.mp4` or
  `.webm`; iPhone HEVC `.mov` files don't play in Chrome on Linux, so convert
  those first.
* **Shelf scans from store videos:** **Add scan** also accepts a store video
  (`.mov`/`.mp4`/`.m4v`, e.g. iPhone HEVC originals). It becomes a scan tagged **shelf**, and a
  *convert* job turns it into a textured 3D scan (about 8–15 min). The job runs
  `shelf_convert.py`, which uses the separate `../video_to_3d` environment (`--video-to-3d DIR`).
  The steps are keyframes, COLMAP camera solve, Depth Anything V2 Small depth, TSDF mesh, and a
  texture atlas. When it finishes, the model loads and a GPS-free H.264 copy is attached as the
  scan's video. The scale is **approximate** (it assumes a scanned height of 1.9 m,
  `--shelf-height-m`) until calibrated. Shelf scans don't run the room object detection; gap
  detection is planned. A failed conversion can be retried from the panel.
* **Scale and measuring (shelf scans):** the **Scale** panel shows whether a shelf scan is
  *approximate* (assumed 1.9 m tall) or *calibrated*. **Measure…** works as a ruler: click two
  points on the model to see their distance. Typing the real distance in cm and pressing
  **Calibrate** rescales the whole scan. The scale is stored as a uniform `xformOp:scale` in
  `scene.usda`, so the source file is untouched, and **Reset scale** undoes it. Shelf scans open
  face-on. Automatic bay detection (a 1200 mm bay default) was tried and dropped: on these scans
  the uprights are hidden by products and the bay pitch was not reliably found.
* **Empty-slot labels (ground truth, shelf scans):** in the video panel click **Label** (or
  **Label…** in the *Empty-slot labels* panel). The keyframe nearest the video time opens large;
  drag a box around each empty slot and choose *Empty* or *Pushed back*. Use ◀ ▶ or the arrow
  keys to change keyframe, and Delete to remove a box. Boxes are saved in 2D
  (`labels/empty_slots.json`). The server places each one on the **shelf front** in 3D
  (`shelf_labels.py`), taking the front depth from the products around the box, because the dark
  shelf backs behind a slot are usually not reconstructed, and shows it in the viewer with its
  size in cm.
* **Gap detection (shelf scans, baselines):** **Detect gaps** in the *Gap detection* panel runs
  `gaps_runner.py` on the original keyframes. Its two baselines are **AI**: Grounding DINO + SAM,
  prompts "empty shelf space", "gap on shelf", "missing product on shelf"; and **depth**: regions
  more than 10 cm behind the local shelf front, from the conversion's depth maps. Detections are
  placed on the shelf front like labels and merged across keyframes; a gap needs ≥ 2 keyframes.
  The panel shows each method in its own colour, with a score against your labels: found / missed
  / false gaps / width error. On the first scan (IMG_5198, fully stocked) the AI method returned
  5 wide, thin bands and depth returned 27 patches, most of them visually false. Neither is
  validated until real empty slots are labelled.
* **Tests:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SKIP_GLTF_VALIDATOR=1 .venv/bin/python -m pytest tests/`
  (119 pass).
