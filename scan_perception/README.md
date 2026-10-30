# scan_perception — open-vocabulary 3D objects from an unsegmented USD scan

Takes an unsegmented OpenUSD scan (photogrammetry mesh, converted splat). It renders the scan from many virtual cameras, detects text-prompted objects in 2D, lifts the detections into 3D and fuses them across views. The result goes to `scene_objects.json`.

```
render_views.py      Isaac Sim python   USD  -> views/ (rgb, depth, K, extrinsics)
detect_2d.py         torch env          views/ -> views/detections.json (+ SAM masks)
lift_and_cluster.py  numpy/sklearn env  views/ + detections -> scene_objects.json
geometry.py, views.py                   shared math + I/O (numpy only, imported by all three)
```

The stages run in separate processes and hand off through files. Isaac Sim's Python never needs torch, the GPU detector never needs Kit, and you can re-run clustering with new parameters without rendering or detecting again.

## The math: pixel (u, v, depth) → world (x, y, z)

**1. Pixel → camera frame.** The intrinsics are

    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]

The renderer saves depth `d` as distance to the image plane, which is the camera-frame z, not the length of the ray. A pixel then back-projects through the inverse of K:

    P_c = d · K⁻¹ · [u + ½, v + ½, 1]ᵀ
        = [ (u + ½ − cx)·d / fx ,  (v + ½ − cy)·d / fy ,  d ]ᵀ

The ½ puts the sample at the pixel centre. For a USD camera, `fx = fy = focalLength / horizontalAperture · W`, `cx = W/2`, `cy = H/2`.

**2. Camera → world.** The extrinsics give a world-to-camera transform `T_wc = [R | t]`, so `P_c = R·P_w + t`. Inverting it gives

    P_w = Rᵀ · (P_c − t)        or, homogeneously,   [P_w; 1] = T_wc⁻¹ · [P_c; 1]

Here `T_wc⁻¹ = [Rᵀ | −Rᵀt]`. All together:

    [x, y, z, 1]ᵀ = T_wc⁻¹ · [ d·K⁻¹·[u+½, v+½, 1]ᵀ ; 1 ]

**3. Two conventions that cause most bugs:**

- **Camera axes.** USD and OpenGL cameras look down −Z with +Y up. The OpenCV formula above assumes +Z forward with +Y down. The renderer converts with `T_cw_cv = T_cw_usd · diag(1, −1, −1, 1)` before saving.
- **Row vs column vectors.** `Gf.Matrix4d` is row-vector (`p' = p·M`, translation in the bottom row), so a USD world matrix is the *transpose* of the column-vector matrix used here.

The saved `manifest.json` is already in OpenCV and column-vector form. To use it, just apply `pixels_to_world` from `geometry.py`.

## Fusion and boxes

- **Pixel selection.** With `--sam`, a detection's pixels are its eroded SAM mask. Without SAM they are its box: shrunk 10% per side, keeping only pixels whose depth lies within `0.15 m + ½·object size` of the box centre's median depth. This drops the wall or floor visible behind the object.
- **Clustering, per class.** Points are voxel-downsampled (2 cm) and clustered with DBSCAN (eps 6 cm); each point inherits its voxel's label. Voxelizing evens out density, so a camera close to an object does not dominate, and keeps DBSCAN fast.
- **Multi-view consensus.** Each detection is assigned to the cluster holding most of its points. A cluster survives only if detections from at least `--min-views` (default 2) different views landed in it. This removes single-view hallucinations and background fragments. `confidence` is the mean score of the cluster's detections.
- **Box and position.** The AABB is a per-axis percentile range (1–99%), robust to stray points. `position_world` is the centroid of the inlier points. Note that this centroid is biased toward the surfaces the cameras saw.
- **Cross-class suppression.** If two objects of different classes overlap with 3D IoU > 0.5 (e.g. "monitor" vs "tv"), only the more confident one is kept.

All CLI lengths are in metres and are converted using the stage's `metersPerUnit`. Output coordinates are in stage units and the stage's up axis; `room_scan.usda` is Y-up with 1 unit = 1 m.

## Running headless

### 1. Render (Isaac Sim 4.x / 5.x)

```bash
cd /home/saumy/pranatree_work/room_scan
~/isaacsim/python.sh scan_perception/render_views.py \
    --usd room_scan.usda --out out/views --mode interior
# pip-installed Isaac Sim: activate that env and use `python` instead of python.sh
```

`--headless` is on by default; `--no-headless` opens the viewport. Isaac Sim's first launch compiles shaders and can take several minutes.

- **`--mode interior` (default).** A 3×3 grid of eye-height (1.4 m) positions, each with 8 headings × 2 pitches, for 144 views. Use this for rooms; a dome placed outside a closed room only sees the outside of the walls.
- **`--mode dome`.** Three rings of 16 cameras around the scene centre. Use this for open scenes or tabletop objects.
- **`--color-source albedo` (default).** Saves the unlit texture colour. Scans already have the real lighting baked into their textures, and this needs no light setup. If your Replicator version lacks the `DiffuseAlbedo` annotator, the script falls back to `lit`. That mode path-traces with a dome light plus a headlight riding on the camera, so closed rooms are still lit.
- The script cross-checks every saved pose against Replicator's `camera_params` annotator and re-renders if they disagree. Any `WARNING ... camera_params` line means extrinsics may be off for that view.
- Nothing is written to the source USD; the camera and lights live on the session layer.

### 2. Detect (CV env with a GPU)

```bash
python -m venv .venv-cv && . .venv-cv/bin/activate
pip install -r scan_perception/requirements-cv.txt   # install the torch build matching your CUDA first

python scan_perception/detect_2d.py --views out/views \
    --prompts chair monitor "coffee cup" table --sam --overlays
```

- Detectors: `grounding-dino` (default, `IDEA-Research/grounding-dino-base`), `owlv2`, `owlvit`.
- Weights download from the Hugging Face hub on first run. On an offline box, set `HF_HUB_OFFLINE=1` after caching them.
- `--overlays` writes `out/views/overlays/*.jpg`. Look at a few of them before tuning anything in 3D.

### 3. Lift, cluster, export (CPU)

```bash
python scan_perception/lift_and_cluster.py --views out/views \
    --out out/scene_objects.json --debug-ply out/objects.ply
```

Open `objects.ply` in MeshLab or CloudCompare on top of the scan to check the clusters. Tuning:

| Symptom | Fix |
|---|---|
| Two neighbouring objects merged | lower `--eps` (e.g. 0.04) |
| One object split in two | raise `--eps`, or render more views |
| Real objects missing | `--min-views 1`, lower `--box-threshold` in stage 2 |
| Boxes include floor/wall | use `--sam`, or lower `--depth-band` |

### One-shot

```bash
ISAAC_PY=~/isaacsim/python.sh CV_PY=.venv-cv/bin/python \
  scan_perception/run_pipeline.sh room_scan.usda out chair monitor "coffee cup"
```

## Tests

```bash
python -m pytest scan_perception/tests -q
```

The tests ray-cast a synthetic Y-up room with known boxes into the exact on-disk format and fake detections from the object-id buffer. They then check that the full lift/cluster stage recovers each AABB and rejects a planted one-view false positive. Tolerances are 3 cm with masks and 10 cm with boxes only. This covers the geometry and USD→OpenCV conventions end to end. It does not exercise Isaac Sim or the detectors.
