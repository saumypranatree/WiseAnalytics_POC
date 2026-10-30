# video_to_3d: phone video to textured USDZ scan (prototype)

This turns one handheld phone video of a shelf into a UV-textured triangle mesh `.usdz` that
`open_vocab_3d` (scan_prep, render_views_lite, usd_to_gltf) ingests. It also writes a
browser-playable H.264 copy of the video for the app's video panel. Everything runs offline once the
model is cached. See `INSTALLED.md` for packages, sizes and licenses.

## Pipeline

| Step | Script | What it does | IMG_5198 runtime |
|---|---|---|---|
| 1 | `scripts/01_extract.py` | PyAV decodes HEVC 10-bit **HLG** (BT.2020). It tone-maps to SDR sRGB (HLG inverse OETF, OOTF, BT.2020 to BT.709, soft shoulder), applies the display rotation (-90 becomes portrait 1080x1920), writes all frames as JPEG (no EXIF) plus a Laplacian-variance blur score, and keeps the sharpest frame in each of N equal windows (N = `--keyframes auto`: 5 per second of video, clamped to 100..300). It also writes the H.264 MP4 (libx264 CRF 20, yuv420p, BT.709 tags, faststart, no audio, no Apple/GPS metadata). | 50 s |
| 2 | `scripts/02_sfm.py` | pycolmap 4.2.1, CPU: SIFT (8k features, 1600 px), sequential matching (overlap 15, quadratic), incremental mapping. One shared SIMPLE_RADIAL camera with a focal **prior** of 0.833 x the long side (iPhone 1x wide lens, 26 mm-equiv), refined by bundle adjustment. | 226 s (extract 21, match 30, map 175) |
| 3 | `scripts/03_depth.py` | Undistorts the frames to pinhole at half resolution (540x960). Depth Anything V2 **Small** (fp16, GPU) gives relative inverse depth. Each frame gets a robust affine fit `1/z = a*d + b` to the SfM sparse depths, then a smooth multiplicative correction field (16x28 grid, Gaussian-filled) fitted to the same points. | 35 s |
| 4 | `scripts/04_fuse.py` | Drops mono-depth edge pixels (relative-gradient test), then Open3D tensor VoxelBlockGrid TSDF on **CPU** (voxel = 0.006 x median depth, trunc 5 voxels, weight >= 2). Marching cubes, then removal of components under 1% of the largest, then quadric decimation to <= 400k tris. | 6 s |
| 4b | `scripts/04b_gravity.py` | The facade normal is the dominant camera-facing normal. Up is the in-facade direction that maximises horizontal-triangle area (shelf boards, product tops). | 3 s |
| 5 | `scripts/05_texture.py` | xatlas UV unwrap (one atlas of about 4.7k x 4.7k). The best view per triangle is the one with the most first-hit pixels in a ray-cast of that keyframe (occlusion-aware) x a sharpness weight, followed by 3 neighbour-majority smoothing passes. Baking ray-casts the UV layout, projects each texel into its triangle's full-resolution undistorted keyframe and samples bilinearly. 8 px of chart padding. | 134 s (xatlas alone ~100 s) |
| 6 | `scripts/06_export_usdz.py` | Rotates so +Y = up and +Z = facade normal (toward the aisle), with origin y = 0 at the lowest geometry. Applies a **provisional** scale (scanned height := 1.9 m). Writes `UsdGeom.Mesh` (faceVarying `st`) + `UsdPreviewSurface` <- `UsdUVTexture` (JPEG), packaged with `UsdUtils.CreateNewUsdzPackage`. metersPerUnit = 1, upAxis = Y. | 1 s |

Total is about 8 min on this laptop (RTX 5060 8 GB, 32 threads). The GPU is used only in step 3.

## Reproduce

```bash
cd /home/saumy/pranatree_work/room_scan/video_to_3d
# one-time environment (already done; see INSTALLED.md)
env -u PYTHONPATH python3.12 -m venv .venv
env -u PYTHONPATH .venv/bin/pip install pycolmap==4.2.1 av==19.0.1 opencv-python-headless numpy open3d==0.20.0 \
    usd-core==26.8 xatlas==0.0.11 pillow scipy transformers huggingface_hub safetensors
env -u PYTHONPATH .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# first run only: allow the 95 MB model download
HF_HUB_OFFLINE=0 ./run_all.sh "/home/saumy/pranatree_work/room_scan/video/IMG_5198 (2).MOV" IMG_5198
# later runs are fully offline
./run_all.sh "<video>" <STEM> [scanned_height_m=1.9]
```

Outputs: `out/<STEM>/<STEM>.usdz` and `out/<STEM>/<STEM>_h264.mp4`. Intermediates are in `work/<STEM>/`
(`frames/`, `sfm_sr/`, `dense/`, `mesh/`, `tex/`, `*_info.json`).
Debug preview (real frame | vertex-colour mesh | oblique view):
`env -u PYTHONPATH .venv/bin/python scripts/preview.py work/<STEM> work/<STEM>/mesh/fused.ply out.png`

### Validation against the app (read-only use of open_vocab_3d, outputs in validate/)

```bash
cd /home/saumy/pranatree_work/room_scan/open_vocab_3d
V=/home/saumy/pranatree_work/room_scan/video_to_3d/validate
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "import scan_prep,json; \
  r=scan_prep.prepare_scan('$V/../out/IMG_5198/IMG_5198.usdz','$V/scans','IMG_5198.usdz',scan_id='IMG_5198_check'); \
  json.dump(r,open('$V/prepare_scan_record.json','w'),indent=1)"
S=$V/scans/IMG_5198_check/scene.usda
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python render_views_lite.py --usd $S --out $V/renders \
    --mode dome --num-views 6 --width 640 --height 360 --verify
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python render_views_lite.py --usd $S --out $V/renders_front \
    --mode focus --focus-domes "0,1.0,0.1,1.6,16" --focus-rings "0,12" --width 960 --height 540 --verify
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python usd_to_gltf.py --usd $S --out $V/viewer/model.glb
```

IMG_5198 results:
- `prepare_scan`: 1 mesh, 353,622 triangles, upAxis Y, mpu 1, bounds about 2.84 x 2.03 x 0.84 m. Orientation is **no flip**, confidence low (score -1.38). That is correct: the scene is already upright, and there is no floor for the heuristic to rely on.
- Renders: the textured shelves are recognisable, with legible brand names and price labels (`validate/renders_front/rgb/0005.png`, `0013.png`). Dome mode is a poor fit for a one-sided facade: 4/6 views look from behind and are skipped as empty. Use `--mode focus` with a frontal dome, as above.
- GLB: `validate/viewer/model.glb` (42 MB) is written.

## Changes after the prototype (2026-10-15)
- **Tiled UV unwrap (`05_texture.py`).** On IMG_5199, one xatlas call on the whole 400k-triangle facade ran for more than 20 min without
  finishing. Unwrapping 7 tiles of about 57k triangles along the facade's long axis took 33 s in total, and the whole texture step took 87 s.
  Each tile gets its own 2048-class atlas, so the .usdz has one Mesh + material per tile, which the app supports.
  The old single-atlas script is kept as `scripts/05_texture.py.bak_single_atlas`.
- **Automatic keyframe count (`01_extract.py`).** On IMG_5200 (47 s, 720p), 100 keyframes split the SfM into 2 pieces
  (30 + 49 of 100 frames registered), and the scan covered only a fragment. With 250 keyframes the largest piece has 129 frames and covers the
  whole 5.6 m bottle facade (`out/IMG_5200_k250/`). The default is now `auto`: 5 per second, 100..300.
- These scripts are also run by the app: `open_vocab_3d/shelf_convert.py` converts store videos uploaded in the UI into shelf scans.

## Known limitations
- **Scale is provisional.** Monocular video has no metric scale. Step 6 sets the scanned vertical extent to 1.9 m (the pan
  covers roughly the full gondola height and the floor is barely visible). The resulting plausibility is a shelf pitch of about 16 cm and a camera distance of about 1.25 m. It is not measured. The error could be +-20% or more. A fix would be a known reference (shelf-label height, gondola height, or ARKit/LiDAR depth).
- **Geometry is mono-depth based.** Depth comes from Depth Anything V2 Small, pinned to the SfM sparse points. Each surface is about 1% off the SfM points (median), but product-level relief is smoothed. Thin shelf lips and the dark shelf backs are partly missing, which shows as grey holes behind products. The facade shows a mild bow of about 14 cm over 2.8 m, which may be SfM drift.
- One-sided scan: only the front is textured. About 10% of triangles (small or back-facing) are seen by no view and get the nearest-centre view's colours.
- Texture seams: per-triangle single-view selection, with no exposure compensation between frames. Seams are faint because exposure was stable in this clip.
- The HLG to SDR tone map is a hand-written approximation, slightly bright and low-contrast compared with the iPhone's own SDR rendering.
- Open3D 0.20's CUDA kernels **hang** on the RTX 5060 (sm_120). Everything Open3D does runs on the CPU. The legacy `ScalableTSDFVolume` returned empty meshes, so the tensor `VoxelBlockGrid` is used.
- The pycolmap pip wheel is CPU-only (no PatchMatch MVS). SfM mapping of 100 frames takes about 3 min. Thread oversubscription made it 8 min in the first try; it is now capped at 12 threads.
- The first SfM attempt (OPENCV model, no focal prior) converged to fx 2155 / fy 2116 px with a stray 7-image model. That is a focal/depth ambiguity of a near-planar pan. The focal prior (SIMPLE_RADIAL, 1600 px start, converged to 1685 px) fixed it. It is kept in `work/IMG_5198/sfm/` for reference only.
- The H.264 copy drops the audio track. It carries no GPS/location metadata (verified by container metadata and a byte search).

## Reproducibility check
`./run_all.sh "<IMG_5198 (2).MOV>" IMG_5198_repro` ran end-to-end with exit 0 in about 7 min
(extract 48 s, sfm 207 s, depth 14 s, fuse 3 s, gravity 2 s, texture 148 s, export <1 s).
The result is close but **not bit-identical**, because multithreaded SfM is non-deterministic: 100/100 registered,
f = 1689 vs 1685 px, 363k vs 354k triangles, bounds within about 10 cm. The repro outputs were deleted afterwards.
