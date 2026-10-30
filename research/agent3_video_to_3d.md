# Agent 3: handheld aisle video to textured 3D mesh (.usdz) for open_vocab_3d

Date: 2026-10-15. Research only. No project files were changed and nothing was installed.

Evidence tags used below:
- **[code: path]**: read in the local repo.
- **[local probe]**: read-only inspection of this machine or of the video files (a scratchpad parser; nothing installed).
- **[src: URL]**: a page or file I opened.
- **unverified**: no source confirmed it.

No runtimes or VRAM figures are invented. Where a project publishes no number, this report says so.

---

## 1. What the app needs from a scan (from the code)

| Requirement | Detail | Evidence |
|---|---|---|
| File type | A `.usdz`, i.e. a zip, uploaded via `POST /api/scans`. The server rejects any other extension (400). | [code: app/server.py:1061-1067], [code: scan_prep.py:434-435] |
| Size limits | At most 500 MB upload (configurable), at most 2 GB uncompressed, at most 2000 zip entries. No absolute paths, `..` entries or symlinks. | [code: scan_prep.py:62-64, 148-196] |
| Root layer | The **first** `.usd/.usda/.usdc` entry in the zip. It must open with `Usd.Stage.Open`. | [code: scan_prep.py:176-180, 391-399] |
| Geometry | At least one visible `UsdGeom.Mesh` with purpose default or render, with points, faceVertexCounts and faceVertexIndices. Polygons are fan-triangulated. World transforms are honoured. If there is no `defaultPrim`, exactly one root prim may contain meshes. | [code: scan_prep.py:202-250, 405-417] |
| Units | `metersPerUnit` must be > 0 and finite. Every stage multiplies coordinates by it (scan_prep, renderer, GLB). An up-to-scale model can therefore carry its scale in `metersPerUnit`, or have it baked into the points. | [code: scan_prep.py:400-402], [code: render_views_lite.py:353,442], [code: usd_to_gltf.py:129] |
| Up axis | The stage `upAxis` is used (Y or Z). The app's output frame is "upright, Y-up, meters", so write **`upAxis = "Y"`**, like the working reference `room_scan.usda`. | [code: app/API_V2.md "Frame: upright, Y-up, meters"], [code: ../room_scan.usda header] |
| Orientation | `scan_prep` runs a **floor-based flip heuristic**. It takes the dominant horizontal surface by area (\|n·up\| > 0.9, 5 cm bins) and votes on its height, its normal sign, the support surfaces 0.25-1.25 m from it, and the vertical mass. When the score is > 0 it auto-applies a 180° X rotation. Horizontal triangles need consistent winding (normals pointing into the room) for the normal vote to work. | [code: scan_prep.py:20-44, 259-328] |
| Colour | **A UV texture is required in practice.** The renderer follows the material bound to the Mesh prim: `UsdPreviewSurface.inputs:diffuseColor` connected to a `UsdUVTexture` `inputs:file`, with UVs from the `st` primvar (or the `UsdPrimvarReader_float2` `varname`), interpolation faceVarying or vertex/varying. Without a texture, it uses a constant diffuseColor or **only `displayColor[0]`** (`dc[0]`). Per-vertex colours are ignored and the whole mesh renders as one flat colour. A vertex-coloured mesh is therefore **not** enough. | [code: render_views_lite.py:78-112, 157-177, 241-256] |
| Textures per mesh | One texture per Mesh prim. The material is resolved on the Mesh prim itself, so GeomSubset per-face materials are not handled. A mesh with several texture pages must be split into one Mesh prim per page. The reference scan does exactly this (`chunk_A`, `chunk_B_main`, `chunk_C`, each with its own material and JPEG). | [code: render_views_lite.py:82,157], [code: ../room_scan.usda] |
| Texture files | Relative asset paths inside the usdz, e.g. `@./textures/x.jpeg@`. JPEG and PNG bytes go into the GLB as-is; anything else is re-encoded to PNG. Textures are loaded as RGB (no alpha) and sampled with repeat wrap. The V coordinate is flipped (USD origin bottom-left). | [code: usd_to_gltf.py:92-108, 138-149], [code: render_views_lite.py:115-120, 185-196] |
| Shading | Unlit. Baked photographic colour is what the app wants. There is no lighting, transparency, subdivision or splat support. | [code: render_views_lite.py:13-17], [code: README.md] |
| GLB size | `usd_to_gltf` writes **unwelded** vertices: 3 corners per triangle, each with position, normal and UV (32 bytes). That is about 96 bytes per triangle, so 1 M triangles gives a GLB of roughly 96 MB plus textures (arithmetic from the code). Decimate to about 0.3-1 M triangles per aisle. | [code: usd_to_gltf.py:129-141] |
| Camera layout downstream | Pass 1 uses 30 dome views (rings 15/35/55°, radius-scale 0.4) around the bbox centre. Pass 2 uses focus domes around "anchor" objects. Dome mode culls back faces. For a mostly 2.5D shelf facade, half of the dome sees its back. Views with more than 95% empty depth are skipped. This is a pipeline-tuning concern for later agents and does not block ingestion. | [code: app/pipeline_config.json], [code: render_views_lite.py:443-447] |

Implications for aisle scans:
1. Write Y-up, right-handed geometry with consistent outward winding.
2. Include some **floor** in the reconstruction, so the floor-based orientation heuristic has something to find. Shelf boards are also horizontal and can confuse it. After upload, check `scan.json.orientation.evidence` and use the manual flip if needed.
3. Use a texture atlas with faceVarying `st`, one material per Mesh, and JPEG textures.

---

## 2. Facts about the input videos

From a **[local probe]**: a read-only MP4 atom parser written to the scratchpad. No ffmpeg or exiftool was used.

- Make and model: **Apple iPhone 13 mini**, iOS 26.6.1. The video-track metadata gives `camera.lens_model` = "iPhone 13 mini back camera 5.1mm f/1.6", plus a `camera.focal_length.35mm_equivalent` field. The loose parse gives 26 mm; confirm with `exiftool -ee`. The files also contain **GPS location** (`com.apple.quicktime.location.ISO6709`). Strip it before archiving or sharing.
- `mebx` timed-metadata tracks in the three .MOV files:
  - `com.apple.quicktime.video-orientation`: 1 sample. ExifTool says the values follow the EXIF orientation codes, 1 to 8, where 6 is "Rotate 90 CW". [src: https://exiftool.org/TagNames/QuickTime.html]
  - `com.apple.quicktime.detected-face`: fields bounds, face-id, roll-angle and yaw-angle. Face detection only, with no camera data. [src: exiftool QuickTime tags]
  - `com.apple.quicktime.live-photo-info`: 540 samples (one per frame) of 144 bytes each. Its setup data is a bplist naming framework versions, including "CoreMotion". ExifTool lists the tag with **no documented structure**. Whether it holds per-frame motion or exposure data is **unverified and undocumented**. [src: exiftool QuickTime tags; https://exiftool.org/forum/index.php?msg=80221]
  - `com.apple.quicktime.scene-illuminance`: per-frame scene brightness in milli-lux. [src: exiftool QuickTime tags]
  - `com.apple.quicktime.segment-identifier`: a UUID.
- **Conclusion on mebx.** None of the documented tracks contains camera intrinsics, poses, gyro or accelerometer data. ExifTool's motion tags (camm, Accelerometer and similar) come from other camera brands' formats, not from these keys. The only useful metadata are the lens model and the 35 mm-equivalent focal length (an initial focal guess), plus the rotation and display matrix. Scene illuminance could flag exposure changes. Intrinsics must be **self-calibrated** in SfM. iPhone video crops the sensor and may stabilise electronically, so the photo focal length is only a prior (the stabilisation effect is unverified).
- `IMG_5200 (2).MP4` is a recompressed H.264 stream with none of these tracks.
- Frame counts at 30 fps: 540, 258, 918 and about 1413. After 2-5 fps keyframe selection this leaves roughly 40-280 frames per aisle.

---

## 3. Comparison table

Licence covers both code and weights. "Prod OK?" means usable for commercial on-prem production without buying a licence. Have counsel confirm before shipping.

| # | Tool | Code licence | Weights licence | Install on this box (no sudo) | CUDA / sm_120 | Output | Prod OK? |
|---|---|---|---|---|---|---|---|
| 1 | **COLMAP 4.2.1** (SfM + global mapper + PatchMatch MVS + Poisson/Delaunay meshing + **mesh_texturer**) | BSD-3 [src: github.com/colmap/colmap LICENSE]. The build pulls in CGAL (GPL/LGPL) and LSD (**AGPL-3**) when enabled [src: colmap src/thirdparty/LSD/LICENSE; https://www.cgal.org/license.html] | ALIKED BSD-3 and LightGlue Apache-2.0 (optional ONNX features) [src: gh cvg/LightGlue README; gh Shiaoming/ALIKED] | conda-forge `colmap` 4.2.1 has `cuda_129` and `cuda_134` linux-64 builds [src: api.anaconda.org/package/conda-forge/colmap]. The recipe builds with `CMAKE_CUDA_ARCHITECTURES=all`, ONNX on and cgal-cpp in host [src: conda-forge/colmap-feedstock recipe/build.sh, recipe.yaml] | "all" on CUDA 12.9 should include sm_120 (inferred from CMake semantics; **verify by running**) | sparse model, depth maps, fused.ply, mesh, **textured/mesh.ply + texture.png** [src: colmap doc/format.rst] | **Yes** (BSD core). GPL/AGPL parts matter only if you redistribute; internal on-prem use is generally fine, but check with legal |
| 2 | pycolmap / **pycolmap-cuda12** 4.2.1 (pip) | BSD-3 [src: pypi.org/pypi/pycolmap-cuda12] | n/a | `pip install pycolmap-cuda12`, manylinux_2_28, cp310-cp314 [src: PyPI JSON] | Built with CUDA 12.9, `CMAKE_CUDA_ARCHITECTURES="all-major"` [src: colmap python/ci/install-colmap-almalinux.sh] | Same API. **But** the wheel is built with `ONNX_ENABLED=OFF`, `CGAL_ENABLED=OFF` and `LSD_ENABLED=OFF` (same file): no ALIKED/LightGlue/LoMa, no Delaunay mesher (needs CGAL [src: colmap src/colmap/mvs/delaunay_meshing.h]), no Manhattan aligner, and texturing **skips occlusion testing** ("CGAL is disabled; occlusion testing will be skipped" [src: colmap src/colmap/mvs/texture_mapping.cc]) | Yes (clean BSD), with reduced features |
| 3 | GLOMAP | BSD-3 | n/a | **Deprecated.** It now ships inside COLMAP as `global_mapper` [src: gh colmap/glomap README; COLMAP 4.0.0 release notes] | via COLMAP | COLMAP sparse model | Yes |
| 4 | OpenMVS 2.4.0 | **AGPL-3.0** [src: gh api license cdcseacave/openMVS] | n/a | Release ships `OpenMVS_Ubuntu_x64.zip` (CUDA build only for Windows) [src: gh releases v2.4.0] | Ubuntu binary CUDA support unverified | dense cloud, mesh, refine, **TextureMesh** [src: openMVS README] | Risky: AGPL obligations apply if you modify it and offer it over a network or distribute it. Legal review needed |
| 5 | Meshroom / AliceVision 2025.1.0 | MPL-2.0 [src: COPYING.md for both] | (DepthEstimation plugin models: unverified) | Linux binaries; "built with CUDA-12, compute capability >= 5.0" (from the release notes, via [src: https://github.com/alicevision/meshroom/releases]) | Depth maps need CUDA [src: https://meshroom-manual.readthedocs.io/en/latest/faq/needs-cuda/needs-cuda.html]; sm_120 support unverified | textured OBJ | Yes (MPL-2 is file-level copyleft) |
| 6 | OpenMVG | MPL-2.0 [src: gh api] | n/a | build from source | CPU | SfM only (pairs with OpenMVS) | Yes, but adds nothing over COLMAP here |
| 7 | mvs-texturing (Waechter 2014) | BSD-3 [src: gh nmoehrle/mvs-texturing LICENSE.txt] | n/a | cmake build from source (sudo-free build unverified) | CPU | textured OBJ | Yes. COLMAP's `mesh_texturer` implements the same method [src: colmap doc/tutorial.rst "[waechter2014]"] |
| 8 | DUSt3R / MASt3R(-SfM) | **CC BY-NC-SA 4.0** [src: gh naver/dust3r, naver/mast3r LICENSE] | NC | pip/conda | torch cu128 OK | point maps + poses | **No** |
| 9 | VGGT | "VGGT License" (commercial use permitted except military, since 2025-07-29) [src: gh facebookresearch/vggt README, LICENSE.txt] | Only **VGGT-1B-Commercial** (gated, form) is commercial; the original checkpoint is NC [src: same; https://huggingface.co/facebook/VGGT-1B-Commercial] | pip (torch, einops, safetensors) | torch cu128 OK (local torch 2.11+cu128 lists sm_120 [local probe]) | depth, point maps, cameras, COLMAP export with optional BA (`demo_colmap.py --use_ba`) [src: README] | Yes (Commercial checkpoint). The README no longer has a VRAM table, so the 8 GB frame budget is **unverified** |
| 9b | VGGT-Omega | **FAIR Noncommercial** [src: gh facebookresearch/vggt-omega LICENSE] | NC | pip | | Published peak memory (1B-512): 1 frame 6.02 GB, 10: 6.67, **25: 7.80**, 50: 9.66 GB [src: vggt-omega README] | **No** |
| 10 | Pi3 / Pi3X | BSD-3 [src: gh yyfz/Pi3] | **CC BY-NC 4.0** [src: same README] | pip | | points, local points, conf, c2w poses | **No** (weights) |
| 11 | MapAnything | Apache-2.0 | `facebook/map-anything-apache` is **Apache-2.0**; `facebook/map-anything` is CC-BY-NC [src: gh facebookresearch/map-anything] | pip, py3.12 | | **metric** pts3d, poses, intrinsics; accepts optional intrinsics/poses/depth; COLMAP export; "up to 2000 views on 140 GB" in memory-efficient mode [src: same] | Yes (apache weights). 8 GB capacity unverified |
| 12 | Depth Anything 3 (DA3) | Apache-2.0 [src: gh ByteDance-Seed/Depth-Anything-3] | DA3-BASE, DA3-SMALL, DA3METRIC-LARGE and DA3MONO-LARGE are **Apache-2.0**. GIANT, LARGE and NESTED are CC BY-NC 4.0 [src: README model table] | pip (+xformers) | | depth, rays, poses (any-view); COLMAP-format extrinsics; glb/ply export. DA3-Streaming for long video: peak VRAM 11.5-15.9 GB at 504x154, 18.7-28.3 GB at 504x378 (chunk 30-120; model size not stated) [src: da3_streaming/README.md] | Yes for Base/Small/Metric. Streaming does not fit 8 GB at the published settings |
| 13 | Fast3R | FAIR Noncommercial [src: gh facebookresearch/fast3r LICENSE] | NC | | | | **No** |
| 14 | Spann3R, CUT3R | CC BY-NC-SA 4.0 [src: their LICENSE files] | NC | | | | **No** |
| 15 | 3DGS (Inria), 2DGS, Gaussian Opacity Fields, SuGaR | Inria/MPII "Gaussian-Splatting License": "research and/or evaluation purposes only… cannot use… for commercial purposes without prior and explicit consent" [src: gh graphdeco-inria/gaussian-splatting LICENSE.md; the 2DGS, GOF and SuGaR repos carry the same licence text] | | | | splats; meshes via TSDF/marching (2DGS/GOF/SuGaR) | **No** (unless Inria licenses it) |
| 16 | **gsplat** (incl. 2DGS trainer) / nerfstudio | Apache-2.0 [src: gh api, both] | n/a | `pip install gsplat` JIT-compiles CUDA on first run; needs PyTorch 2.7+, "CUDA 12.8 and 13.2 build compatibility" [src: gh nerfstudio-project/gsplat README]. **The system nvcc is 12.0** [local probe], which cannot target sm_120, so a conda `cuda-nvcc` 12.8+ is needed (unverified) | sm_120 via cu128 | splats; `simple_trainer_2dgs.py` exists [src: gsplat repo tree]. nerfstudio `ns-export tsdf/poisson`; Poisson needs predicted normals [src: https://docs.nerf.studio/quickstart/export_geometry.html] | Yes (code). Mesh extraction from splats is DIY (render depth, then TSDF) |
| 17 | Monocular depth: Depth Anything V2 | Apache-2.0 code | **Small: Apache-2.0**; Base/Large/Giant: CC-BY-NC-4.0 [src: DA-V2 README; HF DA-V2-Large card] | pip | | relative depth (metric fine-tunes exist) | Small only |
| 17b | Metric3D v2 | BSD-2 code | "For further commercial inquiries, please contact…" [src: Metric3D README]. Weights terms are **unclear/unverified** | pip | | metric depth + normals | Unclear |
| 17c | UniDepth | **CC BY-NC 4.0** [src: LICENSE] | NC | | | metric depth + K | **No** |
| 17d | Depth Pro (Apple) | Apple licence: "use, reproduce, modify and redistribute"; no patent grant [src: gh apple/ml-depth-pro LICENSE] | Same licence (README: "model weights are released under the LICENSE terms") | pip | | metric depth + focal | Probably yes. The licence grants no patent rights, so legal should review |
| 18 | **Open3D 0.20** (already in the app venv) | MIT [src: gh isl-org/Open3D LICENSE] | n/a | installed [local probe] | CPU / CUDA | `ScalableTSDFVolume`, `t.geometry.VoxelBlockGrid`, Poisson, and colour-map optimisation (`run_rigid_optimizer` / `run_non_rigid_optimizer`) are all present [local probe] | Yes |
| 19 | Video decode: **PyAV 19.0.1** | BSD-3 bindings, wheels bundle FFmpeg [src: PyPI; gh PyAV-Org/PyAV README] | | `pip install av` (Python >= 3.12) | | `VideoFrame.rotation` exposes the display rotation [src: PyAV av/video/frame.pyi] | Yes. HEVC decode in the bundled FFmpeg is **unverified until tested**. FFmpeg LGPL and HEVC patent questions belong with legal |
| 20 | USDZ writing: usd-core 26.8 (`pxr`, already in venv) | LicenseRef-TOST-1.0 (Pixar's modified Apache) [src: PyPI usd-core] | | installed; `UsdUtils.CreateNewUsdzPackage` present [local probe] | | .usdz | Yes |

Runtimes: none of these projects documents a runtime for an RTX 5060 Laptop on this kind of footage. Measure them; none are estimated here.

---

## 4. Notes on each family, for THIS footage

The footage is close-range sideways pans of a near-planar shelf face, with dense repeated packs, yellow price strips, reflective bottles and some blur.

### 4.1 Classical SfM + MVS (COLMAP, OpenMVS, Meshroom, OpenMVG)
- **Repetitive packs** are the main SfM failure mode. Identical facings produce false matches far apart in time. Mitigations:
  - Use the **`sequential_matcher`** for video. It matches consecutive frames and runs vocab-tree loop detection only every N frames.
  - Restrict loop candidates with `--SequentialMatching.loop_detection_min_index_distance`, or disable loop detection, because a one-way pan has no real loops. [src: colmap doc/tutorial.rst, "Sequential Matching"]
- **Planar facade**: fundamental-matrix estimation degenerates on planes. COLMAP 4.2 added optional **DEGENSAC** "for scenes dominated by a plane" (`TwoViewGeometry.use_degensac`) and 6-point shared-focal solvers for unknown intrinsics. [src: COLMAP 4.2.0 release notes]
- **Low overlap / weak texture**: ALIKED and LightGlue are recommended for limited overlap and little texture [src: colmap doc/faq.rst]. They need ONNX, so use the conda-forge build, not the pip wheel. For weakly textured MVS, use a larger `PatchMatchStereo.window_radius`, a higher `max_image_size` and a lower `filter_min_ncc` [src: colmap faq.rst].
- **Reflective bottles and foils**: COLMAP's tutorial says "Avoid specularities on shiny surfaces" [src: tutorial.rst]. Expect holes or noise there. Delaunay meshing is "more robust to outliers" than Poisson, and the two can be combined [src: faq.rst "Surface mesh reconstruction"].
- **Sideways pan**: translation gives good parallax, which suits SfM. Pure rotation does not ("Do not take images from the same location by only rotating the camera") [src: tutorial.rst].
- **Texturing**: COLMAP's `mesh_texturer` (new in 4.0) does Waechter-style view selection with global colour correction and writes **one atlas `texture.png` plus a PLY with per-face UVs** [src: colmap doc/format.rst; texture_mapping.h options `apply_color_correction`, `texture_scale_factor`, `inpaint_radius`]. One atlas maps to one USD Mesh with one material, which is exactly what the app supports.
- OpenMVS has the most mature mesh refinement and texturing in open source, but it is AGPL. Meshroom is MPL-2 and a usable fallback, but it is a heavier GUI-centric stack and sm_120 support is unverified.

### 4.2 Feed-forward reconstruction (DUSt3R family, VGGT, Pi3, MapAnything, DA3)
- Licences remove most options. DUSt3R, MASt3R, Spann3R, CUT3R, Fast3R, Pi3/Pi3X weights and VGGT-Omega are all non-commercial. The commercially usable ones are **VGGT-1B-Commercial** (gated), **MapAnything-apache** and **DA3-BASE/SMALL/METRIC-LARGE**.
- **8 GB VRAM is the binding constraint.** The only published memory curve (VGGT-Omega-1B-512, NC) reaches 7.80 GB at 25 frames. DA3-Streaming's published settings all need more than 11.5 GB. Feed-forward models on this GPU will need chunking (about 20-30 frames per chunk) plus Sim(3) chunk alignment. Chunk alignment is exactly where scale drift appears, so a long 47 s aisle is a risk.
- Strengths: robust to blur and low texture, and they give **initial poses and intrinsics**. VGGT and MapAnything export to COLMAP format, which can seed COLMAP or be refined with BA. VGGT's README also notes you can mask reflective regions by zeroing pixels [src: vggt README].
- Weaknesses: the predicted depth is at reduced resolution (inputs about 512 px; MapAnything and VGGT resize). Feed-forward depth is not as sharp at slot edges as PatchMatch on full-resolution frames, and cm-level slot measurement depends on edge sharpness (an inference, not a benchmark).

### 4.3 Neural / Gaussian
- The app cannot render splats [code: render_views_lite.py:17], so a splat model would still need meshing and texturing.
- The Inria-licensed mesh-from-splat methods (2DGS, GOF, SuGaR) are non-commercial.
- **gsplat** is Apache-2.0 and includes a 2DGS trainer, so a commercial path exists: COLMAP poses, then gsplat 2DGS, then rendered depth, then Open3D TSDF, then mesh, then COLMAP `mesh_texturer` on the real frames. That is custom glue (unverified quality), and the CUDA JIT build needs nvcc 12.8+, while the system has 12.0.
- Use it only as a phase-2 quality upgrade, e.g. if PatchMatch leaves holes on glossy packs.

### 4.4 Depth-based fusion (monocular depth + poses + TSDF)
- Open3D TSDF is MIT and already installed. The problem is scale: per-frame monocular depth (DA-V2-Small relative, DA3METRIC-LARGE metric, Depth Pro metric) has a frame-to-frame scale/shift that must be aligned to the SfM sparse points, for example with a per-frame least-squares scale/shift fit against COLMAP 3D points.
- Without that alignment, the TSDF smears. With it, this is a decent **hole-filling fallback**, but it is less precise at edges than MVS (inference).
- The metric nets also give a **rough scale prior** (0.8-1.2x accuracy is unverified). The user has chosen fixture sizes for final scale anyway.

### 4.5 Texturing
- **First choice: COLMAP `mesh_texturer`.** It is BSD-licensed, built into the same tool and writes a single atlas.
  - The conda build has CGAL, so occlusion-aware view selection works.
  - The pip wheel skips occlusion testing. That gives wrong textures where shelf lips occlude products.
  - `texture_scale_factor` defaults to 1.0, meaning "native source-image resolution" [src: texture_mapping.h].
- Alternatives:
  - mvs-texturing (BSD-3): the same paper.
  - OpenMVS TextureMesh: AGPL.
  - Open3D colour-map optimisation: per-vertex colour only. **Not sufficient for the app**, which ignores per-vertex colours.

---

## 5. Recommended pipeline (per video; on-prem, commercial-safe)

Primary stack: **conda env with conda-forge `colmap` (CUDA 12.9 build) + PyAV + OpenCV + Open3D + usd-core.**
- The legal-cleanest variant is `pycolmap-cuda12` from pip, accepting no occlusion testing and no ALIKED/LightGlue/Delaunay.
- The richer variant is the conda-forge build, which links CGAL (GPL) and probably LSD (AGPL). The build script does not disable them; whether LSD is actually linked is unverified. That is fine for internal use, but needs a legal check before you redistribute binaries.

1. **Decode and rotate**
   - Decode with PyAV (`pip install av`). Apply `frame.rotation` (from the 90° display matrix) to get upright **1080x1920 portrait** frames.
   - Verify that the bundled FFmpeg decodes HEVC on one file first. If it does not, use conda-forge `ffmpeg`, whose CLI auto-rotates by default (unverified for this build).
   - Write lossless PNG, or JPEG at quality 95 or higher.
   - Do not resize below native size: texture sharpness drives cm-level measurement.
   - Strip GPS: work from decoded frames only.
2. **Select frames and handle blur**
   - Compute a sharpness score per frame, e.g. Laplacian variance on a centre crop.
   - In each window of about 0.2-0.33 s (6-10 frames at 30 fps), keep the sharpest frame. That gives about 3-5 fps.
   - Drop frames below a per-video percentile threshold, but keep enough frames for coverage (target at least 70-80% overlap between neighbours).
   - Optionally use scene-illuminance from mebx to flag exposure jumps.
   - Skip learned deblurring. It invents texture and hurts both matching and measurement (a judgement, not a verified claim).
3. **Poses (SfM)**
   - Run `feature_extractor` with a single shared camera (`--ImageReader.single_camera 1`) and the OPENCV or SIMPLE_RADIAL model. Use an initial focal from the 35 mm-equivalent field (about 26 mm, to be confirmed) and let BA refine it. Use SIFT, or with the conda build ALIKED plus LightGlue.
   - Run `sequential_matcher` with a small overlap window (about 10-20). Turn loop detection off, or set a large `loop_detection_min_index_distance`, against repetitive packs. Enable DEGENSAC (`TwoViewGeometry.use_degensac`) for the planar facade.
   - Run the incremental `mapper`, or `global_mapper` for speed. For blurry stretches, COLMAP 4.0 adds a "structure-less image registration fallback" for frames without 3-view overlap [src: 4.0.0 notes].
   - **Check that a single model registered most frames.** If it splits into several, re-run with more frames in that segment. With the global mapper, `multiple_models` reports the split components [src: 4.2.0 notes].
   - Fallback for poses: VGGT-1B-Commercial or MapAnything-apache in chunks of about 20-30 frames, exported to COLMAP format, with COLMAP BA / `point_triangulator` refinement.
4. **Dense geometry**
   - Run `image_undistorter` with `--max_image_size` about 1600-2000. Then `patch_match_stereo --PatchMatchStereo.geom_consistency true`, then `stereo_fusion` [src: colmap doc/cli.rst].
   - On 8 GB of VRAM, start at 1600 px and raise it if it fits. PatchMatch VRAM use for this size is unverified.
   - Optionally mask glossy or blown-out pixels.
5. **Meshing**
   - Use Poisson. Better: Delaunay first to reject outliers, then Poisson (`delaunay_mesher` needs CGAL, i.e. the conda build) [src: faq.rst].
   - Trim low-density Poisson faces, and crop to the aisle bounding box so no bubble surface is left behind.
   - Decimate with `mesh_simplifier --MeshSimplification.target_face_ratio` to about 0.3-1 M faces (GLB size, section 1).
6. **Texturing**
   - Run `colmap mesh_texturer --workspace_path dense --input_path mesh.ply --output_path dense/textured`, which writes `mesh.ply` with per-face UVs and `texture.png` [src: cli.rst, format.rst].
   - Keep `apply_color_correction` on. Convert the PNG atlas to JPEG (quality about 90-95). Very large atlases may exceed browser texture limits; split the atlas if needed (browser limits unverified).
7. **Gravity alignment (Y-up) and frame**. There is no IMU in the files (section 2), so estimate up from the scene:
   - (a) Initial up = mean of the camera "up" vectors: the negated image y axis of each upright portrait frame, in world coordinates. Handheld phones are held roughly upright.
   - (b) Refine with RANSAC planes. Fit the dominant **vertical shelf-front plane**; its normal must be perpendicular to up. Fit horizontal shelf boards and the floor (if captured); their normals are parallel to up. Price-strip lines are horizontal.
   - (c) With the conda build, `model_orientation_aligner` offers a Manhattan-world alignment via vanishing points (requires LSD) [src: cli.rst, faq.rst].
   - Then rotate so that +Y = up, +X = along the aisle (the pan direction) and +Z = shelf-front normal pointing towards the aisle. Put the origin on the floor (lowest dense points), or on the bottom shelf if there is no floor.
   - Make sure the triangle winding gives normals facing the camera side, so the floor normal faces up.
8. **Scale (consistent, up-to-scale for now)**
   - Each video is **one** connected SfM model, so it has a single global scale. Do not merge separately scaled chunks.
   - Store `scale_status: "unscaled"` and a reconstruction-unit factor in a sidecar JSON. Write `metersPerUnit = 1` (unscaled units).
   - Later, compute s from fixture sizes, such as shelf pitch or standard gondola widths (user decision), and either rewrite the points by s or set `metersPerUnit = s`. The app honours `metersPerUnit` everywhere (section 1).
   - Optional sanity prior: the DA3METRIC-LARGE (Apache) median metric depth against the SfM depth.
9. **Export to .usdz**, with `pxr` already in the venv:
   - Build `scan.usda` with `defaultPrim="World"`, `upAxis="Y"`, `metersPerUnit=1`. Under `/World`, add one `Mesh` per atlas page with points, faceVertexCounts and faceVertexIndices, and a `primvars:st` texCoord2f[] with **faceVarying** interpolation, matching the PLY per-face UVs.
   - Bind one `Material` per Mesh: `UsdPreviewSurface` with diffuseColor connected to a `UsdUVTexture` (`inputs:file=@./textures/aisle_N.jpg@`, st from `UsdPrimvarReader_float2` with varname "st"), roughness 1, metallic 0. Copy the structure of the working `../room_scan.usda`.
   - Package with `UsdUtils.CreateNewUsdzPackage`, which keeps the root layer first.
   - Validate offline with `python scan_prep.py prepare out.usdz --scans-root /tmp/x` and then `usd_to_gltf.py`. Run these from the scratchpad, not the app's `out/`. Check `orientation.flip_recommended` and `confidence`.
10. **QA**: re-project a few original frames onto the mesh with the SfM poses and compare (colour RMSE, edge alignment). Measure a known fixture dimension to confirm scale.

**Fallback pipeline**, for when COLMAP fails to register a video (blur or repetition), or MVS leaves big holes:
- Poses: VGGT-1B-Commercial or MapAnything-apache, chunked about 20-30 frames, then COLMAP BA.
- Dense: DA3-BASE (Apache) or DA-V2-Small depth, scale-aligned to the sparse points, then Open3D `ScalableTSDFVolume`, then marching-cubes mesh.
- Texture: COLMAP `mesh_texturer`, then the same USD export.
- The off-the-shelf alternative is Meshroom 2025.1 (MPL-2), if its CUDA 12 binary runs on sm_120 (unverified).

**Do not use for production** (licence): DUSt3R, MASt3R, Spann3R, CUT3R, Fast3R, Pi3/Pi3X weights, VGGT-Omega, original VGGT weights, Inria 3DGS, 2DGS, GOF, SuGaR, UniDepth, and the DA-V2 Base/Large and DA3 Large/Giant/Nested weights.

---

## 6. Capture guidelines for future store videos

From the COLMAP tutorial ("good texture", "similar illumination", "avoid specularities", "high visual overlap… each object seen in at least 3 images", "different viewpoints… not only rotating") [src: colmap doc/tutorial.rst], applied to aisles:
1. **Landscape or portrait, but never switch mid-clip.** Lock the lens to the 1x wide camera so the phone cannot switch lenses. Do not zoom.
2. **Use 4K at 30 fps if storage allows, or 1080p at 60 fps** to reduce blur. Use the native .MOV straight from the phone. Do not re-encode through messaging apps; IMG_5200 is a recompressed 720p file.
3. **Walk slowly and parallel to the shelf** at about 0.8-1.5 m, at roughly 1 bay per 5-8 s. Keep the phone steady; do not sweep it by rotating your wrist.
4. **Make two passes:** one at chest height looking straight at the shelf, and one higher or lower angled about 20-30° toward the top and bottom shelves. Add a short final pass that includes the **floor** and the **gondola ends** (fixtures for scale and orientation).
5. **Start and end on a non-repetitive feature**, such as the aisle-end sign or a promo display. Include some floor in every clip.
6. **Lighting**: avoid pointing at fridge glass and lights. Tilt slightly off-axis to avoid glare on bottles and foils. Keep exposure stable: tap-and-hold AE/AF lock.
7. **Turn off** Action mode, Cinematic mode and HDR video if possible. Whether standard stabilisation can be disabled in the stock camera is unverified. A third-party pro-video app may allow locked focus, exposure and stabilisation.
8. **One aisle face per clip**, at most 60 s. Split long aisles into overlapping clips, with about 1 bay of overlap.
9. **Privacy and residency**: avoid shoppers' faces (iOS writes face-detection tracks). Turn off location tagging for store captures, or strip GPS on ingest.
10. Optional for scale: put a known-size marker, such as a printed AprilTag-style board or a 1 m bar, on a shelf in each clip. The user has chosen fixture-based scale, but a marker makes QA easy.

---

## 7. Open items to verify by running (no installs were done)
- The conda-forge colmap `cuda_129` build runs PatchMatch on sm_120, and its PatchMatch VRAM use at 1600-2000 px fits in 8 GB.
- PyAV's bundled FFmpeg decodes these HEVC files and honours the rotation.
- The exact 35 mm-equivalent focal value (`exiftool -ee`).
- The contents of `live-photo-info` (undocumented).
- The orientation heuristic's result on aisle scans that contain only a little floor.
- Legal review: CGAL GPL and LSD AGPL in the conda COLMAP build; OpenMVS AGPL (if used); the Depth Pro licence (no patent grant); the VGGT-1B-Commercial terms (gated, AUP); the FFmpeg/HEVC codec terms.
