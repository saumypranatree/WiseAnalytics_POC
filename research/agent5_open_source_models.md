# Agent 5 — Open-source model / tool catalogue for the shelf-gap pipeline

Pipeline: phone video of aisle -> frames -> poses + dense 3D -> textured mesh -> 2D det/seg on views -> empty-slot (void) detection -> gap size in cm (scale from fixture sizes) -> JSON.

Research date: 2026-10-15. Constraints assumed: commercial production use, fully on-prem (India data residency, no cloud APIs), 8 GB RTX 5060 Laptop (Blackwell, sm_120), 23 GB RAM, Ubuntu 24.04.

## How this was verified

- Code licenses: read from the GitHub API (`/repos/{r}/license`, which returns the LICENSE file text) and README "License" sections, via `gh`.
- Weight licenses: read from the Hugging Face model API (`cardData.license`, tags, gating), and checked against README statements. **Where the HF tag and the README disagree, both are given and the stricter one is assumed.**
- Package and wheel facts: read from the PyPI JSON API and download.pytorch.org.
- "Params" are Hugging Face safetensors parameter totals, or the README's own table. "~X GB bf16" means **my weights-only estimate (params x 2 bytes)**. It is NOT a measured VRAM figure: activations come on top. Figures documented by a source are labelled "documented".
- Nothing was installed or run. "Runs on 8 GB sm_120?" is an engineering assessment, not a test result:
  - **Yes** = pure PyTorch, small enough.
  - **Likely** = needs a custom CUDA extension compiled for sm_120 (CUDA 12.8+ toolkit, `TORCH_CUDA_ARCH_LIST="12.0"`), or tight on memory.
  - **No** = doesn't fit, or cloud-only.
- PyTorch Blackwell support (verified on download.pytorch.org): cu128 wheels exist for torch 2.9.1 to 2.11.0, cu129 for up to 2.13.0, and cu130 for up to 2.14.1 (the latest on PyPI). Use a cu128+ build.

Legend: **NC** = not allowed for commercial use. **COPYLEFT** = commercial use allowed but with GPL/AGPL obligations. These obligations matter if the software is distributed, or (AGPL) served over a network. **GATED** = HF access request or approval needed.

---

## 1. Video decoding / frame extraction

| Name | What it does | Code license | Weights | Commercial? | Size/VRAM | 8 GB sm_120? | Install | Link |
|---|---|---|---|---|---|---|---|---|
| FFmpeg (CLI/libs) | Decode/encode, `select`/`mpdecimate` filters for frame sampling | LGPL-2.1+ core; GPL if built with GPL parts (e.g. libx265) | n/a | Yes (LGPL build). HEVC patent licensing is a separate legal question | CPU; NVDEC optional | Yes | apt `ffmpeg` / static build | https://github.com/FFmpeg/FFmpeg , https://ffmpeg.org/legal.html |
| PyAV 19.0.1 | Pythonic FFmpeg bindings | BSD-3-Clause | n/a | Yes | CPU | Yes | `pip install av`. README: "Binary wheels ... with FFmpeg bundled". Python >= 3.12 required | https://github.com/PyAV-Org/PyAV |
| torchcodec 0.17.0 | Meta video decoder to tensors, CPU and CUDA | BSD-style (Meta; PyPI field empty, unverified) | n/a | Yes (unverified) | GPU decode optional | Yes | `pip install torchcodec`. README: "uses the version of FFmpeg you already have installed" (not bundled) | https://github.com/meta-pytorch/torchcodec |
| decord 0.6.0 | Fast random-access video reader | Apache-2.0 | n/a | Yes | CPU/GPU | Not recommended: last PyPI release 2021-06, wheels only cp36 to cp38 | `pip install decord` (old) / `eva-decord` fork (cp38 to cp311, 2023) | https://github.com/dmlc/decord |
| OpenCV `VideoCapture` (opencv-python 5.0.0.93) | Decode + Laplacian blur metric in one lib | Apache-2.0 | n/a | Yes | CPU | Yes | `pip install opencv-python` | https://pypi.org/project/opencv-python/ |
| imageio-ffmpeg 0.6.0 | Ships an ffmpeg binary for Python | BSD-2-Clause (bundled ffmpeg binary carries its own license) | n/a | Check the bundled binary build | CPU | Yes | pip | https://pypi.org/project/imageio-ffmpeg/ |

HEVC note: iPhone/Android videos are often HEVC. FFmpeg's built-in `hevc` decoder is part of libavcodec. I did not confirm that the PyAV wheel enables it. Check with `python -c "import av; print('hevc' in av.codecs_available)"`. Safe fallback: `ffmpeg -i in.mov -vf fps=4 -q:v 2 f_%05d.jpg`.

## 2. Keyframe selection / blur detection / deblurring

| Name | What it does | Code license | Weights | Commercial? | Size/VRAM | 8 GB? | Install | Link |
|---|---|---|---|---|---|---|---|---|
| Variance-of-Laplacian / Tenengrad (OpenCV) | Classic sharpness score; pick sharpest frame per window | Apache-2.0 (OpenCV) | n/a | Yes | CPU | Yes | pip opencv | https://pypi.org/project/opencv-python/ |
| FFmpeg `mpdecimate` / `select='gt(scene,x)'` | Drop near-duplicate frames | LGPL | n/a | Yes | CPU | Yes | ffmpeg | https://ffmpeg.org/ |
| RAFT / SEA-RAFT | Optical flow; select frames by parallax / baseline | BSD-3-Clause (both) | weights: not stated separately | Yes (code) | not stated | Yes | source | https://github.com/princeton-vl/RAFT , https://github.com/princeton-vl/SEA-RAFT |
| NAFNet | Single-image deblur (GoPro) / denoise | MIT | not stated separately (README links) | Code yes; training data license not checked | not stated | Likely | source (BasicSR-based) | https://github.com/megvii-research/NAFNet |
| Restormer | Image restoration incl. motion deblur | MIT | not stated separately | Code yes | not stated | Likely (large images: tile) | source | https://github.com/swz30/Restormer |
| FFTformer | Transformer deblurring | MIT | not stated | Code yes | not stated | Likely | source | https://github.com/kkkls/FFTformer |
| DeblurGAN-v2 | GAN deblur, light | BSD-3-Clause | not stated | Code yes | not stated | Yes | source | https://github.com/VITA-Group/DeblurGANv2 |
| BasicVSR++ (also in MMagic) | Video restoration incl. video deblur | Apache-2.0 | not stated separately | Code yes | not stated | Likely | source / mmagic | https://github.com/ckkelvinchan/BasicVSR_PlusPlus , https://github.com/open-mmlab/mmagic |
| VRT / RVRT | Video restoration transformer (deblur) | **NC** (VRT: CC BY-NC 4.0) | **NC** | **No** | not stated | Unlikely at full res | source | https://github.com/JingyunLiang/VRT |

Recommendation: frame selection (sharpness + parallax) is better value than deblurring. Deblurred pixels can corrupt SfM/MVS.

## 3. Structure-from-Motion / MVS / texturing (classical)

| Name | What it does | Code license | Weights | Commercial? | Size/VRAM | 8 GB sm_120? | Install | Link |
|---|---|---|---|---|---|---|---|---|
| COLMAP (4.x) | Incremental + **global SfM (GLOMAP now merged in as "global" mapper)**, PatchMatch MVS, Poisson/Delaunay meshing | BSD (new BSD; dependencies may affect the resulting license) | n/a | Yes | MVS needs CUDA | Likely (build with CUDA 12.8, `-DCMAKE_CUDA_ARCHITECTURES=native`) | apt (no CUDA) / source / Docker | https://github.com/colmap/colmap |
| pycolmap 4.2.1 / pycolmap-cuda12 4.2.1 | Python bindings. The README says "CUDA-enabled wheels are available at pycolmap-cuda12" | BSD-3-Clause | n/a | Yes | — | Yes for SfM. Whether the cuda12 wheel includes sm_120 is **unverified** | `pip install pycolmap` / `pycolmap-cuda12` | https://pypi.org/project/pycolmap/ |
| GLOMAP (standalone) | Global SfM. **Deprecated**: the repo says it is "fully migrated to COLMAP" | BSD-3-Clause | n/a | Yes | CPU | Yes | use COLMAP | https://github.com/colmap/glomap |
| hloc + LightGlue | Learned feature matching -> COLMAP | Apache-2.0 (both) | LightGlue weights for SuperPoint/DISK/ALIKED/SIFT | Yes **only if you avoid SuperPoint** | small | Yes | pip/source | https://github.com/cvg/Hierarchical-Localization , https://github.com/cvg/LightGlue |
| SuperPoint (Magic Leap) | Keypoint detector, common default in hloc | **NC**: "ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY" | **NC** | **No** | small | Yes | — | https://github.com/magicleap/SuperPointPretrainedNetwork |
| ALIKED / DISK | Commercial-friendly keypoints for LightGlue | BSD-3-Clause / Apache-2.0 | not stated separately | Yes (code) | small | Yes | pip | https://github.com/Shiaoming/ALIKED , https://github.com/cvlab-epfl/disk |
| RoMa / DeDoDe | Dense / learned matching | MIT | not stated | Code yes | RoMa heavy (not stated) | Likely | pip/source | https://github.com/parskatt/RoMa , https://github.com/Parskatt/DeDoDe |
| OpenMVS | Densify, mesh, refine, **TextureMesh** | **COPYLEFT AGPL-3.0** | n/a | Yes with AGPL obligations | CPU (+optional CUDA) | Yes | source | https://github.com/cdcseacave/openMVS |
| AliceVision / Meshroom | Full photogrammetry incl. texturing | MPL-2.0 | n/a | Yes (MPL file-level copyleft) | CUDA for depth maps | Likely | binaries/source | https://github.com/alicevision/AliceVision |
| mvs-texturing (texrecon) | Texture a mesh from posed images | BSD-3-Clause | n/a | Yes | CPU | Yes | source | https://github.com/nmoehrle/mvs-texturing |
| Open3D 0.20.0 | TSDF fusion, Poisson, mesh ops, scale/plane fitting | MIT | n/a | Yes | CPU/CUDA | Yes | `pip install open3d` | https://pypi.org/project/open3d/ |
| PyMeshLab | Mesh cleanup/decimation | **COPYLEFT GPL-3** | n/a | Yes with GPL obligations | CPU | Yes | pip | https://pypi.org/project/pymeshlab/ |
| xatlas | UV unwrapping for texture atlases | MIT | n/a | Yes | CPU | Yes | pip | https://pypi.org/project/xatlas/ |

## 4. Learned feed-forward 3D reconstruction

| Name | What it does | Code license | Weights license | Commercial? | Params / VRAM | 8 GB sm_120? | Install | Link |
|---|---|---|---|---|---|---|---|---|
| **VGGT-1B-Commercial** | Feed-forward poses, depth, point maps, tracks; COLMAP export script | VGGT License (v1, 2025-07-29). README: "permit commercial use (excluding military applications)" | VGGT License, **GATED (manual approval)** | **Yes (this checkpoint only)** | 1.26B (~2.5 GB bf16 weights). VRAM per frame count not stated in current README | Likely, with a small number of frames per chunk | source | https://github.com/facebookresearch/vggt , https://huggingface.co/facebook/VGGT-1B-Commercial |
| VGGT-1B (original) | Same | VGGT License | **NC: CC BY-NC 4.0** | **No** | 1.26B | Likely | source | https://huggingface.co/facebook/VGGT-1B |
| VGGT-Omega (2026-05, CVPR 2026) | VGGT successor | **NC: FAIR Noncommercial Research License** | **NC, GATED** | **No** | Documented peak memory: 1 frame 6.02 GB, 10 frames 6.67 GB, 25 frames 7.80 GB, 50 frames 9.66 GB | Yes up to about 20 to 25 frames | source | https://github.com/facebookresearch/vggt-omega , https://vggt-omega.github.io/ |
| **MapAnything (apache variant)** | Universal metric 3D recon from N views; optional intrinsics/pose/depth inputs | Apache-2.0 | `facebook/map-anything-apache`: **Apache-2.0** (`facebook/map-anything` is **NC**) | **Yes (apache ckpt)** | 1.23B. Has `memory_efficient_inference` with minibatch 1. Exact GB per view not stated in text (plot only) | Likely | `pip install -e .` from source | https://github.com/facebookresearch/map-anything , https://huggingface.co/facebook/map-anything-apache |
| **Depth Anything 3 (DA3-SMALL/BASE)** | Any-view depth + pose estimation | Apache-2.0 | SMALL, BASE, METRIC-LARGE, MONO-LARGE: **Apache-2.0**. LARGE, GIANT, NESTED (incl. -1.1): **NC (CC BY-NC 4.0)** | Yes for Apache variants | README: Small 0.08B, Base 0.12B. DA3-Streaming: "less than 12GB GPU memory" for long video | Yes (Small/Base). Streaming at 8 GB is unverified | `pip install -e .` | https://github.com/ByteDance-Seed/Depth-Anything-3 |
| π³ / Pi3 and Pi3X | Permutation-equivariant feed-forward recon; Pi3X adds pose/intrinsics/depth conditioning and approximate metric scale | BSD-3-Clause | README: **NC: CC BY-NC 4.0, "Strictly Non-Commercial"** (HF tag for Pi3 says bsd-2-clause. This conflicts with the README; assume NC) | **No** | Pi3 0.96B, Pi3X 1.36B | Likely | source | https://github.com/yyfz/Pi3 , https://huggingface.co/yyfz233/Pi3X |
| DUSt3R | Pairwise point maps + global alignment | **NC: CC BY-NC-SA 4.0** | **NC** (plus training-dataset licenses) | **No** | 571M | Likely (few views) | source | https://github.com/naver/dust3r |
| MASt3R | DUSt3R + matching head, metric | **NC: CC BY-NC-SA 4.0** | **NC** | **No** | 689M | Likely | source | https://github.com/naver/mast3r |
| MASt3R-SLAM | Real-time SLAM on MASt3R | **NC: CC BY-NC-SA 4.0** | **NC** | **No** | not stated | Likely | source | https://github.com/rmurai0610/MASt3R-SLAM |
| MUSt3R | Multi-view DUSt3R, online | **NC: NAVER Non-Commercial License** | **NC** | **No** | not stated | Likely | source | https://github.com/naver/must3r |
| Fast3R | 1000+ images in one pass | **NC: FAIR Noncommercial Research License** | **NC** | **No** | not stated | Unlikely at scale | source | https://github.com/facebookresearch/fast3r |
| CUT3R | Continuous/streaming recon | **NC: CC BY-NC-SA 4.0** | **NC** | **No** | not stated | Likely | source | https://github.com/CUT3R/CUT3R |
| Spann3R | Spatial-memory incremental recon | **NC: CC BY-NC-SA 4.0** | **NC** | **No** | not stated | Likely | source | https://github.com/HengyiWang/spann3r |
| VGGSfM | Differentiable SfM | **NC: CC BY-NC 4.0** | **NC** | **No** | not stated | Likely | source | https://github.com/facebookresearch/vggsfm |
| MonST3R | Dynamic-scene DUSt3R | **NC** (per README/LICENSE, CC BY-NC-SA style; derived from DUSt3R) | **NC** | **No** | not stated | Likely | source | https://github.com/Junyi42/monst3r |

## 5. Gaussian splatting / neural surface reconstruction with mesh export

| Name | What it does | Code license | Weights | Commercial? | VRAM | 8 GB sm_120? | Install | Link |
|---|---|---|---|---|---|---|---|---|
| **gsplat 1.5.3** | CUDA 3DGS rasterizer + trainer. Includes **2DGS** trainer (`simple_trainer_2dgs.py`), MCMC, 3DGUT. "up to 4x less GPU memory" than the official 3DGS | Apache-2.0 | n/a (per-scene) | **Yes** | Documented only relatively ("4x less") | Likely: README says "PyTorch 2.7+ ... CUDA 12.8 ... build compatibility". JIT build on first run | `pip install gsplat` (JIT) or prebuilt wheels at docs.gsplat.studio/whl | https://github.com/nerfstudio-project/gsplat |
| nerfstudio 1.1.5 | Framework (splatfacto, nerfacto); exports point clouds/meshes (TSDF/Poisson) | Apache-2.0 | n/a | Yes | not stated | Likely | pip. Note: last PyPI release 2024-11 (low maintenance) | https://github.com/nerfstudio-project/nerfstudio |
| SDFStudio | Neural SDF surfaces (NeuS, MonoSDF, BakedSDF) -> mesh | Apache-2.0 | n/a | Yes | not stated | Uncertain (old deps, last push 2024-07) | source | https://github.com/autonomousvision/sdfstudio |
| Brush | Cross-platform splatting trainer (Rust/WebGPU) | Apache-2.0 | n/a | Yes | not stated | Yes (no CUDA build) | binaries | https://github.com/ArthurBrussee/brush |
| OpenSplat | C++ splatting trainer | **COPYLEFT AGPL-3.0** | n/a | Yes with AGPL | not stated | Likely | source | https://github.com/pierotofy/OpenSplat |
| 2DGS (official) | Surfel splatting, TSDF mesh export | **NC: Inria/MPII Gaussian-Splatting License, "CANNOT USE ... FOR COMMERCIAL PURPOSES"** | n/a | **No** | not stated | Likely | source | https://github.com/hbb1/2d-gaussian-splatting |
| Gaussian Opacity Fields (GOF) | Compact unbounded meshes from 3DGS | **NC: Inria license** | n/a | **No** | not stated | Likely | source | https://github.com/autonomousvision/gaussian-opacity-fields |
| SuGaR | Surface-aligned GS -> mesh | **NC: Inria license** | n/a | **No** | not stated | Likely | source | https://github.com/Anttwo/SuGaR |
| MILo | Mesh-in-the-loop GS | **NC: Inria license** | n/a | **No** | not stated | Likely | source | https://github.com/Anttwo/MILo |
| PGSR | Planar-based GS surface recon (good for flat shelves) | **NC**: "For commercial uses ... please send email" (ZJU) | n/a | **No (without separate license)** | not stated | Likely | source | https://github.com/zju3dv/PGSR |

Commercial-safe meshing route: posed frames -> gsplat (3DGS or its own 2DGS trainer) -> render depth -> Open3D TSDF -> mesh. Or COLMAP MVS -> Poisson -> mvs-texturing.

## 6. Monocular depth

| Name | What it does | Code license | Weights license | Commercial? | Params | 8 GB? | Install | Link |
|---|---|---|---|---|---|---|---|---|
| Depth Anything V2 Small | Relative depth | Apache-2.0 | **Apache-2.0** | Yes | small (~25M, not re-checked; HF total not exposed) | Yes | pip/transformers | https://github.com/DepthAnything/Depth-Anything-V2 |
| Depth Anything V2 Base/Large | Relative depth | Apache-2.0 | **NC: CC BY-NC 4.0** | **No** | — | Yes | — | https://huggingface.co/depth-anything/Depth-Anything-V2-Large |
| Depth Anything V2 Metric-Indoor-Large | Metric indoor depth | Apache-2.0 | not stated on HF card. Large-based, so assume **NC** (unverified) | Treat as No | 335M | Yes | transformers | https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf |
| **Depth Anything 3 METRIC-LARGE / MONO-LARGE** | Monocular metric depth (+ sky seg) | Apache-2.0 | **Apache-2.0** | **Yes** | 0.35B (~0.7 GB bf16) | Yes | source | https://huggingface.co/depth-anything/DA3METRIC-LARGE |
| **MoGe / MoGe-2 / MoGe-3** | Monocular metric point map, depth, normals, FOV (MoGe-3 paper arXiv 2607.17967) | MIT (DINOv2 parts Apache-2.0) | **MIT** (HF: moge-vitl, moge-2-vitl-normal, moge-3-vitl) | **Yes** | ViT-L/B/S variants | Yes | `pip install git+https://github.com/microsoft/MoGe.git`. README shows cu128 install | https://github.com/microsoft/MoGe |
| Metric3D v2 | Metric depth + normals; torch.hub and ONNX | BSD-2-Clause | not stated on HF card (JUGGHM/Metric3D) | Code yes; weights **unverified** | ViT-S/L/g2 | Yes (S/L) | torch.hub | https://github.com/YvanYin/Metric3D |
| UniDepth v2 | Metric depth + camera intrinsics | **NC: CC BY-NC 4.0** | not stated on HF card; covered by repo NC | **No** | ViT-S/B/L (vitl14 354M) | Yes | source | https://github.com/lpiccinelli-eth/UniDepth |
| Depth Pro (Apple) | Sharp metric depth + focal | GitHub LICENSE is the Apple sample-code license. README says weights use "the LICENSE terms" | HF card: **apple-amlr: "sole purpose of scientific research"** | **Treat as No** (conflicting sources; HF AMLR is research-only) | not stated | Likely | source | https://github.com/apple/ml-depth-pro , https://huggingface.co/apple/DepthPro |

## 7. Open-vocabulary detection

| Name | What it does | Code license | Weights | Commercial? | Params | 8 GB sm_120? | Install | Link |
|---|---|---|---|---|---|---|---|---|
| **Grounding DINO (T/B)** | Text-prompted boxes | Apache-2.0 | **Apache-2.0** (HF grounding-dino-tiny/base) | Yes | tiny 172M, base 233M | Yes via `transformers` (avoids the custom CUDA op of the original repo) | transformers / source | https://github.com/IDEA-Research/GroundingDINO |
| **MM-Grounding-DINO** | Stronger open re-train of Grounding DINO (MMDetection) | Apache-2.0 | Apache-2.0 (HF openmmlab-community/mm_grounding_dino_large_all) | Yes | large 344M | Yes | transformers / mmdet | https://github.com/open-mmlab/mmdetection |
| **LLMDet** | Grounding-DINO-style detector co-trained with an LLM | Apache-2.0 | Apache-2.0 (iSEE-Laboratory/llmdet_large) | Yes | large 344M | Yes | transformers / source | https://github.com/iSEE-Laboratory/LLMDet |
| **OWLv2** | Open-vocab + image-query (one-shot) detection | Apache-2.0 | Apache-2.0 | Yes | B/16 155M, L/14 438M | Yes | transformers | https://huggingface.co/google/owlv2-large-patch14-ensemble |
| **OmDet-Turbo** | Real-time open-vocab detection | Apache-2.0 | Apache-2.0 | Yes | 115M | Yes | transformers | https://github.com/om-ai-lab/OmDet |
| **Florence-2 (B/L)** | Captioning, OD, phrase grounding, OCR | MIT | **MIT** | Yes | B 232M, L 777M | Yes | transformers | https://huggingface.co/microsoft/Florence-2-large |
| YOLO-World | Real-time open-vocab YOLO | **COPYLEFT GPL-3.0** ("supported for commercial usage"; commercial license on request) | GPL-3.0 | Yes with GPL obligations | not stated | Yes | source / ultralytics | https://github.com/AILab-CVC/YOLO-World |
| YOLOE | Open-prompt YOLO detection + seg | **COPYLEFT AGPL-3.0** | AGPL-3.0 (jameslahm/yoloe-v8l-seg) | Yes with AGPL (or Ultralytics enterprise license) | v8l-seg 53.5M | Yes | ultralytics | https://github.com/THU-MIG/yoloe |
| Rex-Omni | 3B MLLM detector (next-token boxes), strong on dense objects | **IDEA License 1.0**; built on Qwen2.5-VL-3B (**Qwen RESEARCH license**) | same | **No / unverified**: inherits the Qwen research (non-commercial) terms | 3B (~7 GB bf16) | Tight; needs 4-bit | source | https://github.com/IDEA-Research/Rex-Omni |
| Grounding DINO 1.5 / 1.6 Pro, DINO-X | SOTA open-set detection | API SDK Apache-2.0 | **Weights not released: cloud API only** (token from deepdataspace; paid quota) | **Excluded: violates on-prem/no-cloud** | — | No (cloud) | `dds-cloudapi-sdk` | https://github.com/IDEA-Research/Grounding-DINO-1.5-API , https://github.com/IDEA-Research/DINO-X-API |

Closed-set detectors useful for fine-tuning on shelf data:
- RF-DETR: Apache-2.0 core models; **XL/2XL and Atto/Femto/Pico are PML 1.0** (Roboflow license); has instance segmentation. https://github.com/roboflow/rf-detr
- D-FINE (Apache-2.0): https://github.com/Peterande/D-FINE
- RT-DETR (Apache-2.0): https://github.com/lyuwenyu/RT-DETR
- Ultralytics YOLO (**AGPL-3.0**): https://github.com/ultralytics/ultralytics
- DEIM (custom company license, not checked in detail): https://github.com/ShihuaHuang95/DEIM

## 8. Segmentation

| Name | What it does | Code license | Weights | Commercial? | Params | 8 GB sm_120? | Install | Link |
|---|---|---|---|---|---|---|---|---|
| SAM (v1) | Promptable masks | Apache-2.0 | Apache-2.0 | Yes | ViT-H ~636M (not re-checked) | Yes | pip | https://github.com/facebookresearch/segment-anything |
| **SAM 2 / SAM 2.1** | Image + video masks with memory (track masks across frames) | Apache-2.0 | **Apache-2.0** (HF sam2.1-hiera-*) | Yes | large 224M, small 46M | Yes | `pip install sam2` (sdist, builds an optional CUDA ext) / source | https://github.com/facebookresearch/sam2 |
| **SAM 3 / SAM 3.1** | **Concept (text/exemplar) promptable segmentation + tracking**. SAM 3.1 (2026-03-27) adds Object Multiplex | SAM License (2025-11-19) | SAM License, **GATED (manual)** | **Yes**: broad grant; prohibits ITAR/military/sanctioned uses; no NC clause found | sam3 860M (~1.7 GB bf16) | Likely: README needs Python 3.12+, PyTorch 2.7+, CUDA 12.6+; example uses `torch==2.10.0 ... cu128` | source | https://github.com/facebookresearch/sam3 , https://huggingface.co/facebook/sam3.1 |
| EfficientSAM | Light SAM | Apache-2.0 | Apache-2.0 | Yes | not stated | Yes | source | https://github.com/yformer/EfficientSAM |
| MobileSAM | Very light SAM | Apache-2.0 | MIT (HF dhkim2810/MobileSAM) | Yes | ~10M (not re-checked) | Yes | source | https://github.com/ChaoningZhang/MobileSAM |
| Grounded-SAM-2 | Glue: Grounding DINO / Florence-2 + SAM 2 tracking | Apache-2.0 | per component | Yes (if components are) | — | Yes | source | https://github.com/IDEA-Research/Grounded-SAM-2 |

## 9. Retail-specific (dense products, voids, price tags, planograms)

There are **no mature, permissively licensed, pretrained "empty-shelf/void" models** from a reputable source. The ecosystem is mostly student/hobby repos with no license (no license = all rights reserved = not usable). Plan to fine-tune your own detector on commercially usable data, or derive voids geometrically from product boxes + shelf lines + 3D depth.

| Name | Type | License | Commercial? | Notes | Link |
|---|---|---|---|---|---|
| SKU-110K | Dataset, 11.7k dense shelf images | README: "solely for academic and non-commercial purposes" | **NC** | Gold standard for dense product boxes. **Any model trained on it inherits the restriction** (legal view; HF models like `is36e/detr-resnet-50-sku110k` or `benjamintli/dfine-xl_sku110k` are tagged Apache-2.0, but the data is NC) | https://github.com/eg4000/SKU110K_CVPR19 |
| **SKUs_on_shelves_PL (ShelfWise)** | Dataset, ~27k shelf images, ~2M boxes, ~8k SKUs, COCO, Poland | **CC BY-4.0** | **Yes (attribution)** | Best verified commercially usable dense-product dataset found | https://huggingface.co/datasets/shelfwise-by-form/SKUs_on_shelves_PL |
| Unitail (Det + OCR) | Dataset: quadrilateral products (1.8M), price/text OCR | "academic purposes only". Unitail-Det no longer public | **NC** | ECCV 2022 | https://unitedretail.github.io/ |
| Roboflow Universe empty-shelf sets ("Empty Shelves" hostview, "retail-shelf-object-detection" w/ `empty` class, "CE+S", "Empty Spaces Detection in Shelf Data") | Small datasets | Search snippets say CC BY 4.0 / Public Domain. **Unverified** (pages returned HTTP 403 to my fetcher) | Probably yes; verify each page | Small and noisy, but give a starting `empty` class | https://universe.roboflow.com/hostview/empty-shelves-pelqt-rft4u , https://universe.roboflow.com/retailshelfobjectdetection/retail-shelf-object-detection |
| Retail Product Checkout (RPC) mirror | Checkout (not shelf) dataset | CC BY-NC-SA 2.0 (HF tag) | **NC** | Not shelf imagery | https://huggingface.co/datasets/benjamintli/retail-product-checkout |
| Alijanloo/Retail-Shelf-Monitoring | OSS system: OOS + misplacement, planogram grid, SORT, ONNX/OpenVINO | MIT (code) | Code yes. Its detector is "YOLOv12-M fine-tuned on SKU-110k" (data NC; YOLO implementation likely AGPL) | Good reference architecture (grid mapping + temporal consensus) | https://github.com/Alijanloo/Retail-Shelf-Monitoring |
| laitalaj/cvpce | Planogram compliance (thesis code, 2021) | MIT | Yes | Old, reference only | https://github.com/laitalaj/cvpce |
| Moh-Tayyab/shelf-void-detection | Void detection repo (tiny, 2026) | MIT | Yes | Unreviewed quality, 1 star | https://github.com/Moh-Tayyab/shelf-void-detection |
| Grocery Store Dataset | Fine-grained grocery item classification (hand-held photos) | MIT | Yes | For SKU recognition, not voids | https://github.com/marcusklasson/GroceryStoreDataset |

Price-tag detection: no reputable open model was found. Options: (a) open-vocab detection ("price tag", "shelf label") with OWLv2 / Grounding DINO / Florence-2, or SAM 3 concept prompts; (b) fine-tune RF-DETR / D-FINE on your own labels. Shelf-edge/price-rail strips are also the most reliable **scale reference**: standard gondola shelf depth/height and label-strip height.

## 10. Local vision-language models (verification step)

bf16 weight sizes below are params x 2 bytes, an estimate. The 8 GB fit includes room for the KV cache and vision tokens.

| Name | Code / weights license | Commercial? | Params (HF) | 8 GB? | Install | Link |
|---|---|---|---|---|---|---|
| **Qwen3.5-2B / 4B / 9B** (2026-02) | Apache-2.0 | Yes | 2.27B / 4.66B / 9.65B | 2B: yes (~4.5 GB). 4B: tight in bf16, yes in 8-bit/4-bit. 9B: 4-bit only | transformers / vLLM / llama.cpp | https://huggingface.co/Qwen/Qwen3.5-4B |
| Qwen3-VL-2B / 4B / 8B-Instruct | Apache-2.0 | Yes | 2.13B / 4.44B / 8.77B | 2B yes; 4B tight; 8B 4-bit | transformers | https://github.com/QwenLM/Qwen3-VL |
| Qwen2.5-VL-7B-Instruct | Apache-2.0 | Yes | 8.29B | 4-bit only | transformers | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct |
| Qwen2.5-VL-3B-Instruct | **NC: qwen-research license** | **No** | 3.75B | Yes | — | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct |
| Gemma 4 E2B-it / E4B-it (2026) | HF tag Apache-2.0 (license link: gemma_4_license) | Yes (verify the linked terms) | 5.12B / 8.00B total (HF) | E2B likely; E4B quantized (official QAT q4_0 GGUF exists) | transformers / llama.cpp | https://huggingface.co/google/gemma-4-E4B-it |
| PaliGemma 2 (3B-448) / Gemma 3 4B | Gemma license (commercial allowed under the Gemma Terms and Prohibited Use Policy), **GATED** | Yes, with terms | 3.03B / 4.30B | Yes | transformers | https://huggingface.co/google/paligemma2-3b-pt-448 |
| Molmo2-4B / 8B | Apache-2.0 | Yes | 4.85B / — | 4B quantized; pointing/counting is useful for "count facings" | transformers | https://huggingface.co/allenai/Molmo2-4B |
| InternVL3.5-2B / 4B / 8B | Apache-2.0 (HF) | Yes | 8B: 8.53B | small: yes | transformers | https://huggingface.co/OpenGVLab/InternVL3_5-8B |
| Microsoft Mage-VL | Apache-2.0 | Yes | 4.74B | quantized | transformers | https://huggingface.co/microsoft/Mage-VL |
| Phi-4-multimodal-instruct | MIT | Yes | 5.57B | quantized | transformers | https://huggingface.co/microsoft/Phi-4-multimodal-instruct |
| Florence-2 | MIT | Yes | 0.23B / 0.78B | Yes | transformers | https://huggingface.co/microsoft/Florence-2-large |
| SmolVLM2-2.2B | Apache-2.0 | Yes | 2.25B | Yes | transformers | https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct |
| moondream2 (2025-04-14) | Apache-2.0 | Yes | 1.93B | Yes | transformers | https://huggingface.co/vikhyatk/moondream2 |
| moondream3 / 3.1 (9B-A2B) | **Moondream Model License 1.0** (custom; terms not reviewed) | **Unverified** | 9.27B | 4-bit | — | https://huggingface.co/moondream/moondream3.1-9B-A2B |

---

## Licensing traps (summary)

**Non-commercial, do not ship:**
- DUSt3R, MASt3R, MASt3R-SLAM, MUSt3R, CUT3R, Spann3R, Fast3R, VGGSfM, MonST3R, VGGT-Omega
- Pi3/Pi3X weights
- VGGT-1B original checkpoint
- `facebook/map-anything` (non-apache checkpoint)
- DA3 LARGE/GIANT/NESTED; DA-V2 Base/Large
- UniDepth; Depth Pro (AMLR)
- 2DGS, GOF, SuGaR, MILo (Inria); PGSR
- SuperPoint
- VRT
- Qwen2.5-VL-3B; Rex-Omni
- SKU-110K and Unitail data, and anything trained on them

**Copyleft:**
- AGPL: OpenMVS, Ultralytics/YOLOE, OpenSplat
- GPL: YOLO-World, PyMeshLab
- MPL: AliceVision

**Cloud-only (violates on-prem):** Grounding DINO 1.5/1.6 Pro, DINO-X.

**Gated but commercial-OK:** VGGT-1B-Commercial, SAM 3/3.1, PaliGemma/Gemma 3.

## Recommendations per stage

| Stage | Best permissive (commercial-safe) stack | Best quality regardless of license |
|---|---|---|
| 1 Decode | FFmpeg CLI (LGPL build) or PyAV wheels | same |
| 2 Keyframes/deblur | Laplacian sharpness + SEA-RAFT parallax; NAFNet/Restormer (MIT) only if needed | + VRT video deblur (NC) |
| 3 SfM/MVS | COLMAP 4.x global mapper (ex-GLOMAP) + ALIKED/DISK + LightGlue; COLMAP MVS / Open3D TSDF; mvs-texturing for texture | COLMAP + SuperPoint+LightGlue; OpenMVS (AGPL) densify + TextureMesh |
| 4 Learned 3D | MapAnything-apache, or VGGT-1B-Commercial (gated), chunked, then COLMAP BA; DA3-BASE for poses+depth | VGGT-Omega / Pi3X / MASt3R-SLAM (all NC) |
| 5 Splat/mesh | gsplat (3DGS or built-in 2DGS) -> rendered depth -> Open3D TSDF mesh | 2DGS / GOF / PGSR / MILo (NC) |
| 6 Mono depth | MoGe-2/3 (MIT) or DA3METRIC-LARGE (Apache) | + Depth Pro / UniDepth v2 (NC) for comparison |
| 7 Open-vocab det | Grounding DINO / MM-Grounding-DINO / LLMDet via transformers, OWLv2 for image-query, Florence-2; then fine-tune RF-DETR (Apache tiers) or D-FINE on own shelf data | Grounding DINO 1.6 Pro / DINO-X (cloud, excluded), Rex-Omni (NC) |
| 8 Segmentation | SAM 3/3.1 (SAM License, gated; concept prompts like "empty shelf space") or SAM 2.1 (Apache) for video tracking | SAM 3.1 |
| 9 Retail | Fine-tune on SKUs_on_shelves_PL (CC BY 4.0) + own labelled aisle frames; voids = gaps between product boxes along detected shelf rows, checked in 3D depth | SKU-110K-trained detector (NC) + Unitail (academic) |
| 10 VLM verify | Qwen3.5-4B or Qwen3-VL-4B (Apache), 4/8-bit on 8 GB; Florence-2 for cheap grounding | Qwen3.5-9B / Qwen3-VL-8B 4-bit (still Apache), larger Qwen3.6/3.8 need more VRAM |

8 GB budget note: run stages sequentially and unload models between them. Feed-forward 3D models must be chunked: about 20 to 25 frames per pass for VGGT-class 1B models, per the documented VGGT-Omega table. Every custom CUDA extension (gsplat, SAM 2 ext, COLMAP CUDA, 3DGS rasterizers, original GroundingDINO op) must be compiled against CUDA 12.8+ for sm_120. Prefer `transformers` implementations, which avoid custom ops.

## Sources

Code repositories (GitHub):
- Video and frames: [FFmpeg](https://github.com/FFmpeg/FFmpeg) ([legal](https://ffmpeg.org/legal.html)), [PyAV](https://github.com/PyAV-Org/PyAV), [torchcodec](https://github.com/meta-pytorch/torchcodec), [decord](https://github.com/dmlc/decord)
- SfM, matching and texturing: [COLMAP](https://github.com/colmap/colmap), [GLOMAP](https://github.com/colmap/glomap), [OpenMVS](https://github.com/cdcseacave/openMVS), [AliceVision](https://github.com/alicevision/AliceVision), [mvs-texturing](https://github.com/nmoehrle/mvs-texturing), [hloc](https://github.com/cvg/Hierarchical-Localization), [LightGlue](https://github.com/cvg/LightGlue), [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork), [ALIKED](https://github.com/Shiaoming/ALIKED), [DISK](https://github.com/cvlab-epfl/disk), [RoMa](https://github.com/parskatt/RoMa), [DeDoDe](https://github.com/Parskatt/DeDoDe)
- Flow and deblurring: [RAFT](https://github.com/princeton-vl/RAFT), [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT), [NAFNet](https://github.com/megvii-research/NAFNet), [Restormer](https://github.com/swz30/Restormer), [FFTformer](https://github.com/kkkls/FFTformer), [DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2), [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus), [MMagic](https://github.com/open-mmlab/mmagic), [VRT](https://github.com/JingyunLiang/VRT)
- Learned 3D: [VGGT](https://github.com/facebookresearch/vggt), [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) ([project page](https://vggt-omega.github.io/)), [MapAnything](https://github.com/facebookresearch/map-anything), [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), [Pi3](https://github.com/yyfz/Pi3), [DUSt3R](https://github.com/naver/dust3r), [MASt3R](https://github.com/naver/mast3r), [MUSt3R](https://github.com/naver/must3r), [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM), [Fast3R](https://github.com/facebookresearch/fast3r), [CUT3R](https://github.com/CUT3R/CUT3R), [Spann3R](https://github.com/HengyiWang/spann3r), [VGGSfM](https://github.com/facebookresearch/vggsfm), [MonST3R](https://github.com/Junyi42/monst3r)
- Splatting and surfaces: [gsplat](https://github.com/nerfstudio-project/gsplat), [nerfstudio](https://github.com/nerfstudio-project/nerfstudio), [SDFStudio](https://github.com/autonomousvision/sdfstudio), [Brush](https://github.com/ArthurBrussee/brush), [OpenSplat](https://github.com/pierotofy/OpenSplat), [2DGS](https://github.com/hbb1/2d-gaussian-splatting), [GOF](https://github.com/autonomousvision/gaussian-opacity-fields), [SuGaR](https://github.com/Anttwo/SuGaR), [MILo](https://github.com/Anttwo/MILo), [PGSR](https://github.com/zju3dv/PGSR)
- Monocular depth: [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2), [Metric3D](https://github.com/YvanYin/Metric3D), [UniDepth](https://github.com/lpiccinelli-eth/UniDepth), [ml-depth-pro](https://github.com/apple/ml-depth-pro), [MoGe](https://github.com/microsoft/MoGe)
- Detection: [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [Grounding-DINO-1.5-API](https://github.com/IDEA-Research/Grounding-DINO-1.5-API), [DINO-X-API](https://github.com/IDEA-Research/DINO-X-API), [Rex-Omni](https://github.com/IDEA-Research/Rex-Omni), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [YOLOE](https://github.com/THU-MIG/yoloe), [LLMDet](https://github.com/iSEE-Laboratory/LLMDet), [OmDet](https://github.com/om-ai-lab/OmDet), [MMDetection](https://github.com/open-mmlab/mmdetection), [RF-DETR](https://github.com/roboflow/rf-detr), [D-FINE](https://github.com/Peterande/D-FINE), [RT-DETR](https://github.com/lyuwenyu/RT-DETR), [DEIM](https://github.com/ShihuaHuang95/DEIM), [Ultralytics](https://github.com/ultralytics/ultralytics)
- Segmentation: [SAM](https://github.com/facebookresearch/segment-anything), [SAM 2](https://github.com/facebookresearch/sam2), [SAM 3](https://github.com/facebookresearch/sam3), [EfficientSAM](https://github.com/yformer/EfficientSAM), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
- Retail: [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19), [Retail-Shelf-Monitoring](https://github.com/Alijanloo/Retail-Shelf-Monitoring), [cvpce](https://github.com/laitalaj/cvpce), [shelf-void-detection](https://github.com/Moh-Tayyab/shelf-void-detection), [GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset)
- VLMs: [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)

Hugging Face model cards (license fields read via `huggingface.co/api/models/...`):
- Learned 3D: [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B), [facebook/VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial), [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega), [facebook/map-anything](https://huggingface.co/facebook/map-anything), [facebook/map-anything-apache](https://huggingface.co/facebook/map-anything-apache), [yyfz233/Pi3](https://huggingface.co/yyfz233/Pi3), [yyfz233/Pi3X](https://huggingface.co/yyfz233/Pi3X)
- Depth: [depth-anything/DA3-*](https://huggingface.co/depth-anything), [Depth-Anything-V2-Small/Base/Large/Metric-Indoor-Large-hf](https://huggingface.co/depth-anything), [apple/DepthPro](https://huggingface.co/apple/DepthPro), [Ruicheng/moge-vitl](https://huggingface.co/Ruicheng/moge-vitl), [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal), [Ruicheng/moge-3-vitl](https://huggingface.co/Ruicheng/moge-3-vitl), [lpiccinelli/unidepth-v2-vitl14](https://huggingface.co/lpiccinelli/unidepth-v2-vitl14), [JUGGHM/Metric3D](https://huggingface.co/JUGGHM/Metric3D)
- Detection: [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny), [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base), [openmmlab-community/mm_grounding_dino_large_all](https://huggingface.co/openmmlab-community/mm_grounding_dino_large_all), [iSEE-Laboratory/llmdet_large](https://huggingface.co/iSEE-Laboratory/llmdet_large), [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble), [google/owlv2-large-patch14-ensemble](https://huggingface.co/google/owlv2-large-patch14-ensemble), [omlab/omdet-turbo-swin-tiny-hf](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf), [jameslahm/yoloe-v8l-seg](https://huggingface.co/jameslahm/yoloe-v8l-seg)
- Segmentation: [facebook/sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large), [facebook/sam2.1-hiera-small](https://huggingface.co/facebook/sam2.1-hiera-small), [facebook/sam3](https://huggingface.co/facebook/sam3), [facebook/sam3.1](https://huggingface.co/facebook/sam3.1), [yunyangx/EfficientSAM](https://huggingface.co/yunyangx/EfficientSAM), [dhkim2810/MobileSAM](https://huggingface.co/dhkim2810/MobileSAM)
- VLMs: [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base), [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large), [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it), [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), [google/paligemma2-3b-pt-448](https://huggingface.co/google/paligemma2-3b-pt-448), [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it), [allenai/Molmo2-4B](https://huggingface.co/allenai/Molmo2-4B), [OpenGVLab/InternVL3_5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B), [microsoft/Mage-VL](https://huggingface.co/microsoft/Mage-VL), [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct), [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct), [vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2), [moondream/moondream3.1-9B-A2B](https://huggingface.co/moondream/moondream3.1-9B-A2B)
- SKU-110K-trained models: [is36e/detr-resnet-50-sku110k](https://huggingface.co/is36e/detr-resnet-50-sku110k), [benjamintli/dfine-xl_sku110k](https://huggingface.co/benjamintli/dfine-xl_sku110k)

Datasets: [shelfwise-by-form/SKUs_on_shelves_PL](https://huggingface.co/datasets/shelfwise-by-form/SKUs_on_shelves_PL), [benjamintli/retail-product-checkout](https://huggingface.co/datasets/benjamintli/retail-product-checkout), [Unitail](https://unitedretail.github.io/) ([paper](https://arxiv.org/abs/2204.00298)), Roboflow Universe [Empty Shelves](https://universe.roboflow.com/hostview/empty-shelves-pelqt-rft4u) and [retail-shelf-object-detection](https://universe.roboflow.com/retailshelfobjectdetection/retail-shelf-object-detection) (license from search results only, unverified).

Packages (PyPI JSON API): av, decord, eva-decord, torchcodec, opencv-python, imageio-ffmpeg, pycolmap, pycolmap-cuda12, gsplat, nerfstudio, rfdetr, ultralytics, transformers, sam2, open3d, pymeshlab, xatlas, basicsr. PyTorch wheel indices: https://download.pytorch.org/whl/cu128, /cu129, /cu130.
