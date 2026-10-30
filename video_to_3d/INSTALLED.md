# Installed packages and models (video_to_3d only)

Everything lives in `video_to_3d/.venv` (python3.12 venv, created with
`env -u PYTHONPATH python3.12 -m venv .venv`) and `video_to_3d/models/hf` (HF cache).
Nothing was installed system-wide, in conda base, or in `open_vocab_3d/.venv`.
Total on disk: `.venv` 8.7 GB (about 6 GB of it is torch + NVIDIA CUDA libraries), models 95 MB.
Download volume: about 4-5 GB (estimate; the torch wheel alone was 820 MB, plus the CUDA wheels).

## Python packages (direct installs)

| Package | Version | Size on disk | License | Used for |
|---|---|---|---|---|
| torch | 2.11.0+cu128 | 1.6 GB (+ nvidia-* 4.3 GB, triton 0.6 GB) | BSD-3-Clause (NVIDIA CUDA libs: NVIDIA EULA, redistributable) | Depth Anything V2 inference (GPU, sm_120) |
| torchvision | 0.26.0+cu128 | small | BSD-3-Clause | dependency |
| pycolmap | 4.2.1 (CPU-only wheel, `has_cuda=False`) | 135 MB | BSD-3-Clause | SIFT, sequential matching, incremental SfM |
| av (PyAV) | 19.0.1 (bundled FFmpeg, av.ffmpeg_version_info = 9.0.2, libavcodec 63.1) | 27 MB + av.libs | PyAV BSD-3-Clause. **The bundled FFmpeg includes libx264/libx265, i.e. a GPL build. See note.** | HEVC decode, H.264 encode |
| opencv-python-headless | 5.0.0.93 | 72 MB | Apache-2.0 | image IO, undistortion, blur score |
| numpy | 2.5.4 | 43 MB | BSD-3-Clause | |
| scipy | 1.18.1 | 109 MB | BSD-3-Clause | (installed; not essential) |
| open3d | 0.20.0 | 907 MB | MIT | TSDF fusion (CPU), mesh cleanup/decimation, Embree ray casting |
| usd-core | 26.8 | 153 MB | Tomorrow Open Source Technology License 1.0 (Apache-2.0 derived) | USD / USDZ writing |
| xatlas (python-xatlas) | 0.0.11 | <1 MB | MIT | UV atlas generation |
| pillow | 12.3.0 | 7 MB | MIT-CMU (HPND) | EXIF check |
| trimesh | 5.1.1 | 5 MB | MIT | (installed; not used in the final pipeline) |
| transformers | 5.19.0 | 116 MB | Apache-2.0 | Depth Anything V2 loading |
| huggingface_hub | 2.2.0 | small | Apache-2.0 | model download (run offline afterwards) |
| safetensors / tokenizers | 0.8.0 / 0.23.3 | small | Apache-2.0 | dependencies |

`pip list` in the venv shows 113 distributions in total, all pulled in as dependencies of the above.

## Models

| Model | Source | Size | License |
|---|---|---|---|
| Depth Anything V2 **Small** (`depth-anything/Depth-Anything-V2-Small-hf`, 24.8M params) | Hugging Face | 95 MB | **Apache-2.0** (commercial OK) |

Not used on purpose: Depth Anything V2 Base/Large (CC-BY-NC-4.0); DA-V2 Metric-Indoor (no license tag on the HF page);
VGGT / MASt3R / DUSt3R (non-commercial licenses).

## License flags
- **GPL (FFmpeg + libx264/x265 inside the PyAV wheel):** fine for internal or on-prem use. If the tool is
  *distributed* to a client, GPL obligations apply. Alternatives: an LGPL FFmpeg build plus OpenH264
  (BSD, Cisco binary), or encoding on the client side.
- HEVC/H.264 decoding and encoding are patent-encumbered codecs (MPEG-LA/Access Advance pools). Flag for legal review if this is productised.
- No non-commercial components are used in the final pipeline.

## Offline operation
After the first run the model is in `models/hf`. All later runs use `HF_HUB_OFFLINE=1` (the default in
`run_all.sh`), and step 3 was re-run successfully with `HF_HUB_OFFLINE=1`.
