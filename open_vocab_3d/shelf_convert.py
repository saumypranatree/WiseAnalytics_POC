#!/usr/bin/env python3
"""Shelf-scan conversion job for app v2: store video -> textured .usdz + H.264 copy.

    python -u shelf_convert.py --scan-dir out/scans/<id> --job-id <job_id> \
        --video-to-3d ../video_to_3d [--height-m 1.9]

Runs the video_to_3d step scripts (01_extract ... 06_export_usdz) with video_to_3d's own
interpreter (<video-to-3d>/.venv/bin/python), so its packages never mix with this app's venv.
Everything is written under <scan>/convert/: work/ (frames, SfM model with the keyframe
camera poses, depth, mesh, texture; kept for labelling and gap detection) and out/.

The server does the rest when this exits 0: scan_prep.prepare_generated (no auto-flip),
GLB build, and attaching out/video_h264.mp4 as the scan's source video.

Stdout protocol (same as pipeline_runner.py; everything else is log):
    PROGRESS <0..1> <stage> <message>
    RESULT <path to convert/out/scan.usdz>     last line, exit 0
    ERROR <message>                            last line, exit != 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# (stage, script, progress window, user-facing message); SfM and texturing dominate the runtime
STEPS = [
    ("extract", "01_extract.py", (0.00, 0.12), "decoding video, picking sharp keyframes"),
    ("sfm", "02_sfm.py", (0.12, 0.55), "solving camera positions"),
    ("depth", "03_depth.py", (0.55, 0.62), "estimating depth per keyframe"),
    ("fuse", "04_fuse.py", (0.62, 0.65), "fusing depth into a mesh"),
    ("gravity", "04b_gravity.py", (0.65, 0.67), "aligning the scan upright"),
    ("texture", "05_texture.py", (0.67, 0.96), "baking the texture from the keyframes"),
    ("export", "06_export_usdz.py", (0.96, 0.99), "writing the .usdz"),
    ("cameras", "07_cameras.py", (0.99, 1.00), "exporting keyframe cameras (for labelling)"),
]


def say(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def step_argv(name: str, script: Path, py: Path, video: Path, work: Path, out: Path,
              height_m: float) -> list[str]:
    argv = [str(py), "-u", str(script)]
    if name == "extract":
        return argv + [str(video), str(work), "--mp4", str(out / "video_h264.mp4")]
    if name == "sfm":
        return argv + [str(work), "--name", "sfm_sr"]
    if name in ("depth", "texture"):
        return argv + [str(work), "--sfm", "sfm_sr"]
    if name == "export":
        return argv + [str(work), str(out / "scan.usdz"), "--height-m", str(height_m)]
    return argv + [str(work)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-dir", required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--video-to-3d", required=True, help="video_to_3d dir (scripts/, .venv/, models/hf/)")
    ap.add_argument("--height-m", type=float, default=1.9,
                    help="provisional scale: scanned height in metres (until the scan is calibrated)")
    a = ap.parse_args(argv)

    scan_dir = Path(a.scan_dir).resolve()
    v3d = Path(a.video_to_3d).resolve()
    py = v3d / ".venv" / "bin" / "python"
    rec = json.loads((scan_dir / "scan.json").read_text())
    video = scan_dir / rec["source_path"]
    for need, what in ((py, "video_to_3d interpreter"), (v3d / "scripts", "video_to_3d scripts"),
                       (video, "uploaded video")):
        if not need.exists():
            say(f"ERROR {what} not found: {need}")
            return 2

    work, out = scan_dir / "convert" / "work", scan_dir / "convert" / "out"
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # ROS etc. must not leak in
    env.update(HF_HOME=str(v3d / "models" / "hf"), HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")

    timings = {}
    for name, script, (p0, p1), msg in STEPS:
        say(f"PROGRESS {p0:.4f} {name} {msg}")
        t0 = time.time()
        cmd = step_argv(name, v3d / "scripts" / script, py, video, work, out, a.height_m)
        say(f"$ {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, cwd=str(v3d), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, bufsize=1, errors="replace")
        last = ""
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                last = line
                say(f"  [{name}] {line}")  # prefixed, so it can never look like a protocol line
        rc = proc.wait()
        timings[name] = round(time.time() - t0, 1)
        if rc != 0:
            say(f"ERROR {name} step failed (exit {rc}): {last[:300]}")
            return 1
        say(f"PROGRESS {p1:.4f} {name} {msg}: done in {timings[name]:.0f} s")

    usdz = out / "scan.usdz"
    if not usdz.is_file():
        say("ERROR export step finished but wrote no scan.usdz")
        return 1
    info = {"job_id": a.job_id, "seconds": timings, "height_m_assumed": a.height_m,
            "video_to_3d": str(v3d), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (scan_dir / "convert" / "convert_info.json").write_text(json.dumps(info, indent=2))
    say(f"RESULT {usdz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
