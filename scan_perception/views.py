"""Read/write the on-disk view dataset produced by render_views.py.

Layout of a views directory:
    manifest.json            scene metadata + one entry per view (K, extrinsics, file paths)
    rgb/view_0000.png        uint8 RGB
    depth/view_0000.npy      float32 z-depth in stage units; non-finite or <= 0 means no hit
    detections.json          written by detect_2d.py
    masks/view_0000.npz      optional SAM masks, written by detect_2d.py
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MANIFEST = "manifest.json"
DETECTIONS = "detections.json"


@dataclass
class View:
    view_id: str
    rgb_path: Path
    depth_path: Path
    K: np.ndarray                # 3x3
    world_to_camera: np.ndarray  # 4x4, OpenCV camera frame

    def load_depth(self):
        return np.load(self.depth_path).astype(np.float32)

    def load_rgb(self):
        from PIL import Image
        return Image.open(self.rgb_path).convert("RGB")


def load_manifest(views_dir):
    views_dir = Path(views_dir)
    manifest = json.loads((views_dir / MANIFEST).read_text())
    views = [
        View(
            view_id=v["view_id"],
            rgb_path=views_dir / v["rgb"],
            depth_path=views_dir / v["depth"],
            K=np.asarray(v["K"], dtype=np.float64),
            world_to_camera=np.asarray(v["world_to_camera"], dtype=np.float64),
        )
        for v in manifest["views"]
    ]
    return manifest, views


def write_json(path, data):
    """Write atomically so a crash never leaves a half-written manifest."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
