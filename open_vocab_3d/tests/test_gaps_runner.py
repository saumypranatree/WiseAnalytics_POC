"""gaps_runner: merging per-keyframe detections into gaps, and the pixel helpers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gaps_runner as gr  # noqa: E402


def det(kf, x0, x1, y0, y1, score=0.5):
    return {"keyframe": kf, "box_px": [0, 0, 1, 1], "score": score,
            "bmin": [x0, y0, -0.3], "bmax": [x1, y1, 0.0]}


def test_merge_requires_views_and_overlap():
    dets = [det("00001", 0.0, 0.2, 1.0, 1.2), det("00004", 0.01, 0.21, 1.0, 1.21),   # same slot, 2 views
            det("00004", 0.02, 0.2, 1.01, 1.2),                                        # 3rd detection, same view
            det("00009", 1.0, 1.3, 0.2, 0.4),                                          # seen once: dropped
            det("00001", 0.5, 0.6, 1.0, 1.2), det("00002", 0.55, 0.65, 1.0, 1.2)]     # IoU 1/3: merged
    gaps = gr.merge(dets, iou=0.3, min_views=2)
    assert [g["id"] for g in gaps] == ["gap_001", "gap_002"]
    a, b = gaps
    assert (a["num_views"], a["num_detections"]) == (2, 3) and a["bmin"][0] < 0.02
    assert (b["num_views"], b["num_detections"]) == (2, 2)
    # stricter overlap: the IoU-1/3 pair no longer merges, each is then seen once and dropped
    strict = gr.merge(dets, iou=0.5, min_views=2)
    assert len(strict) == 1 and strict[0]["num_detections"] == 3


def test_rect_iou():
    assert gr.rect_iou((0, 1, 0, 1), (0, 1, 0, 1)) == 1.0
    assert gr.rect_iou((0, 1, 0, 1), (2, 3, 0, 1)) == 0.0
    assert abs(gr.rect_iou((0, 2, 0, 1), (1, 3, 0, 1)) - 1 / 3) < 1e-9


def test_distort_matches_shelf_labels_undistort():
    import shelf_labels as sl
    f, cx, cy, k1 = 1400.0, 540.0, 960.0, -0.04
    u = np.array([100.0, 540.0, 1000.0]); v = np.array([200.0, 960.0, 1800.0])
    ud, vd = gr.distort_px(u, v, f, cx, cy, k1)
    x, y = sl.undistort((ud - cx) / f, (vd - cy) / f, k1)
    np.testing.assert_allclose(x * f + cx, u, atol=1e-6); np.testing.assert_allclose(y * f + cy, v, atol=1e-6)
