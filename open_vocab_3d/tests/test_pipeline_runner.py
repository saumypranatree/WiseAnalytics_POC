"""Tests for pipeline_runner.py with FAKE stage scripts (no GPU, no USD rendering):
stdout protocol, monotonic progress, the two-pass merge, focus-centre choice, the
no-anchor skip and the error path. Plus a check of the real config's commands.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_pipeline_runner.py -q
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import export_json  # noqa: E402
import pipeline_runner as pr  # noqa: E402

STAGES = {"render1", "detect1", "lift1", "render2", "detect2", "lift", "export"}

FAKE_RENDER = r'''
import argparse, json, os, sys
import numpy as np
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("--usd"); ap.add_argument("--out"); ap.add_argument("--mode", default="dome")
ap.add_argument("--num-views", type=int, default=4); ap.add_argument("--focus-domes")
ap.add_argument("--focus-rings"); ap.add_argument("--width", type=int, default=8)
ap.add_argument("--height", type=int, default=6)
a, _ = ap.parse_known_args()
assert os.path.isfile(a.usd), a.usd
n = 0 if a.mode == "focus" else a.num_views
if a.focus_domes:
    n += sum(int(s.split(",")[4]) for s in a.focus_domes.split(";"))
os.makedirs(f"{a.out}/rgb"); os.makedirs(f"{a.out}/depth")
json.dump(sys.argv[1:], open(f"{a.out}/argv.json", "w"))
frames = []
for i in range(n):
    fid = f"{i:04d}"
    Image.fromarray(np.full((a.height, a.width, 3), i, np.uint8)).save(f"{a.out}/rgb/{fid}.png")
    np.save(f"{a.out}/depth/{fid}.npy", np.ones((a.height, a.width), np.float32))
    frames.append({"frame_id": fid, "rgb": f"rgb/{fid}.png", "depth": f"depth/{fid}.npy",
                   "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "T_world_to_cam": None,
                   "T_cam_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
    print(f"[render_lite] view {i} -> frame {fid}", flush=True)
json.dump({"scene_id": "fake", "up_axis": "Y", "image_width": a.width, "image_height": a.height,
           "renderer": "open3d-raycast", "render_mode": a.mode,
           "scene_bounds_world": {"min": [0, 0, 0], "max": [1, 1, 1]}, "frames": frames},
          open(f"{a.out}/meta.json", "w"))
'''

FAKE_DETECT = r'''
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument("--renders"); ap.add_argument("--out"); ap.add_argument("--prompts")
a, _ = ap.parse_known_args()
if os.environ.get("FAKE_DETECT_FAIL"):
    print("CUDA out of memory (fake)"); raise SystemExit(3)
meta = json.load(open(f"{a.renders}/meta.json"))
os.makedirs(f"{a.out}/masks")
prompts = a.prompts.split(",")
frames = []
for fr in meta["frames"]:
    did = f"{fr['frame_id']}_00"
    open(f"{a.out}/masks/{did}.png", "wb").write(b"mask " + did.encode())
    frames.append({"frame_id": fr["frame_id"], "detections": [
        {"det_id": did, "class_name": prompts[0], "score": 0.5, "bbox_xyxy": [0, 0, 2, 2],
         "mask": f"masks/{did}.png"}]})
    print(f"INFO detect_2d: frame {fr['frame_id']}: 1 detections", flush=True)
json.dump({"detector": "fake-dino", "segmenter": "fake-sam", "prompts": prompts,
           "frames": frames}, open(f"{a.out}/detections.json", "w"))
'''

FAKE_LIFT = r'''
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument("--renders"); ap.add_argument("--detections"); ap.add_argument("--out")
ap.add_argument("--save-ply"); ap.add_argument("--min-views")
a, _ = ap.parse_known_args()
meta = json.load(open(f"{a.renders}/meta.json"))
dets = json.load(open(f"{a.detections}/detections.json"))
objs = json.loads(os.environ["FAKE_OBJECTS"])
for o in objs:
    o.setdefault("num_views", len(meta["frames"]))
json.dump({"scene_id": "fake", "objects": objs,
           "seen_frames": [f["frame_id"] for f in dets["frames"]],
           "min_views_arg": a.min_views}, open(a.out, "w"))
if a.save_ply:
    open(a.save_ply, "w").write("ply\n")
'''


def obj(cls, c, half, conf=0.5, views=None):
    o = {"class_name": cls, "confidence": conf,
         "position_world": {"x": c[0], "y": c[1], "z": c[2]},
         "bounding_box_3d": {"min": [x - half for x in c], "max": [x + half for x in c]}}
    if views is not None:
        o["num_views"] = views
    return o


@pytest.fixture
def env(tmp_path):
    fakes = tmp_path / "fakes"
    fakes.mkdir()
    for name, src in (("render", FAKE_RENDER), ("detect", FAKE_DETECT), ("lift", FAKE_LIFT)):
        (fakes / f"fake_{name}.py").write_text(textwrap.dedent(src))
    cfg = {
        "_notes": "test",
        "prompts": ["table", "phone"],
        "stages": {
            "render": {"cmd": ["{python}", "-u", str(fakes / "fake_render.py")],
                       "args": {"--width": 8, "--height": 6}},
            "detect": {"script": str(fakes / "fake_detect.py"), "args": {"--sam": True}},
            "lift": {"cmd": ["{python}", "-u", str(fakes / "fake_lift.py")],
                     "args": {"--min-views": 5, "--save-ply": True}},
        },
        "pass1": {"render_args": {"--mode": "dome", "--num-views": 4}, "lift_args": {}},
        "two_pass": {"enabled": True, "anchor_classes": ["table", "bench seat"],
                     "fallback_min_extent_m": 0.4, "merge_distance_m": 1.5, "max_focus": 3,
                     "focus_radius_m": 2.0, "focus_num_views": 3,
                     "render_args": {"--mode": "focus", "--focus-rings": "25,45,65"},
                     "lift_args": {"--min-views": 7}},
    }
    cfg_path = tmp_path / "pipeline_config.json"
    cfg_path.write_text(json.dumps(cfg))
    scans = tmp_path / "out" / "scans"
    sd = scans / "scan_20261007_141500_ab12"
    sd.mkdir(parents=True)
    (sd / "scene.usda").write_text("#usda 1.0\n")
    rec = {"scan_id": sd.name, "source_file": "room.usdz",
           "uploaded_at": "2026-10-07T14:15:00+05:30", "up_axis": "Y",
           "orientation": {"flip_applied": True, "source": "auto"},
           "scene_bounds": {"min": [0, 0, 0], "max": [5, 3, 4]}, "status": "running"}
    (sd / "scan.json").write_text(json.dumps(rec))
    return {"cfg": cfg_path, "cfg_dict": cfg, "scan_dir": sd, "tmp": tmp_path}


def run_cli(env, objects, job="job1", prompts="table,phone", extra_env=None, cfg=None):
    e = {**os.environ, "FAKE_OBJECTS": json.dumps(objects), **(extra_env or {})}
    if cfg is not None:
        env["cfg"].write_text(json.dumps(cfg))
    p = subprocess.run([sys.executable, "-u", str(PROJ / "pipeline_runner.py"),
                        "--scan-dir", str(env["scan_dir"]), "--job-id", job,
                        "--prompts", prompts, "--config", str(env["cfg"])],
                       capture_output=True, text=True, env=e, timeout=120)
    return p, p.stdout.splitlines()


def progress_lines(lines):
    out = []
    for ln in lines:
        if ln.startswith("PROGRESS "):
            _, p, stage, msg = ln.split(" ", 3)
            out.append((float(p), stage, msg))
    return out


def check_protocol(lines):
    prog = progress_lines(lines)
    assert prog, lines
    vals = [p for p, _, _ in prog]
    assert vals == sorted(vals), "progress must be monotonic"
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert {s for _, s, _ in prog} <= STAGES
    for ln in lines:      # only protocol lines may start with a protocol keyword
        assert not ln.startswith(("RESULT", "ERROR")) or ln is lines[-1]
    return prog


def test_two_pass_protocol_and_merge(env):
    objects = [obj("table", [1.0, -0.5, -1.4], 0.5, 0.6), obj("bench seat", [1.5, -0.8, -1.0], 0.4),
               obj("table", [4.5, -0.5, 1.0], 0.4, 0.4), obj("phone", [1.7, -0.7, -1.1], 0.05)]
    p, lines = run_cli(env, objects)
    assert p.returncode == 0, p.stdout + p.stderr
    prog = check_protocol(lines)
    stages = [s for _, s, _ in prog]
    for s in ("render1", "detect1", "lift1", "render2", "detect2", "lift", "export"):
        assert s in stages
    assert prog[-1][0] == 1.0
    assert lines[-1].startswith("RESULT ")
    out = Path(lines[-1].split(" ", 1)[1])
    sd = env["scan_dir"]
    assert out == sd / "results" / "scene_objects_raw.json"
    doc = json.loads(out.read_text())
    assert export_json.validate(doc) == []
    pl = doc["scan"]["pipeline"]
    assert pl["two_pass"] is True and pl["pass2_skipped_reason"] is None
    assert len(pl["focus_centres"]) == 2                 # table+bench merged, far table alone
    assert pl["detector"] == "fake-dino" and pl["segmenter"] == "fake-sam"
    assert pl["num_frames"] == {"pass1": 4, "pass2": 6}
    assert set(pl["stage_seconds"]) >= {"render1", "detect1", "lift1", "render2", "detect2", "lift"}
    assert doc["scan"]["prompts"] == ["table", "phone"] and doc["scan"]["job_id"] == "job1"
    outbox = env["tmp"] / "out" / "backend_outbox" / f"{sd.name}_raw.json"
    assert json.loads(outbox.read_text()) == doc

    run = sd / "runs" / "job1"
    argv2 = json.loads((run / "pass2" / "renders" / "argv.json").read_text())
    assert "--mode=focus" in argv2 and "--focus-rings=25,45,65" in argv2
    fd = next(a for a in argv2 if a.startswith("--focus-domes="))[len("--focus-domes="):]
    domes = [list(map(float, s.split(","))) for s in fd.split(";")]
    assert len(domes) == 2 and all(d[3:] == [2.0, 3.0] for d in domes)
    assert "--num-views=4" not in argv2
    # merged views: unique ids, both passes, files linked, masks renamed
    meta = json.loads((run / "merged" / "renders" / "meta.json").read_text())
    fids = [f["frame_id"] for f in meta["frames"]]
    assert len(fids) == len(set(fids)) == 10
    assert sum(f.startswith("p1_") for f in fids) == 4 and sum(f.startswith("p2_") for f in fids) == 6
    for f in meta["frames"]:
        assert (run / "merged" / "renders" / f["rgb"]).is_file()
        assert (run / "merged" / "renders" / f["depth"]).is_file()
    dets = json.loads((run / "merged" / "detections" / "detections.json").read_text())
    dids = [d["det_id"] for fr in dets["frames"] for d in fr["detections"]]
    assert len(dids) == len(set(dids)) == 10
    d0 = dets["frames"][5]["detections"][0]
    assert d0["det_id"] == "p2_0001_00" and d0["mask"] == "masks/p2_0001_00.png"
    assert (run / "merged" / "detections" / d0["mask"]).read_bytes() == b"mask 0001_00"
    final = json.loads((run / "merged" / "scene_objects.json").read_text())
    assert sorted(final["seen_frames"]) == sorted(fids)
    assert final["min_views_arg"] == "7"                                 # two_pass.lift_args
    assert json.loads((run / "pass1" / "scene_objects.json").read_text())["min_views_arg"] == "5"
    assert (run / "merged" / "clusters.ply").is_file()
    for s in ("render1", "detect1", "lift1", "render2", "detect2", "lift"):
        assert (run / "logs" / f"{s}.log").is_file()
    info = json.loads((run / "run_info.json").read_text())
    assert info["state"] == "done" and info["num_objects"] == 4
    assert [o["object_id"] for o in doc["objects"]] == ["obj_001", "obj_002", "obj_003", "obj_004"]


def test_no_anchor_skips_pass2(env):
    p, lines = run_cli(env, [obj("phone", [1, 0, 0], 0.05), obj("phone", [2, 0, 0], 0.05)])
    assert p.returncode == 0, p.stdout + p.stderr
    prog = check_protocol(lines)
    stages = [s for _, s, _ in prog]
    assert "render2" not in stages
    assert any("skipped" in m for _, _, m in prog)
    assert lines[-1].startswith("RESULT ")
    doc = json.loads(Path(lines[-1][7:]).read_text())
    pl = doc["scan"]["pipeline"]
    assert pl["two_pass"] is False and pl["focus_centres"] == []
    assert pl["pass2_skipped_reason"] == "pass 1 found no anchor objects"
    run = env["scan_dir"] / "runs" / "job1"
    assert not (run / "pass2").exists()
    meta = json.loads((run / "merged" / "renders" / "meta.json").read_text())
    assert [f["frame_id"] for f in meta["frames"]] == ["p1_0000", "p1_0001", "p1_0002", "p1_0003"]
    final = json.loads((run / "merged" / "scene_objects.json").read_text())
    assert final["min_views_arg"] == "5"            # pass-2 lift args not used


def test_fallback_large_object(env):
    p, lines = run_cli(env, [obj("window", [4, 0, -1], 0.6), obj("phone", [1, 0, 0], 0.05)])
    assert p.returncode == 0, p.stdout
    doc = json.loads(Path(lines[-1][7:]).read_text())
    assert doc["scan"]["pipeline"]["focus_centre_source"] == "fallback"
    assert doc["scan"]["pipeline"]["focus_centres"] == [[4.0, 0.0, -1.0]]


def test_disabled_two_pass(env):
    cfg = json.loads(json.dumps(env["cfg_dict"]))
    cfg["two_pass"]["enabled"] = False
    p, lines = run_cli(env, [obj("table", [1, 0, 0], 0.5)], cfg=cfg)
    assert p.returncode == 0
    doc = json.loads(Path(lines[-1][7:]).read_text())
    assert doc["scan"]["pipeline"]["two_pass"] is False
    assert "disabled" in doc["scan"]["pipeline"]["pass2_skipped_reason"]


def test_error_path(env):
    p, lines = run_cli(env, [obj("table", [1, 0, 0], 0.5)], extra_env={"FAKE_DETECT_FAIL": "1"})
    assert p.returncode != 0
    assert lines[-1].startswith("ERROR ")
    assert "detect1 failed (exit 3)" in lines[-1] and "out of memory" in lines[-1]
    assert not any(ln.startswith("RESULT") for ln in lines)
    check_protocol(lines)
    run = env["scan_dir"] / "runs" / "job1"
    assert json.loads((run / "run_info.json").read_text())["state"] == "error"
    assert "out of memory" in (run / "logs" / "detect1.log").read_text()
    assert not (env["scan_dir"] / "results" / "scene_objects_raw.json").exists()


def test_error_bad_inputs(env):
    p, lines = run_cli(env, [], prompts=" , ")
    assert p.returncode != 0 and lines[-1] == "ERROR no prompts given"
    p, lines = run_cli(env, [], job="../evil")
    assert p.returncode != 0 and lines[-1].startswith("ERROR invalid job id")
    (env["scan_dir"] / "scene.usda").unlink()
    p, lines = run_cli(env, [])
    assert p.returncode != 0 and "scene.usda" in lines[-1]


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #
TP = {"anchor_classes": ["table", "bench seat"], "fallback_min_extent_m": 0.4,
      "merge_distance_m": 1.5, "max_focus": 3}


def test_choose_focus_centres_merge_and_cap():
    objs = [obj("table", [0, 0, 0], 0.5, 0.6, 60), obj("bench seat", [1.0, 0, 0], 0.4, 0.4, 30),
            obj("Table", [10, 0, 0], 0.5, 0.5, 20), obj("table", [20, 0, 0], 0.5, 0.5, 10),
            obj("table", [30, 0, 0], 0.5, 0.5, 40), obj("phone", [0, 0, 0], 0.05, 0.9, 99)]
    cs, src = pr.choose_focus_centres(objs, TP)
    assert src == "anchor" and len(cs) == 3
    assert cs[0]["views"] == 90 and len(cs[0]["members"]) == 2
    assert cs[0]["centre"][0] == pytest.approx(0.4, abs=1e-3)    # 0.6*0 + 0.4*1
    assert [c["centre"][0] for c in cs[1:]] == [30.0, 10.0]      # by supporting views


def test_choose_focus_centres_fallback_and_none():
    objs = [obj("window", [0, 0, 0], 0.3), obj("phone", [5, 0, 0], 0.05)]
    cs, src = pr.choose_focus_centres(objs, TP)
    assert src == "fallback" and len(cs) == 1 and cs[0]["centre"] == [0.0, 0.0, 0.0]
    tall = {"class_name": "pole", "confidence": 0.5,
            "bounding_box_3d": {"min": [0, 0, 0], "max": [0.1, 2.0, 0.1]}}
    assert pr.choose_focus_centres([tall], TP) == ([], "none")      # vertical extent only (Y up)
    assert len(pr.choose_focus_centres([tall], TP, up_axis="Z")[0]) == 1
    assert pr.choose_focus_centres([], TP) == ([], "none")


def test_args_and_prompts():
    assert pr.args_to_cli({"--a": True, "--b": False, "--c": None, "--d": [1, 2], "--e": 0.5,
                           "_x": 1}) == ["--a", "--d=1,2", "--e=0.5"]
    assert pr.merged_args({"--a": 1, "--b": 2}, {"--b": None, "--c": 3}) == \
        {"--a": 1, "--b": None, "--c": 3}
    assert pr.parse_prompts(" table , bench  seat,,table ") == ["table", "bench seat"]
    with pytest.raises(pr.PipelineError):
        pr.parse_prompts(",".join(f"p{i}" for i in range(65)))


def test_real_config_commands(tmp_path):
    cfg = pr.load_config(pr.DEFAULT_CONFIG)
    assert "_notes" in cfg and len(cfg["prompts"]) == 13
    py = sys.executable
    r1 = pr._render_cmd(cfg, py, "/x/scene.usda", tmp_path / "r1", cfg["pass1"]["render_args"])
    assert r1[:2] == [py, "-u"] and r1[2].endswith("render_views_lite.py")
    for f in ("--mode=dome", "--num-views=30", "--rings=15,35,55", "--radius-scale=0.4",
              "--width=1280", "--height=720"):
        assert f in r1
    assert not any(a.startswith("--focus-domes") for a in r1)
    tp = cfg["two_pass"]
    assert (tp["focus_radius_m"], tp["focus_num_views"], tp["max_focus"]) == (2.0, 30, 3)
    fd = pr.focus_domes_arg([{"centre": [1.15, -0.5, -1.4]}], tp)
    assert fd == "1.1500,-0.5000,-1.4000,2.0,30"
    r2 = pr._render_cmd(cfg, py, "/x/scene.usda", tmp_path / "r2", tp["render_args"], focus=fd)
    for f in ("--mode=focus", "--focus-rings=25,45,65", f"--focus-domes={fd}", "--width=1280"):
        assert f in r2
    assert not any(a.startswith(("--num-views", "--rings=")) for a in r2)
    d = pr._detect_cmd(cfg, py, tmp_path / "r", tmp_path / "d", cfg["prompts"])
    for f in ("--model-id=IDEA-Research/grounding-dino-base", "--box-threshold=0.2",
              "--text-threshold=0.25", "--prompt-group-size=2", "--sam", "--device=auto",
              "--prompts=" + ",".join(cfg["prompts"])):
        assert f in d
    lf = pr._lift_cmd(cfg, py, tmp_path / "r", tmp_path / "d", tmp_path / "o.json",
                      tmp_path / "c.ply", tp.get("lift_args"))
    for f in ("--min-score=0.3", "--min-views=5",
              "--min-views-per-class=table=10,bench seat=10,phone=6", "--cross-class-nms",
              "--nms-iou=0.3", "--extended", f"--save-ply={tmp_path / 'c.ply'}",
              "--det-cluster-classes=decorative plate,phone,headphones,light switch,cushion"):
        assert f in lf
    assert any(a.startswith("--max-extent-per-class=table=1.6,") for a in lf)


def test_reporter_monotonic():
    import io
    buf = io.StringIO()
    r = pr.Reporter(buf)
    r.at("detect1", 0.5, "a")
    r.at("render1", 1.0, "b  c\nd")       # earlier window: clamped, message one line
    vals = [float(ln.split()[1]) for ln in buf.getvalue().splitlines()]
    assert vals == sorted(vals) and buf.getvalue().splitlines()[1].endswith("render1 b c d")
