"""Tests for app/server.py (API v2). No GPU, no real pipeline run.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SKIP_GLTF_VALIDATOR=1 .venv/bin/python -m pytest tests/test_server.py -q

scan_prep / usd_to_gltf / export_json are replaced by in-memory fakes (via
sys.modules), and pipeline_runner.py by a fake script injected through
server.PIPELINE_CMD_HOOK. test_real_modules_* use the real modules when importable.
"""
import io
import json
import os
import secrets
import sys
import textwrap
import time
import types
import zipfile
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))
import server  # noqa: E402

ROOM_USDZ = ROOT.parent.parent / "room_scan.usdz"


# --------------------------------------------------------------------------- fakes
def make_usdz(extra: bytes = b"", with_usd: bool = True) -> bytes:
    """Deterministic bytes: entries carry a fixed timestamp, so two calls made a moment apart
    (across a 2 s zip-time tick) still compare equal."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        if with_usd:
            z.writestr(zipfile.ZipInfo("scene.usdc", date_time=(2026, 1, 1, 0, 0, 0)), b"#usda 1.0\n" + extra)
        z.writestr(zipfile.ZipInfo("textures/a.png", date_time=(2026, 1, 1, 0, 0, 0)), b"png")
    return buf.getvalue()


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write(path: Path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2))


def fake_scan_prep(calls):
    m = types.ModuleType("scan_prep")

    def prepare_scan(usdz_path, scans_root, original_name):
        calls.append(("prepare_scan", str(usdz_path), original_name))
        with zipfile.ZipFile(usdz_path) as z:
            names = z.namelist()
        if not any(n.endswith((".usd", ".usdc", ".usda")) for n in names):
            raise ValueError("no USD layer in the usdz")
        sid = f"scan_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"
        d = Path(scans_root) / sid
        (d / "source").mkdir(parents=True)
        (d / "source" / original_name).write_bytes(Path(usdz_path).read_bytes())
        (d / "scene.usda").write_text("#usda 1.0\n# flipped\n")
        rec = {"scan_id": sid, "source_file": original_name, "uploaded_at": _now(),
               "status": "uploaded", "stage": "scene.usda",
               "orientation": {"flip_applied": True, "source": "auto",
                               "flip_recommended": True, "confidence": "high",
                               "evidence": {"top_third": True}}}
        _write(d / "scan.json", rec)
        return rec

    def set_flip(scan_dir, flip, source):
        calls.append(("set_flip", Path(scan_dir).name, flip, source))
        d = Path(scan_dir)
        (d / "scene.usda").write_text(f"#usda 1.0\n# flip={flip}\n")
        glb = d / "viewer" / "model.glb"
        if glb.exists():
            glb.unlink()
        rec = json.loads((d / "scan.json").read_text())
        rec["orientation"].update(flip_applied=flip, source=source)
        rec["status"] = "bogus-from-scan-prep"  # server-owned field: must be ignored
        _write(d / "scan.json", rec)
        return rec

    def prepare_generated(scan_dir, usdz_path):
        calls.append(("prepare_generated", Path(scan_dir).name, Path(usdz_path).name))
        with zipfile.ZipFile(usdz_path) as z:
            assert any(n.endswith(".usdc") for n in z.namelist())
        (Path(scan_dir) / "scene.usda").write_text("#usda 1.0\n# generated\n")
        return {"root_layer": "extracted/scan.usdc", "stage": "scene.usda", "up_axis": "Y",
                "meters_per_unit": 1.0, "num_meshes": 1, "num_triangles": 1234,
                "orientation": {"flip_applied": False, "source": "auto", "flip_recommended": False,
                                "confidence": "low",
                                "evidence": {"reason": "gravity-aligned during video conversion"}}}

    def set_scale(scan_dir, scale):
        calls.append(("set_scale", Path(scan_dir).name, scale.get("factor")))
        d = Path(scan_dir)
        (d / "scene.usda").write_text(f"#usda 1.0\n# scale={scale.get('factor')}\n")
        glb = d / "viewer" / "model.glb"
        if glb.exists():
            glb.unlink()
        rec = json.loads((d / "scan.json").read_text())
        rec["scale"] = scale
        _write(d / "scan.json", rec)
        return rec

    m.prepare_scan, m.set_flip, m.prepare_generated, m.set_scale = prepare_scan, set_flip, prepare_generated, set_scale
    return m


def fake_usd_to_gltf(calls):
    m = types.ModuleType("usd_to_gltf")

    def export_glb(usd_path, out_path):
        calls.append(("export_glb", usd_path, out_path))
        Path(out_path).write_bytes(b"glTF" + Path(usd_path).read_bytes())
        return {"up_axis": "Y"}

    m.export_glb = export_glb
    return m


def fake_export_json():
    m = types.ModuleType("export_json")

    def build_reviewed(raw, decisions):
        by_id = {d["object_id"]: d for d in decisions}
        objs, rejected, acc, ren = [], [], 0, 0
        for o in raw["objects"]:
            d = by_id[o["object_id"]]
            if d["decision"] == "reject":
                rejected.append(o["object_id"])
                continue
            renamed = d["class_name"] != o["class_name"]
            ren += renamed
            acc += not renamed
            objs.append(dict(o, class_name=d["class_name"],
                             review={"decision": "renamed" if renamed else "accepted",
                                     "original_class_name": o["class_name"]}))
        scan = dict(raw["scan"], reviewed_at=_now())
        return dict(raw, status="reviewed", scan=scan, objects=objs,
                    summary={"num_objects": len(objs)},
                    review_summary={"accepted": acc, "renamed": ren, "rejected_object_ids": rejected})

    def validate(doc):
        return [] if doc.get("status") in ("raw", "reviewed") else ["bad status"]

    m.build_reviewed, m.validate = build_reviewed, validate
    return m


FAKE_RUNNER = textwrap.dedent("""
    import argparse, json, os, sys, time
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir"); ap.add_argument("--job-id"); ap.add_argument("--prompts")
    ap.add_argument("--config"); ap.add_argument("--gate"); ap.add_argument("--order")
    ap.add_argument("--fail", action="store_true"); ap.add_argument("--chatty", action="store_true")
    a = ap.parse_args()
    d = Path(a.scan_dir)
    if a.order:
        with open(a.order, "a") as f: f.write(f"start {d.name}\\n")
    print("PROGRESS 0.05 render1 rendering wide dome", flush=True)
    print("some noisy log line", flush=True)
    print("PROGRESS 0.02 render1 bogus backwards progress", flush=True)
    if a.gate:
        t0 = time.time()
        while not os.path.exists(a.gate) and time.time() - t0 < 30:
            time.sleep(0.02)
    if a.chatty:
        for i in range(50):
            print(f"chatty line {i}", flush=True)
    if a.fail:
        print("PROGRESS 0.3 detect1 detecting", flush=True)
        print("ERROR CUDA out of memory", flush=True)
        sys.exit(3)
    for p, st in ((0.3, "detect1"), (0.5, "lift1"), (0.7, "detect2"), (0.9, "lift"), (0.97, "export")):
        print(f"PROGRESS {p} {st} stage {st}", flush=True)
    prompts = a.prompts.split(",")
    objs = [{"object_id": f"obj_{i+1:03d}", "class_name": p, "position": {"x": 0, "y": 0, "z": 0},
             "bounding_box": {"min": [0, 0, 0], "max": [1, 1, 1], "size": [1, 1, 1]},
             "confidence": 0.5, "num_views": 9} for i, p in enumerate(prompts)]
    doc = {"schema_version": "1.0", "status": "raw",
           "scan": {"scan_id": d.name, "job_id": a.job_id, "reviewed_at": None, "prompts": prompts},
           "objects": objs, "summary": {"num_objects": len(objs)}}
    res = d / "results" / "scene_objects_raw.json"
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(doc))
    ob = d.parent.parent / "backend_outbox"; ob.mkdir(exist_ok=True)
    (ob / f"{d.name}_raw.json").write_text(json.dumps(doc))
    (d / "runs" / a.job_id / "argv.json").write_text(json.dumps(sys.argv))
    if a.order:
        with open(a.order, "a") as f: f.write(f"end {d.name}\\n")
    print(f"RESULT {res}", flush=True)
""")


FAKE_GAPS = textwrap.dedent("""
    import argparse, json, sys
    from pathlib import Path
    ap = argparse.ArgumentParser(); ap.add_argument("--scan-dir"); ap.add_argument("--job-id"); ap.add_argument("--fail", action="store_true")
    a = ap.parse_args()
    d = Path(a.scan_dir)
    print("PROGRESS 0.3 dino keyframe 1/2", flush=True)
    if a.fail:
        print("ERROR CUDA out of memory", flush=True); sys.exit(3)
    g = lambda i, lo, hi: {"id": f"gap_{i:03d}", "bmin": lo, "bmax": hi, "score": 0.5, "num_views": 3,
                           "num_detections": 3, "detections": []}
    doc = {"schema": 1, "frame": "usd", "job_id": a.job_id, "num_keyframes": 2, "params": {},
           "methods": {"dino": {"description": "fake", "gaps": [g(1, [0.10, 0.70, -0.3], [0.31, 0.90, 0.0]),
                                                               g(2, [1.0, 1.0, -0.3], [1.2, 1.2, 0.0])]},
                       "depth": {"description": "fake", "gaps": []}}}
    out = d / "results" / "gaps.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc))
    print(f"RESULT {out}", flush=True)
""")


FAKE_CONVERTER = textwrap.dedent("""
    import argparse, io, json, os, sys, time, zipfile
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir"); ap.add_argument("--job-id"); ap.add_argument("--video-to-3d")
    ap.add_argument("--height-m"); ap.add_argument("--gate"); ap.add_argument("--fail", action="store_true")
    a = ap.parse_args()
    d = Path(a.scan_dir)
    rec = json.loads((d / "scan.json").read_text())
    assert (d / rec["source_path"]).is_file()
    print("PROGRESS 0.05 extract decoding video", flush=True)
    if a.gate:
        t0 = time.time()
        while not os.path.exists(a.gate) and time.time() - t0 < 30:
            time.sleep(0.02)
    if a.fail:
        print("ERROR sfm step failed (exit 1): no camera model", flush=True)
        sys.exit(1)
    print("PROGRESS 0.5 sfm solving camera positions", flush=True)
    out = d / "convert" / "out"; out.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("scan.usdc", b"#usda 1.0\\n")
    (out / "scan.usdz").write_bytes(buf.getvalue())
    (out / "video_h264.mp4").write_bytes(b"\\x00\\x00\\x00\\x18ftypisom" + b"\\x00" * 64)
    work = d / "convert" / "work"; (work / "sfm_sr" / "images").mkdir(parents=True, exist_ok=True)
    (work / "sfm_sr" / "images" / "00012.jpg").write_bytes(b"\\xff\\xd8\\xff\\xe0fakejpeg")
    kf = lambda n: {"name": n, "frame": int(n), "time_s": int(n) / 30, "R_cam_from_usd": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                    "C_usd": [0, 1, 1.4]}
    (work / "cameras_usd.json").write_text(json.dumps({"camera": {"model": "SIMPLE_RADIAL", "width": 1080,
        "height": 1920, "f": 1400, "cx": 540, "cy": 960, "k1": 0}, "fps": 30.0, "keyframes": [kf("00012"), kf("00040")]}))
    (d / "runs" / a.job_id / "argv.json").write_text(json.dumps(sys.argv))
    print(f"RESULT {out / 'scan.usdz'}", flush=True)
""")


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def env(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "scan_prep", fake_scan_prep(calls))
    monkeypatch.setitem(sys.modules, "usd_to_gltf", fake_usd_to_gltf(calls))
    monkeypatch.setitem(sys.modules, "export_json", fake_export_json())
    runner = tmp_path / "fake_runner.py"
    runner.write_text(FAKE_RUNNER)
    gate, order = tmp_path / "gate", tmp_path / "order.log"
    opts = {"gated": False, "fail": False, "chatty": False}

    converter = tmp_path / "fake_converter.py"
    converter.write_text(FAKE_CONVERTER)

    gaps_fake = tmp_path / "fake_gaps.py"
    gaps_fake.write_text(FAKE_GAPS)

    def hook(argv, ctx):
        if ctx.get("kind") == "gaps":
            assert argv[:3] == [sys.executable, "-u", "gaps_runner.py"]
            return [sys.executable, "-u", str(gaps_fake)] + argv[3:] + (["--fail"] if opts["fail"] else [])
        if ctx.get("kind") == "convert":
            assert argv[:3] == [sys.executable, "-u", "shelf_convert.py"]
            extra = (["--gate", str(gate)] if opts["gated"] else []) + (["--fail"] if opts["fail"] else [])
            return [sys.executable, "-u", str(converter)] + argv[3:] + extra
        assert argv[:3] == [sys.executable, "-u", "pipeline_runner.py"]
        extra = ["--order", str(order)]
        if opts["gated"]:
            extra += ["--gate", str(gate)]
        if opts["fail"]:
            extra.append("--fail")
        if opts["chatty"]:
            extra.append("--chatty")
        return [sys.executable, "-u", str(runner)] + argv[3:] + extra

    monkeypatch.setattr(server, "PIPELINE_CMD_HOOK", hook)
    # label lifting (ray casting on the real mesh) is tested in test_shelf_labels.py; here a
    # stand-in: 1 px = 0.1 cm, so the server's storage / validation / API can be checked alone
    import shelf_labels
    def fake_lift(scan_dir, rec, labels):
        f = float((rec.get("scale") or {}).get("factor", 1.0))
        out = []
        for l in labels:
            x0, y0, x1, y1 = (v / 1000 * f for v in l["box_px"])     # 1 px = 1 mm on the shelf front
            out.append(dict(l, width_cm=round((x1 - x0) * 100, 1), height_cm=round((y1 - y0) * 100, 1),
                            bbox={"min": [x0, y0, -0.3 * f], "max": [x1, y1, 0.0], "size": [x1 - x0, y1 - y0, 0.3 * f]}))
        return out
    monkeypatch.setattr(shelf_labels, "lift_labels", fake_lift)
    cfg = tmp_path / "pipeline_config.json"
    cfg.write_text(json.dumps({"prompts": ["table", "chair", "laptop"]}))
    out = tmp_path / "out"
    apps = []

    def make_app(**kw):
        app = server.create_app(out, cfg, **kw)
        apps.append(app)
        return TestClient(app), app.state.backend

    client, be = make_app()
    yield types.SimpleNamespace(client=client, be=be, out=out, cfg=cfg, calls=calls, opts=opts,
                                gate=gate, order=order, make_app=make_app, tmp=tmp_path)
    gate.touch()
    for a in apps:
        a.state.backend.shutdown()


def upload(client, data=None, name="room scan.usdz", **fields):
    data = make_usdz() if data is None else data
    form = {k: str(v).lower() if isinstance(v, bool) else v for k, v in fields.items()}
    return client.post("/api/scans", files={"file": (name, data, "model/vnd.usdz+zip")}, data=form)


def wait_job(client, sid, states=("done", "error"), timeout=20.0):
    seen, t0 = [], time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/api/scans/{sid}/job")
        if r.status_code == 200:
            seen.append(r.json())
            if r.json()["state"] in states:
                return r.json(), seen
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {states}: {seen[-1:] or None}")


def scan_dirs(out):
    root = out / "scans"
    return [p for p in root.iterdir() if server.SCAN_ID_RE.fullmatch(p.name)] if root.exists() else []


# --------------------------------------------------------------------------- config / upload validation
def test_config(env):
    r = env.client.get("/api/config")
    assert r.status_code == 200
    assert r.json() == {"default_prompts": ["table", "chair", "laptop"], "max_upload_mb": 500}
    assert r.headers["cache-control"] == "no-store"
    assert env.client.get("/api/scans").json() == []


@pytest.mark.parametrize("data,name,status", [
    (make_usdz(), "scan.zip", 400),             # wrong extension
    (make_usdz(), "scan.usdz.txt", 400),
    (b"#usda 1.0\nnot a zip", "scan.usdz", 400),  # right extension, not a zip
    (b"", "scan.usdz", 400),                     # empty
    (make_usdz(with_usd=False), "scan.usdz", 400),  # prepare_scan rejects it
])
def test_upload_rejects(env, data, name, status):
    r = upload(env.client, data, name)
    assert r.status_code == status, r.text
    assert isinstance(r.json()["detail"], str)
    assert scan_dirs(env.out) == []
    assert not list((env.out / "scans" / ".uploads").glob("*"))


def test_upload_missing_file_and_not_multipart(env):
    r = env.client.post("/api/scans", data={"prompts": "a"})
    assert r.status_code == 400
    r = env.client.post("/api/scans", json={"file": "x"})
    assert r.status_code == 400


def test_upload_too_large(env):
    client, be = env.make_app(max_upload_mb=0.01, start_worker=False)  # ~10 KB
    assert client.get("/api/config").json()["max_upload_mb"] == 0.01
    big = make_usdz(extra=os.urandom(64 * 1024))
    r = upload(client, big)
    assert r.status_code == 413, r.text
    assert scan_dirs(env.out) == []
    assert not list((env.out / "scans" / ".uploads").glob("*"))
    # chunked (no Content-Length) bodies are capped while streaming too
    def gen():
        body = (b"--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.usdz\"\r\n"
                b"Content-Type: application/octet-stream\r\n\r\n" + big + b"\r\n--B--\r\n")
        for i in range(0, len(body), 4096):
            yield body[i:i + 4096]
    r = client.post("/api/scans", content=gen(),
                    headers={"content-type": "multipart/form-data; boundary=B"})
    assert r.status_code == 413, r.text


def test_upload_bad_fields(env):
    assert upload(env.client, auto_run="maybe").status_code == 422
    assert upload(env.client, prompts="ok,-rm").status_code == 422
    assert scan_dirs(env.out) == []


# --------------------------------------------------------------------------- happy path
def test_upload_and_run_happy_path(env):
    c = env.client
    r = upload(c, name="../../evil dir/room scan.USDZ", prompts=" chair , table,,chair ")
    assert r.status_code == 200, r.text
    s = r.json()
    sid = s["scan_id"]
    assert server.SCAN_ID_RE.fullmatch(sid)
    assert s["source_file"] == "room_scan.usdz"
    assert s["status"] in ("queued", "running", "done")
    assert s["orientation"] == {"flip_applied": True, "source": "auto",
                                "flip_recommended": True, "confidence": "high"}
    d = env.out / "scans" / sid
    assert (d / "source" / "room_scan.usdz").read_bytes() == make_usdz()
    assert (d / "viewer" / "model.glb").read_bytes().startswith(b"glTF")
    assert ("prepare_scan", env.calls[0][1], "room_scan.usdz") == env.calls[0]

    final, seen = wait_job(c, sid)
    assert final["state"] == "done", final
    assert final["progress"] == 1.0
    assert final["queue_position"] is None
    progs = [j["progress"] for j in seen]
    assert progs == sorted(progs), "progress must be monotonic"
    assert set(final) >= {"job_id", "state", "progress", "stage", "message", "log_tail", "queue_position"}
    assert any("some noisy log line" in l for l in final["log_tail"])
    assert final["prompts"] == ["chair", "table"]

    run_dir = d / "runs" / final["job_id"]
    log = (run_dir / "runner.log").read_text()
    assert "PROGRESS 0.05 render1" in log and "RESULT " in log
    argv = json.loads((run_dir / "argv.json").read_text())
    assert argv[1:] == ["--scan-dir", str(d), "--job-id", final["job_id"], "--prompts", "chair,table",
                        "--config", str(env.cfg), "--order", str(env.order)]
    assert json.loads((run_dir / "job.json").read_text())["state"] == "done"

    rec = c.get(f"/api/scans/{sid}").json()
    assert rec["status"] == "done" and rec["has_raw"] and not rec["has_reviewed"]
    assert rec["num_objects"] == 2 and rec["stale"] is False
    assert rec["orientation"]["evidence"] == {"top_third": True}  # full scan_prep record kept
    disk = json.loads((d / "scan.json").read_text())
    assert disk["status"] == "done" and disk["job_id"] == final["job_id"]
    assert disk["stage"] == "scene.usda"  # scan_prep's own fields survive server writes
    assert not list(d.glob(".*.tmp"))

    objs = c.get(f"/api/scans/{sid}/objects")
    assert objs.status_code == 200 and objs.headers["cache-control"] == "no-store"
    assert [o["class_name"] for o in objs.json()["objects"]] == ["chair", "table"]
    assert c.get(f"/api/scans/{sid}/objects?version=raw").json() == objs.json()
    assert c.get(f"/api/scans/{sid}/objects?version=reviewed").status_code == 404
    assert c.get(f"/api/scans/{sid}/objects?version=bogus").status_code == 422

    lst = c.get("/api/scans").json()
    assert [x["scan_id"] for x in lst] == [sid]
    assert set(lst[0]) >= {"scan_id", "source_file", "uploaded_at", "status", "progress", "message",
                           "num_objects", "has_raw", "has_reviewed", "orientation"}

    g = c.get(f"/api/scans/{sid}/model.glb")
    assert g.status_code == 200 and g.content.startswith(b"glTF")
    assert g.headers["content-type"] == "model/gltf-binary"


def test_default_prompts_log_tail_and_list_order(env):
    c = env.client
    env.opts["chatty"] = True
    s1 = upload(c, auto_run=False).json()
    time.sleep(1.1)  # distinct uploaded_at / scan_id seconds
    s2 = upload(c, auto_run=False).json()
    assert s1["status"] == "uploaded"
    assert c.get(f"/api/scans/{s1['scan_id']}/job").status_code == 404
    assert [x["scan_id"] for x in c.get("/api/scans").json()] == [s2["scan_id"], s1["scan_id"]]

    r = c.post(f"/api/scans/{s1['scan_id']}/run", json={})
    assert r.status_code == 200, r.text
    assert r.json()["prompts"] == ["table", "chair", "laptop"]
    final, _ = wait_job(c, s1["scan_id"])
    assert final["state"] == "done"
    assert len(final["log_tail"]) == 30
    full = (env.out / "scans" / s1["scan_id"] / "runs" / final["job_id"] / "runner.log").read_text()
    assert "chatty line 0" in full and "chatty line 0" not in "\n".join(final["log_tail"])

    assert c.post(f"/api/scans/{s2['scan_id']}/run", json={"prompts": ["a,b"]}).status_code == 422
    assert c.post(f"/api/scans/{s2['scan_id']}/run", json={"prompts": "x"}).status_code == 200


def test_runner_error(env):
    env.opts["fail"] = True
    sid = upload(env.client).json()["scan_id"]
    final, _ = wait_job(env.client, sid)
    assert final["state"] == "error"
    assert final["message"] == "CUDA out of memory"
    rec = env.client.get(f"/api/scans/{sid}").json()
    assert rec["status"] == "error" and rec["has_raw"] is False
    assert env.client.get(f"/api/scans/{sid}/objects").status_code == 404


# --------------------------------------------------------------------------- queue / 409
def test_fifo_queue_and_409s(env):
    c = env.client
    env.opts["gated"] = True
    a = upload(c).json()["scan_id"]
    wait_job(c, a, states=("running",))
    b = upload(c).json()["scan_id"]
    jb = c.get(f"/api/scans/{b}/job").json()
    assert jb["state"] == "queued" and jb["queue_position"] == 1
    assert c.get(f"/api/scans/{a}/job").json()["queue_position"] is None
    assert c.get(f"/api/scans/{b}").json()["status"] == "queued"

    for sid in (a, b):  # running and queued scans are locked
        assert c.post(f"/api/scans/{sid}/run", json={}).status_code == 409
        assert c.post(f"/api/scans/{sid}/orientation", json={"flip": False}).status_code == 409
        assert c.delete(f"/api/scans/{sid}").status_code == 409
        assert c.post(f"/api/scans/{sid}/send", json={"decisions": []}).status_code == 409
    time.sleep(0.3)
    assert c.get(f"/api/scans/{b}/job").json()["state"] == "queued"  # one job at a time

    env.gate.touch()
    fa, _ = wait_job(c, a)
    fb, _ = wait_job(c, b)
    assert fa["state"] == fb["state"] == "done"
    assert env.order.read_text().split("\n")[:4] == [f"start {a}", f"end {a}", f"start {b}", f"end {b}"]
    assert fa["finished"] <= fb["started"]
    assert c.post(f"/api/scans/{a}/run", json={"prompts": ["x"]}).status_code == 200


# --------------------------------------------------------------------------- orientation
def test_orientation_flip_marks_stale(env):
    c = env.client
    sid = upload(c, auto_run=False).json()["scan_id"]
    n_glb = sum(1 for x in env.calls if x[0] == "export_glb")
    # no results yet -> not stale
    r = c.post(f"/api/scans/{sid}/orientation", json={"flip": False})
    assert r.status_code == 200, r.text
    assert r.json()["orientation"]["flip_applied"] is False
    assert r.json()["orientation"]["source"] == "manual"
    assert r.json()["stale"] is False
    assert r.json()["status"] == "uploaded"  # scan_prep's status write is ignored
    assert ("set_flip", sid, False, "manual") in env.calls
    assert sum(1 for x in env.calls if x[0] == "export_glb") == n_glb + 1
    glb = env.out / "scans" / sid / "viewer" / "model.glb"
    assert b"flip=False" in glb.read_bytes()

    c.post(f"/api/scans/{sid}/run", json={})
    assert wait_job(c, sid)[0]["state"] == "done"
    r = c.post(f"/api/scans/{sid}/orientation", json={"flip": True})
    assert r.json()["stale"] is True and r.json()["status"] == "done"
    assert c.get(f"/api/scans/{sid}").json()["stale"] is True
    assert b"flip=True" in c.get(f"/api/scans/{sid}/model.glb").content

    # re-running clears stale
    c.post(f"/api/scans/{sid}/run", json={})
    assert wait_job(c, sid)[0]["state"] == "done"
    assert c.get(f"/api/scans/{sid}").json()["stale"] is False

    for bad in ({}, {"flip": "yes"}, {"flip": 1}):
        assert c.post(f"/api/scans/{sid}/orientation", json=bad).status_code == 422


def test_glb_built_lazily(env):
    c = env.client
    sid = upload(c, auto_run=False).json()["scan_id"]
    glb = env.out / "scans" / sid / "viewer" / "model.glb"
    glb.unlink()
    r = c.get(f"/api/scans/{sid}/model.glb")
    assert r.status_code == 200 and glb.is_file()
    assert r.headers["cache-control"] == "no-cache"


# --------------------------------------------------------------------------- send / reviewed
def test_send_reviewed(env):
    c = env.client
    sid = upload(c, prompts="table,chair,laptop").json()["scan_id"]
    assert wait_job(c, sid)[0]["state"] == "done"
    url = f"/api/scans/{sid}/send"
    ok = [{"object_id": "obj_001", "decision": "accept", "class_name": "table"},
          {"object_id": "obj_002", "decision": "accept", "class_name": " armchair "},
          {"object_id": "obj_003", "decision": "reject", "class_name": "laptop"}]
    bad_cases = [
        {},
        {"decisions": "x"},
        {"decisions": ok[:2]},                                      # missing obj_003
        {"decisions": ok + [ok[0]]},                                # duplicate
        {"decisions": ok[:2] + [dict(ok[2], decision="maybe")]},
        {"decisions": ok[:2] + [dict(ok[2], class_name="  ")]},
        {"decisions": ok[:2] + [dict(ok[2], class_name=None)]},
        {"decisions": ok[:2] + [dict(ok[2], object_id="obj_999")]},
        {"decisions": ok + [{"object_id": "obj_999", "decision": "accept", "class_name": "x"}]},
    ]
    for body in bad_cases:
        r = c.post(url, json=body)
        assert r.status_code == 422, (body, r.text)
        assert isinstance(r.json()["detail"], str)
    assert not (env.out / "scans" / sid / "results" / "scene_objects_reviewed.json").exists()

    r = c.post(url, json={"decisions": ok})
    assert r.status_code == 200, r.text
    res = r.json()
    rev_path = env.out / "scans" / sid / "results" / "scene_objects_reviewed.json"
    outbox = env.out / "backend_outbox" / f"{sid}_reviewed.json"
    assert res == {"saved": True, "path": str(rev_path), "outbox_path": str(outbox), "num_objects": 2}
    doc = json.loads(rev_path.read_text())
    assert json.loads(outbox.read_text()) == doc
    assert [o["class_name"] for o in doc["objects"]] == ["table", "armchair"]
    assert doc["review_summary"]["rejected_object_ids"] == ["obj_003"]

    s = c.get(f"/api/scans/{sid}").json()
    assert s["status"] == "reviewed" and s["has_reviewed"] and s["num_objects"] == 2
    assert c.get(f"/api/scans/{sid}/objects").json()["status"] == "reviewed"
    assert c.get(f"/api/scans/{sid}/objects?version=raw").json()["status"] == "raw"
    # re-send overwrites
    assert c.post(url, json={"decisions": [dict(d, decision="accept") for d in ok]}).json()["num_objects"] == 3

    # a new run supersedes the old review
    c.post(f"/api/scans/{sid}/run", json={"prompts": ["table"]})
    assert wait_job(c, sid)[0]["state"] == "done"
    s = c.get(f"/api/scans/{sid}").json()
    assert s["status"] == "done" and not s["has_reviewed"]
    assert list((env.out / "scans" / sid / "results").glob("scene_objects_reviewed.superseded_*.json"))
    assert outbox.is_file()


def test_send_without_raw(env):
    sid = upload(env.client, auto_run=False).json()["scan_id"]
    assert env.client.post(f"/api/scans/{sid}/send", json={"decisions": []}).status_code == 404


# --------------------------------------------------------------------------- delete
def test_delete(env):
    c = env.client
    sid = upload(c).json()["scan_id"]
    assert wait_job(c, sid)[0]["state"] == "done"
    raw_outbox = env.out / "backend_outbox" / f"{sid}_raw.json"
    assert raw_outbox.is_file()
    r = c.delete(f"/api/scans/{sid}")
    assert r.status_code == 200 and r.json() == {"deleted": True, "scan_id": sid}
    assert not (env.out / "scans" / sid).exists()
    assert raw_outbox.is_file()
    assert c.get(f"/api/scans/{sid}").status_code == 404
    assert c.get(f"/api/scans/{sid}/job").status_code == 404
    assert c.delete(f"/api/scans/{sid}").status_code == 404
    assert c.get("/api/scans").json() == []


# --------------------------------------------------------------------------- source video
MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + bytes(range(256)) * 4
WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 300


def put_video(client, sid, data=MP4, name="room_scan.mp4"):
    return client.post(f"/api/scans/{sid}/video", files={"file": (name, data, "video/mp4")})


def test_video_attach_stream_replace_delete(env):
    c = env.client
    sid = upload(c, auto_run=False).json()["scan_id"]
    assert c.get(f"/api/scans/{sid}").json()["source_video"] is None
    assert c.get("/api/scans").json()[0]["has_video"] is False
    assert c.get(f"/api/scans/{sid}/video").status_code == 404
    assert c.delete(f"/api/scans/{sid}/video").status_code == 404

    r = put_video(c, sid, name="../My Scan (1).mp4")
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["file"] == "My Scan _1_.mp4" and meta["bytes"] == len(MP4)
    assert meta["content_type"] == "video/mp4" and meta["url"] == f"/api/scans/{sid}/video"
    assert (env.out / "scans" / sid / "source" / "video.mp4").read_bytes() == MP4
    assert c.get(f"/api/scans/{sid}").json()["source_video"] == meta
    assert c.get("/api/scans").json()[0]["has_video"] is True
    assert not list((env.out / "scans" / ".uploads").glob("*"))

    r = c.get(meta["url"])
    assert r.status_code == 200 and r.content == MP4
    assert r.headers["content-type"] == "video/mp4"
    assert r.headers["cache-control"] == "no-cache"
    r = c.get(meta["url"], headers={"Range": "bytes=4-11"})      # seeking
    assert r.status_code == 206 and r.content == MP4[4:12]
    assert r.headers["content-range"] == f"bytes 4-11/{len(MP4)}"

    r = put_video(c, sid, WEBM, "capture.webm")                   # replace: old file removed
    assert r.status_code == 200 and r.json()["content_type"] == "video/webm"
    assert not (env.out / "scans" / sid / "source" / "video.mp4").exists()
    assert c.get(f"/api/scans/{sid}/video").content == WEBM

    r = c.delete(f"/api/scans/{sid}/video")
    assert r.status_code == 200 and r.json() == {"deleted": True, "scan_id": sid}
    assert r.headers["cache-control"] == "no-store"
    assert not (env.out / "scans" / sid / "source" / "video.webm").exists()
    assert c.get(f"/api/scans/{sid}").json()["source_video"] is None
    assert (env.out / "scans" / sid / "source" / "room_scan.usdz").is_file()  # the scan itself is untouched


@pytest.mark.parametrize("data,name", [
    (MP4, "clip.mov"),                 # not a browser-safe container
    (MP4, "clip"),
    (b"not a video at all", "clip.mp4"),
    (WEBM, "clip.mp4"),                # magic does not match the extension
    (MP4, "clip.webm"),
    (b"", "clip.mp4"),
])
def test_video_rejects(env, data, name):
    c = env.client
    sid = upload(c, auto_run=False).json()["scan_id"]
    r = put_video(c, sid, data, name)
    assert r.status_code == 400, r.text
    assert c.get(f"/api/scans/{sid}").json()["source_video"] is None
    assert not list((env.out / "scans" / sid / "source").glob("video*"))
    assert not list((env.out / "scans" / ".uploads").glob("*"))


def test_video_too_large_and_unknown_scan(env):
    client, _ = env.make_app(max_upload_mb=0.01, start_worker=False)
    sid = upload(client, auto_run=False).json()["scan_id"]
    assert put_video(client, sid, MP4 + os.urandom(64 * 1024)).status_code == 413
    assert client.get(f"/api/scans/{sid}").json()["source_video"] is None
    assert put_video(client, "scan_20261007_141500_ab12").status_code == 404


def test_video_path_outside_source_is_ignored(env):
    c = env.client
    sid = upload(c, auto_run=False).json()["scan_id"]
    assert put_video(c, sid).status_code == 200
    for bad in ("../scan.json", "../../../../etc/passwd", "source/../scan.json"):
        env.be.update_scan(sid, source_video={"path": bad, "content_type": "video/mp4"})
        assert c.get(f"/api/scans/{sid}/video").status_code == 404
        assert c.get(f"/api/scans/{sid}").json()["source_video"] is None


# --------------------------------------------------------------------------- shelf scans (store video)
MOV = b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00qt  " + bytes(range(256)) * 4


def upload_video(client, data=MOV, name="IMG_5198 (2).MOV"):
    return client.post("/api/scans", files={"file": (name, data, "video/quicktime")})


def test_shelf_video_upload_converts(env):
    c = env.client
    r = upload_video(c)
    assert r.status_code == 200, r.text
    s = r.json()
    sid = s["scan_id"]
    assert server.SCAN_ID_RE.fullmatch(sid)
    assert s["scan_type"] == "shelf" and s["status"] in ("queued", "running", "done")
    sdir = env.out / "scans" / sid
    assert (sdir / "source" / "IMG_5198_2.mov").read_bytes() == MOV
    assert not list((env.out / "scans" / ".uploads").glob("*"))

    job, seen = wait_job(c, sid)
    assert job["state"] == "done", job
    assert job["kind"] == "convert" and job["message"].startswith("converted: 1,234 triangles")
    argv = json.loads(next((sdir / "runs").glob("*/argv.json")).read_text())
    assert "--video-to-3d" in argv and "--height-m" in argv

    rec = c.get(f"/api/scans/{sid}").json()
    assert rec["scan_type"] == "shelf" and rec["status"] == "done"
    assert rec["num_triangles"] == 1234 and rec["orientation_summary"]["flip_applied"] is False
    assert rec["scale"] == {"status": "approximate", "method": "assumed_height", "assumed_height_m": 1.9,
                            "factor": 1.0}
    assert rec["source_file"] == "IMG_5198_2.mov" and rec["converted_at"]
    assert rec["source_video"]["file"] == "IMG_5198_2.mp4"            # H.264 copy attached
    assert c.get(rec["source_video"]["url"]).content.startswith(b"\x00\x00\x00\x18ftyp")
    assert not (sdir / "convert" / "out" / "video_h264.mp4").exists()   # moved into source/
    assert c.get(f"/api/scans/{sid}/model.glb").status_code == 200
    assert ("prepare_generated", sid, "scan.usdz") in env.calls
    assert c.get("/api/scans").json()[0]["scan_type"] == "shelf"

    r = c.post(f"/api/scans/{sid}/run", json={})                       # room detection: not for shelves
    assert r.status_code == 422 and "shelf" in r.json()["detail"]
    r = c.post(f"/api/scans/{sid}/convert")                             # re-convert works
    assert r.status_code == 200 and r.json()["kind"] == "convert"
    assert wait_job(c, sid)[0]["state"] == "done"


def test_shelf_scale_measure_reset_and_reconvert(env):
    c = env.client
    sid = upload_video(c).json()["scan_id"]
    assert wait_job(c, sid)[0]["state"] == "done"
    url = f"/api/scans/{sid}/scale"
    # the user clicked two points 0.5 m apart (scene frame) and says they are 60 cm apart
    r = c.post(url, json={"method": "measured", "p1": [0, 1, 0], "p2": [0.3, 1.4, 0], "real_cm": 60})
    assert r.status_code == 200, r.text
    sc = r.json()["scale"]
    assert sc["status"] == "calibrated" and sc["factor"] == pytest.approx(1.2)
    assert sc["measured"]["scene_distance_m"] == pytest.approx(0.5) and sc["measured"]["previous_factor"] == 1.0
    assert ("set_scale", sid, pytest.approx(1.2)) in env.calls
    assert c.get(f"/api/scans/{sid}").json()["scale"]["factor"] == pytest.approx(1.2)
    assert c.get("/api/scans").json()[0]["scale_status"] == "calibrated"
    assert c.get(f"/api/scans/{sid}/model.glb").status_code == 200        # GLB rebuilt
    # a second measurement is relative to the current (already scaled) frame
    r = c.post(url, json={"method": "measured", "p1": [0, 0, 0], "p2": [1, 0, 0], "real_cm": 150})
    assert r.json()["scale"]["factor"] == pytest.approx(1.8)
    r = c.post(url, json={"method": "reset"})
    assert r.status_code == 200 and r.json()["scale"] == {"status": "approximate", "method": "assumed_height",
                                                         "assumed_height_m": 1.9, "factor": 1.0}
    # bad input
    for body in ({"method": "measured", "p1": [0, 0], "p2": [1, 0, 0], "real_cm": 10},
                 {"method": "measured", "p1": [0, 0, 0], "p2": [0.001, 0, 0], "real_cm": 10},
                 {"method": "measured", "p1": [0, 0, 0], "p2": [1, 0, 0], "real_cm": 0},
                 {"method": "measured", "p1": [0, 0, 0], "p2": [1, 0, 0], "real_cm": 100000},
                 {"method": "guess"}):
        assert c.post(url, json=body).status_code == 422, body
    # re-conversion makes new geometry: the calibration is reset
    c.post(url, json={"method": "measured", "p1": [0, 0, 0], "p2": [1, 0, 0], "real_cm": 120})
    assert c.post(f"/api/scans/{sid}/convert").status_code == 200
    assert wait_job(c, sid)[0]["state"] == "done"
    assert c.get(f"/api/scans/{sid}").json()["scale"]["status"] == "approximate"


def test_scale_only_for_converted_shelf_scans(env):
    c = env.client
    room = upload(c, auto_run=False).json()["scan_id"]
    body = {"method": "reset"}
    assert c.post(f"/api/scans/{room}/scale", json=body).status_code == 422
    env.opts["gated"] = True
    shelf = upload_video(c).json()["scan_id"]
    assert c.post(f"/api/scans/{shelf}/scale", json=body).status_code == 409   # not converted yet
    env.gate.touch()
    assert wait_job(c, shelf)[0]["state"] == "done"
    assert c.post(f"/api/scans/{shelf}/scale", json=body).status_code == 200


def test_shelf_labels_crud(env):
    c = env.client
    sid = upload_video(c).json()["scan_id"]
    assert wait_job(c, sid)[0]["state"] == "done"
    k = c.get(f"/api/scans/{sid}/keyframes").json()
    assert k["width"] == 1080 and k["height"] == 1920 and [x["name"] for x in k["keyframes"]] == ["00012", "00040"]
    r = c.get(f"/api/scans/{sid}/keyframes/00012.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.headers["cache-control"] == "no-cache"
    for bad in ("00040.jpg", "0012.jpg", "..%2Fscan.json", "00012.png"):
        assert c.get(f"/api/scans/{sid}/keyframes/{bad}").status_code == 404, bad
    url = f"/api/scans/{sid}/labels"
    assert c.get(url).json()["labels"] == []
    r = c.post(url, json={"keyframe": "00012", "box_px": [300, 900, 100, 700], "kind": "empty", "note": " top shelf "})
    assert r.status_code == 200, r.text
    lab = r.json()
    assert lab["id"] == "lab_001" and lab["box_px"] == [100, 700, 300, 900] and lab["note"] == "top shelf"
    assert lab["width_cm"] == 20.0 and lab["height_cm"] == 20.0
    lab2 = c.post(url, json={"keyframe": "00040", "box_px": [10, 10, 60, 40], "kind": "pushed_back"}).json()
    assert lab2["id"] == "lab_002" and lab2["kind"] == "pushed_back"
    # validation
    for body in ({"keyframe": "99999", "box_px": [0, 0, 10, 10]},
                 {"keyframe": "00012", "box_px": [0, 0, 2, 2]},
                 {"keyframe": "00012", "box_px": [0, 0, 2000, 10]},
                 {"keyframe": "00012", "box_px": [0, 0, 10]},
                 {"keyframe": "00012", "box_px": [0, 0, 10, 10], "kind": "full"},
                 {"keyframe": "00012", "box_px": [0, 0, 10, 10], "note": "x" * 501}):
        assert c.post(url, json=body).status_code == 422, body
    # update / delete
    r = c.patch(f"{url}/lab_001", json={"kind": "pushed_back", "box_px": [100, 700, 400, 900]})
    assert r.status_code == 200 and r.json()["kind"] == "pushed_back" and r.json()["width_cm"] == 30.0
    assert c.patch(f"{url}/lab_999", json={"kind": "empty"}).status_code == 404
    assert c.delete(f"{url}/lab_002").json() == {"deleted": True, "id": "lab_002"}
    assert c.delete(f"{url}/lab_002").status_code == 404
    labs = c.get(url).json()["labels"]
    assert [l["id"] for l in labs] == ["lab_001"]
    assert c.post(url, json={"keyframe": "00012", "box_px": [0, 0, 10, 10]}).json()["id"] == "lab_002"
    # sizes follow a calibration (labels are stored in 2D, lifted on read)
    c.post(f"/api/scans/{sid}/scale", json={"method": "measured", "p1": [0, 0, 0], "p2": [1, 0, 0], "real_cm": 200})
    assert c.get(url).json()["labels"][0]["width_cm"] == 60.0
    stored = json.loads((env.out / "scans" / sid / "labels" / "empty_slots.json").read_text())
    assert stored["labels"][0]["box_px"] == [100, 700, 400, 900] and "width_cm" not in stored["labels"][0]
    # re-conversion: new keyframes, the old labels are archived
    assert c.post(f"/api/scans/{sid}/convert").status_code == 200
    assert wait_job(c, sid)[0]["state"] == "done"
    assert c.get(url).json()["labels"] == []
    assert list((env.out / "scans" / sid / "labels").glob("empty_slots.superseded_*.json"))


def test_gap_detection_job_and_scores(env):
    c = env.client
    sid = upload_video(c).json()["scan_id"]
    assert wait_job(c, sid)[0]["state"] == "done"
    assert c.get(f"/api/scans/{sid}/gaps").status_code == 404
    # one labelled slot (drawn twice, on two keyframes) + one more label elsewhere
    url = f"/api/scans/{sid}/labels"
    c.post(url, json={"keyframe": "00012", "box_px": [100, 700, 300, 900]})
    c.post(url, json={"keyframe": "00040", "box_px": [105, 705, 305, 905]})
    c.post(url, json={"keyframe": "00040", "box_px": [500, 100, 600, 160]})
    r = c.post(f"/api/scans/{sid}/gaps/run")
    assert r.status_code == 200 and r.json()["kind"] == "gaps"
    job, _ = wait_job(c, sid)
    assert job["state"] == "done" and job["message"] == "gaps: dino 2, depth 0", job
    assert c.get("/api/scans").json()[0]["has_gaps"] is True
    g = c.get(f"/api/scans/{sid}/gaps").json()
    assert g["labelled_slots"] == 2 and g["scale_changed_since_run"] is False
    d = g["scores"]["dino"]
    assert (d["found"], d["missed"], d["false_gaps"], d["recall"], d["precision"]) == (1, 1, 1, 0.5, 0.5)
    assert d["matches"][0]["gap_id"] == "gap_001" and sorted(d["matches"][0]["label_ids"]) == ["lab_001", "lab_002"]
    assert g["methods"]["dino"]["gaps"][0]["width_cm"] == 21.0
    assert g["scores"]["depth"]["found"] == 0 and g["scores"]["depth"]["precision"] is None
    # calibrating rescales the gaps too (stored in the usd frame, mapped on read)
    c.post(f"/api/scans/{sid}/scale", json={"method": "measured", "p1": [0, 0, 0], "p2": [1, 0, 0], "real_cm": 200})
    g2 = c.get(f"/api/scans/{sid}/gaps").json()
    assert g2["methods"]["dino"]["gaps"][0]["width_cm"] == 42.0 and g2["scale_changed_since_run"] is True
    assert g2["scores"]["dino"]["found"] == 1        # labels and gaps scale together
    # a failing run keeps the previous results; re-conversion archives them
    env.opts["fail"] = True
    c.post(f"/api/scans/{sid}/gaps/run")
    assert wait_job(c, sid)[0]["state"] == "error"
    assert c.get(f"/api/scans/{sid}/gaps").status_code == 200
    env.opts["fail"] = False
    assert c.post(f"/api/scans/{sid}/convert").status_code == 200
    assert wait_job(c, sid)[0]["state"] == "done"
    assert c.get(f"/api/scans/{sid}/gaps").status_code == 404
    assert list((env.out / "scans" / sid / "results").glob("gaps.superseded_*.json"))


def test_gaps_rejected_for_room_and_unconverted(env):
    c = env.client
    room = upload(c, auto_run=False).json()["scan_id"]
    assert c.post(f"/api/scans/{room}/gaps/run").status_code == 422
    assert c.get(f"/api/scans/{room}/gaps").status_code == 422
    env.opts["gated"] = True
    shelf = upload_video(c).json()["scan_id"]
    assert c.post(f"/api/scans/{shelf}/gaps/run").status_code == 409
    env.gate.touch()


def test_labels_only_for_shelf_scans(env):
    c = env.client
    room = upload(c, auto_run=False).json()["scan_id"]
    assert c.get(f"/api/scans/{room}/labels").status_code == 422
    assert c.get(f"/api/scans/{room}/keyframes").status_code == 422
    env.opts["gated"] = True
    shelf = upload_video(c).json()["scan_id"]
    assert c.get(f"/api/scans/{shelf}/keyframes").status_code == 409      # no cameras before conversion
    assert c.post(f"/api/scans/{shelf}/labels", json={"keyframe": "00012", "box_px": [0, 0, 9, 9]}).status_code == 409
    env.gate.touch()


def test_room_scan_cannot_be_converted(env):
    c = env.client
    sid = upload(c, auto_run=False).json()["scan_id"]
    assert c.get("/api/scans").json()[0]["scan_type"] == "room"
    assert c.post(f"/api/scans/{sid}/convert").status_code == 422


@pytest.mark.parametrize("data,name", [
    (b"not a video at all, no atoms here", "clip.MOV"),
    (WEBM, "clip.mp4"),
    (b"", "clip.mov"),
])
def test_shelf_video_rejects(env, data, name):
    r = upload_video(env.client, data, name)
    assert r.status_code == 400, r.text
    assert scan_dirs(env.out) == []
    assert not list((env.out / "scans" / ".uploads").glob("*"))


def test_shelf_convert_failure_and_retry(env):
    c = env.client
    env.opts["fail"] = True
    sid = upload_video(c).json()["scan_id"]
    job, _ = wait_job(c, sid)
    assert job["state"] == "error" and "no camera model" in job["message"]
    rec = c.get(f"/api/scans/{sid}").json()
    assert rec["status"] == "error" and rec["source_video"] is None
    assert c.get(f"/api/scans/{sid}/model.glb").status_code == 404
    env.opts["fail"] = False
    assert c.post(f"/api/scans/{sid}/convert").status_code == 200
    assert wait_job(c, sid)[0]["state"] == "done"
    assert c.get(f"/api/scans/{sid}").json()["status"] == "done"


def test_shelf_orientation_before_conversion_and_409(env):
    c = env.client
    env.opts["gated"] = True
    sid = upload_video(c).json()["scan_id"]
    assert c.post(f"/api/scans/{sid}/orientation", json={"flip": True}).status_code == 409
    assert c.post(f"/api/scans/{sid}/convert").status_code == 409     # already queued/running
    env.gate.touch()
    assert wait_job(c, sid)[0]["state"] == "done"
    assert c.post(f"/api/scans/{sid}/orientation", json={"flip": True}).status_code == 200


def test_shelf_convert_requeued_after_restart(env):
    c0, _ = env.make_app(start_worker=False)
    sid = upload_video(c0).json()["scan_id"]
    assert c0.get(f"/api/scans/{sid}/job").json()["state"] == "queued"
    env.be.shutdown()
    c1, _ = env.make_app()
    job, _ = wait_job(c1, sid)
    assert job["state"] == "done" and job["kind"] == "convert"


# --------------------------------------------------------------------------- restart
def test_restart_recovery(env):
    c0, be0 = env.make_app(start_worker=False)   # second app on the same out dir, no worker
    a = upload(c0).json()["scan_id"]
    b = upload(c0).json()["scan_id"]
    assert c0.get(f"/api/scans/{a}/job").json()["state"] == "queued"
    # simulate a crash while job A was running
    ja = be0.jobs[be0.active_job(a).job_id]
    ja.state, ja.started = "running", time.time()
    ja.persist()
    be0.update_scan(a, status="running", progress=0.4)

    env.be.shutdown()
    c1, be1 = env.make_app()
    s_a = c1.get(f"/api/scans/{a}").json()
    assert s_a["status"] == "error" and s_a["message"] == "interrupted (server restarted)"
    j_a = c1.get(f"/api/scans/{a}/job").json()
    assert j_a["state"] == "error" and j_a["message"] == "interrupted (server restarted)"
    fb, _ = wait_job(c1, b)          # the queued job was re-queued and runs
    assert fb["state"] == "done"
    assert c1.get(f"/api/scans/{b}").json()["status"] == "done"
    # A can be re-run after the interruption
    assert c1.post(f"/api/scans/{a}/run", json={}).status_code == 200
    assert wait_job(c1, a)[0]["state"] == "done"


# --------------------------------------------------------------------------- ids / static
@pytest.mark.parametrize("bad", [
    "..", "scan_..", "scan_20261007_141500_ABCD", "scan_20261007_141500_abc",
    "scan_20261007_141500_ab12x", "..%2F..%2Fetc", "%2E%2E", "scan_20261007_141500_ab12%2F..",
])
def test_bad_scan_ids(env, bad):
    c = env.client
    for method, path in (("get", ""), ("get", "/job"), ("get", "/objects"), ("get", "/model.glb"),
                         ("delete", ""), ("post", "/run"), ("post", "/send"), ("post", "/orientation"),
                         ("get", "/video"), ("post", "/video"), ("delete", "/video"),
                         ("post", "/scale"), ("post", "/convert"), ("get", "/labels"), ("post", "/labels"),
                         ("get", "/keyframes"), ("get", "/keyframes/00012.jpg"),
                         ("patch", "/labels/lab_001"), ("delete", "/labels/lab_001"),
                         ("get", "/gaps"), ("post", "/gaps/run")):
        kw = {"json": {"flip": True, "decisions": []}} if method in ("post", "patch") else {}
        r = getattr(c, method)(f"/api/scans/{bad}{path}", **kw)
        assert r.status_code in (404, 405, 422), (method, bad, path, r.status_code)


def test_unknown_valid_id(env):
    assert env.client.get("/api/scans/scan_20261007_141500_ab12").status_code == 404


def test_static_served(env):
    if not (server.STATIC_DIR / "index.html").is_file():
        pytest.skip("UI not present")
    assert env.client.get("/").status_code == 200


def test_sanitize_filename():
    assert server.sanitize_filename("../../a b/c d.USDZ") == "c_d.usdz"
    assert server.sanitize_filename("..\\..\\x.usdz") == "x.usdz"
    assert server.sanitize_filename(".usdz") == "scan.usdz"
    assert server.sanitize_filename("ü.usdz") == "scan.usdz"


# --------------------------------------------------------------------------- real modules
def _real(name):
    sys.modules.pop(name, None)
    try:
        return __import__(name)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"real {name} not importable: {e}")


def test_real_modules_upload_and_flip(tmp_path, monkeypatch):
    if not ROOM_USDZ.is_file():
        pytest.skip(f"{ROOM_USDZ} missing")
    for n in ("scan_prep", "usd_to_gltf"):
        monkeypatch.delitem(sys.modules, n, raising=False)
    sp = _real("scan_prep")
    _real("usd_to_gltf")
    if not hasattr(sp, "prepare_scan") or not hasattr(sp, "set_flip"):
        pytest.skip("scan_prep incomplete")
    app = server.create_app(tmp_path / "out", server.DEFAULT_CONFIG, start_worker=False)
    try:
        c = TestClient(app)
        r = c.post("/api/scans", files={"file": ("room_scan.usdz", ROOM_USDZ.read_bytes())},
                   data={"auto_run": "false"})
        assert r.status_code == 200, r.text
        s = r.json()
        assert s["status"] == "uploaded"
        assert s["orientation"]["flip_recommended"] is True
        assert s["orientation"]["flip_applied"] is True
        sid = s["scan_id"]
        d = tmp_path / "out" / "scans" / sid
        assert (d / "source" / "room_scan.usdz").read_bytes() == ROOM_USDZ.read_bytes()
        g1 = c.get(f"/api/scans/{sid}/model.glb")
        assert g1.status_code == 200 and g1.content[:4] == b"glTF"
        r = c.post(f"/api/scans/{sid}/orientation", json={"flip": False})
        assert r.status_code == 200, r.text
        assert r.json()["orientation"]["flip_applied"] is False
        assert r.json()["orientation"]["source"] == "manual"
        g2 = c.get(f"/api/scans/{sid}/model.glb")
        assert g2.status_code == 200 and g2.content != g1.content
    finally:
        app.state.backend.shutdown()


def test_real_export_json_send(env, monkeypatch):
    monkeypatch.delitem(sys.modules, "export_json", raising=False)
    ej = _real("export_json")
    if not all(hasattr(ej, f) for f in ("build_raw", "build_reviewed", "validate")):
        pytest.skip("export_json incomplete")
    c = env.client
    sid = upload(c, auto_run=False).json()["scan_id"]
    d = env.out / "scans" / sid
    lift = tmp = env.tmp / "lift.json"
    tmp.write_text(json.dumps({"scene_id": "x", "objects": [
        {"class_name": "table", "confidence": 0.6, "num_views": 20, "num_points": 100,
         "position_world": {"x": 1.0, "y": 0.5, "z": -1.0},
         "bounding_box_3d": {"min": [0.5, 0.0, -1.5], "max": [1.5, 0.8, -0.5]}},
        {"class_name": "laptop", "confidence": 0.5, "num_views": 9, "num_points": 50,
         "position_world": {"x": 1.1, "y": 0.8, "z": -1.1},
         "bounding_box_3d": {"min": [1.0, 0.75, -1.2], "max": [1.3, 0.85, -1.0]}}]}))
    rec = json.loads((d / "scan.json").read_text())
    try:
        raw = ej.build_raw(rec, str(lift), "job_x", ["table", "laptop"],
                           {"detector": "d", "segmenter": "s", "renderer": "r", "two_pass": False,
                            "focus_centres": []})
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"build_raw signature/inputs differ: {e}")
    assert ej.validate(raw) == [], ej.validate(raw)
    server.atomic_write_json(d / "results" / "scene_objects_raw.json", raw)
    ids = [o["object_id"] for o in raw["objects"]]
    dec = [{"object_id": ids[0], "decision": "accept", "class_name": "desk"},
           {"object_id": ids[1], "decision": "reject", "class_name": "laptop"}]
    r = c.post(f"/api/scans/{sid}/send", json={"decisions": dec})
    assert r.status_code == 200, r.text
    doc = c.get(f"/api/scans/{sid}/objects").json()
    assert doc["status"] == "reviewed"
    assert [o["class_name"] for o in doc["objects"]] == ["desk"]
    assert ej.validate(doc) == []
