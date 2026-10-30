#!/usr/bin/env python3
"""Two-pass pipeline orchestrator for app v2 (see app/API_V2.md, "pipeline_runner.py").

    python -u pipeline_runner.py --scan-dir out/scans/<id> --job-id <job_id> \
        --prompts "a,b,c" [--config app/pipeline_config.json]

  pass 1   render1 (wide dome) -> detect1 -> lift1
  centres  focus centres from the pass-1 anchor objects (fallback: large objects)
  pass 2   render2 (focus domes only, render_views_lite --mode focus) -> detect2
  merge    pass-1 + pass-2 frames/detections renamed p1_*/p2_* (hard links)
  final    lift over the merged views -> export raw JSON (+ outbox copy)

Stdout protocol (the only lines the server parses; everything else is log, and
stage output is echoed with a "  [stage] " prefix so it can never match):
    PROGRESS <0..1> <stage> <message>     monotonic; stage in render1, detect1,
                                          lift1, render2, detect2, lift, export
    RESULT <path to results/scene_objects_raw.json>    last line, exit 0
    ERROR <message>                                    last line, exit != 0

Everything intermediate goes to <scan>/runs/<job_id>/: pass1/, pass2/, merged/,
logs/<stage>.log, run_info.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import export_json  # noqa: E402

DEFAULT_CONFIG = HERE / "app" / "pipeline_config.json"
MAX_PROMPTS = 64   # same limit as app/server.py
MAX_PROMPT_LEN = 100

# progress windows [start, end) per stage
WINDOWS = {
    "render1": (0.00, 0.06), "detect1": (0.06, 0.30), "lift1": (0.30, 0.33),
    "render2": (0.33, 0.40), "detect2": (0.40, 0.88), "lift": (0.88, 0.96),
    "export": (0.96, 1.00),
}
_RE_VIEW = re.compile(r"\bview (\d+)\b")
_RE_FRAME = re.compile(r"\bframe (\S+): \d+ detections")


class PipelineError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# protocol
# --------------------------------------------------------------------------- #
class Reporter:
    def __init__(self, out=None):
        self.out = out or sys.stdout
        self.last = 0.0

    def _line(self, s: str):
        self.out.write(s + "\n")
        self.out.flush()

    def progress(self, p: float, stage: str, msg: str):
        p = min(1.0, max(self.last, float(p)))
        self.last = p
        msg = " ".join(str(msg).split()) or "-"
        self._line(f"PROGRESS {p:.4f} {stage} {msg}")

    def at(self, stage: str, frac: float, msg: str):
        a, b = WINDOWS[stage]
        self.progress(a + (b - a) * min(1.0, max(0.0, frac)), stage, msg)

    def log(self, msg: str):
        for ln in str(msg).splitlines() or [""]:
            self._line(f"[runner] {ln}")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(path) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    st = cfg.get("stages")
    if not isinstance(st, dict):
        raise PipelineError(f"{path}: missing 'stages' object")
    for name in ("render", "detect", "lift"):
        s = st.get(name)
        if not isinstance(s, dict) or not (s.get("script") or s.get("cmd")):
            raise PipelineError(f"{path}: stages.{name} needs 'script' or 'cmd'")
        if not isinstance(s.get("args", {}), dict):
            raise PipelineError(f"{path}: stages.{name}.args must be an object")
    cfg.setdefault("pass1", {})
    tp = cfg.setdefault("two_pass", {})
    tp.setdefault("enabled", True)
    tp.setdefault("anchor_classes", ["table", "bench seat", "desk", "counter", "shelf",
                                     "sofa", "chair"])
    tp.setdefault("fallback_min_extent_m", 0.4)
    tp.setdefault("merge_distance_m", 1.5)
    tp.setdefault("max_focus", 3)
    tp.setdefault("focus_radius_m", 2.0)
    tp.setdefault("focus_num_views", 30)
    tp.setdefault("render_args", {"--mode": "focus"})
    return cfg


def args_to_cli(args: dict) -> list:
    out = []
    for k, v in (args or {}).items():
        if k.startswith("_") or v is None or v is False:
            continue
        if v is True:
            out.append(k)
        elif isinstance(v, (list, tuple)):
            out.append(f"{k}={','.join(str(x) for x in v)}")
        else:
            out.append(f"{k}={v}")
    return out


def merged_args(*dicts) -> dict:
    """Later dicts override earlier ones; None removes the flag (kept as None)."""
    out: dict = {}
    for d in dicts:
        out.update(d or {})
    return out


def stage_prefix(spec: dict, python: str) -> list:
    if spec.get("cmd"):
        return [python if a == "{python}" else str(a) for a in spec["cmd"]]
    script = Path(spec["script"])
    if not script.is_absolute():
        script = HERE / script
    return [python, "-u", str(script)]


def parse_prompts(text) -> list:
    if isinstance(text, (list, tuple)):
        items = [str(x) for x in text]
    else:
        items = str(text or "").split(",")
    out = []
    for p in items:
        p = " ".join(p.split())
        if p and p not in out:
            out.append(p)
    if not out:
        raise PipelineError("no prompts given")
    if len(out) > MAX_PROMPTS:
        raise PipelineError(f"too many prompts ({len(out)} > {MAX_PROMPTS})")
    for p in out:
        if len(p) > MAX_PROMPT_LEN:
            raise PipelineError(f"prompt too long: {p[:40]!r}...")
    return out


# --------------------------------------------------------------------------- #
# focus centres
# --------------------------------------------------------------------------- #
def _box_centre(o) -> list:
    bb = o["bounding_box_3d"]
    return [0.5 * (a + b) for a, b in zip(bb["min"], bb["max"])]


def _dist(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def choose_focus_centres(objects: list, tp: dict, up_axis: str = "Y") -> tuple[list, str]:
    """Returns ([{"centre": [x,y,z], "members": [...], "views": n}], source) where
    source is 'anchor', 'fallback' or 'none'."""
    anchors = {a.strip().lower() for a in tp.get("anchor_classes") or []}
    cands = [o for o in objects if str(o["class_name"]).strip().lower() in anchors]
    source = "anchor"
    if not cands:
        hidx = [0, 1] if str(up_axis).upper() == "Z" else [0, 2]
        thr = float(tp.get("fallback_min_extent_m", 0.4))
        cands = []
        for o in objects:
            bb = o["bounding_box_3d"]
            if max(bb["max"][i] - bb["min"][i] for i in hidx) >= thr:
                cands.append(o)
        source = "fallback"
    if not cands:
        return [], "none"
    cands.sort(key=lambda o: (-int(o.get("num_views", 0)), -float(o.get("confidence", 0))))
    merge_d = float(tp.get("merge_distance_m", 1.5))
    groups: list[list] = []

    def centre(g):
        w = [max(1e-6, float(o.get("confidence", 0))) for o in g]
        cs = [_box_centre(o) for o in g]
        return [sum(wi * c[k] for wi, c in zip(w, cs)) / sum(w) for k in range(3)]

    for o in cands:
        c = _box_centre(o)
        best = None
        for g in groups:
            d = _dist(c, centre(g))
            if d < merge_d and (best is None or d < best[0]):
                best = (d, g)
        if best:
            best[1].append(o)
        else:
            groups.append([o])
    changed = True                      # merged centres may drift together
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if _dist(centre(groups[i]), centre(groups[j])) < merge_d:
                    groups[i] += groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    groups.sort(key=lambda g: -sum(int(o.get("num_views", 0)) for o in g))
    out = []
    for g in groups[:max(0, int(tp.get("max_focus", 3)))]:
        out.append({"centre": [round(x, 4) for x in centre(g)],
                    "members": [f"{o['class_name']}@{[round(x, 2) for x in _box_centre(o)]}"
                                for o in g],
                    "views": sum(int(o.get("num_views", 0)) for o in g)})
    return out, source


def focus_domes_arg(centres: list, tp: dict) -> str:
    r, n = float(tp["focus_radius_m"]), int(tp["focus_num_views"])
    return ";".join(f"{c['centre'][0]:.4f},{c['centre'][1]:.4f},{c['centre'][2]:.4f},{r},{n}"
                    for c in centres)


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def _link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def merge_passes(passes: list, out_renders: Path, out_dets: Path) -> dict:
    """passes: [(prefix, renders_dir, detections_dir)]. Writes one renders dir and one
    detections dir with frame ids '<prefix>_<frame_id>' and det ids
    '<prefix>_<det_id>'. Files are hard-linked (copied if that fails)."""
    for d in (out_renders, out_dets):
        if d.exists():
            shutil.rmtree(d)
    meta_out, dets_out, fids = None, None, set()
    for prefix, rdir, ddir in passes:
        rdir, ddir = Path(rdir), Path(ddir)
        meta = json.loads((rdir / "meta.json").read_text())
        dets = json.loads((ddir / "detections.json").read_text())
        if meta_out is None:
            meta_out = {k: v for k, v in meta.items() if k not in ("frames", "skipped_views")}
            meta_out.update({"frames": [], "render_mode": "merged", "passes": []})
            dets_out = {k: v for k, v in dets.items() if k != "frames"}
            dets_out["frames"] = []
        else:
            for k in ("image_width", "image_height"):
                if meta.get(k) != meta_out.get(k):
                    meta_out["image_size_varies"] = True
            if dets.get("prompts") != dets_out.get("prompts"):
                raise PipelineError("passes were detected with different prompts")
        meta_out["passes"].append({"prefix": prefix, "renders": str(rdir),
                                   "render_mode": meta.get("render_mode"),
                                   "num_frames": len(meta["frames"])})
        for fr in meta["frames"]:
            fid = f"{prefix}_{fr['frame_id']}"
            if fid in fids:
                raise PipelineError(f"duplicate frame id {fid}")
            fids.add(fid)
            ext = os.path.splitext(fr["rgb"])[1] or ".png"
            _link(rdir / fr["rgb"], out_renders / "rgb" / f"{fid}{ext}")
            _link(rdir / fr["depth"], out_renders / "depth" / f"{fid}.npy")
            meta_out["frames"].append({**fr, "frame_id": fid, "rgb": f"rgb/{fid}{ext}",
                                       "depth": f"depth/{fid}.npy", "pass": prefix,
                                       "source_frame_id": fr["frame_id"]})
        for fr in dets.get("frames", []):
            fid = f"{prefix}_{fr['frame_id']}"
            recs = []
            for d in fr.get("detections", []):
                did = f"{prefix}_{d['det_id']}"
                mask = None
                if d.get("mask"):
                    mask = f"masks/{did}.png"
                    _link(ddir / d["mask"], out_dets / mask)
                recs.append({**d, "det_id": did, "mask": mask})
            dets_out["frames"].append({"frame_id": fid, "detections": recs})
    out_renders.mkdir(parents=True, exist_ok=True)
    out_dets.mkdir(parents=True, exist_ok=True)
    export_json.write_json(out_renders / "meta.json", meta_out)
    export_json.write_json(out_dets / "detections.json", dets_out)
    return {"num_frames": len(meta_out["frames"]),
            "num_detections": sum(len(f["detections"]) for f in dets_out["frames"])}


# --------------------------------------------------------------------------- #
# stage execution
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, rep: Reporter, logs_dir: Path, python: str):
        self.rep = rep
        self.logs_dir = logs_dir
        self.python = python
        self.child: subprocess.Popen | None = None
        self.timings: dict = {}
        self.commands: dict = {}

    def run(self, stage: str, cmd: list, *, total: int = 0, regex=None, label: str = ""):
        log_path = self.logs_dir / f"{stage}.log"
        self.commands[stage] = cmd
        self.rep.log(f"{stage}: {' '.join(cmd)}")
        self.rep.at(stage, 0.0, f"{label or stage} started")
        t0 = time.time()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        tail: list = []
        done = 0
        with open(log_path, "w") as lf:
            lf.write("$ " + " ".join(cmd) + "\n")
            lf.flush()
            self.child = subprocess.Popen(cmd, cwd=str(HERE), stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True, bufsize=1,
                                          env=env, errors="replace")
            for line in self.child.stdout:
                lf.write(line)
                line = line.rstrip("\n")
                print(f"  [{stage}] {line}", flush=True)
                tail = (tail + [line])[-15:]
                if regex is not None and total > 0 and regex.search(line):
                    done += 1
                    self.rep.at(stage, done / total, f"{label or stage} {min(done, total)}/{total}")
            rc = self.child.wait()
            self.child = None
            dt = time.time() - t0
            lf.write(f"\n# exit {rc} after {dt:.1f}s\n")
        self.timings[stage] = round(dt, 1)
        if rc != 0:
            last = next((t for t in reversed(tail) if t.strip()), "")
            raise PipelineError(f"{stage} failed (exit {rc}) after {dt:.0f}s: {last[:200]} "
                                f"(log: {log_path})")
        self.rep.at(stage, 1.0, f"{label or stage} done in {dt:.0f}s")
        return dt

    def kill(self):
        c = self.child
        if c is not None and c.poll() is None:
            c.terminate()
            try:
                c.wait(timeout=10)
            except subprocess.TimeoutExpired:
                c.kill()


def _render_cmd(cfg, python, usd, out_dir, extra: dict, focus: str | None = None) -> list:
    st = cfg["stages"]["render"]
    args = merged_args(st.get("args"), extra)
    for k in ("--usd", "--out", "--focus-domes"):
        args.pop(k, None)
    cmd = stage_prefix(st, python) + [f"--usd={usd}", f"--out={out_dir}"] + args_to_cli(args)
    if focus:
        cmd.append(f"--focus-domes={focus}")
    return cmd


def _detect_cmd(cfg, python, renders, out_dir, prompts) -> list:
    st = cfg["stages"]["detect"]
    args = merged_args(st.get("args"))
    for k in ("--renders", "--out", "--prompts", "--prompts-file"):
        args.pop(k, None)
    return (stage_prefix(st, python) + [f"--renders={renders}", f"--out={out_dir}",
                                        f"--prompts={','.join(prompts)}"] + args_to_cli(args))


def _lift_cmd(cfg, python, renders, dets, out_json, ply, extra: dict) -> list:
    st = cfg["stages"]["lift"]
    args = merged_args(st.get("args"), extra)
    for k in ("--renders", "--detections", "--out"):
        args.pop(k, None)
    if args.get("--save-ply") is True:
        args["--save-ply"] = str(ply)
    return (stage_prefix(st, python) + [f"--renders={renders}", f"--detections={dets}",
                                        f"--out={out_json}"] + args_to_cli(args))


def _num_frames(renders: Path) -> int:
    meta = json.loads((renders / "meta.json").read_text())
    n = len(meta.get("frames", []))
    if n == 0:
        raise PipelineError(f"{renders}: no frames rendered")
    return n


def _expect(path: Path, stage: str):
    if not path.is_file():
        raise PipelineError(f"{stage} finished but {path} is missing")


def run_pipeline(scan_dir, job_id: str, prompts, config_path=DEFAULT_CONFIG,
                 rep: Reporter | None = None, python: str | None = None,
                 runner_holder: list | None = None) -> Path:
    rep = rep or Reporter()
    python = python or sys.executable
    scan_dir = Path(scan_dir).resolve()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", job_id or ""):
        raise PipelineError(f"invalid job id {job_id!r}")
    rec_path = scan_dir / "scan.json"
    if not rec_path.is_file():
        raise PipelineError(f"{rec_path} not found")
    stage_usd = scan_dir / "scene.usda"
    if not stage_usd.is_file():
        raise PipelineError(f"{stage_usd} not found")
    rec = json.loads(rec_path.read_text())
    prompts = parse_prompts(prompts)
    cfg = load_config(config_path)
    tp = cfg["two_pass"]
    p1 = cfg.get("pass1") or {}

    run_dir = scan_dir / "runs" / job_id
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    export_json.write_json(run_dir / "pipeline_config.used.json", cfg)
    r = Runner(rep, logs, python)
    if runner_holder is not None:
        runner_holder.append(r)
    info: dict = {"job_id": job_id, "scan_id": rec.get("scan_id"), "prompts": prompts,
                  "started_at": export_json.now_iso(), "config": str(config_path)}
    t_start = time.time()

    def save_info(**kw):
        info.update(kw, stage_seconds=r.timings, commands=r.commands)
        export_json.write_json(run_dir / "run_info.json", info)

    try:
        # ---------------- pass 1 ----------------
        P1 = run_dir / "pass1"
        n1_expected = int(merged_args((cfg["stages"]["render"].get("args")),
                                      p1.get("render_args")).get("--num-views") or 0)
        r.run("render1", _render_cmd(cfg, python, stage_usd, P1 / "renders",
                                     p1.get("render_args")),
              total=n1_expected, regex=_RE_VIEW, label="pass-1 render")
        n1 = _num_frames(P1 / "renders")
        r.run("detect1", _detect_cmd(cfg, python, P1 / "renders", P1 / "detections", prompts),
              total=n1, regex=_RE_FRAME, label="pass-1 detect")
        _expect(P1 / "detections" / "detections.json", "detect1")
        r.run("lift1", _lift_cmd(cfg, python, P1 / "renders", P1 / "detections",
                                 P1 / "scene_objects.json", P1 / "clusters.ply",
                                 p1.get("lift_args")), label="pass-1 lift")
        _expect(P1 / "scene_objects.json", "lift1")
        objs1 = json.loads((P1 / "scene_objects.json").read_text()).get("objects", [])
        meta1 = json.loads((P1 / "renders" / "meta.json").read_text())

        # ---------------- focus centres + pass 2 ----------------
        if tp.get("enabled", True):
            centres, csrc = choose_focus_centres(objs1, tp, meta1.get("up_axis", "Y"))
        else:
            centres, csrc = [], "disabled"
        skip_reason = None
        if not tp.get("enabled", True):
            skip_reason = "two_pass disabled in the config"
        elif not centres:
            skip_reason = "pass 1 found no anchor objects"
        rep.log(f"pass-1: {len(objs1)} objects; focus centres ({csrc}): "
                + (json.dumps(centres) if centres else "none"))
        passes = [("p1", P1 / "renders", P1 / "detections")]
        n2 = 0
        if skip_reason is None:
            P2 = run_dir / "pass2"
            fd = focus_domes_arg(centres, tp)
            r.run("render2", _render_cmd(cfg, python, stage_usd, P2 / "renders",
                                         tp.get("render_args"), focus=fd),
                  total=len(centres) * int(tp["focus_num_views"]), regex=_RE_VIEW,
                  label=f"pass-2 render ({len(centres)} focus dome"
                        f"{'s' if len(centres) != 1 else ''})")
            n2 = _num_frames(P2 / "renders")
            r.run("detect2", _detect_cmd(cfg, python, P2 / "renders", P2 / "detections",
                                         prompts),
                  total=n2, regex=_RE_FRAME, label="pass-2 detect")
            _expect(P2 / "detections" / "detections.json", "detect2")
            passes.append(("p2", P2 / "renders", P2 / "detections"))
        else:
            rep.at("detect2", 1.0, f"pass 2 skipped: {skip_reason}")

        # ---------------- merge + final lift ----------------
        M = run_dir / "merged"
        mstat = merge_passes(passes, M / "renders", M / "detections")
        rep.log(f"merged: {mstat['num_frames']} frames, {mstat['num_detections']} detections")
        r.run("lift", _lift_cmd(cfg, python, M / "renders", M / "detections",
                                M / "scene_objects.json", M / "clusters.ply",
                                tp.get("lift_args") if skip_reason is None else None),
              label="final lift")
        _expect(M / "scene_objects.json", "lift")

        # ---------------- export ----------------
        rep.at("export", 0.0, "writing raw JSON")
        t0 = time.time()
        rec = json.loads(rec_path.read_text())          # re-read: orientation may have changed
        dets_meta = json.loads((M / "detections" / "detections.json").read_text())
        pinfo = {
            "detector": dets_meta.get("detector") or export_json.DEFAULT_PIPELINE["detector"],
            "segmenter": dets_meta.get("segmenter"),
            "renderer": meta1.get("renderer", "open3d-raycast"),
            "two_pass": skip_reason is None,
            "focus_centres": [c["centre"] for c in centres] if skip_reason is None else [],
            "focus_centre_source": csrc,
            "pass2_skipped_reason": skip_reason,
            "num_frames": {"pass1": n1, "pass2": n2},
        }
        doc = export_json.build_raw(rec, M / "scene_objects.json", job_id, prompts, pinfo)
        doc["scan"]["pipeline"]["stage_seconds"] = dict(r.timings)
        problems = export_json.validate(doc)
        if problems:
            raise PipelineError("export failed validation: " + "; ".join(problems[:5]))
        out = scan_dir / "results" / "scene_objects_raw.json"
        export_json.write_json(out, doc)
        outbox = scan_dir.parent.parent / "backend_outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        export_json.write_json(outbox / f"{rec['scan_id']}_raw.json", doc)
        r.timings["export"] = round(time.time() - t0, 1)
        save_info(state="done", finished_at=export_json.now_iso(),
                  total_seconds=round(time.time() - t_start, 1), focus_centres=centres,
                  focus_centre_source=csrc, pass2_skipped_reason=skip_reason,
                  num_objects=len(doc["objects"]), result=str(out))
        msg = f"{len(doc['objects'])} objects"
        if skip_reason:
            msg += f" (pass 2 skipped: {skip_reason})"
        rep.at("export", 1.0, msg)
        return out
    except BaseException as e:
        try:
            save_info(state="error", error=str(e), finished_at=export_json.now_iso())
        except Exception:
            pass
        raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scan-dir", required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--prompts", required=True, help="comma-separated")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    a = ap.parse_args(argv)
    rep = Reporter()
    holder: list = []

    def on_term(signum, _frame):
        for r in holder:
            r.kill()
        raise PipelineError(f"cancelled (signal {signum})")

    signal.signal(signal.SIGTERM, on_term)
    try:
        out = run_pipeline(a.scan_dir, a.job_id, a.prompts, a.config, rep, runner_holder=holder)
    except KeyboardInterrupt:
        for r in holder:
            r.kill()
        print("ERROR cancelled", flush=True)
        return 130
    except Exception as e:
        for r in holder:
            r.kill()
        msg = " ".join(str(e).split()) or type(e).__name__
        print(f"ERROR {msg}", flush=True)
        return 1
    print(f"RESULT {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
