#!/usr/bin/env python3
"""open_vocab_3d app v2 backend: upload .usdz -> prepare -> pipeline -> raw/reviewed JSON.

Contract: app/API_V2.md ("Server HTTP API v2").

    .venv/bin/python app/server.py --out out --port 8000 [--host 127.0.0.1] \
        [--config app/pipeline_config.json] [--max-upload-mb 500]

Relative --out/--config paths resolve against open_vocab_3d/, not the CWD.
No auth: binds to 127.0.0.1 by default.

Design notes
- out/scans/<scan_id>/scan.json is the source of truth for each scan; every
  write is a read-modify-write under one lock, written atomically, so fields
  written by scan_prep / pipeline_runner are preserved.
- One worker thread runs pipeline jobs FIFO, one at a time. A job is ONE
  subprocess (pipeline_runner.py); only its PROGRESS/RESULT/ERROR stdout lines
  are interpreted, everything is logged to runs/<job_id>/runner.log.
- scan_prep / export_json / usd_to_gltf are imported lazily (sys.modules
  lookups), so the server starts without them and tests can inject fakes.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent  # open_vocab_3d/
APP_DIR = BASE_DIR / "app"
STATIC_DIR = APP_DIR / "static"
DEFAULT_CONFIG = APP_DIR / "pipeline_config.json"

SCAN_ID_RE = re.compile(r"^scan_\d{8}_\d{6}_[0-9a-f]{4}$")
JOB_ID_RE = re.compile(r"^job_\d{8}_\d{6}_[0-9a-f]{6}$")
ZIP_MAGIC = b"PK\x03\x04"
LOG_TAIL = 30
ACTIVE = ("queued", "running")
MULTIPART_SLACK = 1024 * 1024        # form overhead allowed on top of the file cap
MAX_TEXT_FIELD = 64 * 1024
MAX_PROMPTS = 64
MAX_PROMPT_LEN = 100

JOB_KINDS = ("pipeline", "convert", "gaps")
GAPS_NAME = "gaps.json"
RAW_NAME = "scene_objects_raw.json"
REVIEWED_NAME = "scene_objects_reviewed.json"

# Optional capture video per scan (the footage the scan was made from); shown in the UI only,
# never read by the pipeline. Browser-playable containers only; stored as source/video<ext>.
VIDEO_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm"}

# Store videos that become shelf scans (converted by shelf_convert.py). .mov is allowed here
# (iPhone originals, often HEVC): it is converted, never played directly.
SHELF_VIDEO_EXTS = (".mov", ".mp4", ".m4v")
DEFAULT_VIDEO_TO_3D = BASE_DIR.parent / "video_to_3d"
DEFAULT_SHELF_HEIGHT_M = 1.9   # provisional scale until the scan is calibrated

# Fields of scan.json owned by the server (kept when scan_prep returns a record).
SERVER_KEYS = ("status", "progress", "job_stage", "message", "job_id", "prompts", "stale",
               "num_objects", "error", "glb_error", "reviewed_at", "queued_at", "updated_at")

# Test hook: if set, called as PIPELINE_CMD_HOOK(default_argv, ctx) -> argv.
# ctx = {"scan_dir", "job_id", "prompts", "config"}.
PIPELINE_CMD_HOOK: Optional[Callable[[list, dict], list]] = None

RE_PROGRESS = re.compile(r"^PROGRESS\s+(\S+)\s+(\S+)(?:\s+(.*))?$")
RE_RESULT = re.compile(r"^RESULT\s+(.+)$")
RE_ERROR = re.compile(r"^ERROR\s*(.*)$")


# --------------------------------------------------------------------------- helpers
def resolve_path(p: str | os.PathLike) -> Path:
    p = Path(p).expanduser()
    return (p if p.is_absolute() else BASE_DIR / p).resolve()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_module(name: str):
    """Lazy import of a sibling pipeline module (scan_prep, export_json, usd_to_gltf)."""
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    try:
        return importlib.import_module(name)
    except Exception as e:  # ImportError or a broken module
        raise HTTPException(503, detail=f"{name} unavailable ({type(e).__name__}: {e})")


def sanitize_filename(name: Optional[str]) -> str:
    """Client filename -> safe '<stem>.usdz' (never used as a path as-is)."""
    base = re.split(r"[\\/]", name or "")[-1]
    stem = base[:-5] if base.lower().endswith(".usdz") else base
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:100]
    return (stem or "scan") + ".usdz"


def video_ext(name: Optional[str]) -> str:
    """Lower-case extension of the client filename's basename ('' if none)."""
    return Path(re.split(r"[\\/]", name or "")[-1]).suffix.lower()


def sniff_video(head: bytes, ext: str) -> bool:
    """Container magic matches the extension: ISO-BMFF 'ftyp' box for mp4/m4v, EBML for webm;
    QuickTime .mov starts with an ftyp/moov/mdat/wide/free atom."""
    if ext in (".mp4", ".m4v"):
        return head[4:8] == b"ftyp"
    if ext == ".mov":
        return head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip")
    if ext == ".webm":
        return head[:4] == b"\x1a\x45\xdf\xa3"
    return False


def clean_prompts(value: Any, default: list) -> list[str]:
    """None/empty -> default; comma string or list of strings -> validated list."""
    if value is None:
        items: list = []
    elif isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list):
        if not all(isinstance(p, str) for p in value):
            raise HTTPException(422, detail="prompts must be strings")
        items = value
    else:
        raise HTTPException(422, detail="prompts must be a list of strings or a comma-separated string")
    out: list[str] = []
    for p in items:
        p = " ".join(p.split())
        if p and p not in out:
            out.append(p)
    if not out:
        out = list(default or [])
        if not out:
            raise HTTPException(422, detail="no prompts given and the config has no default prompts")
    if len(out) > MAX_PROMPTS:
        raise HTTPException(422, detail=f"at most {MAX_PROMPTS} prompts")
    for p in out:
        if "," in p:
            raise HTTPException(422, detail=f"prompt {p!r} must not contain a comma")
        if p.startswith("-"):
            raise HTTPException(422, detail=f"prompt {p!r} must not start with '-'")
        if len(p) > MAX_PROMPT_LEN or any(ord(c) < 32 for c in p):
            raise HTTPException(422, detail=f"prompt {p[:40]!r} is too long or has control characters")
    return out


def parse_bool(v: Any, name: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise HTTPException(422, detail=f"{name} must be true or false")


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# --------------------------------------------------------------------------- jobs
class Job:
    """kind: "pipeline" (pipeline_runner.py: detection -> raw JSON), "convert"
    (shelf_convert.py: shelf video -> textured usdz) or "gaps" (gaps_runner.py: empty-slot
    detection on a shelf scan's keyframes -> results/gaps.json)."""

    def __init__(self, job_id: str, scan_id: str, scan_dir: Path, prompts: list[str],
                 kind: str = "pipeline"):
        self.job_id = job_id
        self.scan_id = scan_id
        self.scan_dir = scan_dir
        self.prompts = prompts
        self.kind = kind
        self.state = "queued"
        self.progress = 0.0
        self.stage: Optional[str] = None
        self.message = "queued"
        self.log: deque[str] = deque(maxlen=LOG_TAIL)
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.pid: Optional[int] = None
        self.result_path: Optional[str] = None
        self.lock = threading.RLock()

    @property
    def run_dir(self) -> Path:
        return self.scan_dir / "runs" / self.job_id

    def to_dict(self) -> dict:
        with self.lock:
            return {"job_id": self.job_id, "scan_id": self.scan_id, "kind": self.kind, "state": self.state,
                    "progress": round(self.progress, 4), "stage": self.stage,
                    "message": self.message, "log_tail": list(self.log),
                    "prompts": self.prompts, "created": self.created, "started": self.started,
                    "finished": self.finished, "pid": self.pid, "result_path": self.result_path}

    @classmethod
    def from_dict(cls, d: dict, scan_dir: Path) -> "Job":
        j = cls(d["job_id"], d["scan_id"], scan_dir, list(d.get("prompts") or []),
                d.get("kind") if d.get("kind") in JOB_KINDS else "pipeline")
        j.state = d.get("state", "error")
        j.progress = float(d.get("progress") or 0.0)
        j.stage, j.message = d.get("stage"), d.get("message", "")
        j.log.extend(d.get("log_tail") or [])
        j.created = d.get("created") or 0.0
        j.started, j.finished = d.get("started"), d.get("finished")
        j.pid, j.result_path = d.get("pid"), d.get("result_path")
        return j

    def persist(self) -> None:
        try:
            atomic_write_json(self.run_dir / "job.json", self.to_dict())
        except OSError:
            pass


class Backend:
    def __init__(self, out_dir: Path, config_path: Path = DEFAULT_CONFIG,
                 max_upload_mb: float = 500, start_worker: bool = True,
                 video_to_3d: Path = DEFAULT_VIDEO_TO_3D, shelf_height_m: float = DEFAULT_SHELF_HEIGHT_M):
        self.out = Path(out_dir).resolve()
        self.video_to_3d = Path(video_to_3d).resolve()
        self.shelf_height_m = float(shelf_height_m)
        self.scans_root = self.out / "scans"
        self.outbox = self.out / "backend_outbox"
        self.uploads = self.scans_root / ".uploads"
        self.config_path = Path(config_path).resolve()
        self.max_upload_mb = max_upload_mb
        self.max_upload_bytes = int(max_upload_mb * 1024 * 1024)
        self.scans_root.mkdir(parents=True, exist_ok=True)

        self.lock = threading.RLock()          # registry, queue, scan.json writes
        self.cv = threading.Condition(self.lock)
        self.upload_lock = threading.Lock()    # serialises prepare_scan
        self.glb_lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.queue: deque[str] = deque()
        self.current: Optional[Job] = None
        self.proc: Optional[subprocess.Popen] = None
        self._stopping = False

        shutil.rmtree(self.uploads, ignore_errors=True)
        self._recover()
        self.worker: Optional[threading.Thread] = None
        if start_worker:
            self.worker = threading.Thread(target=self._worker_loop, daemon=True, name="pipeline-worker")
            self.worker.start()

    # ---------------------------------------------------------------- config
    def load_config(self) -> dict:
        try:
            cfg = read_json(self.config_path)
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(500, detail=f"cannot read config {self.config_path}: {e}")
        if not isinstance(cfg, dict):
            raise HTTPException(500, detail=f"config {self.config_path} is not a JSON object")
        return cfg

    def default_prompts(self) -> list[str]:
        p = self.load_config().get("prompts") or []
        return [s for s in p if isinstance(s, str) and s.strip()]

    # ---------------------------------------------------------------- registry
    def scan_dir(self, scan_id: str) -> Path:
        """Validated scan dir; 404 for malformed ids (path traversal) or unknown scans."""
        if not isinstance(scan_id, str) or not SCAN_ID_RE.fullmatch(scan_id):
            raise HTTPException(404, detail="unknown scan")
        d = self.scans_root / scan_id
        if not (d / "scan.json").is_file():
            raise HTTPException(404, detail="unknown scan")
        return d

    def read_scan(self, scan_id: str) -> dict:
        d = self.scan_dir(scan_id)
        try:
            rec = read_json(d / "scan.json")
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(500, detail=f"scan.json unreadable: {e}")
        return rec if isinstance(rec, dict) else {}

    def update_scan(self, scan_id: str, **fields) -> dict:
        """Read-modify-write scan.json atomically (keeps fields written by other modules)."""
        with self.lock:
            path = self.scans_root / scan_id / "scan.json"
            try:
                rec = read_json(path)
                if not isinstance(rec, dict):
                    rec = {}
            except (OSError, json.JSONDecodeError):
                rec = {}
            rec.update(fields)
            rec["scan_id"] = scan_id
            rec["updated_at"] = now_iso()
            atomic_write_json(path, rec)
            return rec

    def active_job(self, scan_id: str) -> Optional[Job]:
        with self.lock:
            for j in self.jobs.values():
                if j.scan_id == scan_id and j.state in ACTIVE:
                    return j
        return None

    def latest_job(self, scan_id: str, rec: Optional[dict] = None) -> Optional[Job]:
        with self.lock:
            jid = (rec or {}).get("job_id")
            if jid and jid in self.jobs:
                return self.jobs[jid]
            cands = [j for j in self.jobs.values() if j.scan_id == scan_id]
            return max(cands, key=lambda j: j.created) if cands else None

    def queue_position(self, job: Job) -> Optional[int]:
        """1-based position among waiting jobs (1 = runs next); None unless queued."""
        with self.lock:
            if job.state != "queued":
                return None
            try:
                return list(self.queue).index(job.job_id) + 1
            except ValueError:
                return None

    def job_public(self, job: Job) -> dict:
        d = job.to_dict()
        out = {k: d[k] for k in ("job_id", "kind", "state", "progress", "stage", "message", "log_tail",
                                 "prompts", "created", "started", "finished")}
        out["queue_position"] = self.queue_position(job)
        return out

    @staticmethod
    def _orientation(rec: dict) -> dict:
        o = rec.get("orientation") if isinstance(rec.get("orientation"), dict) else {}
        det = o.get("detected") if isinstance(o.get("detected"), dict) else {}
        def pick(*keys, default=None):
            for src in (o, det, rec):
                for k in keys:
                    if k in src and src[k] is not None:
                        return src[k]
            return default
        return {"flip_applied": bool(pick("flip_applied", "flip", default=False)),
                "source": pick("source", "orientation_source"),
                "flip_recommended": pick("flip_recommended"),
                "confidence": pick("confidence")}

    def _live(self, scan_id: str, rec: dict) -> dict:
        """status/progress/message overlaid from the in-memory job (fresher than scan.json)."""
        st = {"status": rec.get("status", "uploaded"), "progress": rec.get("progress", 0.0),
              "message": rec.get("message", ""), "job_stage": rec.get("job_stage")}
        job = self.latest_job(scan_id, rec)
        if job is not None and job.state in ACTIVE and job.job_id == rec.get("job_id"):
            with job.lock:
                st.update(status=job.state, progress=round(job.progress, 4),
                          message=job.message, job_stage=job.stage)
        return st

    def summary(self, scan_id: str, rec: Optional[dict] = None) -> dict:
        rec = rec if rec is not None else self.read_scan(scan_id)
        d = self.scans_root / scan_id
        live = self._live(scan_id, rec)
        return {
            "scan_id": scan_id,
            "source_file": rec.get("source_file") or rec.get("original_name"),
            "uploaded_at": rec.get("uploaded_at"),
            "status": live["status"],
            "progress": live["progress"],
            "message": live["message"],
            "num_objects": rec.get("num_objects"),
            "has_raw": (d / "results" / RAW_NAME).is_file(),
            "has_reviewed": (d / "results" / REVIEWED_NAME).is_file(),
            "stale": bool(rec.get("stale", False)),
            "job_id": rec.get("job_id"),
            "orientation": self._orientation(rec),
            "has_video": self.video_path(scan_id, rec) is not None,
            "scan_type": rec.get("scan_type") or "room",
            "has_model": (d / "scene.usda").is_file(),
            "has_gaps": (d / "results" / GAPS_NAME).is_file(),
            "scale_status": (rec.get("scale") or {}).get("status") if isinstance(rec.get("scale"), dict) else None,
        }

    def list_scans(self) -> list[dict]:
        out = []
        for d in self.scans_root.iterdir():
            if not (SCAN_ID_RE.fullmatch(d.name) and (d / "scan.json").is_file()):
                continue
            try:
                rec = read_json(d / "scan.json")
                out.append(self.summary(d.name, rec if isinstance(rec, dict) else {}))
            except (OSError, json.JSONDecodeError):
                continue
        out.sort(key=lambda s: (s.get("uploaded_at") or "", s["scan_id"]), reverse=True)
        return out

    def full_record(self, scan_id: str) -> dict:
        rec = self.read_scan(scan_id)
        d = self.scans_root / scan_id
        out = dict(rec)
        out.update(self._live(scan_id, rec))
        out["has_raw"] = (d / "results" / RAW_NAME).is_file()
        out["has_reviewed"] = (d / "results" / REVIEWED_NAME).is_file()
        out["stale"] = bool(rec.get("stale", False))
        out["orientation_summary"] = self._orientation(rec)
        out["model_url"] = f"/api/scans/{scan_id}/model.glb"
        out["source_video"] = self.video_public(scan_id, rec)
        out["scan_type"] = rec.get("scan_type") or "room"
        job = self.latest_job(scan_id, rec)
        out["job"] = self.job_public(job) if job else None
        return out

    # ---------------------------------------------------------------- upload
    def create_scan(self, upload_path: Path, original_name: str, prompts: list[str],
                    auto_run: bool) -> dict:
        scan_prep = load_module("scan_prep")
        with self.upload_lock:
            before = {p.name for p in self.scans_root.iterdir()}
            try:
                kw = {}
                try:
                    if "max_upload_bytes" in inspect.signature(scan_prep.prepare_scan).parameters:
                        kw["max_upload_bytes"] = self.max_upload_bytes
                except (TypeError, ValueError):
                    pass
                rec = scan_prep.prepare_scan(str(upload_path), str(self.scans_root), original_name, **kw)
            except Exception as e:  # noqa: BLE001  invalid usdz, unsafe zip, no mesh, ...
                for p in self.scans_root.iterdir():
                    if p.name not in before and SCAN_ID_RE.fullmatch(p.name):
                        shutil.rmtree(p, ignore_errors=True)
                code = 413 if type(e).__name__ == "ScanTooLargeError" else 400
                raise HTTPException(code, detail=f"invalid scan: {e}")
            finally:
                if upload_path.exists():
                    upload_path.unlink()
        if not isinstance(rec, dict) or not SCAN_ID_RE.fullmatch(str(rec.get("scan_id", ""))):
            raise HTTPException(500, detail="prepare_scan returned no valid scan_id")
        scan_id = rec["scan_id"]
        sdir = self.scans_root / scan_id
        if not sdir.is_dir():
            raise HTTPException(500, detail=f"prepare_scan did not create {sdir}")
        with self.lock:
            cur = {}
            if (sdir / "scan.json").is_file():
                try:
                    cur = read_json(sdir / "scan.json")
                except (OSError, json.JSONDecodeError):
                    cur = {}
            merged = {**cur, **rec}
            merged.setdefault("source_file", original_name)
            merged.setdefault("uploaded_at", now_iso())
            merged.update(status="uploaded", progress=0.0, job_stage=None, message="uploaded",
                          prompts=prompts, stale=False, num_objects=None, error=None,
                          scan_id=scan_id, updated_at=now_iso())
            atomic_write_json(sdir / "scan.json", merged)
        try:
            self.build_glb(scan_id, force=True)
            self.update_scan(scan_id, glb_error=None)
        except HTTPException as e:
            self.update_scan(scan_id, glb_error=str(e.detail))
        if auto_run:
            self.enqueue(scan_id, prompts)
        return self.summary(scan_id)

    # ---------------------------------------------------------------- shelf scans (store video)
    def _new_scan_dir(self) -> tuple[str, Path]:
        for _ in range(50):
            sid = f"scan_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
            d = self.scans_root / sid
            try:
                d.mkdir()
                return sid, d
            except FileExistsError:
                continue
        raise HTTPException(500, detail="could not allocate a scan id")

    def create_shelf_scan(self, upload_path: Path, original_name: str) -> dict:
        """A store video becomes a shelf scan: keep the video in source/, queue a convert job.
        The 3D model (scene.usda, GLB) only exists once the convert job has finished."""
        import hashlib
        ext = video_ext(original_name)
        with open(upload_path, "rb") as f:
            head = f.read(12)
        if not sniff_video(head, ext):
            raise HTTPException(400, detail=f"not a valid {ext} video")
        base = re.split(r"[\\/]", original_name)[-1]
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", base[: -len(ext)]).strip("._-")[:100] or "video"
        name = stem + ext
        h = hashlib.sha256()
        with open(upload_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        scan_id, sdir = self._new_scan_dir()
        try:
            (sdir / "source").mkdir()
            size = upload_path.stat().st_size
            os.replace(upload_path, sdir / "source" / name)
            rec = {"schema": 1, "scan_id": scan_id, "scan_type": "shelf",
                   "source_file": name, "source_path": f"source/{name}", "source_bytes": size,
                   "source_sha256": h.hexdigest(), "uploaded_at": now_iso(),
                   "scale": self.default_scale(),
                   "status": "uploaded", "progress": 0.0, "message": "uploaded", "stale": False,
                   "num_objects": None, "updated_at": now_iso()}
            atomic_write_json(sdir / "scan.json", rec)
        except BaseException:
            shutil.rmtree(sdir, ignore_errors=True)
            raise
        self.enqueue(scan_id, [], kind="convert")
        return self.summary(scan_id)

    def default_scale(self) -> dict:
        """Provisional scale of a converted shelf scan (no calibration yet)."""
        return {"status": "approximate", "method": "assumed_height",
                "assumed_height_m": self.shelf_height_m, "factor": 1.0}

    def set_scale(self, scan_id: str, body: dict) -> dict:
        """Calibrate a shelf scan from two points the user clicked on the model and their real
        distance ({"method": "measured", "p1": [x,y,z], "p2": [x,y,z], "real_cm": 120}), or
        go back to the provisional scale ({"method": "reset"}). Points are in the current
        scene frame (metres), i.e. what the viewer shows."""
        sdir = self.scan_dir(scan_id)
        rec = self.read_scan(scan_id)
        if rec.get("scan_type") != "shelf":
            raise HTTPException(422, detail="scale calibration is for shelf scans")
        if not (sdir / "scene.usda").is_file():
            raise HTTPException(409, detail="this scan has no 3D model yet (conversion not finished)")
        method = body.get("method")
        cur = (rec.get("scale") or {}).get("factor") if isinstance(rec.get("scale"), dict) else None
        cur = float(cur) if _num(cur) and cur > 0 else 1.0
        if method == "reset":
            scale = self.default_scale()
        elif method == "measured":
            p1, p2, real_cm = body.get("p1"), body.get("p2"), body.get("real_cm")
            if not all(isinstance(p, list) and len(p) == 3 and all(_num(c) for c in p) for p in (p1, p2)):
                raise HTTPException(422, detail="p1 and p2 must be [x, y, z] numbers")
            if not (_num(real_cm) and 0.5 <= real_cm <= 100000):
                raise HTTPException(422, detail="real_cm must be a number between 0.5 and 100000")
            dist = math.dist(p1, p2)
            if dist < 0.005:
                raise HTTPException(422, detail="the two points are too close together")
            factor = cur * (real_cm / 100.0) / dist
            if not (0.02 <= factor <= 50):
                raise HTTPException(422, detail=f"implausible scale change (x{factor / cur:.3g})")
            scale = {"status": "calibrated", "method": "measured", "factor": round(factor, 6),
                     "measured": {"p1": [round(c, 5) for c in p1], "p2": [round(c, 5) for c in p2],
                                  "scene_distance_m": round(dist, 5), "real_cm": real_cm,
                                  "previous_factor": cur},
                     "updated_at": now_iso()}
        else:
            raise HTTPException(422, detail='method must be "measured" or "reset"')
        scan_prep = load_module("scan_prep")
        with self.lock:
            if self.active_job(scan_id):
                raise HTTPException(409, detail="a job for this scan is queued or running")
            try:
                scan_prep.set_scale(str(sdir), scale)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(500, detail=f"set_scale failed: {type(e).__name__}: {e}")
            if any((sdir / "results" / n).is_file() for n in (RAW_NAME, REVIEWED_NAME)):
                self.update_scan(scan_id, stale=True)
        try:
            self.build_glb(scan_id, force=True)
            self.update_scan(scan_id, glb_error=None)
        except HTTPException as e:
            self.update_scan(scan_id, glb_error=str(e.detail))
            raise
        return {**self.summary(scan_id), "scale": scale}

    # ---------------------------------------------------------------- shelf labels (ground truth)
    def _shelf(self, scan_id: str) -> tuple[Path, dict]:
        sdir = self.scan_dir(scan_id)
        rec = self.read_scan(scan_id)
        if rec.get("scan_type") != "shelf":
            raise HTTPException(422, detail="empty-slot labels are for shelf scans")
        return sdir, rec

    def _cameras(self, sdir: Path) -> dict:
        cams = load_module("shelf_labels").load_cameras(sdir)
        if cams is None:
            raise HTTPException(409, detail="this scan has no keyframe cameras yet (convert it first, "
                                            "or re-convert scans made before labelling existed)")
        return cams

    def keyframes(self, scan_id: str) -> dict:
        sdir, _ = self._shelf(scan_id)
        cams = self._cameras(sdir)
        c = cams["camera"]
        return {"width": c["width"], "height": c["height"], "fps": cams.get("fps"),
                "keyframes": [{"name": k["name"], "frame": k["frame"], "time_s": k["time_s"]}
                              for k in cams["keyframes"]]}

    def keyframe_image(self, scan_id: str, name: str) -> Path:
        if not re.fullmatch(r"\d{5}", name or ""):
            raise HTTPException(404, detail="unknown keyframe")
        sdir, _ = self._shelf(scan_id)
        p = sdir / load_module("shelf_labels").IMAGES_REL / f"{name}.jpg"
        if not p.is_file():
            raise HTTPException(404, detail="unknown keyframe")
        return p

    def _write_labels(self, scan_id: str, sdir: Path, labels: list[dict]) -> None:
        sl = load_module("shelf_labels")
        atomic_write_json(sdir / sl.LABELS_REL, {"schema": 1, "scan_id": scan_id, "labels": labels,
                                                 "updated_at": now_iso()})

    def _validate_label(self, sdir: Path, body: dict, partial: bool) -> dict:
        sl = load_module("shelf_labels")
        out = {}
        if "keyframe" in body or not partial:
            cams = self._cameras(sdir)
            names = {k["name"] for k in cams["keyframes"]}
            if body.get("keyframe") not in names:
                raise HTTPException(422, detail="keyframe must be one of this scan's keyframes")
            out["keyframe"] = body["keyframe"]
        if "box_px" in body or not partial:
            b = body.get("box_px")
            c = self._cameras(sdir)["camera"]
            if not (isinstance(b, list) and len(b) == 4 and all(_num(v) for v in b)):
                raise HTTPException(422, detail="box_px must be [x0, y0, x1, y1] numbers")
            x0, y0, x1, y1 = (float(v) for v in b)
            x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
            if x0 < 0 or y0 < 0 or x1 > c["width"] or y1 > c["height"]:
                raise HTTPException(422, detail="box_px is outside the keyframe image")
            if x1 - x0 < 4 or y1 - y0 < 4:
                raise HTTPException(422, detail="box_px is too small (min 4 px)")
            out["box_px"] = [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
        if "kind" in body or not partial:
            kind = body.get("kind", "empty")
            if kind not in sl.KINDS:
                raise HTTPException(422, detail=f"kind must be one of {list(sl.KINDS)}")
            out["kind"] = kind
        if "note" in body:
            note = body.get("note") or ""
            if not isinstance(note, str) or len(note) > 500:
                raise HTTPException(422, detail="note must be a string of at most 500 characters")
            out["note"] = note.strip()
        return out

    def labels(self, scan_id: str) -> dict:
        sdir, rec = self._shelf(scan_id)
        sl = load_module("shelf_labels")
        labs = sl.read_labels(sdir)
        return {"scan_id": scan_id, "scale": rec.get("scale"), "labels": sl.lift_labels(sdir, rec, labs)}

    def add_label(self, scan_id: str, body: dict) -> dict:
        sdir, rec = self._shelf(scan_id)
        sl = load_module("shelf_labels")
        fields = self._validate_label(sdir, body, partial=False)
        with self.lock:
            labs = sl.read_labels(sdir)
            n = max([int(l["id"][4:]) for l in labs if re.fullmatch(r"lab_\d+", str(l.get("id")))] + [0]) + 1
            lab = {"id": f"lab_{n:03d}", **fields, "note": fields.get("note", ""), "created_at": now_iso()}
            labs.append(lab)
            self._write_labels(scan_id, sdir, labs)
        return sl.lift_labels(sdir, rec, [lab])[0]

    def update_label(self, scan_id: str, label_id: str, body: dict) -> dict:
        sdir, rec = self._shelf(scan_id)
        sl = load_module("shelf_labels")
        fields = self._validate_label(sdir, body, partial=True)
        with self.lock:
            labs = sl.read_labels(sdir)
            hit = [l for l in labs if l.get("id") == label_id]
            if not hit:
                raise HTTPException(404, detail="unknown label")
            hit[0].update(fields, updated_at=now_iso())
            self._write_labels(scan_id, sdir, labs)
        return sl.lift_labels(sdir, rec, [hit[0]])[0]

    def delete_label(self, scan_id: str, label_id: str) -> dict:
        sdir, _ = self._shelf(scan_id)
        sl = load_module("shelf_labels")
        with self.lock:
            labs = sl.read_labels(sdir)
            keep = [l for l in labs if l.get("id") != label_id]
            if len(keep) == len(labs):
                raise HTTPException(404, detail="unknown label")
            self._write_labels(scan_id, sdir, keep)
        return {"deleted": True, "id": label_id}

    def _after_gaps(self, job: Job) -> None:
        out = job.scan_dir / "results" / GAPS_NAME
        if not out.is_file() or (job.started and out.stat().st_mtime < job.started - 1):
            self._finish(job, "error", f"gap detection exited 0 but wrote no results/{GAPS_NAME}")
            return
        try:
            doc = read_json(out)
            n = {k: len(m.get("gaps", [])) for k, m in doc.get("methods", {}).items()}
        except Exception as e:  # noqa: BLE001
            self._finish(job, "error", f"results/{GAPS_NAME} is not valid JSON: {e}")
            return
        rec = self.read_scan(job.scan_id)
        f = (rec.get("scale") or {}).get("factor", 1.0) if isinstance(rec.get("scale"), dict) else 1.0
        self.update_scan(job.scan_id, gaps_at=now_iso(), gaps_scale_factor=f)
        self._finish(job, "done", "gaps: " + ", ".join(f"{k} {v}" for k, v in n.items()))

    def gaps(self, scan_id: str) -> dict:
        """Gap results in the current scene frame, scored against the user's labels."""
        sdir, rec = self._shelf(scan_id)
        p = sdir / "results" / GAPS_NAME
        if not p.is_file():
            raise HTTPException(404, detail="no gap detection results for this scan yet")
        sl = load_module("shelf_labels")
        doc = read_json(p)
        methods = sl.gaps_in_scene(doc, rec)
        labels = sl.lift_labels(sdir, rec, sl.read_labels(sdir))
        slots = sl.unique_slots(labels)
        scores = {k: sl.score_method(slots, m["gaps"]) for k, m in methods.items()}
        f_now = (rec.get("scale") or {}).get("factor", 1.0) if isinstance(rec.get("scale"), dict) else 1.0
        return {"scan_id": scan_id, "created_at": doc.get("created_at"), "job_id": doc.get("job_id"),
                "num_keyframes": doc.get("num_keyframes"), "params": doc.get("params"),
                "methods": methods, "scores": scores, "labelled_slots": len(slots),
                # the depth baseline's cm threshold used the scale of that run
                "scale_changed_since_run": abs(float(f_now) - float(rec.get("gaps_scale_factor", f_now))) > 1e-6}

    def _after_convert(self, job: Job) -> None:
        """convert job exited 0: ingest the usdz (no auto-flip), build the GLB, attach the video."""
        sdir = job.scan_dir
        usdz = Path(job.result_path) if job.result_path else sdir / "convert" / "out" / "scan.usdz"
        usdz = usdz.resolve()
        if sdir.resolve() not in usdz.parents or not usdz.is_file():
            self._finish(job, "error", f"conversion finished but no usdz at {usdz}")
            return
        scan_prep = load_module("scan_prep")
        try:
            fields = scan_prep.prepare_generated(str(sdir), str(usdz))
        except Exception as e:  # noqa: BLE001
            self._finish(job, "error", f"converted scan is not usable: {type(e).__name__}: {e}")
            return
        fields.pop("scan_id", None)
        labels = sdir / "labels" / "empty_slots.json"
        if labels.is_file():   # drawn on the previous conversion's keyframes: no longer placeable
            os.replace(labels, labels.with_name(f"empty_slots.superseded_{time.strftime('%Y%m%d_%H%M%S')}.json"))
        gaps = sdir / "results" / GAPS_NAME
        if gaps.is_file():     # detected on the previous geometry / keyframes
            os.replace(gaps, gaps.with_name(f"gaps.superseded_{time.strftime('%Y%m%d_%H%M%S')}.json"))
        # new geometry, new SfM scale: any earlier calibration no longer applies
        self.update_scan(job.scan_id, **fields, converted_at=now_iso(), stale=False,
                         scale=self.default_scale())
        try:
            self.build_glb(job.scan_id, force=True)
            self.update_scan(job.scan_id, glb_error=None)
        except HTTPException as e:
            self.update_scan(job.scan_id, glb_error=str(e.detail))
        mp4 = sdir / "convert" / "out" / "video_h264.mp4"
        note = ""
        if mp4.is_file():
            rec = self.read_scan(job.scan_id)
            try:
                self.attach_video(job.scan_id, mp4, Path(rec.get("source_file") or "video").stem + ".mp4")
            except HTTPException as e:
                note = f" (video not attached: {e.detail})"
        tris = fields.get("num_triangles")
        self._finish(job, "done", f"converted: {tris:,} triangles{note}" if isinstance(tris, int)
                     else f"converted{note}")

    # ---------------------------------------------------------------- GLB
    def build_glb(self, scan_id: str, force: bool = False) -> Path:
        sdir = self.scans_root / scan_id
        usda = sdir / "scene.usda"
        glb = sdir / "viewer" / "model.glb"
        with self.glb_lock:
            if not usda.is_file():
                raise HTTPException(404, detail="scene.usda missing for this scan")
            if (not force and glb.is_file()
                    and glb.stat().st_mtime >= usda.stat().st_mtime):
                return glb
            mod = load_module("usd_to_gltf")
            glb.parent.mkdir(parents=True, exist_ok=True)
            tmp = glb.with_name("model.tmp.glb")
            try:
                mod.export_glb(str(usda), str(tmp))
                os.replace(tmp, glb)
            except HTTPException:
                raise
            except ImportError as e:
                raise HTTPException(503, detail=f"GLB export unavailable ({type(e).__name__}: {e})")
            except Exception as e:  # noqa: BLE001
                raise HTTPException(500, detail=f"GLB export failed: {type(e).__name__}: {e}")
            finally:
                if tmp.exists():
                    tmp.unlink()
            return glb

    # ---------------------------------------------------------------- orientation
    def set_orientation(self, scan_id: str, flip: bool) -> dict:
        sdir = self.scan_dir(scan_id)
        if not (sdir / "scene.usda").is_file():
            raise HTTPException(409, detail="this scan has no 3D model yet (conversion not finished)")
        scan_prep = load_module("scan_prep")
        with self.lock:
            if self.active_job(scan_id):
                raise HTTPException(409, detail="a pipeline job for this scan is queued or running")
            before = self.read_scan(scan_id)
            was = self._orientation(before)["flip_applied"]
            try:
                ret = scan_prep.set_flip(str(sdir), bool(flip), "manual")
            except Exception as e:  # noqa: BLE001
                raise HTTPException(500, detail=f"set_flip failed: {type(e).__name__}: {e}")
            cur = self.read_scan(scan_id)  # set_flip may have written scan.json itself
            merged = {**cur, **(ret if isinstance(ret, dict) else {})}
            for k in SERVER_KEYS:
                if k in before:
                    merged[k] = before[k]
            has_results = any((sdir / "results" / n).is_file() for n in (RAW_NAME, REVIEWED_NAME))
            if has_results and bool(flip) != was:
                merged["stale"] = True
            merged["scan_id"] = scan_id
            merged["updated_at"] = now_iso()
            atomic_write_json(sdir / "scan.json", merged)
        try:
            self.build_glb(scan_id, force=True)
            self.update_scan(scan_id, glb_error=None)
        except HTTPException as e:
            self.update_scan(scan_id, glb_error=str(e.detail))
            raise
        return self.summary(scan_id)

    # ---------------------------------------------------------------- queue
    def enqueue(self, scan_id: str, prompts: list[str], kind: str = "pipeline") -> Job:
        sdir = self.scan_dir(scan_id)
        shelf = (self.read_scan(scan_id).get("scan_type") == "shelf")
        if kind == "pipeline" and shelf:
            raise HTTPException(422, detail="object detection runs on room scans; shelf scans get gap detection")
        if kind == "convert" and not shelf:
            raise HTTPException(422, detail="only shelf scans (store videos) can be converted")
        if kind == "gaps":
            if not shelf:
                raise HTTPException(422, detail="gap detection is for shelf scans")
            self._cameras(sdir)      # 409 when the scan has no keyframe cameras
        with self.lock:
            if self.active_job(scan_id):
                raise HTTPException(409, detail="a pipeline job for this scan is already queued or running")
            if self._stopping:
                raise HTTPException(503, detail="server is shutting down")
            job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            job = Job(job_id, scan_id, sdir, prompts, kind)
            job.run_dir.mkdir(parents=True, exist_ok=True)
            job.persist()
            self.jobs[job_id] = job
            self.queue.append(job_id)
            self.update_scan(scan_id, status="queued", progress=0.0, job_stage=None, message="queued",
                             job_id=job_id, prompts=prompts, error=None, queued_at=now_iso())
            self.cv.notify_all()
            return job

    def _worker_loop(self) -> None:
        while True:
            with self.cv:
                while not self.queue and not self._stopping:
                    self.cv.wait()
                if self._stopping:
                    return
                job = self.jobs.get(self.queue.popleft())
                if job is None or job.state != "queued":
                    continue
                with job.lock:
                    job.state, job.started, job.message = "running", time.time(), "starting"
                self.current = job
                job.persist()
                self.update_scan(job.scan_id, status="running", progress=0.0, message="starting")
            try:
                self._run_job(job)
            except Exception as e:  # noqa: BLE001
                self._finish(job, "error", f"internal error: {type(e).__name__}: {e}")
            finally:
                with self.lock:
                    self.current = None
                    self.proc = None

    def build_command(self, job: Job) -> list[str]:
        if job.kind == "gaps":
            argv = [sys.executable, "-u", "gaps_runner.py", "--scan-dir", str(job.scan_dir),
                    "--job-id", job.job_id]
        elif job.kind == "convert":
            argv = [sys.executable, "-u", "shelf_convert.py", "--scan-dir", str(job.scan_dir),
                    "--job-id", job.job_id, "--video-to-3d", str(self.video_to_3d),
                    "--height-m", str(self.shelf_height_m)]
        else:
            argv = [sys.executable, "-u", "pipeline_runner.py", "--scan-dir", str(job.scan_dir),
                    "--job-id", job.job_id, "--prompts", ",".join(job.prompts),
                    "--config", str(self.config_path)]
        if PIPELINE_CMD_HOOK is not None:
            argv = list(PIPELINE_CMD_HOOK(argv, {"scan_dir": job.scan_dir, "job_id": job.job_id,
                                                 "prompts": job.prompts, "config": self.config_path,
                                                 "kind": job.kind}))
        return argv

    def _run_job(self, job: Job) -> None:
        argv = self.build_command(job)
        job.run_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        error_msg: Optional[str] = None
        with open(job.run_dir / "runner.log", "w") as logf:
            logf.write(f"$ {' '.join(argv)}\n")
            logf.flush()
            with job.lock:
                job.log.append(f"$ {' '.join(argv)}")
            try:
                proc = subprocess.Popen(argv, cwd=str(BASE_DIR), stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        text=True, bufsize=1, errors="replace", env=env,
                                        start_new_session=True)
            except OSError as e:
                self._finish(job, "error", f"cannot start pipeline: {e}")
                return
            with self.lock:
                self.proc = proc
            with job.lock:
                job.pid = proc.pid
            job.persist()
            last_write = 0.0
            for line in proc.stdout:
                line = line.rstrip("\n")
                logf.write(line + "\n")
                logf.flush()
                s = line.strip()
                if not s:
                    continue
                with job.lock:
                    job.log.append(s)
                force_write = False
                m = RE_PROGRESS.match(s)
                if m:
                    try:
                        p = float(m.group(1))
                    except ValueError:
                        p = None
                    if p is not None and math.isfinite(p):
                        with job.lock:
                            force_write = m.group(2) != job.stage
                            job.progress = max(job.progress, min(1.0, max(0.0, p)))  # monotonic
                            job.stage = m.group(2)
                            job.message = (m.group(3) or m.group(2)).strip()
                elif RE_RESULT.match(s):
                    with job.lock:
                        job.result_path = RE_RESULT.match(s).group(1).strip()
                elif RE_ERROR.match(s):
                    error_msg = RE_ERROR.match(s).group(1).strip() or "pipeline error"
                now = time.time()
                if force_write or now - last_write > 1.0:
                    last_write = now
                    with job.lock:
                        pr, st, msg = round(job.progress, 4), job.stage, job.message
                    job.persist()
                    self.update_scan(job.scan_id, progress=pr, job_stage=st, message=msg)
            rc = proc.wait()
            logf.write(f"[server] exit code {rc}\n")

        raw = job.scan_dir / "results" / RAW_NAME
        if self._stopping and rc != 0:
            self._finish(job, "error", "interrupted (server shut down)")
            return
        if rc != 0:
            last = job.log[-1] if job.log else ""
            self._finish(job, "error", error_msg or f"pipeline exited with code {rc}: {last}")
            return
        if error_msg:
            self._finish(job, "error", error_msg)
            return
        if job.kind == "convert":
            self._after_convert(job)
            return
        if job.kind == "gaps":
            self._after_gaps(job)
            return
        if not raw.is_file() or (job.started and raw.stat().st_mtime < job.started - 1):
            self._finish(job, "error", f"pipeline exited 0 but wrote no results/{RAW_NAME}")
            return
        try:
            doc = read_json(raw)
            n = len(doc.get("objects", []))
        except Exception as e:  # noqa: BLE001
            self._finish(job, "error", f"results/{RAW_NAME} is not valid JSON: {e}")
            return
        # A new raw result invalidates an older review (object ids may differ).
        rev = job.scan_dir / "results" / REVIEWED_NAME
        if rev.is_file() and rev.stat().st_mtime < raw.stat().st_mtime:
            os.replace(rev, rev.with_name(f"scene_objects_reviewed.superseded_{time.strftime('%Y%m%d_%H%M%S')}.json"))
        self._finish(job, "done", f"done: {n} objects", num_objects=n, stale=False, reviewed_at=None)

    def _finish(self, job: Job, state: str, message: str, **scan_fields) -> None:
        with self.lock:
            with job.lock:
                job.state, job.message, job.finished = state, message, time.time()
                if state == "done":
                    job.progress = 1.0
                pr, st = round(job.progress, 4), job.stage
            job.persist()
            if (self.scans_root / job.scan_id / "scan.json").is_file():
                self.update_scan(job.scan_id, status=state, progress=pr, job_stage=st, message=message,
                                 error=message if state == "error" else None, **scan_fields)

    # ---------------------------------------------------------------- restart recovery
    @staticmethod
    def _kill_orphan(pid: Optional[int], job_id: str) -> None:
        if not pid:
            return
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return
        if job_id.encode() in cmd:
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass

    def _recover(self) -> None:
        requeue: list[Job] = []
        for d in sorted(self.scans_root.iterdir()):
            if not (SCAN_ID_RE.fullmatch(d.name) and (d / "scan.json").is_file()):
                continue
            try:
                rec = read_json(d / "scan.json")
            except (OSError, json.JSONDecodeError):
                continue
            for jf in (d / "runs").glob("*/job.json") if (d / "runs").is_dir() else []:
                try:
                    job = Job.from_dict(read_json(jf), d)
                except Exception:  # noqa: BLE001
                    continue
                if job.scan_id != d.name or job.run_dir != jf.parent:
                    continue
                self.jobs[job.job_id] = job
            interrupted = "interrupted (server restarted)"
            for j in [x for x in self.jobs.values() if x.scan_id == d.name]:
                if j.state == "running":
                    self._kill_orphan(j.pid, j.job_id)
                    j.state, j.message, j.finished = "error", interrupted, time.time()
                    j.persist()
            jid, status = rec.get("job_id"), rec.get("status")
            job = self.jobs.get(jid) if jid else None
            if status == "queued" and job is None and jid and JOB_ID_RE.fullmatch(str(jid)):
                job = Job(jid, d.name, d, list(rec.get("prompts") or []),  # job.json never written
                          "convert" if rec.get("scan_type") == "shelf" and not (d / "scene.usda").is_file()
                          else "pipeline")
                self.jobs[jid] = job
            for j in [x for x in self.jobs.values() if x.scan_id == d.name and x.state == "queued"]:
                if not (status == "queued" and j is job):
                    j.state, j.message, j.finished = "error", "stale queued job dropped on restart", time.time()
                    j.persist()
            if status == "running":
                self.update_scan(d.name, status="error", message=interrupted, error=interrupted)
            elif status == "queued":
                if job is not None and job.state == "queued" and (job.prompts or job.kind in ("convert", "gaps")):
                    requeue.append(job)
                else:
                    self.update_scan(d.name, status="error", message="queued job lost on restart",
                                     error="queued job lost on restart")
        for job in sorted(requeue, key=lambda j: j.created):
            self.queue.append(job.job_id)

    # ---------------------------------------------------------------- objects / send / delete
    def get_objects(self, scan_id: str, version: Optional[str]) -> dict:
        sdir = self.scan_dir(scan_id)
        if version not in (None, "", "raw", "reviewed"):
            raise HTTPException(422, detail="version must be 'raw' or 'reviewed'")
        if not version:
            version = "reviewed" if (sdir / "results" / REVIEWED_NAME).is_file() else "raw"
        path = sdir / "results" / (RAW_NAME if version == "raw" else REVIEWED_NAME)
        if not path.is_file():
            raise HTTPException(404, detail=f"no {version} results for this scan")
        try:
            return read_json(path)
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(500, detail=f"{path.name} unreadable: {e}")

    @staticmethod
    def validate_decisions(raw: dict, body: Any) -> list[dict]:
        if not isinstance(body, dict) or not isinstance(body.get("decisions"), list):
            raise HTTPException(422, detail="body must be an object with a 'decisions' list")
        raw_ids = [o.get("object_id") for o in raw.get("objects", []) if isinstance(o, dict)]
        errors: list[str] = []
        seen: set = set()
        cleaned: list[dict] = []
        for i, d in enumerate(body["decisions"]):
            p = f"decisions[{i}]"
            if not isinstance(d, dict):
                errors.append(f"{p}: must be an object")
                continue
            oid, dec, cn = d.get("object_id"), d.get("decision"), d.get("class_name")
            if not isinstance(oid, str) or oid not in raw_ids:
                errors.append(f"{p}.object_id: unknown object_id {oid!r}")
            elif oid in seen:
                errors.append(f"{p}.object_id: duplicate {oid!r}")
            else:
                seen.add(oid)
            if dec not in ("accept", "reject"):
                errors.append(f"{p}.decision: must be 'accept' or 'reject'")
            if not isinstance(cn, str) or not cn.strip():
                errors.append(f"{p}.class_name: must be a non-empty string")
            elif len(cn.strip()) > 200:
                errors.append(f"{p}.class_name: too long")
            if isinstance(oid, str) and dec in ("accept", "reject") and isinstance(cn, str):
                cleaned.append({"object_id": oid, "decision": dec, "class_name": " ".join(cn.split())})
        missing = [o for o in raw_ids if o not in seen]
        if missing:
            errors.append(f"missing decisions for: {', '.join(map(str, missing))}")
        if errors:
            raise HTTPException(422, detail="; ".join(errors))
        return cleaned

    def send(self, scan_id: str, body: Any) -> dict:
        sdir = self.scan_dir(scan_id)
        export_json = load_module("export_json")
        with self.lock:
            if self.active_job(scan_id):
                raise HTTPException(409, detail="a pipeline job for this scan is queued or running")
            raw_path = sdir / "results" / RAW_NAME
            if not raw_path.is_file():
                raise HTTPException(404, detail="no raw results for this scan")
            raw = read_json(raw_path)
            decisions = self.validate_decisions(raw, body)
            try:
                doc = export_json.build_reviewed(raw, decisions)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(500, detail=f"build_reviewed failed: {type(e).__name__}: {e}")
            errs = export_json.validate(doc)
            if errs:
                raise HTTPException(500, detail="reviewed JSON failed validation: " + "; ".join(map(str, errs[:20])))
            out_path = sdir / "results" / REVIEWED_NAME
            outbox_path = self.outbox / f"{scan_id}_reviewed.json"
            atomic_write_json(out_path, doc)
            self.outbox.mkdir(parents=True, exist_ok=True)
            atomic_write_json(outbox_path, doc)
            n = len(doc.get("objects", []))
            reviewed_at = (doc.get("scan") or {}).get("reviewed_at") or now_iso()
            self.update_scan(scan_id, status="reviewed", message=f"reviewed: {n} objects sent",
                             num_objects=n, reviewed_at=reviewed_at)
        return {"saved": True, "path": str(out_path), "outbox_path": str(outbox_path), "num_objects": n}

    def delete(self, scan_id: str) -> dict:
        sdir = self.scan_dir(scan_id)
        with self.lock:
            if self.active_job(scan_id):
                raise HTTPException(409, detail="a pipeline job for this scan is queued or running")
            shutil.rmtree(sdir)
            for jid in [j for j, job in self.jobs.items() if job.scan_id == scan_id]:
                del self.jobs[jid]
        return {"deleted": True, "scan_id": scan_id}

    # ---------------------------------------------------------------- source video
    def video_path(self, scan_id: str, rec: Optional[dict] = None) -> Optional[Path]:
        """The scan's video file, or None. scan.json's path is trusted only inside source/."""
        rec = rec if rec is not None else self.read_scan(scan_id)
        v = rec.get("source_video")
        if not isinstance(v, dict) or not isinstance(v.get("path"), str):
            return None
        src = (self.scans_root / scan_id / "source").resolve()
        p = (self.scans_root / scan_id / v["path"]).resolve()
        return p if p.parent == src and p.is_file() else None

    def video_public(self, scan_id: str, rec: Optional[dict] = None) -> Optional[dict]:
        rec = rec if rec is not None else self.read_scan(scan_id)
        if self.video_path(scan_id, rec) is None:
            return None
        v = rec["source_video"]
        return {k: v.get(k) for k in ("file", "bytes", "content_type", "attached_at")} | {
            "url": f"/api/scans/{scan_id}/video"}

    def attach_video(self, scan_id: str, upload_path: Path, original_name: str) -> dict:
        """Move an uploaded video into source/video<ext>, replacing any previous one."""
        sdir = self.scan_dir(scan_id)
        ext = video_ext(original_name)
        with open(upload_path, "rb") as f:
            head = f.read(12)
        if not sniff_video(head, ext):
            raise HTTPException(400, detail=f"not a valid {ext} video")
        dest = sdir / "source" / f"video{ext}"
        size = upload_path.stat().st_size
        with self.lock:
            old = self.video_path(scan_id)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(upload_path, dest)
            if old is not None and old != dest.resolve():
                old.unlink(missing_ok=True)
            base = re.split(r"[\\/]", original_name)[-1]
            shown = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip()[:120] or dest.name
            self.update_scan(scan_id, source_video={
                "file": shown, "path": f"source/{dest.name}", "bytes": size,
                "content_type": VIDEO_TYPES[ext], "attached_at": now_iso()})
        return self.video_public(scan_id)

    def detach_video(self, scan_id: str) -> dict:
        self.scan_dir(scan_id)
        with self.lock:
            p = self.video_path(scan_id)
            if p is None:
                raise HTTPException(404, detail="no source video for this scan")
            p.unlink(missing_ok=True)
            self.update_scan(scan_id, source_video=None)
        return {"deleted": True, "scan_id": scan_id}

    # ---------------------------------------------------------------- shutdown
    def shutdown(self, timeout: float = 5.0) -> None:
        with self.cv:
            self._stopping = True
            proc = self.proc
            self.cv.notify_all()
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
        if self.worker is not None and self.worker is not threading.current_thread():
            self.worker.join(timeout=timeout)


# --------------------------------------------------------------------------- multipart upload
class _TooLarge(Exception):
    pass


async def parse_upload(request: Request, dest: Path, max_bytes: int) -> dict:
    """Stream a multipart body: the 'file' part goes straight to `dest` (size-capped),
    small text fields are returned. Never buffers the file in memory or /tmp."""
    from python_multipart.multipart import MultipartParser, parse_options_header

    ctype, params = parse_options_header(request.headers.get("content-type", ""))
    if ctype != b"multipart/form-data" or not params.get(b"boundary"):
        raise HTTPException(400, detail="expected multipart/form-data with a 'file' field")
    clen = request.headers.get("content-length")
    if clen and clen.isdigit() and int(clen) > max_bytes + MULTIPART_SLACK:
        raise HTTPException(413, detail=f"upload larger than {max_bytes // (1024 * 1024)} MB")

    st: dict = {"hdr_f": b"", "hdr_v": b"", "headers": {}, "kind": None, "name": None,
                "buf": bytearray(), "fields": {}, "filename": None, "size": 0, "files": 0,
                "head": b"", "fh": None, "err": None}

    def on_part_begin():
        st.update(headers={}, kind=None, name=None, buf=bytearray(), hdr_f=b"", hdr_v=b"")

    def on_header_field(data, start, end):
        st["hdr_f"] += data[start:end]

    def on_header_value(data, start, end):
        st["hdr_v"] += data[start:end]

    def on_header_end():
        st["headers"][st["hdr_f"].decode("latin-1").lower()] = st["hdr_v"]
        st["hdr_f"], st["hdr_v"] = b"", b""

    def on_headers_finished():
        _, opts = parse_options_header(st["headers"].get("content-disposition", b""))
        name = opts.get(b"name", b"").decode("utf-8", "replace")
        fname = opts.get(b"filename")
        st["name"] = name
        if fname is not None:
            if name != "file":
                st["kind"] = "skip"
                return
            st["files"] += 1
            if st["files"] > 1:
                st["err"] = "only one file may be uploaded"
                st["kind"] = "skip"
                return
            st["filename"] = fname.decode("utf-8", "replace")
            st["kind"] = "file"
            st["fh"] = open(dest, "wb")
        else:
            st["kind"] = "field"

    def on_part_data(data, start, end):
        chunk = data[start:end]
        if st["kind"] == "file":
            st["size"] += len(chunk)
            if st["size"] > max_bytes:
                raise _TooLarge()
            if len(st["head"]) < 4:
                st["head"] += chunk[: 4 - len(st["head"])]
            st["fh"].write(chunk)
        elif st["kind"] == "field":
            st["buf"] += chunk
            if len(st["buf"]) > MAX_TEXT_FIELD:
                raise HTTPException(400, detail=f"form field {st['name']!r} too large")

    def on_part_end():
        if st["kind"] == "file" and st["fh"] is not None:
            st["fh"].close()
            st["fh"] = None
        elif st["kind"] == "field":
            st["fields"][st["name"]] = bytes(st["buf"]).decode("utf-8", "replace")

    parser = MultipartParser(params[b"boundary"], {
        "on_part_begin": on_part_begin, "on_header_field": on_header_field,
        "on_header_value": on_header_value, "on_header_end": on_header_end,
        "on_headers_finished": on_headers_finished, "on_part_data": on_part_data,
        "on_part_end": on_part_end})
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_bytes + MULTIPART_SLACK:
                raise _TooLarge()
            parser.write(chunk)
        parser.finalize()
    except _TooLarge:
        raise HTTPException(413, detail=f"upload larger than {max_bytes / (1024 * 1024):g} MB")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001  malformed multipart
        raise HTTPException(400, detail=f"malformed multipart body: {e}")
    finally:
        if st["fh"] is not None:
            st["fh"].close()
    if st["err"]:
        raise HTTPException(400, detail=st["err"])
    if st["filename"] is None:
        raise HTTPException(400, detail="missing 'file' field")
    return {"filename": st["filename"], "size": st["size"], "head": st["head"],
            "fields": st["fields"]}


# --------------------------------------------------------------------------- app
def create_app(out_dir: str | os.PathLike = "out", config_path: str | os.PathLike = DEFAULT_CONFIG,
               max_upload_mb: float = 500, start_worker: bool = True,
               video_to_3d: str | os.PathLike = DEFAULT_VIDEO_TO_3D,
               shelf_height_m: float = DEFAULT_SHELF_HEIGHT_M) -> FastAPI:
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    be = Backend(resolve_path(out_dir), resolve_path(config_path), max_upload_mb, start_worker,
                 resolve_path(video_to_3d), shelf_height_m)
    @asynccontextmanager
    async def lifespan(_app):
        yield
        be.shutdown()

    app = FastAPI(title="open_vocab_3d app v2", lifespan=lifespan)
    app.state.backend = be

    @app.middleware("http")
    async def cache_headers(request: Request, call_next):
        resp = await call_next(request)
        if request.url.path.startswith("/api"):
            revalidate = request.method == "GET" and request.url.path.endswith((".glb", "/video", ".jpg"))
            resp.headers["Cache-Control"] = "no-cache" if revalidate else "no-store"
        return resp

    @app.get("/")
    def index():
        idx = STATIC_DIR / "index.html"
        if not idx.is_file():
            raise HTTPException(404, detail=f"{idx} not found (UI not built yet)")
        return FileResponse(idx, headers={"Cache-Control": "no-cache"})

    @app.get("/api/config")
    def get_config():
        return {"default_prompts": be.default_prompts(), "max_upload_mb": be.max_upload_mb}

    @app.get("/api/scans")
    def list_scans():
        return be.list_scans()

    @app.post("/api/scans")
    async def upload(request: Request):
        be.uploads.mkdir(parents=True, exist_ok=True)
        dest = be.uploads / f"{uuid.uuid4().hex}.part"
        try:
            up = await parse_upload(request, dest, be.max_upload_bytes)
            ext = video_ext(up["filename"])
            if ext in SHELF_VIDEO_EXTS:
                if up["size"] == 0:
                    raise HTTPException(400, detail="empty file")
                return await run_in_threadpool(be.create_shelf_scan, dest, up["filename"])
            if not up["filename"].lower().endswith(".usdz"):
                raise HTTPException(400, detail="only .usdz scans or store videos (.mov/.mp4/.m4v) are accepted")
            if up["size"] == 0:
                raise HTTPException(400, detail="empty file")
            if up["head"] != ZIP_MAGIC:
                raise HTTPException(400, detail="not a .usdz (zip) file")
            fields = up["fields"]
            auto_run = parse_bool(fields.get("auto_run", "true") or "true", "auto_run")
            prompts = clean_prompts(fields.get("prompts") or None, be.default_prompts())
            name = sanitize_filename(up["filename"])
            return await run_in_threadpool(be.create_scan, dest, name, prompts, auto_run)
        finally:
            if dest.exists():
                dest.unlink()

    @app.get("/api/scans/{scan_id}")
    def get_scan(scan_id: str):
        return be.full_record(scan_id)

    @app.delete("/api/scans/{scan_id}")
    def delete_scan(scan_id: str):
        return be.delete(scan_id)

    @app.get("/api/scans/{scan_id}/model.glb")
    def get_glb(scan_id: str):
        be.scan_dir(scan_id)
        glb = be.build_glb(scan_id)
        return FileResponse(glb, media_type="model/gltf-binary")

    @app.get("/api/scans/{scan_id}/video")
    def get_video(scan_id: str):
        rec = be.read_scan(scan_id)
        p = be.video_path(scan_id, rec)
        if p is None:
            raise HTTPException(404, detail="no source video for this scan")
        # FileResponse answers Range requests (206), which the <video> element needs to seek.
        return FileResponse(p, media_type=rec["source_video"].get("content_type") or "video/mp4")

    @app.post("/api/scans/{scan_id}/video")
    async def upload_video(scan_id: str, request: Request):
        be.scan_dir(scan_id)
        be.uploads.mkdir(parents=True, exist_ok=True)
        dest = be.uploads / f"{uuid.uuid4().hex}.part"
        try:
            up = await parse_upload(request, dest, be.max_upload_bytes)
            if video_ext(up["filename"]) not in VIDEO_TYPES:
                raise HTTPException(400, detail="only .mp4, .m4v or .webm videos are accepted")
            if up["size"] == 0:
                raise HTTPException(400, detail="empty file")
            return await run_in_threadpool(be.attach_video, scan_id, dest, up["filename"])
        finally:
            if dest.exists():
                dest.unlink()

    @app.delete("/api/scans/{scan_id}/video")
    def delete_video(scan_id: str):
        return be.detach_video(scan_id)

    async def json_body(request: Request) -> dict:
        raw = await request.body()
        try:
            body = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            raise HTTPException(422, detail="body is not valid JSON")
        if not isinstance(body, dict):
            raise HTTPException(422, detail="body must be a JSON object")
        return body

    @app.post("/api/scans/{scan_id}/orientation")
    async def orientation(scan_id: str, request: Request):
        be.scan_dir(scan_id)
        body = await json_body(request)
        if not isinstance(body.get("flip"), bool):
            raise HTTPException(422, detail="body must be {\"flip\": true|false}")
        return await run_in_threadpool(be.set_orientation, scan_id, body["flip"])

    @app.post("/api/scans/{scan_id}/run")
    async def run(scan_id: str, request: Request):
        be.scan_dir(scan_id)
        body = await json_body(request)
        prompts = clean_prompts(body.get("prompts"), be.default_prompts())
        return be.job_public(be.enqueue(scan_id, prompts))

    @app.get("/api/scans/{scan_id}/keyframes")
    def keyframes(scan_id: str):
        return be.keyframes(scan_id)

    @app.get("/api/scans/{scan_id}/keyframes/{fname}")
    def keyframe_image(scan_id: str, fname: str):
        if not fname.endswith(".jpg"):
            raise HTTPException(404, detail="unknown keyframe")
        return FileResponse(be.keyframe_image(scan_id, fname[:-4]), media_type="image/jpeg")

    @app.get("/api/scans/{scan_id}/labels")
    async def get_labels(scan_id: str):
        return await run_in_threadpool(be.labels, scan_id)

    @app.post("/api/scans/{scan_id}/labels")
    async def add_label(scan_id: str, request: Request):
        be.scan_dir(scan_id)
        body = await json_body(request)
        return await run_in_threadpool(be.add_label, scan_id, body)

    @app.patch("/api/scans/{scan_id}/labels/{label_id}")
    async def update_label(scan_id: str, label_id: str, request: Request):
        be.scan_dir(scan_id)
        body = await json_body(request)
        return await run_in_threadpool(be.update_label, scan_id, label_id, body)

    @app.delete("/api/scans/{scan_id}/labels/{label_id}")
    def delete_label(scan_id: str, label_id: str):
        return be.delete_label(scan_id, label_id)

    @app.post("/api/scans/{scan_id}/gaps/run")
    def run_gaps(scan_id: str):
        be.scan_dir(scan_id)
        return be.job_public(be.enqueue(scan_id, [], kind="gaps"))

    @app.get("/api/scans/{scan_id}/gaps")
    async def get_gaps(scan_id: str):
        return await run_in_threadpool(be.gaps, scan_id)

    @app.post("/api/scans/{scan_id}/scale")
    async def scale(scan_id: str, request: Request):
        be.scan_dir(scan_id)
        body = await json_body(request)
        return await run_in_threadpool(be.set_scale, scan_id, body)

    @app.post("/api/scans/{scan_id}/convert")
    def convert(scan_id: str):
        """Re-run the video -> 3D conversion of a shelf scan (e.g. after a failure)."""
        be.scan_dir(scan_id)
        return be.job_public(be.enqueue(scan_id, [], kind="convert"))

    @app.get("/api/scans/{scan_id}/job")
    def get_job(scan_id: str):
        rec = be.read_scan(scan_id)
        job = be.latest_job(scan_id, rec)
        if job is None:
            raise HTTPException(404, detail="no job for this scan")
        return be.job_public(job)

    @app.get("/api/scans/{scan_id}/objects")
    def get_objects(scan_id: str, version: Optional[str] = None):
        return be.get_objects(scan_id, version)

    @app.post("/api/scans/{scan_id}/send")
    async def send(scan_id: str, request: Request):
        be.scan_dir(scan_id)
        body = await json_body(request)
        return await run_in_threadpool(be.send, scan_id, body)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static")
    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out", help="relative to open_vocab_3d/")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="relative to open_vocab_3d/")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-upload-mb", type=float, default=500)
    ap.add_argument("--video-to-3d", default=str(DEFAULT_VIDEO_TO_3D),
                    help="video_to_3d dir used to convert store videos into shelf scans")
    ap.add_argument("--shelf-height-m", type=float, default=DEFAULT_SHELF_HEIGHT_M,
                    help="provisional scale for new shelf scans: assumed scanned height in metres")
    a = ap.parse_args(argv)
    import uvicorn
    app = create_app(a.out, a.config, a.max_upload_mb, video_to_3d=a.video_to_3d,
                     shelf_height_m=a.shelf_height_m)
    be = app.state.backend
    print(f"[server] out={be.out} scans={be.scans_root} config={be.config_path} "
          f"max_upload_mb={be.max_upload_mb:g} queued={len(be.queue)}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
