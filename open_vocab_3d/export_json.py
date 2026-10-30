#!/usr/bin/env python3
"""Export JSON for app v2: raw and reviewed scene object lists (app/API_V2.md).

    build_raw(scan_record, lift_json_path, job_id, prompts, pipeline_info) -> dict
    build_reviewed(raw, decisions) -> dict      (ValueError on invalid decisions)
    validate(doc) -> list[str]                  (empty list = valid)
    write_json(path, doc)                       (atomic)

object_ids are assigned once, in build_raw: objects sorted by class name, then by
confidence descending, get obj_001, obj_002, ... The reviewed document keeps
those ids, drops rejected objects (listed in review_summary.rejected_object_ids)
and never assigns new ids.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = "1.0"
DEFAULT_PIPELINE = {
    "detector": "IDEA-Research/grounding-dino-base",
    "segmenter": "facebook/sam-vit-base",
    "renderer": "open3d-raycast",
    "two_pass": True,
    "focus_centres": [],
}
OBJECT_KEYS_RAW = {"object_id", "class_name", "position", "bounding_box", "confidence", "num_views"}
OBJECT_ID_RE = re.compile(r"^obj_\d{3,}$")
MAX_CLASS_NAME = 100


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _r(x) -> float:
    return round(float(x), 4)


def _vec(v) -> list:
    return [_r(x) for x in v]


def write_json(path, doc) -> None:
    """Atomic write: temp file in the same dir, fsync, os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _summary(objects) -> dict:
    return {"num_objects": len(objects),
            "by_class": dict(sorted(Counter(o["class_name"] for o in objects).items()))}


def _coordinate_frame(rec: dict) -> dict:
    o = rec.get("orientation") or {}
    up = str(rec.get("up_axis") or "Y")
    return {
        "up_axis": up, "units": "meters", "handedness": "right",
        "origin": "scan origin (as authored in the source file)",
        "orientation_fix": "rotate_x_180" if o.get("flip_applied") else "none",
        "orientation_source": o.get("source", "auto"),
        "note": f"Positions are in the upright scan frame shown in the viewer: +{up} is up.",
    }


def build_raw(scan_record: dict, lift_json_path, job_id: str, prompts, pipeline_info=None) -> dict:
    """Raw export from a lift_cluster.py output (run with --extended for num_views)."""
    with open(lift_json_path) as f:
        lift = json.load(f)
    items = list(lift.get("objects", []))
    items.sort(key=lambda o: (str(o["class_name"]), -float(o.get("confidence", 0.0))))
    objects = []
    for i, o in enumerate(items, start=1):
        bmin = _vec(o["bounding_box_3d"]["min"])
        bmax = _vec(o["bounding_box_3d"]["max"])
        p = o["position_world"]
        objects.append({
            "object_id": f"obj_{i:03d}",
            "class_name": str(o["class_name"]),
            "position": {"x": _r(p["x"]), "y": _r(p["y"]), "z": _r(p["z"])},
            "bounding_box": {"min": bmin, "max": bmax,
                             "size": [_r(b - a) for a, b in zip(bmin, bmax)]},
            "confidence": _r(o.get("confidence", 0.0)),
            "num_views": int(o.get("num_views", 0)),
        })
    pipeline = {**DEFAULT_PIPELINE, **(pipeline_info or {})}
    pipeline["focus_centres"] = [_vec(c) for c in pipeline.get("focus_centres") or []]
    sb = scan_record.get("scene_bounds") or {}
    if not sb and isinstance(lift.get("scene_bounds_world"), dict):
        sb = lift["scene_bounds_world"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "raw",
        "scan": {
            "scan_id": scan_record["scan_id"],
            "source_file": scan_record.get("source_file"),
            "uploaded_at": scan_record.get("uploaded_at"),
            "processed_at": now_iso(),
            "reviewed_at": None,
            "job_id": job_id,
            "coordinate_frame": _coordinate_frame(scan_record),
            "scene_bounds": {"min": _vec(sb.get("min", [0, 0, 0])),
                             "max": _vec(sb.get("max", [0, 0, 0]))},
            "prompts": [str(p) for p in prompts],
            "pipeline": pipeline,
        },
        "objects": objects,
        "summary": _summary(objects),
    }


def _clean_name(name) -> str:
    return re.sub(r"\s+", " ", str(name if name is not None else "")).strip()


def build_reviewed(raw: dict, decisions) -> dict:
    """decisions: [{"object_id", "decision": "accept"|"reject", "class_name"}] covering
    every raw object exactly once. Raises ValueError (server: 422) otherwise."""
    if raw.get("status") != "raw":
        raise ValueError("build_reviewed needs the raw document")
    if not isinstance(decisions, list):
        raise ValueError("decisions must be a list")
    by_id = {o["object_id"]: o for o in raw["objects"]}
    seen: dict = {}
    for d in decisions:
        if not isinstance(d, dict):
            raise ValueError("each decision must be an object")
        oid = d.get("object_id")
        if oid not in by_id:
            raise ValueError(f"unknown object_id {oid!r}")
        if oid in seen:
            raise ValueError(f"duplicate decision for {oid}")
        dec = d.get("decision")
        if dec not in ("accept", "reject"):
            raise ValueError(f"{oid}: decision must be 'accept' or 'reject', got {dec!r}")
        name = _clean_name(d.get("class_name", by_id[oid]["class_name"]))
        if dec == "accept" and not name:
            raise ValueError(f"{oid}: empty class_name")
        if len(name) > MAX_CLASS_NAME:
            raise ValueError(f"{oid}: class_name longer than {MAX_CLASS_NAME} characters")
        seen[oid] = (dec, name)
    missing = [oid for oid in by_id if oid not in seen]
    if missing:
        raise ValueError(f"no decision for {', '.join(missing)}")

    objects, rejected, n_acc, n_ren = [], [], 0, 0
    for o in raw["objects"]:
        dec, name = seen[o["object_id"]]
        if dec == "reject":
            rejected.append(o["object_id"])
            continue
        orig = o["class_name"]
        renamed = name != orig
        n_ren += renamed
        n_acc += not renamed
        objects.append({**json.loads(json.dumps(o)), "class_name": name,
                        "review": {"decision": "renamed" if renamed else "accepted",
                                   "original_class_name": orig}})
    doc = json.loads(json.dumps(raw))
    doc["status"] = "reviewed"
    doc["scan"]["reviewed_at"] = now_iso()
    doc["objects"] = objects
    doc["summary"] = _summary(objects)
    doc["review_summary"] = {"accepted": n_acc, "renamed": n_ren,
                             "rejected_object_ids": sorted(rejected)}
    return doc


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_vec3(v) -> bool:
    return isinstance(v, list) and len(v) == 3 and all(_is_num(x) for x in v)


def _is_ts(s) -> bool:
    if not isinstance(s, str):
        return False
    try:
        return _dt.datetime.fromisoformat(s).tzinfo is not None
    except ValueError:
        return False


def validate(doc) -> list:
    """Schema check of a raw or reviewed document. Returns a list of problems."""
    err = []
    if not isinstance(doc, dict):
        return ["document is not an object"]
    if doc.get("schema_version") != SCHEMA_VERSION:
        err.append(f"schema_version must be {SCHEMA_VERSION!r}")
    status = doc.get("status")
    if status not in ("raw", "reviewed"):
        err.append("status must be 'raw' or 'reviewed'")
    reviewed = status == "reviewed"

    scan = doc.get("scan")
    if not isinstance(scan, dict):
        err.append("scan missing")
        scan = {}
    for k in ("scan_id", "source_file", "job_id"):
        if not isinstance(scan.get(k), str) or not scan.get(k):
            err.append(f"scan.{k} must be a non-empty string")
    for k in ("uploaded_at", "processed_at"):
        if not _is_ts(scan.get(k)):
            err.append(f"scan.{k} must be ISO-8601 with an offset")
    if reviewed:
        if not _is_ts(scan.get("reviewed_at")):
            err.append("scan.reviewed_at must be ISO-8601 with an offset in a reviewed document")
    elif scan.get("reviewed_at") is not None:
        err.append("scan.reviewed_at must be null in a raw document")
    cf = scan.get("coordinate_frame")
    if not isinstance(cf, dict):
        err.append("scan.coordinate_frame missing")
    else:
        if cf.get("up_axis") not in ("Y", "Z"):
            err.append("coordinate_frame.up_axis must be 'Y' or 'Z'")
        if cf.get("units") != "meters":
            err.append("coordinate_frame.units must be 'meters'")
        if cf.get("handedness") != "right":
            err.append("coordinate_frame.handedness must be 'right'")
        if cf.get("orientation_fix") not in ("rotate_x_180", "none"):
            err.append("coordinate_frame.orientation_fix must be 'rotate_x_180' or 'none'")
        if cf.get("orientation_source") not in ("auto", "manual"):
            err.append("coordinate_frame.orientation_source must be 'auto' or 'manual'")
        for k in ("origin", "note"):
            if not isinstance(cf.get(k), str):
                err.append(f"coordinate_frame.{k} must be a string")
    sb = scan.get("scene_bounds")
    if not (isinstance(sb, dict) and _is_vec3(sb.get("min")) and _is_vec3(sb.get("max"))):
        err.append("scan.scene_bounds must have min/max 3-vectors")
    pr = scan.get("prompts")
    if not (isinstance(pr, list) and all(isinstance(p, str) for p in pr)):
        err.append("scan.prompts must be a list of strings")
    pl = scan.get("pipeline")
    if not isinstance(pl, dict):
        err.append("scan.pipeline missing")
    else:
        for k in ("detector", "renderer"):
            if not isinstance(pl.get(k), str):
                err.append(f"pipeline.{k} must be a string")
        if not isinstance(pl.get("two_pass"), bool):
            err.append("pipeline.two_pass must be a bool")
        fc = pl.get("focus_centres")
        if not (isinstance(fc, list) and all(_is_vec3(c) for c in fc)):
            err.append("pipeline.focus_centres must be a list of 3-vectors")

    objs = doc.get("objects")
    if not isinstance(objs, list):
        err.append("objects must be a list")
        objs = []
    ids = []
    allowed = OBJECT_KEYS_RAW | ({"review"} if reviewed else set())
    for i, o in enumerate(objs):
        tag = f"objects[{i}]"
        if not isinstance(o, dict):
            err.append(f"{tag} is not an object")
            continue
        extra = set(o) - allowed
        missing = allowed - set(o)
        if extra:
            err.append(f"{tag}: unexpected keys {sorted(extra)}")
        if missing:
            err.append(f"{tag}: missing keys {sorted(missing)}")
        oid = o.get("object_id")
        if not (isinstance(oid, str) and OBJECT_ID_RE.match(oid)):
            err.append(f"{tag}: bad object_id {oid!r}")
        ids.append(oid)
        if not (isinstance(o.get("class_name"), str) and o["class_name"].strip()):
            err.append(f"{tag}: class_name must be a non-empty string")
        p = o.get("position")
        if not (isinstance(p, dict) and set(p) == {"x", "y", "z"}
                and all(_is_num(p[k]) for k in "xyz")):
            err.append(f"{tag}: position must be {{x, y, z}} numbers")
        bb = o.get("bounding_box")
        if not (isinstance(bb, dict) and set(bb) == {"min", "max", "size"}
                and all(_is_vec3(bb[k]) for k in ("min", "max", "size"))):
            err.append(f"{tag}: bounding_box must have min/max/size 3-vectors")
        else:
            if any(a > b for a, b in zip(bb["min"], bb["max"])):
                err.append(f"{tag}: bounding_box min > max")
            if any(abs((b - a) - s) > 1e-3 for a, b, s in zip(bb["min"], bb["max"], bb["size"])):
                err.append(f"{tag}: bounding_box size != max - min")
        c = o.get("confidence")
        if not (_is_num(c) and 0.0 <= c <= 1.0):
            err.append(f"{tag}: confidence must be in [0, 1]")
        nv = o.get("num_views")
        if not (isinstance(nv, int) and not isinstance(nv, bool) and nv >= 0):
            err.append(f"{tag}: num_views must be a non-negative int")
        if reviewed:
            rv = o.get("review")
            if not (isinstance(rv, dict) and rv.get("decision") in ("accepted", "renamed")
                    and isinstance(rv.get("original_class_name"), str)):
                err.append(f"{tag}: review must have decision accepted|renamed and "
                           "original_class_name")
            elif (rv["decision"] == "renamed") != (rv["original_class_name"] != o.get("class_name")):
                err.append(f"{tag}: review.decision inconsistent with original_class_name")
    dup = [k for k, n in Counter(ids).items() if n > 1]
    if dup:
        err.append(f"duplicate object_ids {dup}")

    sm = doc.get("summary")
    good_objs = [o for o in objs if isinstance(o, dict) and isinstance(o.get("class_name"), str)]
    if not isinstance(sm, dict):
        err.append("summary missing")
    elif sm != _summary(good_objs):
        err.append("summary does not match objects")

    rs = doc.get("review_summary")
    if reviewed:
        if not (isinstance(rs, dict) and isinstance(rs.get("accepted"), int)
                and isinstance(rs.get("renamed"), int)
                and isinstance(rs.get("rejected_object_ids"), list)):
            err.append("review_summary must have accepted, renamed, rejected_object_ids")
        else:
            if rs["accepted"] + rs["renamed"] != len(objs):
                err.append("review_summary accepted + renamed != number of objects")
            rej = rs["rejected_object_ids"]
            if not all(isinstance(r, str) and OBJECT_ID_RE.match(r) for r in rej):
                err.append("review_summary.rejected_object_ids has bad ids")
            if set(rej) & set(ids):
                err.append("a rejected object_id is still in objects")
            if len(set(rej)) != len(rej):
                err.append("duplicate rejected_object_ids")
            n_ren = sum(1 for o in objs if isinstance(o, dict)
                        and (o.get("review") or {}).get("decision") == "renamed")
            if n_ren != rs["renamed"]:
                err.append("review_summary.renamed does not match objects")
    elif rs is not None:
        err.append("review_summary is only allowed in a reviewed document")
    return err


if __name__ == "__main__":  # python export_json.py doc.json -> validation report
    import sys
    problems = validate(json.load(open(sys.argv[1])))
    print("\n".join(problems) or "valid")
    raise SystemExit(1 if problems else 0)
