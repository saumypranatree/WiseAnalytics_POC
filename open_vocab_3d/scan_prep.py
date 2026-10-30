#!/usr/bin/env python3
"""Scan preparation for app v2 (see app/API_V2.md, "scan_prep.py").

    prepare_scan(usdz_path, scans_root, original_name) -> scan.json record
    detect_orientation(stage_path) -> {"flip_recommended", "confidence", "evidence"}
    set_flip(scan_dir, flip, source) -> updated scan.json record

Steps of prepare_scan:
  1. copy the upload byte-for-byte to <scan>/source/<original_name>.usdz
  2. safe-extract the zip into <scan>/extracted/ (no absolute paths, no '..',
     no symlinks, <= 2000 entries, <= 2 GB uncompressed; sizes are enforced
     while streaming, not only from the headers)
  3. validate the root layer (the first .usd* entry): the stage opens, it has
     >= 1 mesh, and metersPerUnit > 0
  4. run the orientation heuristic on the source stage
  5. write scene.usda (the wrapper every later stage uses), with a 180 degree
     rotation about X only when a flip is recommended
  6. write scan.json atomically

Orientation heuristic
---------------------
Triangles whose unit normal satisfies |n . up| > 0.9 are "horizontal". Their
area is binned by height (5 cm bins, 3-bin window); the window with the most
area is the dominant horizontal surface, normally the floor, because scanners
capture the floor most completely. Four votes in [-1, 1] (positive = flip):

  position  The dominant surface's relative height r in the vertical extent:
            +1 at r >= 2/3 (top third), -1 at r <= 1/3, linear in between.
  normal    Area-weighted mean of n . up over the dominant surface. A floor
            faces up into the room, so a mean <= -0.5 votes flip and >= +0.5
            votes upright. Inconsistent winding (|mean| < 0.5) abstains.
  support   Horizontal area 0.25-1.25 m from the dominant surface on the room
            side (seats, table tops): upward-facing vs downward-facing. Table
            tops face away from the floor, so mostly downward-facing support
            surfaces vote flip. It abstains under 0.05 m^2.
  mass      The area-weighted mean relative height of the non-horizontal
            geometry. Furniture keeps the mass of an upright room low:
            +1 at >= 0.6, -1 at <= 0.4. It has half weight.

The decision is flip when the weighted sum is > 0. The confidence is "high"
when at least two votes are non-zero, they all agree, and |sum| >= 1.5;
otherwise it is "low". Every vote is antisymmetric under a 180 degree
rotation about a horizontal axis, so a flipped stage gets the opposite answer
from its source (unless the sum is exactly 0, which gives no flip, low).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

MAX_ENTRIES = 2000
MAX_UNCOMPRESSED = 2 * 1024 ** 3          # 2 GB
DEFAULT_MAX_UPLOAD_BYTES = 500 * 1024 ** 2  # 500 MB
SCHEMA = 1


class ScanPrepError(ValueError):
    """Bad upload (server: 400). The scan dir has been removed."""


class ScanTooLargeError(ScanPrepError):
    """Upload above the size limit (server: 413)."""


# --------------------------------------------------------------------------- #
# small utils
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    """ISO-8601 with the local UTC offset, seconds precision."""
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path, doc) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_text_atomic(path, text: str) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def new_scan_id() -> str:
    return f"scan_{_dt.datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(2)}"


def sanitize_filename(name: str) -> str:
    """Basename only, conservative charset, keeps the extension."""
    base = re.split(r"[\\/]", str(name or ""))[-1].strip()
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return base[:120] or "upload.usdz"


def load_record(scan_dir) -> dict:
    with open(Path(scan_dir) / "scan.json") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# safe extraction
# --------------------------------------------------------------------------- #
def _check_member_name(name: str) -> PurePosixPath:
    if not name or "\x00" in name:
        raise ScanPrepError(f"usdz entry with an invalid name: {name!r}")
    norm = name.replace("\\", "/")
    if norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
        raise ScanPrepError(f"usdz entry with an absolute path: {name!r}")
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ScanPrepError(f"usdz entry escapes the archive ('..'): {name!r}")
    if not parts:
        raise ScanPrepError(f"usdz entry with an empty path: {name!r}")
    return PurePosixPath(*parts)


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def safe_extract_usdz(usdz_path, dest_dir, *, max_entries: int = MAX_ENTRIES,
                      max_uncompressed: int = MAX_UNCOMPRESSED) -> str:
    """Extract a .usdz into dest_dir. Returns the root layer's relative POSIX path
    (the first .usd/.usda/.usdc entry). Raises ScanPrepError on anything unsafe."""
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        zf = zipfile.ZipFile(usdz_path)
    except (zipfile.BadZipFile, OSError) as e:
        raise ScanPrepError(f"not a valid .usdz (zip) file: {e}") from None
    with zf:
        infos = zf.infolist()
        if len(infos) > max_entries:
            raise ScanPrepError(f"usdz has {len(infos)} entries (limit {max_entries})")
        if sum(i.file_size for i in infos) > max_uncompressed:
            raise ScanPrepError(f"usdz uncompressed size exceeds {max_uncompressed} bytes")
        plan = []
        root = None
        for info in infos:
            rel = _check_member_name(info.filename)
            if _is_symlink(info):
                raise ScanPrepError(f"usdz entry is a symlink: {info.filename!r}")
            target = (dest / rel).resolve()
            if dest != target and dest not in target.parents:
                raise ScanPrepError(f"usdz entry escapes the archive: {info.filename!r}")
            if info.is_dir():
                plan.append((info, target, True))
                continue
            if root is None and rel.suffix.lower() in (".usd", ".usda", ".usdc"):
                root = rel.as_posix()
            plan.append((info, target, False))
        if root is None:
            raise ScanPrepError("usdz contains no .usd/.usda/.usdc layer")
        written = 0
        for info, target, is_dir in plan:
            if is_dir:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_uncompressed:   # headers can lie
                        raise ScanPrepError("usdz uncompressed size exceeds the limit")
                    dst.write(chunk)
    return root


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _stage_triangles(stage):
    """World-space triangles (meters) of all visible default/render meshes.
    Returns (v0, v1, v2) arrays (T, 3) with winding corrected for leftHanded
    orientation and mirroring transforms, plus the mesh count."""
    from pxr import Usd, UsdGeom

    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    tris_all, n_meshes = [], 0
    it = iter(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    for prim in it:
        img = UsdGeom.Imageable(prim)
        if img and img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            it.PruneChildren()
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if img.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not fvi:
            continue
        n_meshes += 1
        P = np.asarray(pts, dtype=np.float64)
        M = np.asarray(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                       dtype=np.float64).T          # USD row-vector -> column-vector
        W = (P @ M[:3, :3].T + M[:3, 3]) * mpu
        counts = np.asarray(counts, dtype=np.int64)
        fvi = np.asarray(fvi, dtype=np.int64)
        if np.all(counts == 3):
            tri = fvi.reshape(-1, 3)
        else:
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            rows = [(fvi[s], fvi[s + k], fvi[s + k + 1])
                    for s, n in zip(starts, counts) for k in range(1, n - 1)]
            tri = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
        flip = (mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded)
        if np.linalg.det(M[:3, :3]) < 0:
            flip = not flip
        if flip:
            tri = tri[:, ::-1]
        tri = tri[(tri < len(W)).all(axis=1)]
        tris_all.append(W[tri])
    if not tris_all:
        return None, 0
    T = np.concatenate(tris_all)
    return T, n_meshes


def _up_vector(stage):
    from pxr import UsdGeom
    return np.array([0.0, 0.0, 1.0]) if str(UsdGeom.GetStageUpAxis(stage)) == "Z" \
        else np.array([0.0, 1.0, 0.0])


def _orientation_from_triangles(T, up) -> dict:
    v0, v1, v2 = T[:, 0], T[:, 1], T[:, 2]
    cr = np.cross(v1 - v0, v2 - v0)
    area2 = np.linalg.norm(cr, axis=1)
    ok = area2 > 1e-12
    cr, area2, cen = cr[ok], area2[ok], ((v0 + v1 + v2) / 3.0)[ok]
    area = 0.5 * area2
    nup = (cr @ up) / area2
    h = cen @ up
    allh = T.reshape(-1, 3) @ up
    lo, hi = float(allh.min()), float(allh.max())
    extent = hi - lo
    hz = np.abs(nup) > 0.9
    ev = {"up_axis_vector": up.tolist(), "vertical_min_m": round(lo, 4),
          "vertical_max_m": round(hi, 4), "vertical_extent_m": round(extent, 4),
          "total_area_m2": round(float(area.sum()), 3),
          "horizontal_area_m2": round(float(area[hz].sum()), 3)}
    votes = {"position": 0.0, "normal": 0.0, "support": 0.0, "mass": 0.0}
    weights = {"position": 1.0, "normal": 1.0, "support": 1.0, "mass": 0.5}
    if extent <= 1e-6 or not hz.any():
        ev["reason"] = "no horizontal surfaces or a flat scene"
        return {"flip_recommended": False, "confidence": "low",
                "evidence": {**ev, "votes": votes, "weights": weights, "score": 0.0}}

    # dominant horizontal surface: 5 cm bins, 3-bin window
    bw = 0.05
    nb = max(1, int(np.ceil(extent / bw)) + 1)
    bi = np.clip(((h - lo) / bw).astype(np.int64), 0, nb - 1)
    hist = np.bincount(bi[hz], weights=area[hz], minlength=nb)
    win = np.convolve(hist, np.ones(3), mode="same")
    k = int(np.argmax(win))
    band = hz & (np.abs(bi - k) <= 1)
    dom_h = float(np.average(h[band], weights=area[band]))
    dom_area = float(area[band].sum())
    dom_n = float(np.average(nup[band], weights=area[band]))
    r = (dom_h - lo) / extent
    votes["position"] = float(np.clip((r - 0.5) * 6.0, -1.0, 1.0))
    if abs(dom_n) >= 0.5:
        votes["normal"] = float(-np.sign(dom_n))
    ev.update({"dominant_surface_height_m": round(dom_h, 4),
               "dominant_surface_area_m2": round(dom_area, 3),
               "dominant_surface_rel_height": round(r, 4),
               "dominant_surface_mean_normal_up": round(dom_n, 4),
               "dominant_surface_share_of_horizontal": round(dom_area / float(area[hz].sum()), 3)})

    # support surfaces on the room side of the dominant surface
    room_dir = 1.0 if r < 0.5 else -1.0           # the room lies away from the extreme
    off = (h - dom_h) * room_dir
    sup = hz & ~band & (off >= 0.25) & (off <= 1.25)
    a_up = float(area[sup & (nup > 0)].sum())
    a_dn = float(area[sup & (nup < 0)].sum())
    if a_up + a_dn >= 0.05:
        votes["support"] = float(-(a_up - a_dn) / (a_up + a_dn))
    ev.update({"support_band_m": [0.25, 1.25], "support_up_facing_area_m2": round(a_up, 3),
               "support_down_facing_area_m2": round(a_dn, 3)})

    # vertical mass of the non-horizontal geometry
    nh = ~hz
    if area[nh].sum() > 0:
        m_rel = float(np.average((h[nh] - lo) / extent, weights=area[nh]))
        votes["mass"] = float(np.clip((m_rel - 0.5) * 10.0, -1.0, 1.0))
        ev["nonhorizontal_mean_rel_height"] = round(m_rel, 4)

    score = float(sum(votes[k_] * weights[k_] for k_ in votes))
    nz = [v for v in votes.values() if abs(v) > 1e-9]
    agree = len(nz) >= 2 and (all(v > 0 for v in nz) or all(v < 0 for v in nz))
    conf = "high" if (agree and abs(score) >= 1.5) else "low"
    ev.update({"votes": {k_: round(v, 4) for k_, v in votes.items()}, "weights": weights,
               "score": round(score, 4)})
    return {"flip_recommended": bool(score > 0), "confidence": conf, "evidence": ev}


def detect_orientation(stage_path) -> dict:
    """Orientation heuristic on a USD stage (see the module docstring)."""
    from pxr import Usd
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise ScanPrepError(f"cannot open stage {stage_path}")
    T, _ = _stage_triangles(stage)
    if T is None:
        raise ScanPrepError("stage contains no meshes")
    return _orientation_from_triangles(T, _up_vector(stage))


def _stage_bounds_m(stage_path):
    from pxr import Usd
    stage = Usd.Stage.Open(str(stage_path))
    T, _ = _stage_triangles(stage)
    if T is None:
        return None
    pts = T.reshape(-1, 3)
    return {"min": [round(float(x), 4) for x in pts.min(0)],
            "max": [round(float(x), 4) for x in pts.max(0)]}


# --------------------------------------------------------------------------- #
# wrapper
# --------------------------------------------------------------------------- #
def _usd_literal(v) -> str:
    v = float(v)
    return str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)


def wrapper_text(root_layer_rel: str, meters_per_unit: float, up_axis: str,
                 flip: bool, ref_prim: str | None = None, nested: bool = False,
                 scale: float = 1.0) -> str:
    """scene.usda: /World references the extracted root layer (its default prim, or
    `ref_prim` when the layer has none) and optionally rotates it 180 deg about X and
    scales it uniformly (shelf-scan calibration; uniform scale commutes with the flip).

    nested=True (the referenced prim has its own xformOps): the reference moves to a
    child /World/Scan, so World's xformOpOrder cannot override the source transform."""
    target = f"<{ref_prim}>" if ref_prim else ""
    lines, order = [], []
    if scale != 1.0:
        sv = _usd_literal(scale)
        lines.append(f"    double3 xformOp:scale = ({sv}, {sv}, {sv})\n")
        order.append('"xformOp:scale"')
    if flip:
        lines.append("    double3 xformOp:rotateXYZ = (180, 0, 0)\n")
        order.append('"xformOp:rotateXYZ"')
    body = ("".join(lines) + f"    uniform token[] xformOpOrder = [{', '.join(order)}]\n") if order else ""
    head = ("#usda 1.0\n(\n    defaultPrim = \"World\"\n"
            f"    metersPerUnit = {_usd_literal(meters_per_unit)}\n"
            f"    upAxis = \"{up_axis}\"\n)\n\n")
    ref = f"references = @{root_layer_rel}@{target}"
    if nested:
        return (head + f"def Xform \"World\"\n{{\n{body}"
                f"    def Xform \"Scan\" (\n        {ref}\n    )\n    {{\n    }}\n}}\n")
    return head + f"def Xform \"World\" (\n    {ref}\n)\n{{\n{body}}}\n"


def scale_factor(rec: dict) -> float:
    """Uniform calibration scale of the scan (1.0 = as converted / as uploaded)."""
    f = (rec.get("scale") or {}).get("factor") if isinstance(rec.get("scale"), dict) else None
    try:
        f = float(f)
    except (TypeError, ValueError):
        return 1.0
    return f if np.isfinite(f) and f > 0 else 1.0


def _write_wrapper(scan_dir: Path, rec: dict, flip: bool) -> None:
    txt = wrapper_text(rec["root_layer"], rec["meters_per_unit"], rec["up_axis"], flip,
                       rec.get("reference_prim"), bool(rec.get("reference_has_xform")),
                       scale_factor(rec))
    _write_text_atomic(scan_dir / "scene.usda", txt)


def _drop_glb(scan_dir: Path) -> None:
    glb = scan_dir / "viewer" / "model.glb"
    for p in (glb, glb.with_name("model_meta.json")):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def validate_root_layer(layer_path) -> dict:
    """Open the stage and check it. Returns stage facts for the record."""
    from pxr import Usd, UsdGeom
    try:
        stage = Usd.Stage.Open(str(layer_path))
    except Exception as e:  # pxr raises Tf.ErrorException
        raise ScanPrepError(f"the USD stage does not open: {str(e).strip()[:300]}") from None
    if stage is None:
        raise ScanPrepError("the USD stage does not open")
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not (mpu > 0 and np.isfinite(mpu)):
        raise ScanPrepError(f"invalid metersPerUnit {mpu}")
    up = str(UsdGeom.GetStageUpAxis(stage))
    T, n_meshes = _stage_triangles(stage)
    if T is None or n_meshes == 0:
        raise ScanPrepError("the stage contains no mesh")
    dp = stage.GetDefaultPrim()
    ref_prim = None
    roots = [p for p in stage.GetPseudoRoot().GetChildren() if p.IsActive()]
    if not dp:
        cand = [p for p in roots if any(q.IsA(UsdGeom.Mesh) for q in Usd.PrimRange(p))]
        if not cand:
            raise ScanPrepError("no root prim contains a mesh")
        ref_prim = str(cand[0].GetPath())
        if len(cand) > 1:
            raise ScanPrepError("the root layer has no defaultPrim and several root prims "
                                "with meshes; cannot reference it")
    ref = dp if dp else stage.GetPrimAtPath(ref_prim)
    has_xf = bool(ref.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(ref).GetOrderedXformOps())
    return {"meters_per_unit": mpu, "up_axis": up, "reference_has_xform": has_xf,
            "default_prim": str(dp.GetPath()) if dp else None,
            "reference_prim": ref_prim, "num_meshes": n_meshes,
            "num_triangles": int(len(T)), "triangles": T, "up_vec": _up_vector(stage)}


def prepare_scan(usdz_path, scans_root, original_name, *,
                 max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
                 scan_id: str | None = None, uploaded_at: str | None = None,
                 auto_orient: bool = True) -> dict:
    """Create out/scans/<scan_id>/ from an uploaded .usdz. Returns the scan.json
    record. Raises ScanTooLargeError / ScanPrepError (the dir is removed)."""
    usdz_path = Path(usdz_path)
    name = sanitize_filename(original_name or usdz_path.name)
    if not name.lower().endswith(".usdz"):
        raise ScanPrepError("only .usdz uploads are accepted")
    size = usdz_path.stat().st_size
    if size > max_upload_bytes:
        raise ScanTooLargeError(f"upload is {size} bytes (limit {max_upload_bytes})")
    if size == 0:
        raise ScanPrepError("empty upload")
    scans_root = Path(scans_root)
    scans_root.mkdir(parents=True, exist_ok=True)
    scan_id = scan_id or new_scan_id()
    scan_dir = scans_root / scan_id
    scan_dir.mkdir(parents=False, exist_ok=False)
    try:
        (scan_dir / "source").mkdir()
        src = scan_dir / "source" / name
        h = hashlib.sha256()
        with open(usdz_path, "rb") as fi, open(src, "wb") as fo:
            for chunk in iter(lambda: fi.read(1 << 20), b""):
                h.update(chunk)
                fo.write(chunk)
        rec = {
            "schema": SCHEMA,
            "scan_id": scan_id,
            "source_file": name,
            "source_path": f"source/{name}",
            "source_bytes": size,
            "source_sha256": h.hexdigest(),
            "uploaded_at": uploaded_at or now_iso(),
            **_ingest_usdz(scan_dir, src, auto_orient),
            "status": "uploaded",
            "stale": False,
        }
        write_json_atomic(scan_dir / "scan.json", rec)
        return rec
    except BaseException:
        shutil.rmtree(scan_dir, ignore_errors=True)
        raise


def _ingest_usdz(scan_dir: Path, usdz_path: Path, auto_orient: bool) -> dict:
    """Extract + validate a usdz into scan_dir/extracted, write scene.usda; returns the
    stage/orientation fields of the scan.json record (no source_* fields)."""
    root_rel = safe_extract_usdz(usdz_path, scan_dir / "extracted")
    facts = validate_root_layer(scan_dir / "extracted" / root_rel)
    if auto_orient:
        orient = _orientation_from_triangles(facts["triangles"], facts["up_vec"])
    else:
        orient = {"flip_recommended": False, "confidence": "low",
                  "evidence": {"reason": "auto orientation disabled"}}
    rec = {
        "root_layer": f"extracted/{root_rel}",
        "stage": "scene.usda",
        "up_axis": facts["up_axis"],
        "meters_per_unit": facts["meters_per_unit"],
        "default_prim": facts["default_prim"],
        "reference_prim": facts["reference_prim"],
        "reference_has_xform": facts["reference_has_xform"],
        "num_meshes": facts["num_meshes"],
        "num_triangles": facts["num_triangles"],
        "orientation": {
            "flip_applied": bool(orient["flip_recommended"]),
            "source": "auto",
            "flip_recommended": bool(orient["flip_recommended"]),
            "confidence": orient["confidence"],
            "evidence": orient["evidence"],
            "updated_at": now_iso(),
        },
    }
    _write_wrapper(scan_dir, rec, rec["orientation"]["flip_applied"])
    rec["scene_bounds"] = _stage_bounds_m(scan_dir / "scene.usda")
    return rec


def prepare_generated(scan_dir, usdz_path, *, auto_orient: bool = False) -> dict:
    """Ingest a .usdz generated on the server (shelf video conversion) into an EXISTING
    scan dir: replaces extracted/ and scene.usda, returns the stage/orientation fields to
    merge into scan.json (the caller keeps source_* and server fields). auto_orient is off
    by default: generated scans are already gravity-aligned, and the floor heuristic is
    unreliable on shelf facades with little floor."""
    scan_dir = Path(scan_dir)
    if not (scan_dir / "scan.json").is_file():
        raise ScanPrepError(f"{scan_dir} is not a scan dir")
    shutil.rmtree(scan_dir / "extracted", ignore_errors=True)
    rec = _ingest_usdz(scan_dir, Path(usdz_path), auto_orient)
    if not auto_orient:
        rec["orientation"]["evidence"] = {"reason": "gravity-aligned during video conversion"}
    return rec


def set_flip(scan_dir, flip: bool, source: str = "manual") -> dict:
    """Rewrite scene.usda with/without the X-180 rotation, delete viewer/model.glb,
    update orientation + scene_bounds in scan.json (other fields untouched)."""
    if source not in ("auto", "manual"):
        raise ValueError("source must be 'auto' or 'manual'")
    scan_dir = Path(scan_dir)
    rec = load_record(scan_dir)
    _write_wrapper(scan_dir, rec, bool(flip))     # keeps the calibration scale
    _drop_glb(scan_dir)
    o = dict(rec.get("orientation") or {})
    o.update({"flip_applied": bool(flip), "source": source, "updated_at": now_iso()})
    rec = load_record(scan_dir)                 # re-read: keep concurrent field edits
    rec["orientation"] = {**(rec.get("orientation") or {}), **o}
    rec["scene_bounds"] = _stage_bounds_m(scan_dir / "scene.usda")
    write_json_atomic(scan_dir / "scan.json", rec)
    return rec


def set_scale(scan_dir, scale: dict) -> dict:
    """Rewrite scene.usda with the uniform scale scale["factor"] (keeping the current
    flip), delete viewer/model.glb, store `scale` + scene_bounds in scan.json."""
    scan_dir = Path(scan_dir)
    f = scale_factor({"scale": scale})
    if f != float(scale.get("factor", 1.0)):
        raise ValueError(f"invalid scale factor {scale.get('factor')!r}")
    rec = load_record(scan_dir)
    rec["scale"] = dict(scale)
    flip = bool((rec.get("orientation") or {}).get("flip_applied"))
    _write_wrapper(scan_dir, rec, flip)
    _drop_glb(scan_dir)
    cur = load_record(scan_dir)                  # re-read: keep concurrent field edits
    cur["scale"] = dict(scale)
    cur["scene_bounds"] = _stage_bounds_m(scan_dir / "scene.usda")
    write_json_atomic(scan_dir / "scan.json", cur)
    return cur


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="prepare a .usdz scan or check orientation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("usdz")
    a.add_argument("--scans-root", required=True)
    b = sub.add_parser("orient")
    b.add_argument("stage")
    c = sub.add_parser("flip")
    c.add_argument("scan_dir")
    c.add_argument("--flip", action=argparse.BooleanOptionalAction, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        out = prepare_scan(args.usdz, args.scans_root, os.path.basename(args.usdz))
    elif args.cmd == "orient":
        out = detect_orientation(args.stage)
    else:
        out = set_flip(args.scan_dir, args.flip, "manual")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
