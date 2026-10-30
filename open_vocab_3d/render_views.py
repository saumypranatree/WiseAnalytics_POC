#!/usr/bin/env python3
"""Stage 1 of the open-vocabulary 3D pipeline: render RGB + depth + cameras.

Opens a USD scene in Isaac Sim (directly, so the world frame stays the stage's
native frame), places a set of virtual pinhole cameras around or inside the
scene, renders them with Omniverse Replicator, and writes the files described
in CONTRACT.md ("Stage 1 output"):

    <out>/meta.json
    <out>/rgb/0000.png      uint8 RGB, HxWx3
    <out>/depth/0000.npy    float32 meters, HxW, distance_to_image_plane, no-hit = 0

How to run (Isaac Sim 4.5 / 5.x bundled python; headless by default):

    ~/isaacsim/python.sh render_views.py --usd ../room_scan.usda --out out/renders \
        --mode interior --num-views 48 --verify

    # orbit the scan from outside instead (objects / small scenes):
    ~/isaacsim/python.sh render_views.py --usd scene.usd --out out/renders \
        --mode dome --rings 15,35,60 --num-views 36

    # re-check an existing output with plain numpy (no Isaac Sim needed):
    python3 render_views.py --verify-only --out out/renders

Coordinate conventions (see CONTRACT.md):
  * World frame = native USD stage frame, in meters (stage units * metersPerUnit).
    The up axis is NOT rotated (a Y-up stage stays Y-up); it is recorded in meta.json.
  * Camera frame = OpenCV: +X right, +Y down, +Z forward. USD cameras look down -Z
    with +Y up, so  T_world_to_cam(cv) = diag(1,-1,-1,1) @ T_world_to_cam(usd).
  * Matrices are 4x4, column-vector convention (p_cam = T_world_to_cam @ p_world_h),
    stored row-major as nested lists. USD/Gf matrices are row-vector convention,
    i.e. the transpose; every conversion goes through usd_to_col()/col_to_usd().
  * Intrinsics: fx = f / h_aperture * W, fy = fx (square pixels; the vertical
    aperture is derived from the aspect ratio). Pixel centers sit at integer
    coordinates (contract), and the optical axis passes through the geometric image
    center, which in that convention is cx = (W-1)/2, cy = (H-1)/2. (Isaac Sim's own
    helpers use W/2, H/2 - half a pixel off; pass --principal-point half for that.)
  * Depth = distance_to_image_plane (z-depth), converted to meters; inf/NaN/
    beyond-far-clip -> 0.

Lights and the camera are authored on the stage's *session layer* under
/PerceptionRig, and the stage is never saved, so the scan on disk is untouched.

Pure numpy helpers (look-at, K, convention conversion, pose generation,
back-projection/verification) are at the top and do not import omni/pxr.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# OpenCV <-> USD camera axis flip (its own inverse).
CV_FROM_USD = np.diag([1.0, -1.0, -1.0, 1.0])

AXIS_VEC = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]),
            "Z": np.array([0.0, 0.0, 1.0])}


# ----------------------------------------------------------------------------
# Pure numpy math (unit-testable, no omni / pxr imports)
# ----------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return v / n


def horizontal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to `up`."""
    up = _normalize(np.asarray(up, float))
    # Start from the world axis least aligned with up for numerical stability.
    seed = np.eye(3)[int(np.argmin(np.abs(up)))]
    e1 = _normalize(seed - up * np.dot(seed, up))
    e2 = np.cross(up, e1)
    return e1, e2


def look_at_cv(eye, target, up) -> np.ndarray:
    """OpenCV camera-to-world 4x4 (column-vector) looking from `eye` at `target`.

    Columns of R are the camera axes in world coords: x=right, y=down, z=forward.
    If forward is (anti)parallel to `up`, a substitute up vector is used so the
    basis stays well defined (camera roll is then arbitrary but valid).
    """
    eye = np.asarray(eye, float)
    up = _normalize(np.asarray(up, float))
    z = _normalize(np.asarray(target, float) - eye)
    right = np.cross(z, up)
    if np.linalg.norm(right) < 1e-6:            # degenerate: looking straight up/down
        alt = np.eye(3)[int(np.argmin(np.abs(z)))]
        right = np.cross(z, alt)
    x = _normalize(right)
    y = np.cross(z, x)                           # = "down" when right = forward x up
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def cv_to_usd_cam_pose(T_cv_to_world: np.ndarray) -> np.ndarray:
    """OpenCV cam->world  ->  USD cam->world (both column-vector)."""
    return T_cv_to_world @ CV_FROM_USD


def usd_to_cv_cam_pose(T_usd_to_world: np.ndarray) -> np.ndarray:
    """USD cam->world  ->  OpenCV cam->world (both column-vector)."""
    return T_usd_to_world @ CV_FROM_USD


def usd_to_col(m_row_vector) -> np.ndarray:
    """USD/Gf (row-vector, p' = p @ M) 4x4 or flat-16 -> column-vector 4x4."""
    return np.asarray(m_row_vector, dtype=float).reshape(4, 4).T.copy()


def col_to_usd(T_col: np.ndarray) -> np.ndarray:
    """Column-vector 4x4 -> USD/Gf row-vector 4x4 (just the transpose)."""
    return np.asarray(T_col, dtype=float).T.copy()


def scale_translation(T: np.ndarray, s: float) -> np.ndarray:
    """Rigid transform with translation multiplied by s (stage units -> meters)."""
    T = np.array(T, dtype=float)
    T[:3, 3] *= s
    return T


def rigid_inverse(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def focal_from_hfov(hfov_deg: float, h_aperture: float) -> float:
    return 0.5 * h_aperture / math.tan(math.radians(hfov_deg) / 2.0)


def hfov_from_focal(focal: float, h_aperture: float) -> float:
    return math.degrees(2.0 * math.atan(0.5 * h_aperture / focal))


def intrinsics_from_camera(focal: float, h_aperture: float, width: int, height: int,
                           principal_point: str = "exact") -> np.ndarray:
    """Pinhole K from USD camera params (square pixels, no aperture offset).

    principal_point="exact": cx=(W-1)/2 (pixel centers at integer coords, contract).
    principal_point="half":  cx=W/2 (Isaac Sim helper convention).
    """
    fx = focal / h_aperture * width
    fy = fx
    if principal_point == "half":
        cx, cy = width / 2.0, height / 2.0
    else:
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def fx_from_projection(proj_flat16, width: int) -> float:
    """fx implied by Replicator's cameraProjection (OpenGL-style, row-major).

    P[0][0] = 2 f / h_aperture  (diagonal is unaffected by row/column layout), and
    NDC x in [-1, 1] spans W pixels, so fx = P00 * W / 2.
    """
    P = np.asarray(proj_flat16, dtype=float).reshape(4, 4)
    return float(P[0, 0]) * width / 2.0


def sanitize_depth(depth, meters_per_unit: float, max_valid_stage: float) -> np.ndarray:
    """Raw distance_to_image_plane (stage units) -> float32 meters, no-hit = 0."""
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:                       # some versions return (H, W, 1)
        d = d[..., 0]
    bad = ~np.isfinite(d) | (d <= 0.0) | (d >= max_valid_stage)
    d = np.where(bad, 0.0, d * meters_per_unit)
    return d.astype(np.float32)


def dome_poses(center, radius: float, rings_deg, num_views: int, up,
               rng: np.random.Generator) -> list[np.ndarray]:
    """OpenCV cam->world poses on a hemisphere (elevation rings) looking at center."""
    center = np.asarray(center, float)
    up = _normalize(np.asarray(up, float))
    e1, e2 = horizontal_basis(up)
    rings = list(rings_deg)
    base, rem = divmod(max(num_views, len(rings)), len(rings))
    poses = []
    for i, el in enumerate(rings):
        n = base + (1 if i < rem else 0)
        phase = rng.uniform(0.0, 2 * math.pi / n)
        el_r = math.radians(el)
        for k in range(n):
            az = phase + 2 * math.pi * k / n
            d = math.cos(el_r) * (math.cos(az) * e1 + math.sin(az) * e2) + math.sin(el_r) * up
            poses.append(look_at_cv(center + radius * d, center, up))
    return poses


def interior_positions(bmin, bmax, up_axis_idx: int, eye_height: float,
                       num_positions: int, offset_frac: float) -> list[np.ndarray]:
    """Camera centers inside the bbox at eye height: bbox center + a ring of offsets."""
    bmin, bmax = np.asarray(bmin, float), np.asarray(bmax, float)
    c = 0.5 * (bmin + bmax)
    floor, ceil = bmin[up_axis_idx], bmax[up_axis_idx]
    h = ceil - floor
    # Clamp to the middle of the vertical extent in case the bbox is short / odd.
    c[up_axis_idx] = float(np.clip(floor + eye_height, floor + 0.2 * h, floor + 0.8 * h))
    pts = [c.copy()]
    horiz = [a for a in range(3) if a != up_axis_idx]
    half = 0.5 * (bmax - bmin)
    for k in range(max(0, num_positions - 1)):
        ang = 2 * math.pi * k / (num_positions - 1) + math.pi / 4
        p = c.copy()
        p[horiz[0]] += offset_frac * half[horiz[0]] * math.cos(ang)
        p[horiz[1]] += offset_frac * half[horiz[1]] * math.sin(ang)
        pts.append(p)
    return pts


def interior_poses(positions, pitches_deg, num_views: int, up,
                   rng: np.random.Generator) -> list[np.ndarray]:
    """From each position, look outward at evenly spaced yaws x the given pitches."""
    up = _normalize(np.asarray(up, float))
    e1, e2 = horizontal_basis(up)
    n_yaw = max(1, math.ceil(num_views / (len(positions) * len(pitches_deg))))
    poses = []
    for pos in positions:
        for j, pitch in enumerate(pitches_deg):
            # stagger yaws between pitch rows / positions for better coverage
            phase = rng.uniform(0.0, 2 * math.pi / n_yaw) + j * math.pi / n_yaw
            p_r = math.radians(pitch)
            for k in range(n_yaw):
                yaw = phase + 2 * math.pi * k / n_yaw
                fwd = math.cos(p_r) * (math.cos(yaw) * e1 + math.sin(yaw) * e2) + math.sin(p_r) * up
                poses.append(look_at_cv(pos, np.asarray(pos) + fwd, up))
    return poses


def backproject_depth(depth_m: np.ndarray, K: np.ndarray, T_cam_to_world: np.ndarray,
                      stride: int = 4) -> np.ndarray:
    """Valid depth pixels -> Nx3 world points (contract formula)."""
    H, W = depth_m.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_m[::stride, ::stride]
    m = z > 0
    u, v, z = u[m].astype(float), v[m].astype(float), z[m].astype(float)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z, np.ones_like(z)], axis=0)
    return (T_cam_to_world @ pc)[:3].T


def fraction_in_bounds(points: np.ndarray, bmin, bmax, margin: float) -> float:
    if len(points) == 0:
        return 0.0
    lo = np.asarray(bmin) - margin
    hi = np.asarray(bmax) + margin
    return float(np.all((points >= lo) & (points <= hi), axis=1).mean())


def verify_renders(out_dir: str, num_frames: int = 2, min_frac: float = 0.98) -> bool:
    """Back-project a few frames and check the points land inside the scene bounds.

    A wrong axis flip, a missing transpose or a units bug puts points behind the
    camera / outside the scene, so this catches convention bugs cheaply.
    """
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    frames = meta["frames"]
    if not frames:
        print("[verify] FAIL: no frames in meta.json")
        return False
    bmin = np.array(meta["scene_bounds_world"]["min"])
    bmax = np.array(meta["scene_bounds_world"]["max"])
    margin = 0.05 * float(np.linalg.norm(bmax - bmin)) + 0.05
    idxs = sorted(set(np.linspace(0, len(frames) - 1, num=min(num_frames, len(frames))).astype(int)))
    ok_all = True
    for i in idxs:
        fr = frames[i]
        depth = np.load(os.path.join(out_dir, fr["depth"]))
        K = np.array(fr["K"])
        T_c2w = np.array(fr["T_cam_to_world"])
        T_w2c = np.array(fr["T_world_to_cam"])
        inv_err = float(np.abs(T_w2c @ T_c2w - np.eye(4)).max())
        pts = backproject_depth(depth, K, T_c2w)
        frac = fraction_in_bounds(pts, bmin, bmax, margin)
        ok = frac >= min_frac and inv_err < 1e-5 and depth.shape == (meta["image_height"], meta["image_width"])
        ok_all &= ok
        print(f"[verify] frame {fr['frame_id']}: {len(pts)} pts, {100 * frac:.1f}% inside bounds "
              f"(+-{margin:.2f} m), |T_w2c@T_c2w - I|={inv_err:.1e} -> {'PASS' if ok else 'FAIL'}")
    print(f"[verify] {'PASS' if ok_all else 'FAIL'} ({len(idxs)} frames checked)")
    return ok_all


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--usd", help="input USD/USDA/USDZ (required unless --verify-only)")
    p.add_argument("--out", default="./out/renders", help="output renders/ directory")
    p.add_argument("--scene-id", default=None, help="defaults to the USD file stem")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mode", choices=["dome", "interior"], default="dome",
                   help="dome: orbit outside looking at bbox center; "
                        "interior: stand inside the room looking outward (room scans)")
    p.add_argument("--num-views", type=int, default=36,
                   help="dome: exact; interior: rounded up to positions*pitches*yaws")
    p.add_argument("--rings", default="15,35,60", help="dome elevation rings, degrees")
    p.add_argument("--radius-scale", type=float, default=1.0,
                   help="dome radius multiplier on the 'fit bounding sphere in view' distance")
    lens = p.add_mutually_exclusive_group()
    lens.add_argument("--focal-length-mm", type=float, default=None)
    lens.add_argument("--hfov", type=float, default=None,
                      help="horizontal FOV in degrees (default: 55 dome, 90 interior)")
    p.add_argument("--h-aperture-mm", type=float, default=20.955, help="USD default aperture")
    p.add_argument("--principal-point", choices=["exact", "half"], default="exact",
                   help="exact: c=(W-1)/2 (contract pixel centers); half: c=W/2")
    p.add_argument("--eye-height", type=float, default=1.5, help="interior: meters above bbox floor")
    p.add_argument("--pitches", default="-30,-5,20", help="interior pitch rows, degrees (+ = up)")
    p.add_argument("--interior-positions", type=int, default=3,
                   help="interior: 1 = bbox center only, N = center + N-1 offset points")
    p.add_argument("--interior-offset", type=float, default=0.35,
                   help="interior: offset points at this fraction of the half-extent")
    p.add_argument("--dome-intensity", type=float, default=1000.0)
    p.add_argument("--distant-intensity", type=float, default=0.0,
                   help="optional sun-like DistantLight (0 = off; scans usually have baked lighting)")
    p.add_argument("--rt-subframes", type=int, default=8,
                   help="RTX subframes per capture (settles accumulation after camera jumps)")
    p.add_argument("--warmup-frames", type=int, default=30,
                   help="app updates/steps before the first capture (texture streaming)")
    p.add_argument("--max-empty-frac", type=float, default=0.95,
                   help="skip views where more than this fraction of depth is empty")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verify", action="store_true", help="back-project 2 frames after rendering")
    p.add_argument("--verify-only", action="store_true",
                   help="only verify an existing --out directory (plain python, no Isaac Sim)")
    args = p.parse_args(argv)
    if not args.verify_only and not args.usd:
        p.error("--usd is required")
    return args


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


# ----------------------------------------------------------------------------
# Isaac Sim part (omni / pxr imported only after SimulationApp exists)
# ----------------------------------------------------------------------------

def _log(msg: str):
    print(f"[render_views] {msg}", flush=True)


def _write_json_atomic(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _save_png(path: str, rgb: np.ndarray):
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except ImportError:  # pragma: no cover - PIL ships with Isaac Sim
        import imageio
        imageio.imwrite(path, rgb)


def run(args, simulation_app) -> int:
    import carb
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    W, H = args.width, args.height
    usd_path = os.path.abspath(args.usd)
    out_dir = os.path.abspath(args.out)
    os.makedirs(os.path.join(out_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # --- settings: synchronous material/texture loads, no capture-on-play ---------
    settings = carb.settings.get_settings()
    for key in ("/rtx/materialDb/syncLoads", "/omni.kit.plugin/syncUsdLoads",
                "/rtx/hydra/materialSyncLoads"):
        settings.set(key, True)
    settings.set("/omni/replicator/captureOnPlay", False)
    try:
        rep.orchestrator.set_capture_on_play(False)
    except Exception:  # older/newer Replicator without this helper
        pass

    # --- open the stage as-is (native world frame) ----------------------------
    ctx = omni.usd.get_context()
    _log(f"opening {usd_path}")
    if not ctx.open_stage(usd_path):
        raise RuntimeError(f"failed to open stage {usd_path}")
    for _ in range(2000):                       # wait for payloads/sublayers
        simulation_app.update()
        try:
            _, loaded, total = ctx.get_stage_loading_status()
            if loaded == 0 and total == 0:
                break
        except Exception:
            break
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("stage is None after open")

    up_axis = str(UsdGeom.GetStageUpAxis(stage))         # "Y" or "Z"
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    up = AXIS_VEC[up_axis]
    up_idx = "XYZ".index(up_axis)
    _log(f"upAxis={up_axis} metersPerUnit={mpu}")

    # --- scene bounds (stage units) over the default prim, before adding the rig ----
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        root = stage.GetPseudoRoot()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                                   useExtentsHint=False)
    rng3 = bbox_cache.ComputeWorldBound(root).ComputeAlignedRange()
    if rng3.IsEmpty():
        raise RuntimeError(f"empty bounds for {root.GetPath()}")
    bmin_s, bmax_s = np.array(rng3.GetMin(), float), np.array(rng3.GetMax(), float)
    diag_s = float(np.linalg.norm(bmax_s - bmin_s))
    center_s = 0.5 * (bmin_s + bmax_s)
    _log(f"bounds (m) min={bmin_s * mpu} max={bmax_s * mpu} diag={diag_s * mpu:.2f}")

    # --- lens -------------------------------------------------------------------
    h_ap = args.h_aperture_mm
    if args.focal_length_mm is not None:
        focal = args.focal_length_mm
    else:
        focal = focal_from_hfov(args.hfov or (90.0 if args.mode == "interior" else 55.0), h_ap)
    v_ap = h_ap * H / W
    hfov = hfov_from_focal(focal, h_ap)
    vfov = math.degrees(2 * math.atan(0.5 * v_ap / focal))
    K = intrinsics_from_camera(focal, h_ap, W, H, args.principal_point)
    _log(f"focal={focal:.3f}mm h_ap={h_ap}mm hfov={hfov:.1f} vfov={vfov:.1f} fx={K[0, 0]:.2f}")

    # --- camera poses (computed in stage units, OpenCV cam->world) -------------
    if args.mode == "dome":
        r_fit = 0.5 * diag_s / math.sin(math.radians(min(hfov, vfov)) / 2.0)
        poses_cv = dome_poses(center_s, args.radius_scale * r_fit, _parse_floats(args.rings),
                              args.num_views, up, rng)
    else:
        positions = interior_positions(bmin_s, bmax_s, up_idx, args.eye_height / mpu,
                                       args.interior_positions, args.interior_offset)
        poses_cv = interior_poses(positions, _parse_floats(args.pitches), args.num_views, up, rng)
    _log(f"{len(poses_cv)} camera poses ({args.mode})")

    # --- rig on the session layer: lights + camera (scan layer untouched) ------
    rig = "/PerceptionRig"
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Scope.Define(stage, rig)
        if args.dome_intensity > 0:
            dome = UsdLux.DomeLight.Define(stage, f"{rig}/DomeLight")
            dome.CreateIntensityAttr(args.dome_intensity)
        if args.distant_intensity > 0:
            sun = UsdLux.DistantLight.Define(stage, f"{rig}/DistantLight")
            sun.CreateIntensityAttr(args.distant_intensity)
            sun.CreateAngleAttr(1.0)
            # light shines down its local -Z, same as a USD camera looks
            e1, _ = horizontal_basis(up)
            T_sun = cv_to_usd_cam_pose(look_at_cv(np.zeros(3), -up + 0.5 * e1, up))
            xf = UsdGeom.Xformable(sun.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d(*col_to_usd(T_sun).flatten().tolist()))
        cam = UsdGeom.Camera.Define(stage, f"{rig}/Camera")
        cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        cam.CreateFocalLengthAttr(float(focal))
        cam.CreateHorizontalApertureAttr(float(h_ap))
        cam.CreateVerticalApertureAttr(float(v_ap))
        cam.CreateHorizontalApertureOffsetAttr(0.0)
        cam.CreateVerticalApertureOffsetAttr(0.0)
        near_s = 0.01 / mpu
        far_s = max(1000.0 / mpu, 10.0 * diag_s)
        cam.CreateClippingRangeAttr(Gf.Vec2f(float(near_s), float(far_s)))
        cam_xf = UsdGeom.Xformable(cam.GetPrim())
        cam_xf.ClearXformOpOrder()
        cam_op = cam_xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)

    def set_cam_pose(T_usd_to_world_stage: np.ndarray):
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            cam_op.Set(Gf.Matrix4d(*col_to_usd(T_usd_to_world_stage).flatten().tolist()))

    # --- one render product, reused for all poses -------------------------------
    set_cam_pose(cv_to_usd_cam_pose(poses_cv[0]))
    rp = rep.create.render_product(str(cam.GetPath()), (W, H))
    ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    ann_depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    ann_cam = rep.AnnotatorRegistry.get_annotator("camera_params")
    for a in (ann_rgb, ann_depth, ann_cam):
        a.attach([rp])

    _log(f"warming up ({args.warmup_frames} frames)")
    for _ in range(args.warmup_frames):
        simulation_app.update()
    rep.orchestrator.step(rt_subframes=args.rt_subframes)

    # --- meta skeleton ----------------------------------------------------------
    meta = {
        "scene_id": args.scene_id or os.path.splitext(os.path.basename(usd_path))[0],
        "scene_usd": usd_path,
        "up_axis": up_axis,
        "meters_per_unit": mpu,
        "image_width": W,
        "image_height": H,
        "depth_type": "distance_to_image_plane",
        "camera_convention": "opencv",
        "scene_bounds_world": {"min": (bmin_s * mpu).tolist(), "max": (bmax_s * mpu).tolist()},
        "frames": [],
        # extras (not required by the contract; harmless to readers)
        "render_mode": args.mode,
        "hfov_deg": hfov,
        "skipped_views": [],
    }
    meta_path = os.path.join(out_dir, "meta.json")
    warned_K = warned_T = False

    t0 = time.time()
    for vi, T_cv_stage in enumerate(poses_cv):
        T_usd_stage = cv_to_usd_cam_pose(T_cv_stage)
        set_cam_pose(T_usd_stage)

        rgb = depth = cparams = None
        for _attempt in range(3):              # annotators can be empty on the first step
            rep.orchestrator.step(rt_subframes=args.rt_subframes)
            rgb = np.asarray(ann_rgb.get_data())
            depth = np.asarray(ann_depth.get_data())
            cparams = ann_cam.get_data()
            if rgb.size and depth.size:
                break
        if not (rgb.size and depth.size):
            _log(f"view {vi}: annotators returned no data, skipping")
            meta["skipped_views"].append({"view_index": vi, "reason": "no data"})
            continue

        # --- cross-checks against what the renderer actually used --------------
        T_w2c_usd_stage = rigid_inverse(T_usd_stage)
        try:
            prim_T = usd_to_col(np.array(UsdGeom.Xformable(cam.GetPrim())
                                         .ComputeLocalToWorldTransform(Usd.TimeCode.Default())))
            if np.abs(prim_T - T_usd_stage).max() > 1e-4 * max(1.0, diag_s):
                _log(f"WARNING view {vi}: camera prim world xform != requested pose")
        except Exception as e:
            _log(f"WARNING: could not read camera prim xform: {e}")
        if cparams:
            try:
                V = usd_to_col(cparams["cameraViewTransform"])      # world->USD view
                err = np.abs(V - T_w2c_usd_stage).max()
                if err > 1e-3 * max(1.0, diag_s) and not warned_T:
                    _log(f"WARNING view {vi}: cameraViewTransform differs from pose by {err:.2e} "
                         f"(stale frame or convention mismatch)\n{V}\nvs\n{T_w2c_usd_stage}")
                    warned_T = True
                fx_p = fx_from_projection(cparams["cameraProjection"], W)
                if abs(fx_p - K[0, 0]) > 1e-3 * K[0, 0] and not warned_K:
                    _log(f"WARNING: fx from cameraProjection={fx_p:.3f} vs K fx={K[0, 0]:.3f}")
                    warned_K = True
            except Exception as e:
                if not warned_T:
                    _log(f"WARNING: camera_params cross-check skipped: {e}")
                    warned_T = True

        # --- images ---------------------------------------------------------------
        depth_m = sanitize_depth(depth, mpu, max_valid_stage=0.999 * far_s)
        if depth_m.shape != (H, W):
            _log(f"view {vi}: unexpected depth shape {depth_m.shape}, skipping")
            meta["skipped_views"].append({"view_index": vi, "reason": f"shape {depth_m.shape}"})
            continue
        empty = float((depth_m <= 0).mean())
        if empty > args.max_empty_frac:
            _log(f"view {vi}: {100 * empty:.1f}% empty depth, skipping")
            meta["skipped_views"].append({"view_index": vi, "reason": f"empty {empty:.3f}"})
            continue
        rgb8 = rgb[..., :3]
        if rgb8.dtype != np.uint8:             # float LDR fallback
            rgb8 = (np.clip(rgb8, 0, 1) * 255 + 0.5).astype(np.uint8)
        rgb8 = np.ascontiguousarray(rgb8)

        fid = f"{len(meta['frames']):04d}"
        _save_png(os.path.join(out_dir, "rgb", f"{fid}.png"), rgb8)
        np.save(os.path.join(out_dir, "depth", f"{fid}.npy"), depth_m)

        # --- extrinsics in meters, OpenCV camera frame ----------------------------
        T_c2w = scale_translation(T_cv_stage, mpu)
        T_w2c = rigid_inverse(T_c2w)
        meta["frames"].append({
            "frame_id": fid,
            "rgb": f"rgb/{fid}.png",
            "depth": f"depth/{fid}.npy",
            "K": K.tolist(),
            "T_world_to_cam": T_w2c.tolist(),
            "T_cam_to_world": T_c2w.tolist(),
            "view_index": vi,
        })
        _write_json_atomic(meta_path, meta)    # keep meta valid if interrupted
        _log(f"view {vi} -> frame {fid} ({100 * (1 - empty):.0f}% depth hits)")

    try:
        rep.orchestrator.wait_until_complete()
    except Exception:
        pass
    _write_json_atomic(meta_path, meta)
    _log(f"wrote {len(meta['frames'])} frames ({len(meta['skipped_views'])} skipped) "
         f"to {out_dir} in {time.time() - t0:.1f}s")
    for a in (ann_rgb, ann_depth, ann_cam):
        try:
            a.detach([rp])
        except Exception:
            pass
    try:
        rp.destroy()
    except Exception:
        pass

    if not meta["frames"]:
        _log("ERROR: no frames written")
        return 1
    if args.verify and not verify_renders(out_dir):
        return 2
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        return 0 if verify_renders(os.path.abspath(args.out)) else 2

    from isaacsim import SimulationApp   # must precede every omni / pxr import
    simulation_app = SimulationApp({"headless": args.headless, "width": args.width,
                                    "height": args.height})
    rc = 1
    try:
        rc = run(args, simulation_app)
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    finally:
        # The stage is never saved; closing discards the session-layer rig.
        simulation_app.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
