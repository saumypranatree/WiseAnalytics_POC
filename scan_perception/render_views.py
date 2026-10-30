"""Stage 1 - render RGB, depth and camera parameters from an unsegmented USD scan.

Run with Isaac Sim's Python (not your CV environment):
    ~/isaacsim/python.sh scan_perception/render_views.py --usd room_scan.usda --out out/views

Everything this script adds (camera, lights) goes on the stage's session layer,
so the source USD is never modified.

Trajectories (--mode):
  interior  Cameras on a grid inside the scan at eye height, each turning through
            several yaw angles and pitches. Use for rooms: from outside, a dome
            only sees the walls.
  dome      Cameras on rings around the scan, looking at its centre. Use for
            open scenes or single objects.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import USD_TO_CV, intrinsics_from_usd, invert_rigid, look_at_usd  # noqa: E402
from views import MANIFEST, write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--usd", required=True, help="Path to the .usd/.usda/.usdc/.usdz scan")
    p.add_argument("--out", required=True, help="Output views directory")
    p.add_argument("--scene-id", default=None, help="Defaults to the USD file stem")
    p.add_argument("--mode", choices=["interior", "dome"], default="interior")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--focal-length", type=float, default=14.0, help="mm; 14 mm on 20.955 mm aperture is ~73 deg HFOV")
    p.add_argument("--horizontal-aperture", type=float, default=20.955, help="mm")
    # interior mode
    p.add_argument("--grid", type=int, default=3, help="interior: NxN camera positions")
    p.add_argument("--grid-margin", type=float, default=0.25, help="interior: fraction of the footprint kept clear at each side")
    p.add_argument("--eye-height", type=float, default=1.4, help="interior: metres above the detected floor")
    p.add_argument("--yaw-steps", type=int, default=8, help="interior: headings per position")
    p.add_argument("--pitches", type=float, nargs="+", default=[-10.0, -40.0], help="interior: degrees, negative looks down")
    # dome mode
    p.add_argument("--dome-elevations", type=float, nargs="+", default=[20.0, 45.0, 70.0], help="dome: degrees above horizon")
    p.add_argument("--dome-per-ring", type=int, default=16)
    p.add_argument("--dome-radius-scale", type=float, default=1.3, help="dome: radius as a multiple of the scene half-diagonal")
    # rendering
    p.add_argument("--color-source", choices=["albedo", "lit"], default="albedo",
                   help="albedo = unlit texture colour (scans already have lighting baked in); lit = path-traced with added lights")
    p.add_argument("--headlight-intensity", type=float, default=3.0e4, help="SphereLight riding on the camera (lit mode)")
    p.add_argument("--dome-light-intensity", type=float, default=1000.0)
    p.add_argument("--rt-subframes", type=int, default=8, help="RTX accumulation frames per capture; raise if images are noisy")
    p.add_argument("--warmup-frames", type=int, default=60, help="app updates before the first capture so textures finish streaming")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    # SimulationApp/Kit may receive its own flags; ignore what we do not know.
    args, _ = p.parse_known_args()
    return args


ARGS = parse_args()

# SimulationApp must be created before any omni/pxr import.
try:
    from isaacsim import SimulationApp  # Isaac Sim 4.x / 5.x
except ImportError:
    from omni.isaac.kit import SimulationApp  # Isaac Sim 2023.x

simulation_app = SimulationApp({
    "headless": ARGS.headless,
    "width": ARGS.width,
    "height": ARGS.height,
    "renderer": "RayTracedLighting",
})

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402

RIG_PATH = "/PerceptionRig"
CAMERA_PATH = RIG_PATH + "/Camera"


# --------------------------------------------------------------------------- stage

def open_stage(usd_path):
    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(usd_path)):
        raise RuntimeError(f"Could not open stage {usd_path}")
    while ctx.get_stage_loading_status()[2] > 0:
        simulation_app.update()
    return ctx.get_stage()


def axes_for(stage):
    """(up, horizontal_a, horizontal_b) unit vectors for the stage's up axis."""
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z:
        return np.array([0.0, 0, 1]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    return np.array([0.0, 1, 0]), np.array([1.0, 0, 0]), np.array([0.0, 0, 1])


def world_vertices(stage):
    """All mesh vertices in world space. Scans often lack authored extents, so
    BBoxCache can come back empty; reading points directly always works."""
    chunks = []
    time = Usd.TimeCode.Default()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility(time) == UsdGeom.Tokens.invisible:
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get(time)
        if not pts:
            continue
        l2w = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)).T
        pts = np.asarray(pts, dtype=np.float64)
        chunks.append(pts @ l2w[:3, :3].T + l2w[:3, 3])
    if not chunks:
        raise RuntimeError("Stage contains no visible UsdGeom.Mesh prims")
    return np.concatenate(chunks)


# --------------------------------------------------------------------- trajectories

def interior_poses(verts, up, ha, hb, mpu):
    """Grid of eye-height positions inside the scan, each with yaw x pitch headings."""
    h_up = verts @ up
    floor = np.percentile(h_up, 2)  # robust to a few stray vertices below the floor
    eye_h = floor + ARGS.eye_height / mpu
    a, b = verts @ ha, verts @ hb
    a_lo, a_hi = np.percentile(a, [5, 95])
    b_lo, b_hi = np.percentile(b, [5, 95])
    m = ARGS.grid_margin
    a_vals = np.linspace(a_lo + m * (a_hi - a_lo), a_hi - m * (a_hi - a_lo), ARGS.grid)
    b_vals = np.linspace(b_lo + m * (b_hi - b_lo), b_hi - m * (b_hi - b_lo), ARGS.grid)

    poses = []
    for av in a_vals:
        for bv in b_vals:
            eye = av * ha + bv * hb + eye_h * up
            for k in range(ARGS.yaw_steps):
                yaw = 2 * math.pi * k / ARGS.yaw_steps
                heading = math.cos(yaw) * ha + math.sin(yaw) * hb
                for pitch in ARGS.pitches:
                    target = eye + heading + math.tan(math.radians(pitch)) * up
                    poses.append(look_at_usd(eye, target, up))
    return poses


def dome_poses(verts, up, ha, hb):
    """Rings of cameras around the scene centre, all looking at it."""
    lo, hi = verts.min(0), verts.max(0)
    centre = (lo + hi) / 2
    radius = ARGS.dome_radius_scale * np.linalg.norm(hi - lo) / 2
    poses = []
    for el_deg in ARGS.dome_elevations:
        el = math.radians(el_deg)
        for k in range(ARGS.dome_per_ring):
            az = 2 * math.pi * k / ARGS.dome_per_ring
            offset = math.cos(el) * (math.cos(az) * ha + math.sin(az) * hb) + math.sin(el) * up
            poses.append(look_at_usd(centre + radius * offset, centre, up))
    return poses


# ------------------------------------------------------------------- rig + capture

def build_rig(stage, mpu, scene_extent):
    """Camera (+ headlight) and a dome light, authored on the session layer."""
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage, RIG_PATH)
        cam = UsdGeom.Camera.Define(stage, CAMERA_PATH)
        cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        cam.CreateFocalLengthAttr(ARGS.focal_length)
        cam.CreateHorizontalApertureAttr(ARGS.horizontal_aperture)
        cam.CreateVerticalApertureAttr(ARGS.horizontal_aperture * ARGS.height / ARGS.width)
        cam.CreateHorizontalApertureOffsetAttr(0.0)
        cam.CreateVerticalApertureOffsetAttr(0.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.02 / mpu, float(4 * scene_extent)))
        xform_op = UsdGeom.Xformable(cam).AddTransformOp()

        if ARGS.color_source == "lit":
            # Rides with the camera, so closed rooms are lit wherever it looks.
            head = UsdLux.SphereLight.Define(stage, CAMERA_PATH + "/Headlight")
            head.CreateRadiusAttr(0.05 / mpu)
            head.CreateIntensityAttr(ARGS.headlight_intensity)
        dome = UsdLux.DomeLight.Define(stage, RIG_PATH + "/DomeLight")
        dome.CreateIntensityAttr(ARGS.dome_light_intensity)
    return cam, xform_op


def attach_annotators(render_product):
    color = None
    if ARGS.color_source == "albedo":
        try:
            color = rep.AnnotatorRegistry.get_annotator("DiffuseAlbedo")
        except Exception as exc:  # name differs on some Replicator versions
            print(f"[render_views] DiffuseAlbedo annotator unavailable ({exc}); falling back to lit rgb")
    if color is None:
        color = rep.AnnotatorRegistry.get_annotator("rgb")
    depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    cam_params = rep.AnnotatorRegistry.get_annotator("camera_params")
    for annot in (color, depth, cam_params):
        annot.attach(render_product)
    return color, depth, cam_params


def replicator_pose_error(cam_params, world_to_camera_usd):
    """Max abs difference between our extrinsics and what Replicator says it rendered.

    camera_params' cameraViewTransform is the USD (row-vector) world-to-camera
    matrix; transposing it gives our column-vector convention. Returns 0.0 if the
    annotator payload is not in the expected shape.
    """
    try:
        view = np.asarray(cam_params["cameraViewTransform"], dtype=np.float64).reshape(4, 4).T
    except (KeyError, TypeError, ValueError):
        return 0.0
    scale = max(1.0, np.abs(world_to_camera_usd[:3, 3]).max())
    return float(np.abs(view - world_to_camera_usd).max() / scale)


def main():
    usd_path = Path(ARGS.usd).resolve()
    out = Path(ARGS.out).resolve()
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    stage = open_stage(usd_path)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up, ha, hb = axes_for(stage)
    verts = world_vertices(stage)  # before adding the rig
    extent = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    poses = interior_poses(verts, up, ha, hb, mpu) if ARGS.mode == "interior" else dome_poses(verts, up, ha, hb)
    print(f"[render_views] {len(verts)} vertices, extent {extent:.2f} units, {len(poses)} views ({ARGS.mode})")

    cam, xform_op = build_rig(stage, mpu, extent)
    render_product = rep.create.render_product(CAMERA_PATH, (ARGS.width, ARGS.height))
    color_annot, depth_annot, cam_annot = attach_annotators(render_product)
    K = intrinsics_from_usd(ARGS.focal_length, ARGS.horizontal_aperture, ARGS.width, ARGS.height)

    for _ in range(ARGS.warmup_frames):
        simulation_app.update()

    manifest = {
        "scene_id": ARGS.scene_id or usd_path.stem,
        "usd_path": str(usd_path),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": mpu,
        "width": ARGS.width,
        "height": ARGS.height,
        "depth_type": "distance_to_image_plane",
        "camera_convention": "opencv (+x right, +y down, +z forward); matrices column-vector",
        "color_source": ARGS.color_source,
        "views": [],
    }

    time = Usd.TimeCode.Default()
    for i, c2w_usd in enumerate(poses):
        view_id = f"view_{i:04d}"
        xform_op.Set(Gf.Matrix4d(*c2w_usd.T.flatten().tolist()))  # Gf is row-vector: transpose
        # Read the pose back from USD so the saved extrinsics are exactly what was set.
        c2w_usd = np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(time)).T
        # Some Replicator versions return the previous frame right after a pose
        # change; re-step until camera_params agrees with the pose we set.
        for attempt in range(3):
            rep.orchestrator.step(rt_subframes=ARGS.rt_subframes)
            err = replicator_pose_error(cam_annot.get_data(), invert_rigid(c2w_usd))
            if err < 1e-4:
                break
        else:
            print(f"[render_views] WARNING {view_id}: camera_params still differs from the set pose (rel err {err:.3g})")
        c2w_cv = c2w_usd @ USD_TO_CV

        rgb = np.asarray(color_annot.get_data())[..., :3].astype(np.uint8)
        depth = np.asarray(depth_annot.get_data(), dtype=np.float32)
        Image.fromarray(rgb).save(out / "rgb" / f"{view_id}.png")
        np.save(out / "depth" / f"{view_id}.npy", depth)

        manifest["views"].append({
            "view_id": view_id,
            "rgb": f"rgb/{view_id}.png",
            "depth": f"depth/{view_id}.npy",
            "K": K.tolist(),
            "world_to_camera": invert_rigid(c2w_cv).tolist(),
            "camera_to_world": c2w_cv.tolist(),
        })
        write_json(out / MANIFEST, manifest)  # every view, so a crash keeps what rendered
        if i % 10 == 0:
            hit = np.isfinite(depth) & (depth > 0)
            print(f"[render_views] {view_id}: {hit.mean():.0%} pixels hit geometry")

    rep.orchestrator.wait_until_complete()
    print(f"[render_views] wrote {len(poses)} views to {out}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
