"""Camera geometry shared by the Isaac Sim renderer and the CV pipeline.

numpy only, so it imports cleanly inside Isaac Sim's bundled Python and in a
plain CV environment.

Conventions used everywhere in this package
-------------------------------------------
* Matrices are 4x4, column-vector convention:  p' = T @ [x, y, z, 1]^T.
  (USD's Gf.Matrix4d is row-vector convention, i.e. the transpose.)
* Saved cameras use the OpenCV camera frame: +X right, +Y down, +Z forward.
  USD/OpenGL cameras are +X right, +Y up, looking down -Z.
* Pixel (u, v) = (column, row) index; its centre sits at (u + 0.5, v + 0.5),
  and the principal point of an un-shifted camera is (W/2, H/2).
* Depth is "distance to image plane" (z in the camera frame), not ray length.
"""

import numpy as np

# Right-multiplying a USD camera-to-world matrix by this flips Y and Z, giving
# the OpenCV camera-to-world matrix for the same physical camera.
USD_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def intrinsics_from_usd(focal_length, horizontal_aperture, width, height):
    """3x3 pinhole K for a USD camera rendered at width x height.

    focal_length and horizontal_aperture just need the same unit (USD uses mm).
    RTX derives the vertical aperture from the aspect ratio, so pixels are square.
    """
    fx = focal_length / horizontal_aperture * width
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fx, height / 2.0],
                     [0.0, 0.0, 1.0]])


def look_at_usd(eye, target, up):
    """Camera-to-world matrix of a USD camera at `eye` looking at `target`."""
    eye, target, up = (np.asarray(a, dtype=np.float64) for a in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:  # looking straight along `up`
        right = np.cross(forward, np.roll(up, 1))
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = cam_up
    c2w[:3, 2] = -forward  # USD cameras look down their local -Z
    c2w[:3, 3] = eye
    return c2w


def invert_rigid(T):
    """Inverse of a rigid [R|t] transform without a general matrix inverse."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def backproject(u, v, depth, K):
    """Pixels (u, v) with z-depth -> (N, 3) points in the OpenCV camera frame."""
    u, v, depth = (np.asarray(a, dtype=np.float64) for a in (u, v, depth))
    x = (u + 0.5 - K[0, 2]) * depth / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * depth / K[1, 1]
    return np.stack([x, y, depth], axis=-1)


def transform_points(T, points):
    """Apply a 4x4 transform to (N, 3) points."""
    return points @ T[:3, :3].T + T[:3, 3]


def pixels_to_world(u, v, depth, K, world_to_camera):
    """The full lift: pixel + depth -> world coordinates, (N, 3)."""
    return transform_points(invert_rigid(world_to_camera), backproject(u, v, depth, K))


def project(points_world, K, world_to_camera):
    """World points -> continuous pixel coords (u, v) and depth z. Inverse of the lift."""
    pc = transform_points(world_to_camera, np.asarray(points_world, dtype=np.float64))
    z = pc[:, 2]
    u = K[0, 0] * pc[:, 0] / z + K[0, 2] - 0.5
    v = K[1, 1] * pc[:, 1] / z + K[1, 2] - 0.5
    return u, v, z


def aabb_iou(a_min, a_max, b_min, b_max):
    """Intersection-over-union of two axis-aligned 3D boxes."""
    a_min, a_max, b_min, b_max = (np.asarray(x, dtype=np.float64) for x in (a_min, a_max, b_min, b_max))
    inter = np.prod(np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0, None))
    union = np.prod(a_max - a_min) + np.prod(b_max - b_min) - inter
    return float(inter / union) if union > 0 else 0.0
