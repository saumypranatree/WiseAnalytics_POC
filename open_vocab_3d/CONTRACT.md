# Inter-stage data contract

Three stages that talk only through files on disk. Each stage can be run,
replaced, or debugged on its own.

```
room_scan.usda ──[1 render_views.py (Isaac Sim python)]──► out/renders/
out/renders/   ──[2 detect_2d.py (torch, transformers)]───► out/detections/
renders + detections ──[3 lift_cluster.py (numpy, sklearn/open3d)]──► out/scene_objects.json
```

## Conventions (all stages MUST obey)

* **World frame** = the native frame of the input USD stage, scaled to **meters**
  (multiply stage units by `metersPerUnit`). Do NOT rotate Y-up to Z-up. The
  output JSON is in this frame, and `up_axis` is recorded in `meta.json`.
* **Camera frame = OpenCV convention**: +X right, +Y down, +Z forward (looking
  direction). USD/Isaac cameras look down -Z with +Y up, so the render stage
  converts: `T_cam_cv = diag(1,-1,-1,1) @ T_cam_usd`.
* **Matrices** are 4x4, column-vector convention (`p_cam = T_world_to_cam @ p_world_h`),
  stored row-major as nested lists `[[r0],[r1],[r2],[r3]]`.
* **Depth** = `distance_to_image_plane` (z-depth along the optical axis, not
  ray length), float32, meters, shape (H, W). Pixels with no hit are `0.0`
  (convert inf/NaN to 0).
* **Pixel coordinates**: u = column, v = row, origin top-left, pixel centers at
  integer coordinates (u, v) — i.e. pixel (u, v) back-projects as
  `x = (u - cx) * z / fx`.

## Stage 1 output: `out/renders/`

```
out/renders/meta.json
out/renders/rgb/0000.png        # uint8 RGB, HxWx3
out/renders/depth/0000.npy      # float32 meters, HxW
```

`meta.json`:
```json
{
  "scene_id": "room_scan",
  "scene_usd": "/abs/path/room_scan.usda",
  "up_axis": "Y",
  "meters_per_unit": 1.0,
  "image_width": 1280,
  "image_height": 720,
  "depth_type": "distance_to_image_plane",
  "camera_convention": "opencv",
  "scene_bounds_world": {"min": [x, y, z], "max": [x, y, z]},
  "frames": [
    {
      "frame_id": "0000",
      "rgb": "rgb/0000.png",
      "depth": "depth/0000.npy",
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "T_world_to_cam": [[...4], [...4], [...4], [...4]],
      "T_cam_to_world": [[...4], [...4], [...4], [...4]]
    }
  ]
}
```
Paths inside `meta.json` are relative to the `renders/` directory.

## Stage 2 output: `out/detections/`

```
out/detections/detections.json
out/detections/masks/0000_00.png   # uint8, HxW, 0 or 255 (only if SAM enabled)
out/detections/vis/0000.jpg        # optional debug overlay
```

`detections.json`:
```json
{
  "detector": "IDEA-Research/grounding-dino-base",
  "segmenter": "facebook/sam-vit-base",          // or null
  "prompts": ["chair", "monitor", "coffee cup"],
  "box_threshold": 0.3,
  "text_threshold": 0.25,
  "frames": [
    {
      "frame_id": "0000",
      "detections": [
        {
          "det_id": "0000_00",
          "class_name": "chair",                  // MUST be one of `prompts` exactly
          "score": 0.61,
          "bbox_xyxy": [x0, y0, x1, y1],          // float pixels, same image size as rgb
          "mask": "masks/0000_00.png"             // relative to detections/, or null
        }
      ]
    }
  ]
}
```

## Stage 3 output: `out/scene_objects.json`

```json
{
  "scene_id": "room_scan",
  "objects": [
    {
      "class_name": "chair",
      "confidence": 0.58,
      "position_world": {"x": 0.0, "y": 0.0, "z": 0.0},
      "bounding_box_3d": {"min": [x, y, z], "max": [x, y, z]}
    }
  ]
}
```
`position_world` is the centroid of the cluster's points. With `--extended`,
each object also carries `"num_views"`, `"num_points"`, `"source_detections"`.
