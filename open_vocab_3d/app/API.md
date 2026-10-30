# Viewer app contract

Local web app: view `room_scan.usda` in a browser with the detected 3D boxes and
class labels, run the pipeline from the UI, and correct the results.

```
browser (app/static/: three.js viewer)
   │  HTTP JSON on http://127.0.0.1:8000
   ▼
app/server.py (FastAPI)
   ├─ GET  model.glb  ◄── usd_to_gltf.py (USD → GLB, cached in out/viewer/)
   ├─ GET/PUT objects ◄─► out/scene_objects.json
   └─ POST run        ──► subprocesses: render_views_lite.py → detect_2d.py → lift_cluster.py
```

Python env for everything: `open_vocab_3d/.venv` (CUDA torch, usd-core, open3d,
sklearn, fastapi, uvicorn). Install list: `requirements-app.txt`.

## File ownership

| Path | Owner |
|---|---|
| `.venv/`, `requirements-app.txt`, GPU changes to `detect_2d.py` | detection agent |
| `usd_to_gltf.py`, `tests/test_usd_to_gltf.py` | USD/simulation agent |
| `app/server.py`, `app/pipeline_config.json`, `tests/test_server.py` | backend/JSON agent |
| `app/static/**` | UI agent |
| tuning of `pipeline_config.json` / `lift_cluster.py` (later) | R&D agent |

Do not touch `../scan_perception/` (another session's project).

## Coordinates

* `scene_objects.json` and the GLB vertex positions are both in the **scan's native
  world frame, in meters** (the app serves `scenes/room_scan_upright.usda`, an upright wrapper around the upside-down `room_scan.usda`; Y-up, metersPerUnit = 1).
* The GLB stores positions as-is: no up-axis rotation is baked into it.
* The viewer puts the model AND all boxes under one `THREE.Group` ("world").
  If `up_axis == "Z"` it rotates that group by −90° about X; if `"Y"`, there's no rotation.
  Boxes therefore line up with the mesh with no per-box conversion.

## Endpoints

### `GET /` → `app/static/index.html` (static files served from `/static/...`)

### `GET /api/scene`
```json
{
  "scene_id": "room_scan",
  "usd_path": "/abs/path/room_scan.usda",
  "model_url": "/api/scene/model.glb",
  "up_axis": "Y",
  "meters_per_unit": 1.0,
  "bounds": {"min": [x, y, z], "max": [x, y, z]},
  "default_prompts": ["window", "cabinet", "decorative plate", "plywood panel", "artificial turf"],
  "has_results": true
}
```

### `GET /api/scene/model.glb`
Binary glTF (`model/gltf-binary`). Built once by `usd_to_gltf.export_glb()` into
`out/viewer/model.glb`; rebuilt if the USD is newer. Meshes use per-corner UVs
(unwelded), the embedded original JPEG textures, and `KHR_materials_unlit`
(the photogrammetry textures already contain the lighting).

### `GET /api/objects`
`scene_objects.json` plus a session `id` (= array index) on each object:
```json
{"scene_id": "room_scan", "objects": [
  {"id": 0, "class_name": "window", "confidence": 0.48,
   "position_world": {"x": 0, "y": 0, "z": 0},
   "bounding_box_3d": {"min": [x, y, z], "max": [x, y, z]},
   "num_views": 6}
]}
```
If there are no results yet: `{"scene_id": "...", "objects": []}`. Extra fields such as
`num_views`/`num_points` are passed through when present.

### `PUT /api/objects`  (human corrections)
Body: the full corrected list, **with rejected objects removed** and renamed objects
carrying their new `class_name`:
```json
{"objects": [{"id": 0, "class_name": "decorative plate", "confidence": 0.48,
              "position_world": {...}, "bounding_box_3d": {...}}]}
```
Server: validates the schema (422 on error), strips `id`, copies the current file
to `scene_objects.json.bak`, then **overwrites** `out/scene_objects.json`
atomically. Returns `{"saved": N, "path": "..."}`. Returns 409 while a pipeline job is running.

### `POST /api/run`
Body (all optional except `prompts`):
```json
{"prompts": ["window", "cabinet"], "num_views": 30, "mode": "dome"}
```
Returns `{"job_id": "..."}`, or 409 if a job is already running. Defaults come from
`app/pipeline_config.json`.

### `GET /api/jobs/{job_id}`
```json
{"job_id": "...", "state": "queued|rendering|detecting|lifting|done|error",
 "progress": 0.42, "message": "detecting frame 12/30", "log_tail": ["...last ~30 lines"],
 "num_objects": 7}
```
`progress` is 0–1 across all three stages (weights: render 0.2, detect 0.7, lift 0.1).
The UI polls this about once a second and reloads `/api/objects` when `state == "done"`.

### `GET /api/jobs/latest` → the same, for the most recent job (or 404).
