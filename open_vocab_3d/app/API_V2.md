# App v2: upload → pipeline → JSON

The user uploads a `.usdz`. The server prepares the scan (extracts it and
fixes its orientation), runs the two-pass pipeline, writes a **raw** JSON, and
shows the model and boxes in the viewer. After human review, **Send to
backend** writes a **reviewed** JSON. For now, delivery means writing files to
`out/backend_outbox/`; there is no network push and no external API consumer.
The server binds to 127.0.0.1 only.

User decisions (2026-10-07):
- Uploads: `.usdz` only.
- Delivery: JSON files on disk.
- JSON fields: class, position, bbox, confidence, views, stable id, scan metadata.
- Two JSONs per scan: raw after the run, reviewed after "Send to backend".
- Orientation: auto-detect plus a manual flip button.
- Prompts: the default 13, editable.
- Cameras: automatic two-pass.
- Frame: upright, Y-up, meters.
- Scan list in the UI.
- One process, localhost.

## Ownership

| Path | Owner |
|---|---|
| `scan_prep.py`, `pipeline_runner.py`, `export_json.py`, `tests/test_scan_prep.py`, `tests/test_pipeline_runner.py`, `tests/test_export_json.py`, `app/pipeline_config.json` | pipeline agent |
| `app/server.py`, `tests/test_server.py` | backend agent |
| `app/static/**` | UI agent |

The existing stage scripts (`render_views_lite.py`, `detect_2d.py`,
`lift_cluster.py`, `usd_to_gltf.py`) may receive small backward-compatible
fixes from the pipeline agent only. Never touch `../scan_perception/`.

## Scan directory layout

```
out/scans/<scan_id>/                 scan_id = "scan_YYYYMMDD_HHMMSS_<4 hex>"
  scan.json                          registry record (below), written atomically
  source/<original_name>.usdz        the upload, byte-for-byte
  source/video.<mp4|m4v|webm>        optional capture video the scan was made from (UI only; the pipeline never reads it)
  source/<video>.<mov|mp4|m4v>       shelf scans: the uploaded store video (kept as uploaded; never served)
  convert/work/, convert/out/        shelf scans: video_to_3d intermediates (keyframes, SfM model with keyframe
                                     camera poses, depth, mesh, texture) and outputs (scan.usdz)
  extracted/                         safe-extracted usdz contents (root layer + textures)
  scene.usda                         wrapper; ALWAYS the stage everything uses
  viewer/model.glb                   built from scene.usda by usd_to_gltf.export_glb
  runs/<job_id>/                     pipeline working dir (pass1/, pass2/, merged/, logs)
  results/scene_objects_raw.json
  results/scene_objects_reviewed.json      (after Send to backend)
out/backend_outbox/<scan_id>_raw.json      copy written when the raw result is ready
out/backend_outbox/<scan_id>_reviewed.json copy written on Send to backend (overwritten on re-send)
```

`scene.usda` wrapper:
```
#usda 1.0
( defaultPrim = "World"  metersPerUnit = <from source>  upAxis = "<from source>" )
def Xform "World" ( references = @extracted/<root_layer>@ )
{ [ double3 xformOp:rotateXYZ = (180, 0, 0)  uniform token[] xformOpOrder = ["xformOp:rotateXYZ"] ] }
```
The bracketed part is present only when flipped. A flip is always a 180°
rotation about the stage's horizontal X axis, never a mirror. For a Z-up
source, flip about X as well; it still maps up to down. If the source's
default prim has its own transform, the reference keeps it.

## `scan_prep.py` (pipeline agent)

```python
prepare_scan(usdz_path, scans_root, original_name) -> dict   # returns the scan.json record
detect_orientation(stage_path) -> {"flip_recommended": bool, "confidence": "high|low", "evidence": {...}}
set_flip(scan_dir, flip: bool, source: "auto"|"manual") -> dict  # rewrites scene.usda, deletes viewer/model.glb
```
- **Safe extraction:** `.usdz` is a zip. Reject absolute paths, `..` entries, symlinks, more than 2000 entries, and more than 2 GB uncompressed. The root layer is the first `.usd*` entry in the zip, per the USDZ spec.
- **Validation:** the stage opens; it contains ≥1 mesh; `metersPerUnit` > 0. The size limit is 500 MB per upload, configurable.
- **Orientation heuristic:** area-weighted horizontal triangles, i.e. those with |normal·up| > 0.9. Compare the height of the biggest horizontal surface with the scene's vertical extent. Use the normal direction too: floor normals point up into the room when the winding is consistent. Recommend a flip when the dominant horizontal surface sits in the top third and/or the normals indicate inversion. Record the evidence numbers.
  - Must recommend flip=True for `../room_scan.usdz`.
  - Must recommend flip=False for that file once its wrapper is flipped.

## `pipeline_runner.py` (pipeline agent)

CLI used by the server as one subprocess:
```
python -u pipeline_runner.py --scan-dir out/scans/<id> --job-id <job_id> --prompts "a,b,c" [--config app/pipeline_config.json]
```
- **Two passes:**
  - **Pass 1:** wide dome views → detect all prompts → lift.
  - **Focus centres:** chosen from the pass-1 objects in `two_pass.anchor_classes` (default: table, bench seat, desk, counter, shelf, sofa, chair). If there are none, use any pass-1 object with a horizontal extent of at least 0.4 m. Merge centres closer than 1.5 m; keep at most `two_pass.max_focus` (3).
  - **Pass 2:** focus-dome views only (`--focus-domes`) → detect.
  - **Final lift:** over the merged pass-1 + pass-2 frames and detections. Frame ids must stay unique.
  - If pass 1 finds no anchors, skip pass 2 and say so in the result.
- **Stdout protocol** (the server parses only these lines; everything else is log):
  - `PROGRESS <0..1 float> <stage> <message>`, where stage ∈ render1, detect1, lift1, render2, detect2, lift, export. Progress must be monotonic.
  - `RESULT <path to results/scene_objects_raw.json>` on success, as the last line.
  - Exit 0 on success; non-zero with a final `ERROR <message>` line.
- **Outputs:**
  - `results/scene_objects_raw.json` via export_json.
  - A copy in `out/backend_outbox/<scan_id>_raw.json`. The outbox is `<scans_root>/../backend_outbox`; create it if missing.
  - Everything intermediate stays under `runs/<job_id>/`.
- The config holds the pass-1 render args, the focus-dome args (radius, views, rings), the detect args (prompt-group-size 2, etc.) and the lift args. There is no scene-specific `--focus-domes` in the config any more.

## `export_json.py` (pipeline agent): the JSON schema

```json
{
  "schema_version": "1.0",
  "status": "raw",                       // "raw" | "reviewed"
  "scan": {
    "scan_id": "scan_20261007_141500_ab12",
    "source_file": "room_scan.usdz",
    "uploaded_at": "2026-10-07T14:15:00+05:30",
    "processed_at": "2026-10-07T14:27:10+05:30",
    "reviewed_at": null,                 // set in reviewed JSON
    "job_id": "...",
    "coordinate_frame": {
      "up_axis": "Y", "units": "meters", "handedness": "right",
      "origin": "scan origin (as authored in the source file)",
      "orientation_fix": "rotate_x_180",  // or "none"
      "orientation_source": "auto",       // or "manual"
      "note": "Positions are in the upright scan frame shown in the viewer: +Y is up."
    },
    "scene_bounds": {"min": [x, y, z], "max": [x, y, z]},
    "prompts": ["table", "..."],
    "pipeline": {"detector": "IDEA-Research/grounding-dino-base", "segmenter": "facebook/sam-vit-base",
                 "renderer": "open3d-raycast", "two_pass": true, "focus_centres": [[x, y, z]]}
  },
  "objects": [
    {
      "object_id": "obj_001",
      "class_name": "laptop",
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "bounding_box": {"min": [x, y, z], "max": [x, y, z], "size": [dx, dy, dz]},
      "confidence": 0.54,
      "num_views": 59
      // reviewed JSON only: "review": {"decision": "accepted" | "renamed", "original_class_name": "laptop"}
    }
  ],
  "summary": {"num_objects": 19, "by_class": {"laptop": 1}},
  // reviewed JSON only:
  "review_summary": {"accepted": 15, "renamed": 2, "rejected_object_ids": ["obj_007", "obj_012"]}
}
```
- `position` is the centroid of the cluster's points (`position_world` from lift_cluster). It is not the box centre.
- `object_id`s are assigned once in the raw JSON (sorted by class, then confidence descending → obj_001…), never change, and are never reused within a scan.
- Floats are rounded to 4 decimals. Timestamps are ISO-8601 with the local offset.
- Functions:
  - `build_raw(scan_record, lift_json_path, job_id, prompts, pipeline_info) -> dict`
  - `build_reviewed(raw: dict, decisions: list[{"object_id", "decision": "accept"|"reject", "class_name"}]) -> dict`
  - `validate(doc) -> list[str]` (empty list = valid)
  - an atomic `write_json(path, doc)`

## Server HTTP API v2 (backend agent; the UI uses only these)

All JSON. Errors are `{"detail": "..."}` with 4xx/5xx.

| Method & path | Purpose |
|---|---|
| `GET /api/config` | `{"default_prompts": [...], "max_upload_mb": 500}` |
| `GET /api/scans` | list of scan summaries, newest first: `{scan_id, source_file, uploaded_at, status, progress, message, num_objects, has_raw, has_reviewed, orientation: {flip_applied, source, flip_recommended, confidence}, has_video}` |
| `POST /api/scans` | multipart upload: `file` (.usdz) + optional `prompts` (comma-separated) + `auto_run` (default true). Saves, runs `prepare_scan` (auto orientation), builds the GLB, and queues a pipeline job if auto_run. Returns the scan summary. Errors: 400 for a bad file or not .usdz, 413 if too large. **Store video** (`.mov`/`.mp4`/`.m4v`, container checked): creates a `scan_type: "shelf"` scan and queues a `convert` job (no prompts, no detection). |
| `POST /api/scans/{id}/scale` | shelf scans only (422 otherwise; 409 before conversion or while a job runs). `{"method": "measured", "p1": [x,y,z], "p2": [x,y,z], "real_cm": 120}` (points in the current scene frame, metres: what the viewer shows) multiplies the scan's uniform scale by `real / distance(p1, p2)`. `{"method": "reset"}` returns to the provisional scale. Writes `xformOp:scale` into scene.usda via `scan_prep.set_scale` (keeps the flip), rebuilds the GLB, and marks existing results stale. Returns the summary + `scale`. |
| `GET /api/scans/{id}/keyframes` | shelf scans: `{width, height, fps, keyframes: [{name, frame, time_s}]}` from `convert/work/cameras_usd.json` (07_cameras.py). 409 before conversion. |
| `GET /api/scans/{id}/keyframes/{name}.jpg` | the original keyframe image (as the camera solve saw it). |
| `GET /api/scans/{id}/labels` | empty-slot ground-truth labels, each lifted to 3D *now* (follows scale/flip): `{id, keyframe, box_px, kind, note, bbox{min,max,size}, center, width_cm, height_cm, depth_cm, front_source, depth_source}` or `error` if it cannot be placed. |
| `POST /api/scans/{id}/labels` | `{"keyframe": "00063", "box_px": [x0,y0,x1,y1], "kind": "empty"|"pushed_back", "note"?}` → the new label (ids `lab_001`, …). 422 on invalid input. Stored 2D-only in `labels/empty_slots.json`. |
| `PATCH` / `DELETE /api/scans/{id}/labels/{label_id}` | change `kind` / `note` / `box_px` / `keyframe`, or delete. A re-conversion archives the labels (`empty_slots.superseded_<ts>.json`): new keyframes and cameras. |
| `POST /api/scans/{id}/gaps/run` | shelf scans: queue a `gaps` job (`gaps_runner.py`, both baselines on the keyframes, ~1.3 s per keyframe on the RTX 5060). 409 without keyframe cameras. Writes `results/gaps.json` in the exported usd frame. |
| `GET /api/scans/{id}/gaps` | gaps of each method (`dino`, `depth`) mapped to the current scene frame (`bbox`, `center`, `width_cm`, `height_cm`, `num_views`, `detections`), plus `scores` per method against the labels: `labelled_slots` (labels of one slot on several keyframes count once), `found`, `missed`, `false_gaps`, `recall`, `precision`, `mean_abs_width_err_cm`, `matches` (shelf-front IoU ≥ 0.25). `scale_changed_since_run` flags that the depth threshold used an older scale. 404 before the first run; a re-conversion archives the results. |
| `POST /api/scans/{id}/convert` | shelf scans only: (re-)queue the video → 3D conversion. 409 if queued/running, 422 for room scans. When the job ends: `scan_prep.prepare_generated` (no auto-flip), GLB build, and the H.264 copy is attached as `source_video`. |
| `GET /api/scans/{id}` | full `scan.json` record plus status; `source_video` is `{file, bytes, content_type, attached_at, url}` or null |
| `GET /api/scans/{id}/video` | the capture video (answers `Range` requests with 206, so the player can seek). 404 if none. |
| `POST /api/scans/{id}/video` | multipart `file` (.mp4/.m4v/.webm, container checked); replaces any previous video. Returns `source_video`. 400 bad file, 413 too large. Allowed while a job runs. |
| `DELETE /api/scans/{id}/video` | removes the video only; the scan and its results are kept. 404 if none. |
| `DELETE /api/scans/{id}` | deletes the scan dir (409 if its job is running); outbox files are kept |
| `GET /api/scans/{id}/model.glb` | the GLB, built lazily if missing |
| `POST /api/scans/{id}/orientation` | `{"flip": bool}` → `set_flip(..., "manual")`, rebuilds the GLB, marks the existing results `stale: true`. 409 if running. |
| `POST /api/scans/{id}/run` | `{"prompts": [...]}` → queues a job (one job runs at a time, FIFO). 409 if this scan is already queued or running. |
| `GET /api/scans/{id}/job` | `{job_id, state: queued/running/done/error, progress, stage, message, log_tail[], queue_position}` |
| `GET /api/scans/{id}/objects?version=raw\|reviewed` | the export JSON (404 if absent). With no version: reviewed if it exists, else raw. |
| `POST /api/scans/{id}/send` | body `{"decisions": [{"object_id", "decision": "accept"\|"reject", "class_name"}]}` covering every raw object → `build_reviewed` → writes `results/scene_objects_reviewed.json` + `out/backend_outbox/<id>_reviewed.json`. Returns `{"saved": true, "path": ..., "outbox_path": ..., "num_objects": N}`. 409 if running; 422 for invalid decisions. |

Summaries also carry `scan_type` (`room` | `shelf`) and `has_model`. Jobs carry `kind`
(`pipeline` | `convert`). Shelf scans reject `POST /run` (422). A shelf scan's `scale` is
`{"status": "approximate", "method": "assumed_height", "assumed_height_m": 1.9, "factor": 1.0}` until
calibrated, then `{"status": "calibrated", "method": "measured", "factor": f, "measured": {...}}`.
Re-converting resets it (new geometry). Summaries carry `scale_status`.

`scan.json` status values: `uploaded`, `queued`, `running`, `done` (raw ready),
`reviewed`, `error`. `stale: true` when the orientation changed after a run.
Jobs survive server restarts as records: a job left running at restart becomes
`error: interrupted`.
