#!/usr/bin/env bash
# Runs the full pipeline headlessly:
#   1) render views inside Isaac Sim's bundled python
#   2) open-vocabulary 2D detection (+SAM) in a regular torch env
#   3) 3D lifting + DBSCAN clustering -> scene_objects.json
#
# Env overrides:
#   ISAAC_SIM_DIR  Isaac Sim install (default ~/isaacsim)
#   PY             python for stages 2-3 (default: python3 on PATH, e.g. an activated venv)
#   PROMPTS        comma-separated class prompts
#   MODE           dome | interior (camera layout; interior suits room scans)
#   NUM_VIEWS      number of rendered viewpoints
#   RENDERER       isaac | lite (lite = CPU ray casting, no Isaac Sim; uses $PY)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USD="${1:-$HERE/scenes/room_scan_upright.usda}"
OUT="${2:-$HERE/out}"

ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-$HOME/isaacsim}"
PY="${PY:-python3}"
PROMPTS="${PROMPTS:-chair,table,monitor,sofa,bed,coffee cup,lamp,door,window}"
MODE="${MODE:-interior}"
NUM_VIEWS="${NUM_VIEWS:-48}"
RENDERER="${RENDERER:-isaac}"

USD="$(realpath "$USD")"
mkdir -p "$OUT"

echo "==> [1/3] Rendering $USD ($RENDERER, $MODE, $NUM_VIEWS views)"
if [[ "$RENDERER" == "lite" ]]; then
    RENDER_CMD=("$PY" "$HERE/render_views_lite.py")
else
    RENDER_CMD=("$ISAAC_SIM_DIR/python.sh" "$HERE/render_views.py")
fi
"${RENDER_CMD[@]}" --usd "$USD" --out "$OUT/renders" \
    --mode "$MODE" --num-views "$NUM_VIEWS" --verify

echo "==> [2/3] Open-vocabulary detection: $PROMPTS"
"$PY" "$HERE/detect_2d.py" \
    --renders "$OUT/renders" --out "$OUT/detections" \
    --prompts "$PROMPTS" --sam --vis

echo "==> [3/3] 3D lifting + clustering"
"$PY" "$HERE/lift_cluster.py" \
    --renders "$OUT/renders" --detections "$OUT/detections" \
    --out "$OUT/scene_objects.json" --save-ply "$OUT/clusters.ply"

echo "==> Done: $OUT/scene_objects.json"
