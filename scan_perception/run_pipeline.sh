#!/usr/bin/env bash
# Usage: run_pipeline.sh <scan.usd> <out_dir> <prompt> [<prompt> ...]
# Env: ISAAC_PY (default ~/isaacsim/python.sh), CV_PY (default python3), RENDER_ARGS, DETECT_ARGS, LIFT_ARGS
set -euo pipefail
[ $# -ge 3 ] || { sed -n 2,3p "$0"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
USD="$1"; OUT="$2"; shift 2
ISAAC_PY="${ISAAC_PY:-$HOME/isaacsim/python.sh}"
CV_PY="${CV_PY:-python3}"

"$ISAAC_PY" "$HERE/render_views.py" --usd "$USD" --out "$OUT/views" ${RENDER_ARGS:-}
"$CV_PY" "$HERE/detect_2d.py" --views "$OUT/views" --prompts "$@" --sam --overlays ${DETECT_ARGS:-}
"$CV_PY" "$HERE/lift_and_cluster.py" --views "$OUT/views" --out "$OUT/scene_objects.json" \
    --debug-ply "$OUT/objects.ply" ${LIFT_ARGS:-}
