#!/usr/bin/env bash
# Phone video -> textured .usdz (+ H.264 .mp4).
# usage: ./run_all.sh "<video>" <STEM> [height_m]     (HF_HUB_OFFLINE=0 for the first model download)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VIDEO="$1"; STEM="$2"; HEIGHT_M="${3:-1.9}"
export HF_HOME="$HERE/models/hf" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
PY="env -u PYTHONPATH $HERE/.venv/bin/python"
W="$HERE/work/$STEM"; O="$HERE/out/$STEM"
mkdir -p "$W" "$O"
step() { local name=$1; shift; local s=$SECONDS; "$@"; echo "[run_all] $name: $((SECONDS - s)) s"; }
step extract  $PY "$HERE/scripts/01_extract.py" "$VIDEO" "$W" --mp4 "$O/${STEM}_h264.mp4"
step sfm      $PY "$HERE/scripts/02_sfm.py" "$W" --name sfm_sr
step depth    $PY "$HERE/scripts/03_depth.py" "$W" --sfm sfm_sr
step fuse     $PY "$HERE/scripts/04_fuse.py" "$W"
step gravity  $PY "$HERE/scripts/04b_gravity.py" "$W"
step texture  $PY "$HERE/scripts/05_texture.py" "$W" --sfm sfm_sr
step export   $PY "$HERE/scripts/06_export_usdz.py" "$W" "$O/$STEM.usdz" --height-m "$HEIGHT_M"
step cameras  $PY "$HERE/scripts/07_cameras.py" "$W"
