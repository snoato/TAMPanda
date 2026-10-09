#!/usr/bin/env bash
# Rebuild the confined-shelf grasp library (library_v1) from scratch, then validate it against online planning.
#
#   scripts/libraries/confined_shelf/rebuild.sh OUT_DIR
#
# Environment (all optional except A19V2_PYLIB):
#   PYTHON       interpreter with tampanda + MuJoCo 3.4.0 (default: python)
#   WORKERS      build processes (default 34 = one per cell, as the shipped build)
#   CELLS        comma-separated cell ids: rebuild only these (a partial library, for spot checks)
#   VALIDATE     0 skips the validation step (default 1)
#   A19V2_PYLIB  REQUIRED: pip --target directories holding frantik and vamp-planner 0.6.4 (see ../README.md)
# Outputs in OUT_DIR: library_v1.json / .npz, and libval.json (validation rows; summary printed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(mkdir -p "${1:?usage: rebuild.sh OUT_DIR}" && cd "$1" && pwd)"
PY="${PYTHON:-python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" OMP_NUM_THREADS=1 AUDIT_CWD="$OUT"
export PYTHONPATH="$HERE/../common:$HERE${PYTHONPATH:+:$PYTHONPATH}"

"$PY" "$HERE/../common/honest_library.py" confined-shelf library_v1 --workers "${WORKERS:-34}" \
    ${CELLS:+--cells "$CELLS"}
if [ "${VALIDATE:-1}" != 0 ] && [ -z "${CELLS:-}" ]; then
    "$PY" "$HERE/../common/validate_library.py" "$OUT/library_v1.json" 240 libval.json --workers 24 \
        --domain confined-shelf
fi
