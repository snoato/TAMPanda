#!/usr/bin/env bash
# Rebuild the multi-level shelf v2 grasp library (geometry M3, library_M3) from scratch, then validate it against
# online planning.
#
#   scripts/libraries/shelf_access_v2/rebuild.sh OUT_DIR
#
# Environment (all optional):
#   PYTHON       interpreter with tampanda + MuJoCo 3.4.0 (default: python)
#   WORKERS      build processes (default 34)
#   CELLS        comma-separated cell ids: rebuild only these (a partial library, for spot checks)
#   GEO          geometry file (default: the shipped tampanda/symbolic/domains/shelf_access_v2/data/M3.kv)
#   VALIDATE     0 skips the validation step (default 1)
#   A19V2_PYLIB  pip --target directories holding frantik and vamp-planner 0.6.4 (see ../README.md)
# Outputs in OUT_DIR: library_M3.json / .npz, and libval.json (validation rows; summary printed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(mkdir -p "${1:?usage: rebuild.sh OUT_DIR}" && cd "$1" && pwd)"
PY="${PYTHON:-python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" OMP_NUM_THREADS=1 AUDIT_CWD="$OUT"
export PYTHONPATH="$HERE/../common:$HERE${PYTHONPATH:+:$PYTHONPATH}"
GEO="${GEO:-$("$PY" -c 'from tampanda.symbolic.domains.shelf_access_v2 import GEOMETRY; print(GEOMETRY)')}"

"$PY" "$HERE/../common/honest_library.py" "msv2:$GEO" library_M3 --workers "${WORKERS:-34}" \
    ${CELLS:+--cells "$CELLS"}
if [ "${VALIDATE:-1}" != 0 ] && [ -z "${CELLS:-}" ]; then
    "$PY" "$HERE/../common/validate_library.py" "$OUT/library_M3.json" 240 libval.json --workers 24 \
        --domain "msv2:$GEO"
fi
