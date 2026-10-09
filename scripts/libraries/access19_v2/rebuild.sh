#!/usr/bin/env bash
# Rebuild the Access-19 v2 grasp library (geometry T17c, library_T17C) from scratch, validate it against online
# planning and execute the two full-shelf unload plans with exact contact checks.
#
#   scripts/libraries/access19_v2/rebuild.sh OUT_DIR
#
# Environment (all optional):
#   PYTHON       interpreter with tampanda + MuJoCo 3.4.0 (default: python)
#   WORKERS      build / validation processes (default: CPU count - 2; the shipped build used 60)
#   GEO          k=v geometry file (default: T17C from access19_v2/layout.py)
#   VALIDATE     0 skips validation and execution (default 1)
#   A19V2_PYLIB  pip --target directories holding frantik and vamp-planner 0.6.4 (see ../README.md)
# Outputs in OUT_DIR: library_T17C.json / .npz, validate_T17C.json, exec_T17C_024.json, exec_T17C_13.json.
# The builder lives in the package: tampanda/symbolic/domains/access19_v2/library.py (validate.py, execute.py).
set -euo pipefail
OUT="$(mkdir -p "${1:?usage: rebuild.sh OUT_DIR}" && cd "$1" && pwd)"
PY="${PYTHON:-python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" OMP_NUM_THREADS=1
M=tampanda.symbolic.domains.access19_v2
GEOARG=(); [ -n "${GEO:-}" ] && GEOARG=(--geo "$GEO")

"$PY" -m $M.library "$OUT/library_T17C" --workers "${WORKERS:-0}" ${GEOARG[@]+"${GEOARG[@]}"}
if [ "${VALIDATE:-1}" != 0 ]; then
    "$PY" -m $M.validate "$OUT/library_T17C" "$OUT/validate_T17C.json" --n 700 --workers "${WORKERS:-0}" ${GEOARG[@]+"${GEOARG[@]}"}
    "$PY" -m $M.execute "$OUT/library_T17C" unload:0,2,4 --report "$OUT/exec_T17C_024.json" ${GEOARG[@]+"${GEOARG[@]}"}
    "$PY" -m $M.execute "$OUT/library_T17C" unload:1,3 --report "$OUT/exec_T17C_13.json" ${GEOARG[@]+"${GEOARG[@]}"}
fi
