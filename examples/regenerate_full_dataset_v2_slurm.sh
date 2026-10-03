#!/bin/bash
#SBATCH --job-name=mlb-full-regen-v2
#SBATCH --output=/work/rleap1/daniel.swoboda/slurm/mlb_full_v2_%j.out
#SBATCH --error=/work/rleap1/daniel.swoboda/slurm/mlb_full_v2_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --partition=rleap_gpu_24gb

# v2 regen of multilevel_blocks — adds two mini-L4 templates at L3
# (mini_bridge_2pillars, mini_tower_on_bridge) to bridge the L3→L4
# horizon gap.  L3 quota bumped 120 → 160 across train_600 and
# train_l0_l4 (each of 4 L3 templates gets ~40 problems).
#
# Produces:
#   data/multilevel_blocks_v2/
#     train/                      640 problems
#     train_200/, train_400/      subsamples
#     val/L{0..5}/, test/L{0..5}/ 20 each (= 120 + 120)
#     train_l0_l4/                670 problems
#
# DESTRUCTIVE to data/multilevel_blocks_v2/ ONLY.  v1 dataset at
# data/multilevel_blocks/ is untouched.

PROJECT_ROOT="/work/rleap1/daniel.swoboda/projects/manipulation"
DATA_DIR="${PROJECT_ROOT}/data/multilevel_blocks_v2"
WORKERS=64

set -e
echo "========================================================================"
echo "SLURM Job: multilevel_blocks v2 regen (mini-L4 at L3)"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Started:       $(date)"
echo "Workers:       ${WORKERS}"
echo "Output dir:    ${DATA_DIR}  (rm -rf'd at start of step 1)"
echo "========================================================================"

PYTHON_BIN="/work/rleap1/daniel.swoboda/.conda/envs/rgnet_fresh/bin/python"
export PATH="$(dirname ${PYTHON_BIN}):${PATH}"
export MUJOCO_GL=egl

cd "${PROJECT_ROOT}"

echo
echo "--- Step 1: full curriculum (train_640 + val + test + subsamples) ---"
WORKERS=${WORKERS} bash examples/generate_mlb_curriculum_v2.sh "${DATA_DIR}"

echo
echo "--- Step 2: train_l0_l4_v2 (670-problem OOD-train split) ---"
WORKERS=${WORKERS} bash examples/generate_mlb_train_l0_l4_v2.sh "${DATA_DIR}"

EXIT_CODE=$?

echo "========================================================================"
echo "Finished:      $(date)"
echo "Exit code:     ${EXIT_CODE}"
echo "--- final layout ---"
for s in train train_200 train_400 train_l0_l4; do
    n=$(find "${DATA_DIR}/${s}" -name 'config_*.pddl' 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && echo "    ${s}: ${n} problems"
done
for L in 0 1 2 3 4 5; do
    for s in val test; do
        n=$(find "${DATA_DIR}/${s}/L${L}" -name 'config_*.pddl' 2>/dev/null | wc -l)
        [ "$n" -gt 0 ] && echo "    ${s}/L${L}: ${n} problems"
    done
done
echo "========================================================================"

exit ${EXIT_CODE}
