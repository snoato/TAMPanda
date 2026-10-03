#!/bin/bash
# v2 driver — same as generate_mlb_curriculum.sh but with the L3 quota
# bumped from 120 → 160 to accommodate the two new mini-L4 templates
# (mini_bridge_2pillars, mini_tower_on_bridge) added at L3.  Total per
# train_600 spec: 60 + 120 + 120 + 160 + 90 + 90 = 640 problems.
#
# Usage: ./examples/generate_mlb_curriculum_v2.sh [data_dir]
# Defaults to data/multilevel_blocks_v2/.

set -e

DATA_DIR="${1:-data/multilevel_blocks_v2}"
WORKERS="${WORKERS:-4}"

rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

gen_one() {
    local spec="$1"
    local split="$2"
    local level="$3"
    local offset="$4"
    echo
    echo "=== $spec  split=$split  level=L$level  offset=$offset ==="
    python -u -m tampanda.symbolic.domains.multilevel_blocks.generate_data \
        --output-dir "$DATA_DIR" \
        --curriculum-spec "$spec" \
        --curriculum-split "$split" \
        --level "$level" \
        --config-offset "$offset" \
        --num-workers "$WORKERS"
}

# train_600 (flat): 60 + 120 + 120 + 160 + 90 + 90 = 640
gen_one train_600 train 0 0
gen_one train_600 train 1 60
gen_one train_600 train 2 180
gen_one train_600 train 3 300
gen_one train_600 train 4 460
gen_one train_600 train 5 550

# val_per_level + test_per_level: per-level subdirs; offset is irrelevant.
for level in 0 1 2 3 4 5; do
    gen_one val_per_level val "$level" 0
done
for level in 0 1 2 3 4 5; do
    gen_one test_per_level test "$level" 0
done

echo
echo "=== train_200 subsample ==="
python -u examples/sample_dataset_subset.py \
    --source "$DATA_DIR/train" \
    --dest "$DATA_DIR/train_200" \
    --size 200 --seed 42

echo
echo "=== train_400 subsample ==="
python -u examples/sample_dataset_subset.py \
    --source "$DATA_DIR/train" \
    --dest "$DATA_DIR/train_400" \
    --size 400 --seed 42

echo
echo "=== DONE ==="
echo "Output: $DATA_DIR"
