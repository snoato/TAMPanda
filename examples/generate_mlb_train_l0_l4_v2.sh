#!/bin/bash
# v2 driver — same as generate_mlb_train_l0_l4.sh but with the L3 quota
# bumped from 120 → 160 to accommodate the two new mini-L4 templates
# at L3.  Total: 140 + 160 + 120 + 160 + 90 = 670 problems.
#
# NON-DESTRUCTIVE relative to other splits (only writes train_l0_l4/).
#
# Usage:
#   ./examples/generate_mlb_train_l0_l4_v2.sh [data_dir]
#
# Defaults: data_dir = data/multilevel_blocks_v2/, WORKERS=32 (env override).

set -e

DATA_DIR="${1:-data/multilevel_blocks_v2}"
WORKERS="${WORKERS:-32}"

mkdir -p "$DATA_DIR"

gen_one() {
    local level="$1"
    local offset="$2"
    echo
    echo "=== train_l0_l4  split=train_l0_l4  level=L$level  offset=$offset ==="
    python -u -m tampanda.symbolic.domains.multilevel_blocks.generate_data \
        --output-dir "$DATA_DIR" \
        --curriculum-spec train_l0_l4 \
        --curriculum-split train_l0_l4 \
        --level "$level" \
        --config-offset "$offset" \
        --num-workers "$WORKERS"
}

# train_l0_l4 (flat): 140 + 160 + 120 + 160 + 90 = 670
rm -rf "$DATA_DIR/train_l0_l4"
gen_one 0 0
gen_one 1 140
gen_one 2 300
gen_one 3 420
gen_one 4 580

echo
echo "=== DONE ==="
echo "Output: $DATA_DIR/train_l0_l4"
n=$(find "$DATA_DIR/train_l0_l4" -name 'config_*.pddl' 2>/dev/null | wc -l)
echo "Problems: $n"
