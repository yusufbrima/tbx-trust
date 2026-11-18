#!/usr/bin/env bash
# Train each direction with several seeds (each seed re-draws the source train/val split and
# the initialisation). Per-site test metrics are in each run's tables/subgroups.csv.
# Usage: bash scripts/multisite/02_train.sh [config ...]   (default: both directions)
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TBMXAI_REPO"
SEEDS=(${MULTISITE_SEEDS:-1 2 3})
configs=("$@")
(( ${#configs[@]} )) || configs=(cxr_from_shenzhen cxr_from_tbx11k)
seed_args=(); for s in "${SEEDS[@]}"; do seed_args+=(--seed "$s"); done
for c in "${configs[@]}"; do
    echo "== $c × seeds ${SEEDS[*]}"
    uv run tbmxai cv "configs/multisite/$c.yaml" --folds 0 "${seed_args[@]}" \
        --output-dir "$TBMXAI_OUTPUT_ROOT/multisite/$c"
done
echo "Done → $TBMXAI_OUTPUT_ROOT/multisite"
