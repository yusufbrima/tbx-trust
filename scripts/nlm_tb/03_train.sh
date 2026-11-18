#!/usr/bin/env bash
# Train on Shenzhen, test externally on Montgomery: demographics-only, CXR-only and CXR+demographics,
# then compare the models statistically on the same Montgomery images.
# Every config loads the same files (labels, tabular, images), so all models see identical samples,
# and the external split is deterministic (test = all of Montgomery).
# Usage: bash scripts/nlm_tb/03_train.sh                  (all three)
#        bash scripts/nlm_tb/03_train.sh cxr_densenet     (one)
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TBMXAI_REPO"
models=("$@")
(( ${#models[@]} )) || models=(demographics_lr cxr_densenet multimodal)
runs=()
for m in "${models[@]}"; do
    uv run tbmxai run "configs/nlm_tb/$m.yaml"
    runs+=("$(ls -d "$TBMXAI_OUTPUT_ROOT/nlm_tb_$m"/*/ | sort | tail -1)")
done
if (( ${#runs[@]} > 1 )); then
    OUT="$TBMXAI_OUTPUT_ROOT/comparisons/nlm_tb-$(date +%Y%m%d-%H%M%S)"
    names=(); for m in "${models[@]}"; do names+=(--name "$m"); done
    uv run tbmxai compare "${runs[@]}" "${names[@]}" --out "$OUT"
    echo "Comparison → $OUT"
fi
