#!/usr/bin/env bash
# Hyperparameter search (Optuna) for the CXR and CXR+demographics models, then repeated seeds of
# the best settings, then a statistical comparison with the demographics baseline.
#   - Trials train on Shenzhen-train and are scored on Shenzhen-validation (AUROC); Montgomery is
#     used once per model, after refitting the best parameters.
#   - Decision thresholds are chosen on validation (Youden) and applied to Montgomery.
#   - Seeds re-draw the Shenzhen train/val split and the initialisation: they show how much of a
#     difference between models is run-to-run noise.
# Run in tmux:  bash scripts/tmux_job.sh tune bash scripts/nlm_tb/04_tune.sh
# Resumable: re-running continues each Optuna study (sqlite in the output folder).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TBMXAI_REPO"
OUT="${NLM_TB_TUNED:-$TBMXAI_OUTPUT_ROOT/nlm_tb_tuned}"
SEEDS=(${NLM_TB_SEEDS:-1 2 3 4 5})
mkdir -p "$OUT"

for m in cxr_densenet multimodal; do
    echo "== tuning $m"
    uv run tbmxai tune "configs/nlm_tb/$m.yaml" --output-dir "$OUT/$m"
done

echo "== demographics baseline (validation-chosen threshold)"
uv run tbmxai run configs/nlm_tb/demographics_lr.yaml --output-dir "$OUT/demographics_lr"

seed_args=(); for s in "${SEEDS[@]}"; do seed_args+=(--seed "$s"); done
for m in cxr_densenet multimodal; do
    echo "== $m: best settings × seeds ${SEEDS[*]}"
    uv run tbmxai cv "$OUT/$m/config_best.yaml" --folds 0 "${seed_args[@]}" --output-dir "$OUT/${m}_seeds"
done

echo "== comparison on Montgomery"
uv run tbmxai compare "$OUT/demographics_lr" "$OUT/cxr_densenet" "$OUT/multimodal" \
    --name demographics_lr --name cxr_tuned --name multimodal_tuned --out "$OUT/comparison"
echo "Done → $OUT (comparison/, *_seeds/summary, each run's report.html)"
