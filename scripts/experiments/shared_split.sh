#!/usr/bin/env bash
# Shared-split variant: six Shenzhen-trained arms, 5 seeds each, on ONE shared
# Shenzhen train / validation / internal-test split (split_seed: 0). Montgomery is the external test
# during training; the saved models are then scored on TBX11K (scripts/analysis/score_site.py).
# Checkpoints are deleted after each arm (model.pkl keeps the final weights) to save disk.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
R="$HOME/tbmxai_runs"; T="$R/transfer/internal_shared"
step() { local name="$1"; shift; echo "=== $(date -u +%H:%M) START $name"; if "$@"; then echo "=== $(date -u +%H:%M) OK $name"; else echo "=== $(date -u +%H:%M) FAILED $name"; fi; }
for arm in convnext_imagenet convnext_mimic densenet_imagenet xrv raddino_frozen raddino_retuned; do
  step "train $arm" uv run tbmxai cv "configs/transfer/internal_shared/$arm.yaml" --folds 0 \
    --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --output-dir "$T/$arm"
  rm -rf -- "$T/$arm"/seed*/checkpoints
  df -h "$HOME" | tail -1
done
step "score tbx11k" uv run python scripts/analysis/score_site.py "$HOME/data/tbmxai/tbx11k" tb \
  "$R/analysis/internal_shared_tbx11k" "$T"/*/seed*
echo "=== $(date -u +%H:%M) shared_split done"
