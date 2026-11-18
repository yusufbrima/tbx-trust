#!/usr/bin/env bash
# Grad-CAM lung focus for the foundation-model arms (TorchXRayVision, RAD-DINO), whose target layer
# comes from the encoder's saliency_layer().
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
R="$HOME/tbmxai_runs"
best=$(uv run python -c "import json;print(json.load(open('$R/transfer/raddino_lr/selection.json'))['selected'].removeprefix('nlm_'))")
echo "=== $(date -u +%H:%M) START saliency foundation models"
if uv run python scripts/analysis/saliency_lungs.py "$R/analysis/saliency_lungs_fm" \
  "Shenzhen→Montgomery: TorchXRayVision=$R/transfer/nlm_xrv/seed*" \
  "Shenzhen→Montgomery: RAD-DINO, frozen=$R/transfer/nlm_raddino_frozen/seed*" \
  "Shenzhen→Montgomery: RAD-DINO, retuned=$R/transfer/raddino_lr/nlm_$best/seed*"; then
  echo "=== $(date -u +%H:%M) OK saliency foundation models"
else echo "=== $(date -u +%H:%M) FAILED saliency foundation models"; fi
echo "=== $(date -u +%H:%M) saliency_foundation_models done"
