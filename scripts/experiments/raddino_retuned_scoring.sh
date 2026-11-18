#!/usr/bin/env bash
# After raddino_learning_rate.sh:
# 1. the retuned RAD-DINO on the MIMIC TB site and Naija-CXR (exploratory sites);
# 2. Grad-CAM lung focus on Montgomery for the foundation-model arms (see also
#    saliency_foundation_models.sh).
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
R="$HOME/tbmxai_runs"
step() { local name="$1"; shift; echo "=== $(date -u +%H:%M) START $name"; if "$@"; then echo "=== $(date -u +%H:%M) OK $name"; else echo "=== $(date -u +%H:%M) FAILED $name"; fi; }
best=$(uv run python -c "import json;print(json.load(open('$R/transfer/raddino_lr/selection.json'))['selected'].removeprefix('nlm_'))")
step "mimic tb site retuned raddino" uv run python scripts/analysis/mimic_tb_site.py score "$R/analysis/mimic_tb_site_raddino_lr" \
  "RAD-DINO, retuned=$R/transfer/raddino_lr/nlm_$best/seed*"
step "naija retuned raddino" uv run python scripts/analysis/naija_site.py score "$R/analysis/naija_site_raddino_lr" \
  "RAD-DINO, retuned=$R/transfer/raddino_lr/nlm_$best/seed*"
step "saliency foundation models" uv run python scripts/analysis/saliency_lungs.py "$R/analysis/saliency_lungs_fm" \
  "Shenzhen→Montgomery: TorchXRayVision=$R/transfer/nlm_xrv/seed*" \
  "Shenzhen→Montgomery: RAD-DINO, frozen=$R/transfer/nlm_raddino_frozen/seed*" \
  "Shenzhen→Montgomery: RAD-DINO, retuned=$R/transfer/raddino_lr/nlm_$best/seed*"
echo "=== $(date -u +%H:%M) raddino_retuned_scoring done"
