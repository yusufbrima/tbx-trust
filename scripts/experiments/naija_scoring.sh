#!/usr/bin/env bash
# Score every Shenzhen-trained NLM-protocol arm on Naija-CXR (exploratory African site).
set -uo pipefail
R="$HOME/tbmxai_runs"
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
echo "=== $(date -u +%H:%M) START naija score"
if uv run python scripts/analysis/naija_site.py score "$R/analysis/naija_site" \
  "DenseNet-121, ImageNet=$R/nlm_tb_tuned/cxr_densenet_seeds/seed*" \
  "DenseNet-121, MIMIC=$R/transfer/nlm_densenet_mimicpt/seed*" \
  "DenseNet-121, MIMIC, frozen=$R/transfer/nlm_densenet_mimicpt_frozen/seed*" \
  "ConvNeXt-T, ImageNet=$R/nlm_tb_tuned/cxr_convnext_seeds/seed*" \
  "ConvNeXt-T, MIMIC=$R/transfer/nlm_convnext_mimicpt/seed*" \
  "ConvNeXt-T, MIMIC, frozen=$R/transfer/nlm_convnext_mimicpt_frozen/seed*" \
  "TorchXRayVision=$R/transfer/nlm_xrv/seed*" \
  "TorchXRayVision, frozen=$R/transfer/nlm_xrv_frozen/seed*" \
  "RAD-DINO=$R/transfer/nlm_raddino/seed*" \
  "RAD-DINO, frozen=$R/transfer/nlm_raddino_frozen/seed*"; then
  echo "=== $(date -u +%H:%M) OK naija score"
else
  echo "=== $(date -u +%H:%M) FAILED naija score"
fi
