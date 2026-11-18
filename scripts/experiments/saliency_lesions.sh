#!/usr/bin/env bash
# Lesion-level Grad-CAM against TBX11K TB boxes for the Shenzhen-trained multi-site arms.
# Needs the TBX11K re-tune (tbx11k_retune.sh).
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
R="$HOME/tbmxai_runs"
echo "=== $(date -u +%H:%M) START saliency boxes"
if uv run python scripts/analysis/saliency_boxes.py "$R/analysis/saliency_boxes" \
  "DenseNet-121, ImageNet=$R/multisite/cxr_from_shenzhen/seed*" \
  "ConvNeXt-T, ImageNet=$R/multisite/cxr_convnext_from_shenzhen/seed*" \
  "ConvNeXt-T, MIMIC=$R/transfer/multisite_convnext_mimicpt_from_shenzhen/seed*" \
  "TorchXRayVision=$R/transfer/multisite_xrv_from_shenzhen/seed*"; then echo "=== $(date -u +%H:%M) OK saliency boxes"
else echo "=== $(date -u +%H:%M) FAILED saliency boxes"; fi
