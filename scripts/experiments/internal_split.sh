#!/usr/bin/env bash
# Internal Shenzhen test split for six arms, then the MIMIC TB site re-scored with per-image
# predictions saved.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
R="$HOME/tbmxai_runs"
step() { local name="$1"; shift; echo "=== $(date -u +%H:%M) START $name"; if "$@"; then echo "=== $(date -u +%H:%M) OK $name"; else echo "=== $(date -u +%H:%M) FAILED $name"; fi; }
for arm in convnext_imagenet convnext_mimic densenet_imagenet xrv raddino_frozen raddino_retuned; do
  step "internal $arm" uv run tbmxai cv "configs/transfer/internal/$arm.yaml" --folds 0 --seed 1 --seed 2 --seed 3 \
    --output-dir "$R/transfer/internal/$arm"
done
best=$(uv run python -c "import json;print(json.load(open('$R/transfer/raddino_lr/selection.json'))['selected'].removeprefix('nlm_'))")
step "mimic tb site rescore" uv run python scripts/analysis/mimic_tb_site.py score "$R/analysis/mimic_tb_site_v2" \
  "DenseNet-121, ImageNet (tuned)=$R/nlm_tb_tuned/cxr_densenet_seeds/seed*" \
  "ConvNeXt-T, ImageNet=$R/nlm_tb_tuned/cxr_convnext_seeds/seed*" \
  "DenseNet-121, MIMIC-pretrained=$R/transfer/nlm_densenet_mimicpt/seed*" \
  "DenseNet-121, MIMIC-pretrained, frozen=$R/transfer/nlm_densenet_mimicpt_frozen/seed*" \
  "ConvNeXt-T, MIMIC-pretrained=$R/transfer/nlm_convnext_mimicpt/seed*" \
  "ConvNeXt-T, MIMIC-pretrained, frozen=$R/transfer/nlm_convnext_mimicpt_frozen/seed*" \
  "TorchXRayVision DenseNet-121=$R/transfer/nlm_xrv/seed*" \
  "TorchXRayVision DenseNet-121, frozen=$R/transfer/nlm_xrv_frozen/seed*" \
  "RAD-DINO=$R/transfer/nlm_raddino/seed*" \
  "RAD-DINO, frozen=$R/transfer/nlm_raddino_frozen/seed*" \
  "RAD-DINO, retuned=$R/transfer/raddino_lr/nlm_$best/seed*"
echo "=== $(date -u +%H:%M) internal_split done"
