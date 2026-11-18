#!/usr/bin/env bash
# Foundation-model references (TorchXRayVision, RAD-DINO; fine-tuned and frozen) on the
# Shenzhen -> Montgomery and multi-site protocols, then every Shenzhen-trained model scored on
# MIMIC as a fourth external TB site. One GPU job at a time.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
R="$HOME/tbmxai_runs"
step() { local name="$1"; shift; echo "=== $(date -u +%H:%M) START $name"; if "$@"; then echo "=== $(date -u +%H:%M) OK $name"; else echo "=== $(date -u +%H:%M) FAILED $name"; fi; }
seeds() { local cfg="$1" out="$2" n="$3"; local a=(); for s in $(seq 1 "$n"); do a+=(--seed "$s"); done
          uv run tbmxai cv "$cfg" --folds 0 "${a[@]}" --output-dir "$out"; }
for fm in xrv raddino; do
  for v in "" "_frozen"; do
    step "nlm $fm$v" seeds "configs/transfer/nlm_$fm$v.yaml" "$R/transfer/nlm_$fm$v" 5
    step "multisite $fm$v" seeds "configs/transfer/multisite_${fm}${v}_from_shenzhen.yaml" "$R/transfer/multisite_${fm}${v}_from_shenzhen" 3
  done
done
step "mimic tb site" uv run python scripts/analysis/mimic_tb_site.py score "$R/analysis/mimic_tb_site" \
  "DenseNet-121, ImageNet (tuned)=$R/nlm_tb_tuned/cxr_densenet_seeds/seed*" \
  "ConvNeXt-T, ImageNet=$R/nlm_tb_tuned/cxr_convnext_seeds/seed*" \
  "DenseNet-121, MIMIC-pretrained=$R/transfer/nlm_densenet_mimicpt/seed*" \
  "DenseNet-121, MIMIC-pretrained, frozen=$R/transfer/nlm_densenet_mimicpt_frozen/seed*" \
  "ConvNeXt-T, MIMIC-pretrained=$R/transfer/nlm_convnext_mimicpt/seed*" \
  "ConvNeXt-T, MIMIC-pretrained, frozen=$R/transfer/nlm_convnext_mimicpt_frozen/seed*" \
  "TorchXRayVision DenseNet-121=$R/transfer/nlm_xrv/seed*" \
  "TorchXRayVision DenseNet-121, frozen=$R/transfer/nlm_xrv_frozen/seed*" \
  "RAD-DINO=$R/transfer/nlm_raddino/seed*" \
  "RAD-DINO, frozen=$R/transfer/nlm_raddino_frozen/seed*"
echo "=== $(date -u +%H:%M) foundation_models done"
