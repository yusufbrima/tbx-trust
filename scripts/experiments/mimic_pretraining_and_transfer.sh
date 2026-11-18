#!/usr/bin/env bash
# MIMIC CheXpert pretraining (DenseNet-121, ConvNeXt-T), then the MIMIC-initialised TB arms and
# their saliency. Each step logs OK/FAILED; a failed step does not stop independent later steps.
# Resumable: pretraining resumes from last.pt.
# Usage: bash scripts/tmux_job.sh mimic_pretraining bash scripts/experiments/mimic_pretraining_and_transfer.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
R="$HOME/tbmxai_runs"
step() { local name="$1"; shift; echo "=== $(date -u +%H:%M) START $name"; if "$@"; then echo "=== $(date -u +%H:%M) OK $name"; else echo "=== $(date -u +%H:%M) FAILED $name"; return 1; fi; }
seeds() { local cfg="$1" out="$2" n="$3"; local a=(); for s in $(seq 1 "$n"); do a+=(--seed "$s"); done
          uv run tbmxai cv "$cfg" --folds 0 "${a[@]}" --output-dir "$out"; }
sal() { local out="$1"; shift; uv run python scripts/analysis/saliency_lungs.py "$out" "$@"; }

step "pretrain densenet121" uv run tbmxai pretrain configs/pretrain/mimic_chexpert_densenet121.yaml --resume; DN=$?
rm -rf "$R/multisite/cxr_convnext_from_tbx11k"
step "multisite convnext from tbx11k" bash scripts/multisite/02_train.sh cxr_convnext_from_tbx11k
step "saliency multisite convnext" sal "$R/analysis/saliency_lungs_convnext_multisite" \
  "Multi-site: ConvNeXt-T trained on Shenzhen=$R/multisite/cxr_convnext_from_shenzhen/seed*" \
  "Multi-site: ConvNeXt-T trained on TBX11K=$R/multisite/cxr_convnext_from_tbx11k/seed*"
if [ "$DN" = 0 ]; then
  step "nlm densenet mimicpt" seeds configs/transfer/nlm_densenet_mimicpt.yaml "$R/transfer/nlm_densenet_mimicpt" 5
  step "nlm densenet mimicpt frozen" seeds configs/transfer/nlm_densenet_mimicpt_frozen.yaml "$R/transfer/nlm_densenet_mimicpt_frozen" 5
fi
step "pretrain convnext_tiny" uv run tbmxai pretrain configs/pretrain/mimic_chexpert_convnext_tiny.yaml --resume; CX=$?
if [ "$CX" = 0 ]; then
  step "nlm convnext mimicpt" seeds configs/transfer/nlm_convnext_mimicpt.yaml "$R/transfer/nlm_convnext_mimicpt" 5
  step "nlm convnext mimicpt frozen" seeds configs/transfer/nlm_convnext_mimicpt_frozen.yaml "$R/transfer/nlm_convnext_mimicpt_frozen" 5
  step "multisite convnext mimicpt from shenzhen" seeds configs/transfer/multisite_convnext_mimicpt_from_shenzhen.yaml "$R/transfer/multisite_convnext_mimicpt_from_shenzhen" 3
  step "multisite convnext mimicpt from tbx11k" seeds configs/transfer/multisite_convnext_mimicpt_from_tbx11k.yaml "$R/transfer/multisite_convnext_mimicpt_from_tbx11k" 3
  step "saliency transfer" sal "$R/analysis/saliency_lungs_transfer" \
    "Shenzhen→Montgomery: ConvNeXt-T MIMIC-pretrained=$R/transfer/nlm_convnext_mimicpt/seed*" \
    "Shenzhen→Montgomery: ConvNeXt-T MIMIC-pretrained, frozen=$R/transfer/nlm_convnext_mimicpt_frozen/seed*" \
    "Multi-site: ConvNeXt-T MIMIC-pretrained, trained on Shenzhen=$R/transfer/multisite_convnext_mimicpt_from_shenzhen/seed*" \
    "Multi-site: ConvNeXt-T MIMIC-pretrained, trained on TBX11K=$R/transfer/multisite_convnext_mimicpt_from_tbx11k/seed*"
fi
echo "=== $(date -u +%H:%M) mimic_pretraining_and_transfer done"
