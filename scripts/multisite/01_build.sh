#!/usr/bin/env bash
# Build NLM and TBX11K with their recipes, pool them (NLM "normal" → non_tb) with a full
# cross-dataset duplicate-image scan, and cache resized images.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TBMXAI_REPO"
uv run tbmxai dataset build configs/datasets/nlm_tb.yaml "$NLM_TB_RAW" "$TBMXAI_NLM_TB"
uv run tbmxai dataset build configs/datasets/tbx11k.yaml "$TBX11K_RAW" "$TBMXAI_TBX11K" --check-images 0
uv run tbmxai dataset merge "$TBMXAI_MULTISITE" "$TBMXAI_NLM_TB" "$TBMXAI_TBX11K" \
    --label tb=non_tb,tb --rename tb:normal=non_tb --name tb_multisite
uv run tbmxai cache-images configs/multisite/cxr_from_shenzhen.yaml
echo "Done. Next: bash scripts/multisite/02_train.sh"
