#!/usr/bin/env bash
# Extract the NLM zips, convert them to a tbmxai standard dataset, validate it and cache resized images.
# Usage: bash scripts/nlm_tb/02_build.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TBMXAI_REPO"
uv run tbmxai dataset build configs/datasets/nlm_tb.yaml "$NLM_TB_RAW" "$TBMXAI_NLM_TB" --check-images 0
uv run tbmxai cache-images configs/nlm_tb/cxr_densenet.yaml
echo "Done. Next: bash scripts/nlm_tb/03_train.sh"
