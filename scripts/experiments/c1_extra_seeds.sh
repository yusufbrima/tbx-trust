#!/usr/bin/env bash
# Seeds 4-10 for the two C1 arms on Shenzhen -> TBX11K, with the
# exact resolved configs of seeds 1-3, so the training-variance analysis has 10 seed pairs. Checkpoints deleted after each arm.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
R="$HOME/tbmxai_runs"
step() { local name="$1"; shift; echo "=== $(date -u +%H:%M) START $name"; if "$@"; then echo "=== $(date -u +%H:%M) OK $name"; else echo "=== $(date -u +%H:%M) FAILED $name"; fi; }
for arm in multisite/cxr_convnext_from_shenzhen transfer/multisite_convnext_mimicpt_from_shenzhen; do
  cp "$R/$arm/seed1/config.yaml" "$R/$arm/config_seed1_resolved.yaml"
  step "seeds 4-10 $arm" uv run tbmxai cv "$R/$arm/config_seed1_resolved.yaml" --folds 0 \
    --seed 4 --seed 5 --seed 6 --seed 7 --seed 8 --seed 9 --seed 10 --output-dir "$R/$arm"
  rm -rf -- "$R/$arm"/seed*/checkpoints
done
echo "=== $(date -u +%H:%M) c1_extra_seeds done"
