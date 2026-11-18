#!/usr/bin/env bash
# Re-tune the TBX11K-trained ConvNeXt-T on TBX11K validation only (fixes target-site hyperparameter
# leakage for TBX11K -> Shenzhen), then refit ImageNet and MIMIC initialisations with those settings,
# 3 seeds each, at the pre-specified epoch budget.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
R="$HOME/tbmxai_runs"; T="$R/multisite_retuned"
step() { local name="$1"; shift; echo "=== $(date -u +%H:%M) START $name"; if "$@"; then echo "=== $(date -u +%H:%M) OK $name"; else echo "=== $(date -u +%H:%M) FAILED $name"; fi; }
step "tune convnext on tbx11k val" uv run tbmxai tune configs/multisite/tune_convnext_tbx11k.yaml --output-dir "$T/tune"
uv run python - "$T" <<'PY'
import sys, yaml
from pathlib import Path
T = Path(sys.argv[1])
best = yaml.safe_load(open(T / "tune" / "config_best.yaml"))
keys = [("model", "params", "image", "size"), ("training", "optimizer", "lr"), ("training", "optimizer", "weight_decay"),
        ("training", "optimizer", "encoder_lr_mult"), ("model", "params", "head", "dropout")]
def get(d, k):
    for x in k: d = d[x]
    return d
def put(d, k, v):
    for x in k[:-1]: d = d[x]
    d[k[-1]] = v
for tag, base in (("imagenet", "configs/multisite/cxr_convnext_from_tbx11k.yaml"),
                  ("mimic", "configs/transfer/multisite_convnext_mimicpt_from_tbx11k.yaml")):
    c = yaml.safe_load(open(base))
    for k in keys:
        put(c, k, get(best, k))
    c["name"] = f"multisite_convnext_{tag}_from_tbx11k_retuned"
    (T / f"config_{tag}.yaml").write_text(f"# {base} with settings tuned on TBX11K validation (tune/config_best.yaml)\n"
                                          + yaml.safe_dump(c, sort_keys=False))
    print(tag, {".".join(k): get(c, k) for k in keys})
PY
for tag in imagenet mimic; do
  step "refit $tag" uv run tbmxai cv "$T/config_$tag.yaml" --folds 0 --seed 1 --seed 2 --seed 3 --output-dir "$T/$tag"
done
echo "=== $(date -u +%H:%M) tbx11k_retune done"
