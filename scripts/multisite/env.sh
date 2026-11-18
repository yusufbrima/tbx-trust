# Settings for the multi-site TB workflow (NLM Shenzhen + Montgomery, TBX11K). Sourced by every script here.
source "$(dirname "${BASH_SOURCE[0]}")/../nlm_tb/env.sh"
export TBX11K_RAW="${TBX11K_RAW:-$HOME/data/tbx11k}"
export TBMXAI_TBX11K="${TBMXAI_TBX11K:-$HOME/data/tbmxai/tbx11k}"
export TBMXAI_MULTISITE="${TBMXAI_MULTISITE:-$HOME/data/tbmxai/tb_multisite}"
