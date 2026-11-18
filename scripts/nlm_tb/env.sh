# Settings for the Shenzhen + Montgomery (NLM) TB workflow. Every script here sources this file.

# Where the NLM zip files are downloaded and extracted (outside the repo).
export NLM_TB_RAW="${NLM_TB_RAW:-$HOME/data/nlm_tb}"

# Where the tbmxai standard dataset is written.
export TBMXAI_NLM_TB="${TBMXAI_NLM_TB:-$HOME/data/tbmxai/nlm_tb}"

# Where experiment runs and the resized-image cache go.
export TBMXAI_OUTPUT_ROOT="${TBMXAI_OUTPUT_ROOT:-$HOME/tbmxai_runs}"
export TBMXAI_IMAGE_CACHE="${TBMXAI_IMAGE_CACHE:-$HOME/data/tbmxai/cxr_cache}"

export TBMXAI_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
