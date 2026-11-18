#!/usr/bin/env bash
# Download TBX11K (Liu et al., CVPR 2020; CC BY 4.0; 3.3 GB) from the authors' Google Drive
# (https://mmcheng.net/tb/). Resumable. NLM sets: bash scripts/nlm_tb/01_download.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$TBX11K_RAW"
cd "$TBX11K_RAW"
uvx --from gdown==5.2.0 gdown --continue -O TBX11K.zip "https://drive.google.com/uc?id=1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR"
python3 -c "import sys, zipfile; sys.exit(zipfile.ZipFile(sys.argv[1]).testzip() is not None)" TBX11K.zip
echo "✓ TBX11K.zip OK. Next: bash scripts/multisite/01_build.sh"
