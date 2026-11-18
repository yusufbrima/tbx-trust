#!/usr/bin/env bash
# Download the public Shenzhen (3.8 GB) and Montgomery (0.6 GB) TB chest X-ray sets from NLM Open-i.
# No login needed. Resumable: re-run if interrupted. Verifies each zip after download.
# Usage: bash scripts/nlm_tb/01_download.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$NLM_TB_RAW"
for url in \
    https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip \
    https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip; do
    file="$NLM_TB_RAW/$(basename "$url")"
    expected=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:" {n=$2} END {print n}')
    if [[ -f "$file" && -n "$expected" && "$(stat -c %s "$file")" == "$expected" ]]; then
        echo "✓ $(basename "$file") already complete"
    else
        wget --continue --no-verbose --show-progress -O "$file" "$url"
    fi
    if python3 -c "import sys, zipfile; sys.exit(zipfile.ZipFile(sys.argv[1]).testzip() is not None)" "$file" 2> /dev/null; then
        echo "✓ $(basename "$file") OK"
    else
        echo "✗ $(basename "$file") is corrupt: delete it and re-run" >&2
        exit 1
    fi
done
echo "Done. Next: bash scripts/nlm_tb/02_build.sh"
