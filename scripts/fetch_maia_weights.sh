#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${CHESS_ML_MAIA_WEIGHTS_DIR:-"$ROOT_DIR/checkpoints/maia"}"
BASE_URL="https://github.com/CSSLab/maia-chess/releases/download/v1.0"
RATINGS=(1100 1500 1900)

mkdir -p "$WEIGHTS_DIR"

for rating in "${RATINGS[@]}"; do
  target="$WEIGHTS_DIR/maia-${rating}.pb.gz"
  if [[ -f "$target" ]]; then
    echo "Maia ${rating} already present: $target"
    continue
  fi

  url="$BASE_URL/maia-${rating}.pb.gz"
  echo "Downloading Maia ${rating} to $target"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --output "$target" "$url"
  else
    python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$url" "$target"
  fi
done

echo "Maia weights ready in $WEIGHTS_DIR"
