#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/autodl-tmp/qwen3.5-nano}"
REPO_DIR="${2:-$HOME/autodl-tmp/third_party/taming-transformers}"
CKPT_PATH="${3:-$HOME/autodl-tmp/models/vqgan_imagenet_f16_16384.ckpt}"
CONFIG_PATH="${4:-$HOME/autodl-tmp/models/vqgan_imagenet_f16_16384.yaml}"

CKPT_URL="https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1"
CONFIG_URL="https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1"

download_if_missing() {
  local url="$1"
  local target="$2"
  local min_size="$3"

  if [ -f "$target" ] && [ "$(stat -c%s "$target")" -ge "$min_size" ]; then
    return 0
  fi

  mkdir -p "$(dirname "$target")"
  rm -f "$target.tmp"
  if command -v wget >/dev/null 2>&1; then
    wget --tries=3 --timeout=30 -O "$target.tmp" "$url"
  else
    curl -L --retry 5 --retry-delay 5 --connect-timeout 15 -o "$target.tmp" "$url"
  fi
  mv "$target.tmp" "$target"
}

mkdir -p "$PROJECT_ROOT" "$(dirname "$REPO_DIR")" "$(dirname "$CKPT_PATH")"

download_if_missing "$CONFIG_URL" "$CONFIG_PATH" 128
download_if_missing "$CKPT_URL" "$CKPT_PATH" 1048576

echo "Taming repo argument ignored: using in-tree inference modules"
echo "VQ config: $CONFIG_PATH"
echo "VQ checkpoint: $CKPT_PATH"
