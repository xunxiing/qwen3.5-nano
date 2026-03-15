#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/autodl-tmp/qwen3.5-nano}"
REPO_DIR="${2:-$HOME/autodl-tmp/third_party/LlamaGen}"
CKPT_PATH="${3:-$HOME/autodl-tmp/models/vq_ds16_t2i.pt}"

mkdir -p "$(dirname "$REPO_DIR")" "$(dirname "$CKPT_PATH")"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/FoundationVision/LlamaGen.git "$REPO_DIR"
fi

if [ ! -f "$CKPT_PATH" ]; then
  rm -f "$CKPT_PATH.tmp"
  if command -v wget >/dev/null 2>&1; then
    wget --tries=2 --timeout=30 -O "$CKPT_PATH.tmp" "https://hf-mirror.com/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt"
  else
    curl -L --retry 2 --retry-delay 5 --connect-timeout 15 --max-time 180 -o "$CKPT_PATH.tmp" "https://hf-mirror.com/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt"
  fi
  if [ ! -f "$CKPT_PATH.tmp" ] || [ "$(stat -c%s "$CKPT_PATH.tmp")" -lt 1048576 ]; then
    rm -f "$CKPT_PATH.tmp"
    echo "LlamaGen checkpoint download failed or returned an incomplete file." >&2
    echo "This environment cannot currently resolve the mirror's xet storage host." >&2
    exit 1
  fi
  mv "$CKPT_PATH.tmp" "$CKPT_PATH"
fi

echo "LlamaGen repo: $REPO_DIR"
echo "VQ checkpoint: $CKPT_PATH"
