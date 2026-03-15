#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/autodl-tmp/qwen3.5-nano}"
TARGET_DIR="${2:-$HOME/autodl-tmp/models/Qwen3.5-9B}"

cd "$PROJECT_ROOT"

if [ -f "$HOME/.bashrc_qwen35_nano_env" ]; then
  source "$HOME/.bashrc_qwen35_nano_env"
fi
source .venv/bin/activate

while tmux has-session -t qwen35_9b_download 2>/dev/null; do
  sleep 60
done

if [ ! -f "$TARGET_DIR/config.json" ]; then
  echo "9B model download did not complete successfully: $TARGET_DIR" >&2
  exit 1
fi

python -m qwen35_nano.train --config configs/smoke_9b_single_gpu.yaml

