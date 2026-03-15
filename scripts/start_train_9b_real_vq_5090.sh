#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/autodl-tmp/qwen3.5-nano}"
CONFIG_PATH="${2:-configs/train_9b_real_vq_5090.yaml}"
LOG_PATH="${3:-$PROJECT_ROOT/tmp/train_9b_real_vq_5090.log}"

cd "$PROJECT_ROOT"

if [ -f "$HOME/.bashrc_qwen35_nano_env" ]; then
  source "$HOME/.bashrc_qwen35_nano_env"
fi
source .venv/bin/activate
export PYTHONPATH=src

mkdir -p "$(dirname "$LOG_PATH")"
nohup python -u -m qwen35_nano.train --config "$CONFIG_PATH" > "$LOG_PATH" 2>&1 < /dev/null &
echo $! > "${LOG_PATH}.pid"
echo "pid=$(cat "${LOG_PATH}.pid")"
echo "log=$LOG_PATH"
