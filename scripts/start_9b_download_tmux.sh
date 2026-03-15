#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/autodl-tmp/qwen3.5-nano}"
SESSION_NAME="${2:-qwen35_9b_download}"
TARGET_DIR="${3:-$HOME/autodl-tmp/models/Qwen3.5-9B}"

cd "$PROJECT_ROOT"

if [ -f "$HOME/.bashrc_qwen35_nano_env" ]; then
  source "$HOME/.bashrc_qwen35_nano_env"
fi
source .venv/bin/activate

mkdir -p "$TARGET_DIR"
tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME" || true
tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_ROOT' && source .venv/bin/activate && bash scripts/download_modelscope_model.sh Qwen/Qwen3.5-9B '$TARGET_DIR' > '$PROJECT_ROOT/qwen35_9b_download.log' 2>&1"
tmux ls | grep "$SESSION_NAME"
sleep 2
tail -n 20 "$PROJECT_ROOT/qwen35_9b_download.log" || true

