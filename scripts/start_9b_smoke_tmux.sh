#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/autodl-tmp/qwen3.5-nano}"
SESSION_NAME="${2:-qwen35_9b_smoke}"

cd "$PROJECT_ROOT"

if [ -f "$HOME/.bashrc_qwen35_nano_env" ]; then
  source "$HOME/.bashrc_qwen35_nano_env"
fi
source .venv/bin/activate

tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME" || true
tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_ROOT' && source .venv/bin/activate && bash scripts/wait_and_smoke_9b.sh '$PROJECT_ROOT' > '$PROJECT_ROOT/qwen35_9b_smoke.log' 2>&1"
tmux ls | grep "$SESSION_NAME"
sleep 2
tail -n 20 "$PROJECT_ROOT/qwen35_9b_smoke.log" || true
