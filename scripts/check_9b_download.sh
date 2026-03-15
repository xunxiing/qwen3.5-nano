#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/autodl-tmp/qwen3.5-nano}"
TARGET_DIR="${2:-$HOME/autodl-tmp/models/Qwen3.5-9B}"

tmux ls | grep qwen35_9b_download || true
du -sh "$TARGET_DIR" 2>/dev/null || true
tail -n 20 "$PROJECT_ROOT/qwen35_9b_download.log" || true
