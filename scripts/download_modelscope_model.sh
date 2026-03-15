#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:?usage: $0 <ModelScope model id> [local_dir]}"
LOCAL_DIR="${2:-}"

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "Activate the project venv first." >&2
  exit 1
fi

if [ -f "$HOME/.bashrc_qwen35_nano_env" ]; then
  source "$HOME/.bashrc_qwen35_nano_env"
fi

if [ -z "$LOCAL_DIR" ]; then
  model_name="$(basename "$MODEL_ID")"
  LOCAL_DIR="$HOME/autodl-tmp/models/$model_name"
fi

mkdir -p "$LOCAL_DIR"
modelscope download --model "$MODEL_ID" --local_dir "$LOCAL_DIR"

