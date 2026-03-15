#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/autodl-tmp/qwen3.5-nano}"
INPUT_META="${2:-}"
IMAGE_ROOT="${3:-}"
MANIFEST_OUT="${4:-$PROJECT_ROOT/data/laion_aesthetics_manifest.jsonl}"
TOKENS_OUT="${5:-$PROJECT_ROOT/data/laion_aesthetics_tokens.jsonl}"
IMAGE_SIZE="${6:-256}"
BATCH_SIZE="${7:-8}"
MAX_SAMPLES="${8:-}"
MIN_AESTHETIC_SCORE="${MIN_AESTHETIC_SCORE:-}"
MIN_WIDTH="${MIN_WIDTH:-256}"
MIN_HEIGHT="${MIN_HEIGHT:-256}"

TAMING_REPO="${TAMING_REPO:-$HOME/autodl-tmp/third_party/taming-transformers}"
TAMING_CKPT="${TAMING_CKPT:-$HOME/autodl-tmp/models/vqgan_imagenet_f16_16384.ckpt}"
TAMING_CONFIG="${TAMING_CONFIG:-$HOME/autodl-tmp/models/vqgan_imagenet_f16_16384.yaml}"

if [ -z "$INPUT_META" ]; then
  echo "Usage: bash scripts/run_real_laion_taming_pipeline.sh <project_root> <input_meta> <image_root> [manifest_out] [tokens_out] [image_size] [batch_size] [max_samples]" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

if [ -f "$HOME/.bashrc_qwen35_nano_env" ]; then
  source "$HOME/.bashrc_qwen35_nano_env"
fi
source .venv/bin/activate
export PYTHONPATH=src

bash scripts/setup_taming_vqgan.sh "$PROJECT_ROOT" "$TAMING_REPO" "$TAMING_CKPT" "$TAMING_CONFIG"

mkdir -p "$(dirname "$MANIFEST_OUT")" "$(dirname "$TOKENS_OUT")"

prepare_args=(
  --input "$INPUT_META"
  --output "$MANIFEST_OUT"
  --min-width "$MIN_WIDTH"
  --min-height "$MIN_HEIGHT"
)
if [ -n "$IMAGE_ROOT" ]; then
  prepare_args+=(--image-root "$IMAGE_ROOT")
fi
if [ -n "$MIN_AESTHETIC_SCORE" ]; then
  prepare_args+=(--min-aesthetic-score "$MIN_AESTHETIC_SCORE")
fi
if [ -n "$MAX_SAMPLES" ]; then
  prepare_args+=(--max-samples "$MAX_SAMPLES")
fi

python -m qwen35_nano.prepare_laion "${prepare_args[@]}"

python -m qwen35_nano.precompute_tokens \
  --manifest "$MANIFEST_OUT" \
  --output "$TOKENS_OUT" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --tokenizer-backend taming_vqgan_f16_16384 \
  --taming-repo "$TAMING_REPO" \
  --taming-ckpt "$TAMING_CKPT" \
  --taming-config "$TAMING_CONFIG"
