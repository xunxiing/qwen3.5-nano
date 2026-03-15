#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/autodl-tmp/qwen3.5-nano}"
SOURCE_MANIFEST="${2:-$PROJECT_ROOT/data/pokemon_blip_tokens.jsonl}"
MAX_STEPS="${3:-1500}"
LR="${4:-0.0002}"
OUTPUT_DIR="${5:-outputs/overfit_one_pokemon}"
CONFIG_PATH="${6:-tmp/overfit_one_pokemon.yaml}"
ONE_MANIFEST_PATH="${7:-$PROJECT_ROOT/data/pokemon_blip_tokens_one.jsonl}"

cd "$PROJECT_ROOT"

if [ -f "$HOME/.bashrc_qwen35_nano_env" ]; then
  source "$HOME/.bashrc_qwen35_nano_env"
fi
source .venv/bin/activate
export PYTHONPATH=src

python - <<'PY' "$SOURCE_MANIFEST" "$ONE_MANIFEST_PATH" "$CONFIG_PATH" "$OUTPUT_DIR" "$MAX_STEPS" "$LR"
from pathlib import Path
import json
import sys

import yaml

source_manifest = Path(sys.argv[1])
one_manifest_path = Path(sys.argv[2])
config_path = Path(sys.argv[3])
output_dir = sys.argv[4]
max_steps = int(sys.argv[5])
lr = float(sys.argv[6])

with source_manifest.open("r", encoding="utf-8") as handle:
    first = next(handle)

row = json.loads(first)
one_manifest_path.parent.mkdir(parents=True, exist_ok=True)
one_manifest_path.write_text(first, encoding="utf-8")

config = {
    "seed": 42,
    "output_dir": output_dir,
    "device": "cuda",
    "dtype": "bfloat16",
    "model": {
        "text_model_name": "/root/autodl-tmp/models/Qwen3.5-9B",
        "trust_remote_code": True,
        "text_max_length": 64,
        "image_vocab_size": 16384,
        "image_seq_len": 256,
        "hidden_size": 1024,
        "num_layers": 12,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "gradient_checkpointing": True,
    },
    "data": {
        "train_manifest": str(one_manifest_path.relative_to(Path.cwd())),
        "batch_size": 1,
        "num_workers": 0,
        "cfg_dropout_prob": 0.0,
        "unconditional_text": "",
    },
    "train": {
        "max_steps": max_steps,
        "lr": lr,
        "weight_decay": 0.0,
        "grad_accum_steps": 1,
        "log_every": 50,
        "save_every": 500,
        "max_grad_norm": 1.0,
    },
}

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

prompt_path = config_path.with_suffix(".prompt.txt")
prompt_path.write_text(row["text"], encoding="utf-8")

print(f"single-sample prompt: {row['text']}")
print(f"single-sample tokens: {len(row['image_tokens'])}")
print(f"manifest: {one_manifest_path}")
print(f"config: {config_path}")
print(f"prompt_file: {prompt_path}")
PY

python -u -m qwen35_nano.train --config "$CONFIG_PATH"
