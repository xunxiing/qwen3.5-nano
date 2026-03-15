#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/autodl-tmp/qwen3.5-nano}"
cd "$PROJECT_ROOT"

if [ -f "$HOME/.bashrc_qwen35_nano_env" ]; then
  source "$HOME/.bashrc_qwen35_nano_env"
fi
source .venv/bin/activate

mkdir -p tmp/smoke_images data

python - <<'PY'
from pathlib import Path

from PIL import Image, ImageDraw

root = Path("tmp/smoke_images")
root.mkdir(parents=True, exist_ok=True)
meta_path = Path("tmp/smoke_meta.jsonl")
records = []
for idx, color in enumerate([(220, 30, 30), (30, 30, 220)]):
    image = Image.new("RGB", (320, 320), color=color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 60, 260, 260), outline=(255, 255, 255), width=8)
    draw.text((80, 145), f"sample-{idx}", fill=(255, 255, 255))
    image_path = root / f"sample_{idx}.png"
    image.save(image_path)
    records.append(
        {
            "caption": f"synthetic laion sample {idx}, framed square, high contrast",
            "image_path": image_path.name,
        }
    )

with meta_path.open("w", encoding="utf-8") as handle:
    for record in records:
        handle.write(__import__("json").dumps(record, ensure_ascii=False) + "\n")
PY

python -m qwen35_nano.prepare_laion \
  --input tmp/smoke_meta.jsonl \
  --image-root tmp/smoke_images \
  --output data/smoke_manifest.jsonl

python -m qwen35_nano.precompute_tokens \
  --manifest data/smoke_manifest.jsonl \
  --output data/smoke_tokens.jsonl \
  --image-size 128 \
  --patch-size 16 \
  --batch-size 2 \
  --max-samples 2

python -m qwen35_nano.train --config configs/smoke_2b_pipeline.yaml

