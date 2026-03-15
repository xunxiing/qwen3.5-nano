#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/autodl-tmp/qwen3.5-nano}"
REPO_DIR="${2:-$HOME/autodl-tmp/third_party/taming-transformers}"
CKPT_PATH="${3:-$HOME/autodl-tmp/models/vqgan_imagenet_f16_16384.ckpt}"
CONFIG_PATH="${4:-$HOME/autodl-tmp/models/vqgan_imagenet_f16_16384.yaml}"

cd "$PROJECT_ROOT"

if [ -f "$HOME/.bashrc_qwen35_nano_env" ]; then
  source "$HOME/.bashrc_qwen35_nano_env"
fi
source .venv/bin/activate

bash scripts/setup_taming_vqgan.sh "$PROJECT_ROOT" "$REPO_DIR" "$CKPT_PATH" "$CONFIG_PATH"

mkdir -p tmp/real_vq_images data tmp/real_vq_recon

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw

root = Path("tmp/real_vq_images")
root.mkdir(parents=True, exist_ok=True)
meta = Path("tmp/real_vq_meta.csv")
rows = ["caption,image_path,aesthetic_score,width,height"]
for idx, color in enumerate([(240, 120, 40), (60, 120, 240)]):
    image = Image.new("RGB", (512, 512), color=color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((96, 96, 416, 416), outline=(255, 255, 255), width=10)
    draw.text((150, 246), f"vq-{idx}", fill=(255, 255, 255))
    path = root / f"vq_{idx}.png"
    image.save(path)
    rows.append(f"synthetic aesthetics sample {idx},{path.name},6.2,512,512")
meta.write_text("\n".join(rows) + "\n", encoding="utf-8")
PY

python -m qwen35_nano.prepare_laion \
  --input tmp/real_vq_meta.csv \
  --image-root tmp/real_vq_images \
  --output data/real_vq_smoke_manifest.jsonl \
  --min-aesthetic-score 5.0 \
  --min-width 512 \
  --min-height 512

python -m qwen35_nano.precompute_tokens \
  --manifest data/real_vq_smoke_manifest.jsonl \
  --output data/real_vq_smoke_tokens.jsonl \
  --image-size 256 \
  --batch-size 2 \
  --tokenizer-backend taming_vqgan_f16_16384 \
  --taming-repo "$REPO_DIR" \
  --taming-ckpt "$CKPT_PATH" \
  --taming-config "$CONFIG_PATH" \
  --save-reconstruction-dir tmp/real_vq_recon \
  --max-samples 2

python -m qwen35_nano.train --config configs/smoke_2b_real_vq_pipeline.yaml
