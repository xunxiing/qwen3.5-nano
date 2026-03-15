from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from qwen35_nano.vision import (
    LlamaGenVQTokenizer,
    MockPatchVisionTokenizer,
    TamingVQGANTokenizer,
    build_image_transform,
    save_tensor_image,
)


class ImageCaptionDataset(Dataset[dict[str, str]]):
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="jsonl with {text, image_path}")
    parser.add_argument("--output", required=True, help="jsonl with {text, image_tokens, image_path}")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--tokenizer-backend",
        choices=["mock", "llamagen_vq16_t2i", "taming_vqgan_f16_16384"],
        default="mock",
    )
    parser.add_argument("--llamagen-repo", default=None)
    parser.add_argument("--llamagen-ckpt", default=None)
    parser.add_argument("--taming-repo", default=None)
    parser.add_argument("--taming-ckpt", default=None)
    parser.add_argument("--taming-config", default=None)
    parser.add_argument("--save-reconstruction-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def collate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return rows


def load_batch(rows: list[dict[str, str]], image_transform) -> tuple[list[dict[str, str]], torch.Tensor]:
    valid_rows: list[dict[str, str]] = []
    image_tensors: list[torch.Tensor] = []
    for row in rows:
        try:
            image = Image.open(row["image_path"])
            image_tensors.append(image_transform(image))
            valid_rows.append(row)
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            continue
    if not image_tensors:
        return [], torch.empty(0)
    return valid_rows, torch.stack(image_tensors, dim=0)


def main() -> None:
    args = parse_args()
    dataset = ImageCaptionDataset(args.manifest)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_rows,
    )
    transform = build_image_transform(args.image_size)
    if args.tokenizer_backend == "mock":
        tokenizer = MockPatchVisionTokenizer(
            image_size=args.image_size,
            patch_size=args.patch_size,
        )
    elif args.tokenizer_backend == "llamagen_vq16_t2i":
        if not args.llamagen_repo or not args.llamagen_ckpt:
            raise ValueError("--llamagen-repo and --llamagen-ckpt are required for llamagen_vq16_t2i")
        tokenizer = LlamaGenVQTokenizer(
            repo_root=args.llamagen_repo,
            checkpoint_path=args.llamagen_ckpt,
            model_name="VQ-16",
            image_size=args.image_size,
            codebook_size=16384,
            codebook_embed_dim=8,
            device=args.device,
            dtype=torch.float32,
        )
    else:
        if not args.taming_ckpt or not args.taming_config:
            raise ValueError(
                "--taming-ckpt and --taming-config are required for taming_vqgan_f16_16384"
            )
        tokenizer = TamingVQGANTokenizer(
            checkpoint_path=args.taming_ckpt,
            config_path=args.taming_config,
            image_size=args.image_size,
            device=args.device,
            dtype=torch.float32,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reconstruction_dir = Path(args.save_reconstruction_dir) if args.save_reconstruction_dir else None
    if reconstruction_dir:
        reconstruction_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for rows in dataloader:
            valid_rows, images = load_batch(rows, transform)
            if not valid_rows:
                continue

            token_batch = tokenizer.encode(images).token_ids  # [B, L]
            for row, token_ids in zip(valid_rows, token_batch, strict=True):
                record = {
                    "text": row["text"],
                    "image_path": row["image_path"],
                    "image_tokens": token_ids.tolist(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if reconstruction_dir and written == 0:
                    decoded = tokenizer.decode(token_ids.unsqueeze(0))[0]
                    save_tensor_image(decoded, reconstruction_dir / "reconstruction.png")
                written += 1
                if args.max_samples is not None and written >= args.max_samples:
                    print(f"wrote {written} tokenized rows to {output_path}")
                    return

    print(f"wrote {written} tokenized rows to {output_path}")
    print(
        f"tokenizer_backend={args.tokenizer_backend} "
        f"vocab_size={tokenizer.vocab_size} seq_len={tokenizer.seq_len}"
    )


if __name__ == "__main__":
    main()
