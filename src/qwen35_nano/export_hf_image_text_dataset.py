from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset id")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--image-format", default="jpg", choices=["jpg", "png"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--resize", type=int, default=None, help="Optional square resize before save")
    return parser.parse_args()


def center_crop_resize(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    image = image.crop((left, top, left + min_dim, top + min_dim))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"

    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    iterator = dataset if args.streaming else range(len(dataset))

    written = 0
    with metadata_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["caption", "image_path", "width", "height"])

        progress = tqdm(iterator, desc="export_hf_dataset")
        for item in progress:
            row = item if args.streaming else dataset[item]
            image = row.get(args.image_column)
            text = str(row.get(args.text_column, "")).strip()
            if image is None or not text:
                continue
            if not isinstance(image, Image.Image):
                continue

            image = image.convert("RGB")
            if args.resize is not None:
                image = center_crop_resize(image, args.resize)

            file_stem = f"{written:05d}"
            image_rel_path = f"images/{file_stem}.{args.image_format}"
            image_path = output_dir / image_rel_path

            if args.image_format == "jpg":
                image.save(image_path, "JPEG", quality=95)
            else:
                image.save(image_path, "PNG")
            (output_dir / f"{file_stem}.txt").write_text(text, encoding="utf-8")

            width, height = image.size
            writer.writerow([text, image_rel_path, width, height])
            written += 1
            progress.set_postfix(written=written)

            if args.max_samples is not None and written >= args.max_samples:
                break

    print(f"wrote {written} samples to {output_dir}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
