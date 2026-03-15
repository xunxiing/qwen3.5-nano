from __future__ import annotations

import argparse

from qwen35_nano.laion import build_laion_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="LAION metadata path: parquet/jsonl/csv or directory.")
    parser.add_argument("--output", required=True, help="Output manifest jsonl.")
    parser.add_argument("--image-root", default=None, help="Optional root folder for relative image paths.")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--image-column", default=None)
    parser.add_argument("--aesthetic-column", default=None)
    parser.add_argument("--min-aesthetic-score", type=float, default=None)
    parser.add_argument("--min-width", type=int, default=None)
    parser.add_argument("--min-height", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = build_laion_manifest(
        input_path=args.input,
        output_path=args.output,
        image_root=args.image_root,
        text_column=args.text_column,
        image_column=args.image_column,
        aesthetic_column=args.aesthetic_column,
        min_aesthetic_score=args.min_aesthetic_score,
        min_width=args.min_width,
        min_height=args.min_height,
        max_samples=args.max_samples,
    )
    print(f"wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
