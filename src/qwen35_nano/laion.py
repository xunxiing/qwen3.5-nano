from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


TEXT_COLUMN_CANDIDATES = ["caption", "text", "TEXT", "prompt", "description"]
IMAGE_COLUMN_CANDIDATES = ["image_path", "path", "filepath", "file_name", "image"]
AESTHETIC_COLUMN_CANDIDATES = ["aesthetic_score", "aesthetic", "score", "aes_score"]
WIDTH_COLUMN_CANDIDATES = ["width", "image_width"]
HEIGHT_COLUMN_CANDIDATES = ["height", "image_height"]


def _iter_input_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_dir():
        for pattern in ("*.parquet", "*.jsonl", "*.csv"):
            yield from sorted(input_path.glob(pattern))
    else:
        yield input_path


def infer_column(columns: list[str], candidates: list[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Column '{explicit}' not found. Available: {columns}")
        return explicit
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not infer column from candidates {candidates}. Available: {columns}")


def read_table(input_file: Path) -> pd.DataFrame:
    suffix = input_file.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_file)
    if suffix == ".jsonl":
        return pd.read_json(input_file, lines=True)
    if suffix == ".csv":
        return pd.read_csv(input_file)
    raise ValueError(f"Unsupported metadata file: {input_file}")


def build_laion_manifest(
    input_path: str | Path,
    output_path: str | Path,
    image_root: str | Path | None = None,
    text_column: str | None = None,
    image_column: str | None = None,
    aesthetic_column: str | None = None,
    min_aesthetic_score: float | None = None,
    min_width: int | None = None,
    min_height: int | None = None,
    max_samples: int | None = None,
) -> int:
    input_root = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image_root_path = Path(image_root) if image_root else None

    written = 0
    with output_file.open("w", encoding="utf-8") as handle:
        for metadata_file in _iter_input_files(input_root):
            table = read_table(metadata_file)
            columns = list(table.columns)
            chosen_text_column = infer_column(columns, TEXT_COLUMN_CANDIDATES, text_column)
            chosen_image_column = infer_column(columns, IMAGE_COLUMN_CANDIDATES, image_column)
            chosen_aesthetic_column = None
            chosen_width_column = None
            chosen_height_column = None
            if min_aesthetic_score is not None:
                chosen_aesthetic_column = infer_column(
                    columns,
                    AESTHETIC_COLUMN_CANDIDATES,
                    aesthetic_column,
                )
            if min_width is not None:
                chosen_width_column = infer_column(columns, WIDTH_COLUMN_CANDIDATES, None)
            if min_height is not None:
                chosen_height_column = infer_column(columns, HEIGHT_COLUMN_CANDIDATES, None)

            for row in table.itertuples(index=False):
                row_dict = row._asdict()
                text = str(row_dict[chosen_text_column]).strip()
                image_value = str(row_dict[chosen_image_column]).strip()
                if chosen_aesthetic_column is not None:
                    if float(row_dict[chosen_aesthetic_column]) < min_aesthetic_score:
                        continue
                if chosen_width_column is not None:
                    if int(row_dict[chosen_width_column]) < min_width:
                        continue
                if chosen_height_column is not None:
                    if int(row_dict[chosen_height_column]) < min_height:
                        continue
                image_path = Path(image_value)
                if image_root_path and not image_path.is_absolute():
                    image_path = image_root_path / image_path
                if not text or not image_path.exists():
                    continue

                record = {"text": text, "image_path": str(image_path)}
                if chosen_aesthetic_column is not None:
                    record["aesthetic_score"] = float(row_dict[chosen_aesthetic_column])
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if max_samples is not None and written >= max_samples:
                    return written
    return written
