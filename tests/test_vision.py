from pathlib import Path

import torch
from PIL import Image

from qwen35_nano.laion import build_laion_manifest
from qwen35_nano.vision import MockPatchVisionTokenizer, build_image_transform


def test_mock_patch_tokenizer_roundtrip_shape() -> None:
    tokenizer = MockPatchVisionTokenizer(image_size=32, patch_size=8, levels_per_channel=8)
    images = torch.zeros(2, 3, 32, 32)
    encoded = tokenizer.encode(images)
    decoded = tokenizer.decode(encoded.token_ids)
    assert encoded.token_ids.shape == (2, 16)
    assert decoded.shape == (2, 3, 32, 32)


def test_build_laion_manifest_filters_missing_images(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_path)
    metadata_path = tmp_path / "meta.jsonl"
    metadata_path.write_text(
        '{"caption":"red square","image_path":"sample.png"}\n'
        '{"caption":"missing","image_path":"missing.png"}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "manifest.jsonl"
    written = build_laion_manifest(metadata_path, output_path, image_root=tmp_path)
    assert written == 1
    assert "red square" in output_path.read_text(encoding="utf-8")


def test_build_laion_manifest_filters_by_aesthetic_and_resolution(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(image_path)
    metadata_path = tmp_path / "meta.csv"
    metadata_path.write_text(
        "caption,image_path,aesthetic_score,width,height\n"
        "keep,sample.png,6.5,1024,1024\n"
        "drop_low,sample.png,4.0,1024,1024\n"
        "drop_small,sample.png,6.5,128,128\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "manifest.jsonl"
    written = build_laion_manifest(
        metadata_path,
        output_path,
        image_root=tmp_path,
        min_aesthetic_score=5.0,
        min_width=512,
        min_height=512,
    )
    content = output_path.read_text(encoding="utf-8")
    assert written == 1
    assert "keep" in content
    assert "drop_low" not in content


def test_image_transform_normalizes_to_unit_range(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(image_path)
    tensor = build_image_transform(32)(Image.open(image_path))
    assert tensor.shape == (3, 32, 32)
    assert float(tensor.max()) <= 1.0
    assert float(tensor.min()) >= -1.0
