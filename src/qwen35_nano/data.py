from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    text: str
    image_tokens: list[int]


class TokenizedImageTextDataset(Dataset[Sample]):
    """Loads text + pre-encoded image token pairs from JSONL."""

    def __init__(self, manifest_path: str | Path, image_seq_len: int) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_seq_len = image_seq_len
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.samples = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        item = self.samples[index]
        tokens = item["image_tokens"][: self.image_seq_len]
        if len(tokens) < self.image_seq_len:
            raise ValueError(
                f"Sample {index} has {len(tokens)} image tokens, expected at least {self.image_seq_len}."
            )
        return Sample(text=item["text"], image_tokens=tokens)


class MockTokenDataset(Dataset[Sample]):
    """Small synthetic dataset for smoke tests before GPU is attached."""

    def __init__(self, num_samples: int, image_seq_len: int, image_vocab_size: int) -> None:
        self.num_samples = num_samples
        self.image_seq_len = image_seq_len
        self.image_vocab_size = image_vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Sample:
        generator = torch.Generator().manual_seed(index)
        tokens = torch.randint(
            low=0,
            high=self.image_vocab_size,
            size=(self.image_seq_len,),
            generator=generator,
        ).tolist()
        return Sample(
            text=f"mock prompt {index}: armor, damaged cloak, cinematic lighting",
            image_tokens=tokens,
        )


class TextImageCollator:
    """
    Builds:
    - text ids for the frozen Qwen encoder
    - causal decoder inputs for the visual AR model
    - labels masked outside image prediction positions
    """

    def __init__(
        self,
        tokenizer: Any,
        text_max_length: int,
        image_seq_len: int,
        image_bos_id: int,
        image_eos_id: int,
        pad_token_id: int,
        cfg_dropout_prob: float = 0.1,
        unconditional_text: str = "",
    ) -> None:
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.image_seq_len = image_seq_len
        self.image_bos_id = image_bos_id
        self.image_eos_id = image_eos_id
        self.pad_token_id = pad_token_id
        self.cfg_dropout_prob = cfg_dropout_prob
        self.unconditional_text = unconditional_text

    def __call__(self, batch: list[Sample]) -> dict[str, torch.Tensor]:
        texts = []
        image_targets = []
        fallback_text = self.unconditional_text
        if not fallback_text:
            fallback_text = self.tokenizer.eos_token or self.tokenizer.pad_token or " "
        for sample in batch:
            text = sample.text
            if torch.rand(()) < self.cfg_dropout_prob:
                text = fallback_text
            elif not text:
                text = fallback_text
            texts.append(text)
            image_targets.append(torch.tensor(sample.image_tokens, dtype=torch.long))

        encoded = self.tokenizer(
            texts,
            max_length=self.text_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        image_targets_tensor = torch.stack(image_targets, dim=0)  # [B, L_img]
        image_bos = torch.full(
            (image_targets_tensor.size(0), 1),
            fill_value=self.image_bos_id,
            dtype=torch.long,
        )
        image_eos = torch.full(
            (image_targets_tensor.size(0), 1),
            fill_value=self.image_eos_id,
            dtype=torch.long,
        )

        decoder_input_ids = torch.cat(
            [image_bos, image_targets_tensor],
            dim=1,
        )  # [B, 1 + L_img]

        labels = torch.cat(
            [image_targets_tensor, image_eos],
            dim=1,
        )  # [B, L_img + 1]

        decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.long)

        return {
            "text_input_ids": encoded["input_ids"],
            "text_attention_mask": encoded["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }
