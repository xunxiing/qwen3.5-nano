from qwen35_nano.data import Sample, TextImageCollator


class DummyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __call__(self, texts, max_length, truncation, padding, return_tensors):
        assert truncation is True
        assert padding is True
        ids = []
        masks = []
        for text in texts:
            encoded = [min(ord(ch), 255) % 17 + 1 for ch in text][:max_length]
            if not encoded:
                encoded = [1]
            ids.append(encoded)
        width = max(len(row) for row in ids)
        for row in ids:
            pad_count = width - len(row)
            masks.append([1] * len(row) + [0] * pad_count)
            row.extend([self.pad_token_id] * pad_count)
        import torch

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def test_collator_shapes() -> None:
    tokenizer = DummyTokenizer()
    collator = TextImageCollator(
        tokenizer=tokenizer,
        text_max_length=16,
        image_seq_len=8,
        image_bos_id=128,
        image_eos_id=129,
        pad_token_id=tokenizer.pad_token_id,
        cfg_dropout_prob=0.0,
    )
    batch = [
        Sample(text="red cape knight", image_tokens=list(range(8))),
        Sample(text="ancient ruins at dusk", image_tokens=list(range(8, 16))),
    ]
    output = collator(batch)
    assert output["text_input_ids"].shape[0] == 2
    assert output["decoder_input_ids"].shape == (2, 9)
    assert output["labels"].shape == (2, 9)
