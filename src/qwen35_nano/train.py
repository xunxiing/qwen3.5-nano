from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import time
from typing import Iterator

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from qwen35_nano.config import Config
from qwen35_nano.data import MockTokenDataset, TextImageCollator, TokenizedImageTextDataset
from qwen35_nano.losses import (
    blank_token_suppression_loss,
    build_token_frequency_weights,
    build_color_token_weights,
    focal_cross_entropy,
    load_blank_token_ids,
    repeat_unlikelihood_loss,
    token_distribution_kl,
)
from qwen35_nano.model import (
    FrozenQwenTextEncoder,
    QwenTextToImageModel,
    VisualARDecoder,
    build_tokenizer,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping[name.lower()]


def build_dataloader(cfg: Config, tokenizer, special_ids):
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    if Path(data_cfg["train_manifest"]).exists():
        dataset = TokenizedImageTextDataset(
            manifest_path=data_cfg["train_manifest"],
            image_seq_len=model_cfg["image_seq_len"],
        )
    else:
        dataset = MockTokenDataset(
            num_samples=max(cfg["train"]["max_steps"] * data_cfg["batch_size"], 8),
            image_seq_len=model_cfg["image_seq_len"],
            image_vocab_size=model_cfg["image_vocab_size"],
        )

    collator = TextImageCollator(
        tokenizer=tokenizer,
        text_max_length=model_cfg["text_max_length"],
        image_seq_len=model_cfg["image_seq_len"],
        image_bos_id=special_ids.image_bos_id,
        image_eos_id=special_ids.image_eos_id,
        pad_token_id=tokenizer.pad_token_id,
        cfg_dropout_prob=data_cfg["cfg_dropout_prob"],
        unconditional_text=data_cfg["unconditional_text"],
    )
    return DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=data_cfg["num_workers"] > 0,
        prefetch_factor=4 if data_cfg["num_workers"] > 0 else None,
        collate_fn=collator,
    )


def infinite_dataloader(dataloader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in dataloader:
            yield batch


def build_loss_config(train_cfg: dict) -> dict[str, float | bool]:
    loss_cfg = train_cfg.get("loss", {})
    return {
        "use_class_balance": bool(loss_cfg.get("use_class_balance", False)),
        "class_balance_alpha": float(loss_cfg.get("class_balance_alpha", 0.5)),
        "class_balance_min_weight": float(loss_cfg.get("class_balance_min_weight", 0.25)),
        "class_balance_max_weight": float(loss_cfg.get("class_balance_max_weight", 6.0)),
        "focal_gamma": float(loss_cfg.get("focal_gamma", 0.0)),
        "label_smoothing": float(loss_cfg.get("label_smoothing", 0.0)),
        "token_kl_weight": float(loss_cfg.get("token_kl_weight", 0.0)),
        "repeat_unlikelihood_weight": float(loss_cfg.get("repeat_unlikelihood_weight", 0.0)),
        "use_color_weights": bool(loss_cfg.get("use_color_weights", False)),
        "color_stats_path": loss_cfg.get("color_stats_path"),
        "color_reward_strength": float(loss_cfg.get("color_reward_strength", 1.0)),
        "blank_penalty_strength": float(loss_cfg.get("blank_penalty_strength", 0.75)),
        "blank_token_top_k": int(loss_cfg.get("blank_token_top_k", 64)),
        "blank_suppression_weight": float(loss_cfg.get("blank_suppression_weight", 0.0)),
    }


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    set_seed(cfg["seed"])

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(cfg["dtype"])
    loss_cfg = build_loss_config(train_cfg)

    tokenizer, special_ids = build_tokenizer(
        model_name=model_cfg["text_model_name"],
        trust_remote_code=model_cfg["trust_remote_code"],
        image_vocab_size=model_cfg["image_vocab_size"],
    )

    text_encoder = FrozenQwenTextEncoder(
        model_name=model_cfg["text_model_name"],
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    decoder = VisualARDecoder(
        image_vocab_size=model_cfg["image_vocab_size"],
        image_seq_len=model_cfg["image_seq_len"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
        dropout=model_cfg["dropout"],
        encoder_hidden_size=text_encoder.hidden_size,
    )
    decoder = decoder.to(device=device, dtype=model_dtype)
    model = QwenTextToImageModel(text_encoder=text_encoder, decoder=decoder).to(device)
    if model_cfg.get("gradient_checkpointing", False):
        for layer in model.decoder.layers:
            layer._gradient_checkpointing = True
    optimizer = AdamW(
        model.decoder.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    if train_cfg.get("resume_from"):
        checkpoint = torch.load(train_cfg["resume_from"], map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = model.decoder.load_state_dict(state_dict, strict=False)
        print(
            json.dumps(
                {
                    "resume_from": train_cfg["resume_from"],
                    "missing_keys": missing,
                    "unexpected_keys": unexpected,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    class_weights = None
    color_weights = None
    blank_token_ids = None
    if loss_cfg["use_class_balance"] and Path(cfg["data"]["train_manifest"]).exists():
        class_weights = build_token_frequency_weights(
            cfg["data"]["train_manifest"],
            model_cfg["image_vocab_size"],
            alpha=loss_cfg["class_balance_alpha"],
            min_weight=loss_cfg["class_balance_min_weight"],
            max_weight=loss_cfg["class_balance_max_weight"],
        ).to(device)
        print(
            json.dumps(
                {
                    "loss_class_balance": True,
                    "class_weight_min": round(float(class_weights.min().item()), 4),
                    "class_weight_max": round(float(class_weights.max().item()), 4),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if loss_cfg["use_color_weights"] and loss_cfg["color_stats_path"]:
        color_weights = build_color_token_weights(
            loss_cfg["color_stats_path"],
            model_cfg["image_vocab_size"],
            color_reward_strength=loss_cfg["color_reward_strength"],
            blank_penalty_strength=loss_cfg["blank_penalty_strength"],
        ).to(device)
        blank_token_ids = load_blank_token_ids(
            loss_cfg["color_stats_path"],
            model_cfg["image_vocab_size"],
            top_k=loss_cfg["blank_token_top_k"],
        ).to(device)
        print(
            json.dumps(
                {
                    "loss_color_weights": True,
                    "color_weight_min": round(float(color_weights.min().item()), 4),
                    "color_weight_max": round(float(color_weights.max().item()), 4),
                    "blank_token_top_k": int(blank_token_ids.numel()),
                    "blank_token_preview": blank_token_ids[:8].tolist(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if class_weights is not None and color_weights is not None:
        class_weights = (class_weights * color_weights).clamp(min=0.1, max=10.0)
        class_weights = class_weights / class_weights.mean()
    elif color_weights is not None:
        class_weights = color_weights

    dataloader = build_dataloader(cfg, tokenizer, special_ids)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_log_start = time.perf_counter()
    last_grad_norm = None

    data_iter = infinite_dataloader(dataloader)
    for step in range(1, train_cfg["max_steps"] + 1):
        batch = next(data_iter)
        batch = {key: value.to(device) for key, value in batch.items()}
        autocast_enabled = device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16)
        with torch.autocast(device_type=device.type, dtype=model_dtype, enabled=autocast_enabled):
            outputs = model(
                text_input_ids=batch["text_input_ids"],
                text_attention_mask=batch["text_attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                labels=None,
            )
            ce_loss = focal_cross_entropy(
                outputs["logits"],
                batch["labels"],
                class_weights=class_weights,
                gamma=loss_cfg["focal_gamma"],
                label_smoothing=loss_cfg["label_smoothing"],
            )
            token_kl = outputs["logits"].new_zeros(())
            if loss_cfg["token_kl_weight"] > 0:
                token_kl = token_distribution_kl(
                    outputs["logits"],
                    batch["labels"],
                    vocab_size=model_cfg["image_vocab_size"],
                )
            repeat_ul = outputs["logits"].new_zeros(())
            if loss_cfg["repeat_unlikelihood_weight"] > 0:
                repeat_ul = repeat_unlikelihood_loss(
                    outputs["logits"],
                    batch["labels"],
                    batch["decoder_input_ids"],
                )
            blank_sup = outputs["logits"].new_zeros(())
            if loss_cfg["blank_suppression_weight"] > 0 and blank_token_ids is not None:
                blank_sup = blank_token_suppression_loss(
                    outputs["logits"],
                    batch["labels"],
                    blank_token_ids,
                )
            raw_loss = (
                ce_loss
                + loss_cfg["token_kl_weight"] * token_kl
                + loss_cfg["repeat_unlikelihood_weight"] * repeat_ul
                + loss_cfg["blank_suppression_weight"] * blank_sup
            )
        loss = raw_loss / train_cfg["grad_accum_steps"]
        loss.backward()

        if step % train_cfg["grad_accum_steps"] == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), train_cfg["max_grad_norm"])
            last_grad_norm = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % train_cfg["log_every"] == 0:
            payload = {
                "step": step,
                "loss": float(raw_loss.item()),
                "ce_loss": round(float(ce_loss.item()), 4),
                "step_time_sec": round(time.perf_counter() - running_log_start, 4),
            }
            if loss_cfg["token_kl_weight"] > 0:
                payload["token_kl"] = round(float(token_kl.item()), 4)
            if loss_cfg["repeat_unlikelihood_weight"] > 0:
                payload["repeat_ul"] = round(float(repeat_ul.item()), 4)
            if loss_cfg["blank_suppression_weight"] > 0 and blank_token_ids is not None:
                payload["blank_sup"] = round(float(blank_sup.item()), 4)
            if last_grad_norm is not None:
                payload["grad_norm"] = round(last_grad_norm, 4)
            if device.type == "cuda":
                payload["gpu_mem_alloc_gb"] = round(torch.cuda.memory_allocated(device) / (1024**3), 3)
                payload["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved(device) / (1024**3), 3)
                payload["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
                torch.cuda.reset_peak_memory_stats(device)
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            running_log_start = time.perf_counter()

        if step % train_cfg["save_every"] == 0:
            ckpt_path = output_dir / f"decoder_step_{step}.pt"
            torch.save({"model": model.decoder.state_dict()}, ckpt_path)

    final_ckpt_path = output_dir / "decoder_final.pt"
    torch.save({"model": model.decoder.state_dict()}, final_ckpt_path)


if __name__ == "__main__":
    main()
