from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from qwen35_nano.config import Config
from qwen35_nano.model import (
    FrozenQwenTextEncoder,
    QwenTextToImageModel,
    VisualARDecoder,
    build_tokenizer,
)
from qwen35_nano.vision import TamingVQGANTokenizer, save_tensor_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", action="append", dest="prompts", default=[])
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--unconditional-text", default="")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--taming-ckpt", required=True)
    parser.add_argument("--taming-config", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = [prompt.strip() for prompt in args.prompts if prompt.strip()]
    if args.prompt_file:
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prompts.append(line)
    if not prompts:
        raise ValueError("Provide at least one --prompt or a --prompt-file")
    return prompts


def top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits
    if top_k > 0 and top_k < filtered.size(-1):
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        scatter_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
        scatter_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        filtered = filtered.masked_fill(scatter_mask, float("-inf"))
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    filtered = logits / temperature
    filtered = top_k_top_p_filter(filtered, top_k=top_k, top_p=top_p)
    probs = torch.softmax(filtered.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1)


def build_model(cfg: Config) -> tuple[QwenTextToImageModel, object, object]:
    model_cfg = cfg["model"]
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
    return QwenTextToImageModel(text_encoder=text_encoder, decoder=decoder), tokenizer, special_ids


@torch.no_grad()
def generate_tokens(
    model: QwenTextToImageModel,
    tokenizer,
    special_ids,
    prompt: str,
    *,
    device: torch.device,
    model_dtype: torch.dtype,
    text_max_length: int,
    image_vocab_size: int,
    image_seq_len: int,
    temperature: float,
    top_k: int,
    top_p: float,
    cfg_scale: float,
    unconditional_text: str,
) -> torch.Tensor:
    if cfg_scale != 1.0:
        encoded = tokenizer(
            [prompt, unconditional_text],
            max_length=text_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        encoder_hidden_states = model.text_encoder(input_ids, attention_mask)
        decoder_input_ids = torch.full(
            (2, 1),
            fill_value=special_ids.image_bos_id,
            dtype=torch.long,
            device=device,
        )
    else:
        encoded = tokenizer(
            [prompt],
            max_length=text_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        encoder_hidden_states = model.text_encoder(input_ids, attention_mask)
        decoder_input_ids = torch.full(
            (1, 1),
            fill_value=special_ids.image_bos_id,
            dtype=torch.long,
            device=device,
        )

    model.decoder.eval()
    generated: list[torch.Tensor] = []
    encoder_hidden_states = encoder_hidden_states.to(model_dtype)

    for _ in range(image_seq_len):
        logits = model.decoder(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )
        next_logits = logits[:, -1, :image_vocab_size]
        if cfg_scale != 1.0:
            cond_logits = next_logits[:1]
            uncond_logits = next_logits[1:]
            next_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            decoder_append = sample_next_token(next_logits, temperature, top_k, top_p)
            decoder_input_ids = torch.cat(
                [decoder_input_ids, decoder_append.expand(2, 1)],
                dim=1,
            )
            generated.append(decoder_append)
        else:
            next_token = sample_next_token(next_logits, temperature, top_k, top_p)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            generated.append(next_token)

    return torch.cat(generated, dim=1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = Config.load(args.config)
    prompts = load_prompts(args)

    model_cfg = cfg["model"]
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(cfg["dtype"])

    model, tokenizer, special_ids = build_model(cfg)
    model = model.to(device)
    model.decoder.to(device=device, dtype=model_dtype)
    model.eval()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.decoder.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "missing_keys": missing,
                    "unexpected_keys": unexpected,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    vq_tokenizer = TamingVQGANTokenizer(
        checkpoint_path=args.taming_ckpt,
        config_path=args.taming_config,
        image_size=args.image_size,
        device=device,
        dtype=torch.float32,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, object]] = []

    for index, prompt in enumerate(prompts):
        image_tokens = generate_tokens(
            model=model,
            tokenizer=tokenizer,
            special_ids=special_ids,
            prompt=prompt,
            device=device,
            model_dtype=model_dtype,
            text_max_length=model_cfg["text_max_length"],
            image_vocab_size=model_cfg["image_vocab_size"],
            image_seq_len=model_cfg["image_seq_len"],
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            cfg_scale=args.cfg_scale,
            unconditional_text=args.unconditional_text,
        )
        decoded = vq_tokenizer.decode(image_tokens)[0]
        image_name = f"{index:02d}.png"
        image_path = output_dir / image_name
        save_tensor_image(decoded, image_path)
        metadata.append(
            {
                "index": index,
                "prompt": prompt,
                "image": image_name,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "cfg_scale": args.cfg_scale,
                "seed": args.seed,
            }
        )
        print(json.dumps(metadata[-1], ensure_ascii=False), flush=True)

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
