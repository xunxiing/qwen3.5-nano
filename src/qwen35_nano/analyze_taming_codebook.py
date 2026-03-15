from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from qwen35_nano.vision import TamingVQGANTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taming-ckpt", required=True)
    parser.add_argument("--taming-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = TamingVQGANTokenizer(
        checkpoint_path=args.taming_ckpt,
        config_path=args.taming_config,
        image_size=args.image_size,
        device=args.device,
        dtype=torch.float32,
    )

    vocab_size = tokenizer.vocab_size
    seq_len = tokenizer.seq_len
    brightness = np.zeros(vocab_size, dtype=np.float32)
    saturation = np.zeros(vocab_size, dtype=np.float32)
    blankness = np.zeros(vocab_size, dtype=np.float32)

    for start in range(0, vocab_size, args.batch_size):
        end = min(start + args.batch_size, vocab_size)
        token_ids = torch.arange(start, end, device=tokenizer.device, dtype=torch.long)
        repeated = token_ids[:, None].repeat(1, seq_len)
        decoded = tokenizer.decode(repeated).float().clamp(-1, 1).add(1.0).mul(0.5)

        rgb_mean = decoded.mean(dim=(1, 2, 3))
        channel_max = decoded.max(dim=1).values
        channel_min = decoded.min(dim=1).values
        chroma = (channel_max - channel_min).mean(dim=(1, 2))
        sat = ((channel_max - channel_min) / channel_max.clamp_min(1e-4)).mean(dim=(1, 2))
        blank = rgb_mean * (1.0 - sat)

        brightness[start:end] = rgb_mean.cpu().numpy()
        saturation[start:end] = sat.cpu().numpy()
        blankness[start:end] = blank.cpu().numpy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        brightness=brightness,
        saturation=saturation,
        blankness=blankness,
    )
    print(
        {
            "output": str(output_path),
            "vocab_size": vocab_size,
            "blankest_token": int(blankness.argmax()),
            "most_colorful_token": int(saturation.argmax()),
        }
    )


if __name__ == "__main__":
    main()
