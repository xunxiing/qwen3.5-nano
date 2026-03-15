from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import yaml

from qwen35_nano.taming_vqgan_modules import Decoder, Encoder, VectorQuantizer2


def build_image_transform(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


@dataclass
class TokenizedImageBatch:
    token_ids: torch.Tensor
    image_size: int
    patch_size: int
    vocab_size: int


class MockPatchVisionTokenizer:
    """
    Fast, deterministic patch tokenizer for smoke tests.

    It is not a production-quality VQ tokenizer. It exists to validate the full
    AR training pipeline before swapping in a learned VQ-VAE/VQ-GAN.
    """

    def __init__(self, image_size: int = 256, patch_size: int = 16, levels_per_channel: int = 16) -> None:
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.levels_per_channel = levels_per_channel
        self.grid_size = image_size // patch_size
        self.seq_len = self.grid_size * self.grid_size
        self.vocab_size = levels_per_channel ** 3

    def encode(self, images: torch.Tensor) -> TokenizedImageBatch:
        """
        images: [B, 3, H, W] normalized to [-1, 1]
        returns token_ids: [B, seq_len]
        """
        if images.ndim != 4:
            raise ValueError(f"Expected [B, 3, H, W], got {tuple(images.shape)}")
        pooled = F.avg_pool2d(
            images.add(1.0).mul(0.5),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )  # [B, 3, H/P, W/P] in [0, 1]
        quantized = torch.clamp(
            (pooled * (self.levels_per_channel - 1)).round().long(),
            min=0,
            max=self.levels_per_channel - 1,
        )
        token_ids = (
            quantized[:, 0]
            + self.levels_per_channel * quantized[:, 1]
            + (self.levels_per_channel**2) * quantized[:, 2]
        ).reshape(images.size(0), -1)
        return TokenizedImageBatch(
            token_ids=token_ids,
            image_size=self.image_size,
            patch_size=self.patch_size,
            vocab_size=self.vocab_size,
        )

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(f"Expected [B, L], got {tuple(token_ids.shape)}")
        grid = token_ids.reshape(token_ids.size(0), self.grid_size, self.grid_size)
        blue = torch.div(grid, self.levels_per_channel**2, rounding_mode="floor")
        green = torch.div(
            grid - blue * (self.levels_per_channel**2),
            self.levels_per_channel,
            rounding_mode="floor",
        )
        red = grid % self.levels_per_channel
        stacked = torch.stack([red, green, blue], dim=1).float()
        stacked = stacked / (self.levels_per_channel - 1)
        upsampled = F.interpolate(
            stacked,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )
        return upsampled.mul(2.0).sub(1.0)


def save_tensor_image(tensor: torch.Tensor, output_path: str | Path) -> None:
    image = tensor.detach().cpu().clamp(-1, 1).add(1.0).mul(127.5).byte()
    image = image.permute(1, 2, 0).numpy()
    Image.fromarray(image).save(output_path)


class LlamaGenVQTokenizer:
    """
    Wrapper around the official LlamaGen image tokenizer.

    Expected setup:
    - clone https://github.com/FoundationVision/LlamaGen
    - download https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt
    """

    def __init__(
        self,
        repo_root: str | Path,
        checkpoint_path: str | Path,
        model_name: str = "VQ-16",
        image_size: int = 512,
        codebook_size: int = 16384,
        codebook_embed_dim: int = 8,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.image_size = image_size
        self.device = torch.device(device)
        self.dtype = dtype

        if not self.repo_root.exists():
            raise FileNotFoundError(f"LlamaGen repo not found: {self.repo_root}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"LlamaGen checkpoint not found: {self.checkpoint_path}")

        repo_root_str = str(self.repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        vq_module = importlib.import_module("tokenizer.tokenizer_image.vq_model")
        vq_models = getattr(vq_module, "VQ_models")
        model_factory = vq_models[model_name]
        self.model = model_factory(
            codebook_size=codebook_size,
            codebook_embed_dim=codebook_embed_dim,
        ).to(device=self.device, dtype=self.dtype)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.patch_size = 16 if model_name == "VQ-16" else 8
        if image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by tokenizer patch size")
        self.grid_size = image_size // self.patch_size
        self.seq_len = self.grid_size * self.grid_size
        self.vocab_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim

    def encode(self, images: torch.Tensor) -> TokenizedImageBatch:
        images = images.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            _, _, info = self.model.encode(images)
        indices = info[2].reshape(images.shape[0], -1).long()
        return TokenizedImageBatch(
            token_ids=indices,
            image_size=self.image_size,
            patch_size=self.patch_size,
            vocab_size=self.vocab_size,
        )

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(device=self.device)
        shape = (
            token_ids.size(0),
            self.codebook_embed_dim,
            self.grid_size,
            self.grid_size,
        )
        with torch.no_grad():
            decoded = self.model.decode_code(token_ids.reshape(-1), shape=shape)
        return decoded


class _TamingVQGANModel(nn.Module):
    def __init__(self, ddconfig: dict, n_embed: int, embed_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer2(n_embed, embed_dim, beta=0.25)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], kernel_size=1)
        self.embed_dim = embed_dim

    def encode_indices(self, images: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(images)  # [B, z_channels, H/16, W/16]
        latent = self.quant_conv(latent)  # [B, embed_dim, H/16, W/16]
        _, _, info = self.quantize(latent)
        return info[2].reshape(images.shape[0], -1).long()  # [B, L]

    def decode_indices(self, token_ids: torch.Tensor, grid_size: int) -> torch.Tensor:
        codebook = self.quantize.get_codebook_entry(
            token_ids.reshape(-1),
            shape=(token_ids.size(0), grid_size, grid_size, self.embed_dim),
        )  # [B, embed_dim, H/16, W/16]
        decoded = self.decoder(self.post_quant_conv(codebook))  # [B, 3, H, W]
        return decoded


class TamingVQGANTokenizer:
    """
    Wrapper around the official CompVis Taming Transformers VQGAN.

    Expected setup:
    - clone https://github.com/CompVis/taming-transformers
    - download the ImageNet f16-16384 checkpoint and config from the official
      Heidelberg model zoo links
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        image_size: int = 256,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.image_size = image_size
        self.device = torch.device(device)
        self.dtype = dtype

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Taming checkpoint not found: {self.checkpoint_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Taming config not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        model_params = config["model"]["params"]
        ddconfig = model_params["ddconfig"]
        n_embed = int(model_params["n_embed"])
        embed_dim = int(model_params["embed_dim"])

        self.model = _TamingVQGANModel(
            ddconfig=ddconfig,
            n_embed=n_embed,
            embed_dim=embed_dim,
        ).to(device=self.device, dtype=self.dtype)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        invalid_unexpected = [key for key in unexpected if not key.startswith("loss.")]
        if invalid_unexpected:
            raise RuntimeError(f"Unexpected Taming checkpoint keys: {invalid_unexpected[:8]}")
        allowed_missing_prefixes = ("loss.",)
        invalid_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
        if invalid_missing:
            raise RuntimeError(f"Missing required Taming checkpoint keys: {invalid_missing[:8]}")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        num_downsamples = len(ddconfig["ch_mult"]) - 1
        self.patch_size = 2 ** num_downsamples  # f16 => 16
        if image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by tokenizer patch size")
        self.grid_size = image_size // self.patch_size
        self.seq_len = self.grid_size * self.grid_size
        self.vocab_size = n_embed
        self.codebook_embed_dim = embed_dim

    def encode(self, images: torch.Tensor) -> TokenizedImageBatch:
        images = images.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            indices = self.model.encode_indices(images)
        return TokenizedImageBatch(
            token_ids=indices,
            image_size=self.image_size,
            patch_size=self.patch_size,
            vocab_size=self.vocab_size,
        )

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(device=self.device)
        with torch.no_grad():
            decoded = self.model.decode_indices(token_ids, grid_size=self.grid_size)
        return decoded
