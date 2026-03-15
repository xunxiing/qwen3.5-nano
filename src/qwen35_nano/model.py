from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer


@dataclass
class SpecialTokenIds:
    image_bos_id: int
    image_eos_id: int
    image_pad_id: int


def build_tokenizer(
    model_name: str,
    trust_remote_code: bool,
    image_vocab_size: int,
) -> tuple[Any, SpecialTokenIds]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[IMG_BOS]", "[IMG_EOS]", "[IMG_PAD]"]}
    )
    ids = SpecialTokenIds(
        image_bos_id=image_vocab_size,
        image_eos_id=image_vocab_size + 1,
        image_pad_id=image_vocab_size + 2,
    )
    return tokenizer, ids


class FrozenQwenTextEncoder(nn.Module):
    """Uses Qwen as a frozen text feature extractor to avoid 9B training activations."""

    def __init__(self, model_name: str, trust_remote_code: bool) -> None:
        super().__init__()
        loaded_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            dtype=torch.bfloat16,
        )
        self.model = getattr(loaded_model, "language_model", loaded_model)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @property
    def hidden_size(self) -> int:
        config = self.model.config
        if getattr(config, "hidden_size", None) is not None:
            return config.hidden_size
        text_config = getattr(config, "text_config", None)
        if text_config is not None and getattr(text_config, "hidden_size", None) is not None:
            return text_config.hidden_size
        for name in ("d_model", "dim", "n_embd"):
            value = getattr(config, name, None)
            if value is not None:
                return value
        raise ValueError(f"Could not infer hidden size from config type: {type(config).__name__}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                cache_position=cache_position,
                return_dict=True,
            )
        return outputs.last_hidden_state


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self._gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attn_mask: torch.Tensor,
        encoder_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_out, _ = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=self_attn_mask,
            need_weights=False,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        attn_out, _ = self.cross_attn(
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            key_padding_mask=encoder_key_padding_mask,
            need_weights=False,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class VisualARDecoder(nn.Module):
    def __init__(
        self,
        image_vocab_size: int,
        image_seq_len: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        encoder_hidden_size: int,
    ) -> None:
        super().__init__()
        grid_size = math.isqrt(image_seq_len)
        if grid_size * grid_size != image_seq_len:
            raise ValueError("image_seq_len must be a perfect square for 2D positional embeddings")
        self.image_seq_len = image_seq_len
        self.grid_size = grid_size
        self.token_embed = nn.Embedding(image_vocab_size + 3, hidden_size)
        self.bos_pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.row_pos_embed = nn.Parameter(torch.zeros(1, grid_size, hidden_size))
        self.col_pos_embed = nn.Parameter(torch.zeros(1, grid_size, hidden_size))
        self.encoder_proj = nn.Linear(encoder_hidden_size, hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, image_vocab_size + 3, bias=False)
        nn.init.normal_(self.bos_pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.row_pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.col_pos_embed, mean=0.0, std=0.02)

    def _position_embeddings(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if seq_len < 1 or seq_len > self.image_seq_len + 1:
            raise ValueError(f"seq_len={seq_len} is out of range for decoder image_seq_len={self.image_seq_len}")
        bos = self.bos_pos_embed.to(device=device, dtype=dtype)
        if seq_len == 1:
            return bos
        token_count = seq_len - 1
        row_ids = torch.arange(token_count, device=device) // self.grid_size
        col_ids = torch.arange(token_count, device=device) % self.grid_size
        image_pos = self.row_pos_embed[:, row_ids, :] + self.col_pos_embed[:, col_ids, :]
        image_pos = image_pos.to(device=device, dtype=dtype)
        return torch.cat([bos, image_pos], dim=1)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.token_embed(decoder_input_ids)  # [B, 1 + L_img, H]
        hidden_states = hidden_states + self._position_embeddings(
            decoder_input_ids.size(1),
            device=decoder_input_ids.device,
            dtype=hidden_states.dtype,
        )
        encoder_hidden_states = encoder_hidden_states.to(self.encoder_proj.weight.dtype)
        encoder_hidden_states = self.encoder_proj(encoder_hidden_states)  # [B, T_txt, H]

        seq_len = decoder_input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=decoder_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        encoder_key_padding_mask = encoder_attention_mask == 0

        for layer in self.layers:
            if self.training and getattr(layer, "_gradient_checkpointing", False):
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    encoder_hidden_states,
                    causal_mask,
                    encoder_key_padding_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    self_attn_mask=causal_mask,
                    encoder_key_padding_mask=encoder_key_padding_mask,
                )

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)  # [B, 1 + L_img, V_img]


class QwenTextToImageModel(nn.Module):
    def __init__(
        self,
        text_encoder: FrozenQwenTextEncoder,
        decoder: VisualARDecoder,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.decoder = decoder

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_hidden_states = self.text_encoder(text_input_ids, text_attention_mask)
        logits = self.decoder(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=text_attention_mask,
        )
        output = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        return output
