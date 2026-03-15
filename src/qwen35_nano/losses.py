from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def build_token_frequency_weights(
    manifest_path: str | Path,
    vocab_size: int,
    *,
    extra_token_count: int = 3,
    alpha: float = 0.5,
    min_weight: float = 0.25,
    max_weight: float = 6.0,
) -> torch.Tensor:
    manifest = Path(manifest_path)
    counts = Counter()
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts.update(row["image_tokens"])

    token_counts = torch.ones(vocab_size, dtype=torch.float32)
    for token_id, count in counts.items():
        if 0 <= token_id < vocab_size:
            token_counts[token_id] += float(count)

    frequencies = token_counts / token_counts.sum()
    reference = frequencies.mean()
    weights = (reference / frequencies).pow(alpha)
    weights = weights.clamp(min=min_weight, max=max_weight)
    weights = weights / weights.mean()
    weights = weights.clamp(min=min_weight, max=max_weight)

    full_weights = torch.ones(vocab_size + extra_token_count, dtype=torch.float32)
    full_weights[:vocab_size] = weights
    return full_weights


def build_color_token_weights(
    stats_path: str | Path,
    vocab_size: int,
    *,
    extra_token_count: int = 3,
    color_reward_strength: float = 1.0,
    blank_penalty_strength: float = 0.75,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
) -> torch.Tensor:
    stats = np.load(Path(stats_path))
    saturation = torch.from_numpy(stats["saturation"]).float()[:vocab_size]
    blankness = torch.from_numpy(stats["blankness"]).float()[:vocab_size]

    sat_norm = (saturation - saturation.min()) / (saturation.max() - saturation.min() + 1e-6)
    blank_norm = (blankness - blankness.min()) / (blankness.max() - blankness.min() + 1e-6)
    weights = 1.0 + color_reward_strength * sat_norm - blank_penalty_strength * blank_norm
    weights = weights.clamp(min=min_weight, max=max_weight)

    full_weights = torch.ones(vocab_size + extra_token_count, dtype=torch.float32)
    full_weights[:vocab_size] = weights
    return full_weights


def load_blank_token_ids(
    stats_path: str | Path,
    vocab_size: int,
    *,
    top_k: int = 64,
) -> torch.Tensor:
    stats = np.load(Path(stats_path))
    blankness = torch.from_numpy(stats["blankness"]).float()[:vocab_size]
    top_k = min(top_k, vocab_size)
    return torch.topk(blankness, k=top_k).indices.long()


def focal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    gamma: float = 0.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    valid_mask = flat_labels != -100
    valid_logits = flat_logits[valid_mask]
    valid_labels = flat_labels[valid_mask]
    if valid_labels.numel() == 0:
        return logits.new_zeros(())

    ce = F.cross_entropy(
        valid_logits,
        valid_labels,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    if gamma > 0:
        probs = torch.softmax(valid_logits.float(), dim=-1)
        pt = probs.gather(dim=-1, index=valid_labels.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)
        ce = ce * (1.0 - pt).pow(gamma)
    return ce.mean()


def token_distribution_kl(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    valid_mask = flat_labels != -100
    valid_logits = flat_logits[valid_mask]
    valid_labels = flat_labels[valid_mask]
    if valid_labels.numel() == 0:
        return logits.new_zeros(())

    predicted_dist = torch.softmax(valid_logits.float(), dim=-1).mean(dim=0)
    target_hist = torch.bincount(valid_labels, minlength=logits.size(-1)).float()
    target_dist = target_hist / target_hist.sum().clamp_min(1.0)

    predicted_dist = predicted_dist[: vocab_size + 3].clamp_min(1e-8)
    target_dist = target_dist[: vocab_size + 3].clamp_min(1e-8)
    return F.kl_div(predicted_dist.log(), target_dist, reduction="sum")


def repeat_unlikelihood_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    decoder_input_ids: torch.Tensor,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    flat_previous = decoder_input_ids.reshape(-1)
    valid_mask = flat_labels != -100
    valid_logits = flat_logits[valid_mask]
    valid_labels = flat_labels[valid_mask]
    valid_previous = flat_previous[valid_mask]

    penalty_mask = valid_previous != valid_labels
    if penalty_mask.sum().item() == 0:
        return logits.new_zeros(())

    penalty_logits = valid_logits[penalty_mask]
    penalty_previous = valid_previous[penalty_mask]
    probs = torch.softmax(penalty_logits.float(), dim=-1)
    repeat_probs = probs.gather(dim=-1, index=penalty_previous.unsqueeze(-1)).squeeze(-1)
    return -torch.log1p(-repeat_probs.clamp(max=1 - 1e-6)).mean()


def blank_token_suppression_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    blank_token_ids: torch.Tensor,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    valid_mask = flat_labels != -100
    valid_logits = flat_logits[valid_mask]
    valid_labels = flat_labels[valid_mask]
    if valid_labels.numel() == 0:
        return logits.new_zeros(())

    blank_token_ids = blank_token_ids.to(device=valid_logits.device)
    probs = torch.softmax(valid_logits.float(), dim=-1)
    blank_mass = probs[:, blank_token_ids].sum(dim=-1)
    target_is_blank = torch.isin(valid_labels, blank_token_ids)
    suppress_mask = ~target_is_blank
    if suppress_mask.sum().item() == 0:
        return logits.new_zeros(())
    return blank_mass[suppress_mask].mean()
