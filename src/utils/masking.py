"""Masking helpers for variable-length sequences."""
from __future__ import annotations

from typing import Optional

import torch


def lengths_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = int(max_len or lengths.max().item())
    range_tensor = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    return range_tensor >= lengths.unsqueeze(1)


def apply_mask(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return tensor
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    return tensor.masked_fill(mask, 0.0)


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=1)
    valid = (~mask).float()
    while valid.dim() < tensor.dim():
        valid = valid.unsqueeze(-1)
    numerator = (tensor * valid).sum(dim=1)
    denominator = valid.sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def masked_sum(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return tensor.sum(dim=1)
    valid = (~mask).float()
    while valid.dim() < tensor.dim():
        valid = valid.unsqueeze(-1)
    return (tensor * valid).sum(dim=1)


__all__ = ["lengths_to_mask", "apply_mask", "masked_mean", "masked_sum"]
