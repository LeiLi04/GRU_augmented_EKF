"""Normalized mean squared error."""
from __future__ import annotations

from typing import Optional

import torch

from ..utils.masking import masked_mean


def nmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute NMSE per batch (lower is better)."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    mse = (pred - target) ** 2
    mse = masked_mean(mse, mask)
    power = masked_mean(target ** 2, mask).clamp_min(1e-12)
    return (mse / power).mean()


__all__ = ["nmse"]
