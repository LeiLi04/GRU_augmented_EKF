"""Consistency metrics: NIS and NEES."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from scipy.stats import chi2

from utils.linear_algebra import safe_cholesky, chol_solve


def _chi2_band(dim: int, alpha: float) -> Tuple[float, float]:
    lower = chi2.ppf(alpha / 2.0, dim)
    upper = chi2.ppf(1 - alpha / 2.0, dim)
    return float(lower), float(upper)


def compute_nis(innov: torch.Tensor, S: torch.Tensor, mask: Optional[torch.Tensor] = None, alpha: float = 0.05) -> Tuple[torch.Tensor, float]:
    chol_S, _ = safe_cholesky(S)
    S_inv_delta = chol_solve(chol_S, innov.unsqueeze(-1)).squeeze(-1)
    nis_vals = (innov * S_inv_delta).sum(dim=-1)
    if mask is not None:
        nis_vals = nis_vals.masked_fill(mask, 0.0)
    lower, upper = _chi2_band(innov.size(-1), alpha)
    valid = nis_vals[~torch.isnan(nis_vals)]
    in_band = (valid >= lower) & (valid <= upper)
    in_band_rate = (in_band.float().mean().item() * 100.0) if valid.numel() > 0 else 0.0
    return nis_vals, in_band_rate


def compute_nees(error: torch.Tensor, covariance: torch.Tensor, mask: Optional[torch.Tensor] = None, alpha: float = 0.05) -> Tuple[torch.Tensor, float]:
    chol_P, _ = safe_cholesky(covariance)
    P_inv_error = chol_solve(chol_P, error.unsqueeze(-1)).squeeze(-1)
    nees_vals = (error * P_inv_error).sum(dim=-1)
    if mask is not None:
        nees_vals = nees_vals.masked_fill(mask, 0.0)
    lower, upper = _chi2_band(error.size(-1), alpha)
    valid = nees_vals[~torch.isnan(nees_vals)]
    in_band = (valid >= lower) & (valid <= upper)
    in_band_rate = (in_band.float().mean().item() * 100.0) if valid.numel() > 0 else 0.0
    return nees_vals, in_band_rate


__all__ = ["compute_nis", "compute_nees"]
