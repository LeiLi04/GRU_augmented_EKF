"""Innovation negative log-likelihood."""
from __future__ import annotations

from typing import Optional

import torch

from ..utils.linear_algebra import chol_logdet, chol_solve, safe_cholesky
from ..utils.masking import masked_mean


def innovation_nll(
    delta_y: torch.Tensor,
    S: torch.Tensor,
    logdet_S: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    jitter: float = 1e-6,
) -> torch.Tensor:
    chol_S, _ = safe_cholesky(S, jitter=jitter)
    S_inv_delta = chol_solve(chol_S, delta_y.unsqueeze(-1)).squeeze(-1)
    maha = (delta_y * S_inv_delta).sum(dim=-1)
    maha = torch.nan_to_num(maha, nan=0.0, posinf=1e6, neginf=-1e6)
    if logdet_S is None:
        logdet_S = chol_logdet(chol_S)
    logdet_S = torch.nan_to_num(logdet_S, nan=0.0, posinf=1e6, neginf=-1e6)
    terms = maha + logdet_S
    terms = torch.nan_to_num(terms, nan=0.0, posinf=1e6, neginf=-1e6)
    return masked_mean(terms.unsqueeze(-1), mask).mean()


__all__ = ["innovation_nll"]
