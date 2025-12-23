"""Linear algebra helpers for EKF computations."""
from __future__ import annotations

from typing import Tuple

import torch


def safe_cholesky(matrix: torch.Tensor, jitter: float = 1e-6, max_tries: int = 5) -> Tuple[torch.Tensor, float]:
    """Stable Cholesky with automatic jitter escalation."""
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)

    chol = None
    used_jitter = jitter
    dim = matrix.size(-1)
    identity = torch.eye(dim, device=matrix.device, dtype=matrix.dtype)
    expand_shape = (1,) * (matrix.dim() - 2) + (dim, dim)
    identity_expanded = identity.view(expand_shape).expand(matrix.shape[:-2] + (dim, dim))

    for _ in range(max_tries):
        try:
            chol = torch.linalg.cholesky(matrix + used_jitter * identity_expanded)
            break
        except RuntimeError:
            used_jitter *= 10

    if chol is None:
        sym_matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
        eps_eye = used_jitter * identity_expanded
        chol = torch.linalg.cholesky(sym_matrix + eps_eye)

    if torch.isnan(chol).any():
        chol = torch.sqrt(torch.tensor(used_jitter, device=matrix.device, dtype=matrix.dtype)) * identity_expanded
    return chol, used_jitter


def chol_solve(chol: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve (L L^T) x = rhs given lower-triangular chol."""
    if chol.dim() > 2:
        n = chol.size(-1)
        rhs_cols = rhs.size(-1)
        chol_flat = chol.reshape(-1, n, n)
        rhs_flat = rhs.reshape(-1, n, rhs_cols)
        sol_flat = torch.cholesky_solve(rhs_flat, chol_flat)
        return sol_flat.reshape(*rhs.shape)
    return torch.cholesky_solve(rhs, chol)


def chol_logdet(chol: torch.Tensor) -> torch.Tensor:
    """Compute log|A| from its Cholesky factor L (A = L L^T)."""
    diag = torch.diagonal(chol, dim1=-2, dim2=-1)
    return 2.0 * torch.log(diag).sum(dim=-1)


__all__ = ["safe_cholesky", "chol_solve", "chol_logdet"]
