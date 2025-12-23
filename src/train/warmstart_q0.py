"""Covariance warm-start utilities (innovation NLL grid search)."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch.utils.data import DataLoader

from models.gru_augmented_ekf.ekf import DifferentiableEKF
from models.gru_augmented_ekf.psd import PSDParameter, ScalarPSDParameter
from utils.linear_algebra import safe_cholesky


def _softplus_inverse(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y = torch.clamp(y, min=eps)
    return y + torch.log(-torch.expm1(-y))


def project_to_psd(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sym = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(sym)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)


def set_psd_parameter_from_matrix(param: torch.nn.Module, matrix: torch.Tensor, jitter: float = 1e-5) -> torch.Tensor:
    if isinstance(param, ScalarPSDParameter):
        param.set_base(matrix, jitter=jitter)
        return param.matrix()
    if not isinstance(param, PSDParameter):
        raise TypeError(f"Unsupported parameter type: {type(param)}")
    device = param.raw.device
    dtype = param.raw.dtype
    target = matrix.to(device=device, dtype=dtype)
    target = 0.5 * (target + target.transpose(-1, -2))
    dim = target.size(0)
    eye = torch.eye(dim, device=device, dtype=dtype)
    chol = torch.linalg.cholesky(target + jitter * eye)
    tril_rows = param.tril_rows
    tril_cols = param.tril_cols
    raw = param.raw.data
    raw.zero_()
    raw.copy_(chol[tril_rows, tril_cols])
    diag_mask = tril_rows == tril_cols
    diag_vals = torch.diagonal(chol)
    diag_target = torch.clamp(diag_vals - param.config.min_diag, min=1e-8)
    raw[diag_mask] = _softplus_inverse(diag_target)
    return chol @ chol.transpose(-1, -2)


def _initial_state(dim: int, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x0 = torch.zeros(batch_size, dim, device=device)
    Sigma0 = torch.eye(dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    return x0, Sigma0


def _prepare_mask(mask: torch.Tensor | None, shape: torch.Size, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    return mask.to(device=device)


def _set_isotropic_q(ekf: DifferentiableEKF, q0: float, device: torch.device, jitter: float = 1e-6) -> torch.Tensor:
    """Set ekf.q_param to an isotropic Q = q0^2 I."""
    q_matrix = torch.eye(ekf.state_dim, device=device) * (q0**2)
    return set_psd_parameter_from_matrix(ekf.q_param, q_matrix, jitter=jitter)


@torch.no_grad()
def _innovation_nll_for_q(
    ekf: DifferentiableEKF,
    dataloader: DataLoader,
    device: torch.device,
    q0: float,
    eps: float,
) -> float:
    """Compute mean innovation NLL for a candidate q0."""
    _set_isotropic_q(ekf, q0, device=device, jitter=eps)
    ekf.eval()
    ekf.dynamics.eval()

    total_nll = torch.zeros((), device=device)
    total_count = torch.zeros((), device=device)

    for obs, mask in dataloader:
        obs = obs.to(device)
        mask_tensor = _prepare_mask(mask, obs.shape[:2], device)
        x0, Sigma0 = _initial_state(ekf.state_dim, obs.size(0), device)
        outputs = ekf(obs, x0, Sigma0, mask=mask_tensor)

        logdet_S = outputs["logdet_S"]  # (B, T)
        whitened = outputs["whitened"]  # (B, T, obs_dim)
        step_nll = whitened.pow(2).sum(dim=-1) + logdet_S

        valid = ~mask_tensor
        total_nll += (step_nll * valid).sum()
        total_count += valid.sum()

    if total_count.item() == 0:
        return float("inf")
    return float((total_nll / total_count).item())


@torch.no_grad()
def covariance_matching_warm_start(
    ekf: DifferentiableEKF,
    dataloader: DataLoader,
    device: torch.device,
    eps: float = 1e-6,
) -> float:
    """Estimate an isotropic Q = q0^2 I via innovation NLL grid search."""
    # Short log-spaced grid around typical small process noise scales.
    grid: Iterable[float] = torch.logspace(-3, 0, steps=7).tolist()
    best_q0, best_nll = None, float("inf")

    for q0 in grid:
        nll = _innovation_nll_for_q(ekf, dataloader, device, float(q0), eps)
        if nll < best_nll:
            best_nll = nll
            best_q0 = float(q0)

    if best_q0 is None:
        return 0.0

    q_matrix = _set_isotropic_q(ekf, best_q0, device=device, jitter=eps)
    # Return the selected q0 (std) for logging.
    return float(best_q0)


__all__ = ["covariance_matching_warm_start", "set_psd_parameter_from_matrix", "project_to_psd"]
