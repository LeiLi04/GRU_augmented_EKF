"""Covariance warm-start utilities."""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from ..models.ekf import DifferentiableEKF
from ..models.psd import PSDParameter, ScalarPSDParameter
from ..utils.linear_algebra import safe_cholesky


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


@torch.no_grad()
def _estimate_q(ekf: DifferentiableEKF, dataloader: DataLoader, device: torch.device, eps: float) -> torch.Tensor:
    state_dim = ekf.state_dim
    obs_dim = ekf.obs_dim

    sum_dy_outer = torch.zeros(obs_dim, obs_dim, device=device)
    count_dy = torch.zeros(1, device=device)

    sum_prop = torch.zeros(state_dim, state_dim, device=device)
    count_prop = torch.zeros(1, device=device)

    R = ekf.r_param.matrix().to(device)

    for obs, mask in dataloader:
        obs = obs.to(device)
        mask_tensor = _prepare_mask(mask, obs.shape[:2], device)
        x0, Sigma0 = _initial_state(ekf.state_dim, obs.size(0), device)
        outputs = ekf(obs, x0, Sigma0, mask=mask_tensor, return_jacobians=True)

        delta = outputs["innovations"]
        valid = (~mask_tensor).unsqueeze(-1).unsqueeze(-1)
        dy_outer = delta.unsqueeze(-1) * delta.unsqueeze(-2)
        sum_dy_outer += (dy_outer * valid).sum(dim=(0, 1))
        count_dy += valid.sum()

        if outputs["Sigma_filt"].size(1) < 2:
            continue
        sigma_filt_prev = outputs["Sigma_filt"][:, :-1]
        F_prev = outputs["F"][:, 1:]
        valid_curr = (~mask_tensor[:, 1:]).unsqueeze(-1).unsqueeze(-1)
        valid_prev = (~mask_tensor[:, :-1]).unsqueeze(-1).unsqueeze(-1)
        transition_mask = valid_curr * valid_prev
        propagated = torch.matmul(F_prev, torch.matmul(sigma_filt_prev, F_prev.transpose(-1, -2)))
        sum_prop += (propagated * transition_mask).sum(dim=(0, 1))
        count_prop += transition_mask.sum()

    if count_dy.item() <= 0 or count_prop.item() <= 0:
        return torch.eye(state_dim, device=device)

    S_hat = sum_dy_outer / count_dy
    Sigma_hat = project_to_psd(S_hat - R, eps)
    mean_prop = sum_prop / count_prop
    Q_hat = project_to_psd(Sigma_hat - mean_prop, eps)
    q0 = torch.clamp(torch.trace(Q_hat) / state_dim, min=eps)
    return torch.eye(state_dim, device=device) * q0


@torch.no_grad()
def covariance_matching_warm_start(
    ekf: DifferentiableEKF,
    dataloader: DataLoader,
    device: torch.device,
    eps: float = 1e-6,
) -> float:
    """Estimate an isotropic Q using a single warm-start pass."""
    ekf.eval()
    ekf.dynamics.eval()
    q_matrix = _estimate_q(ekf, dataloader, device, eps)
    q_matrix = set_psd_parameter_from_matrix(ekf.q_param, q_matrix)
    tau_q = torch.logdet(q_matrix + eps * torch.eye(q_matrix.size(0), device=device))
    return float(tau_q.item())


__all__ = ["covariance_matching_warm_start", "set_psd_parameter_from_matrix", "project_to_psd"]
