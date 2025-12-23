"""Lorenz-63 discrete dynamics utilities."""
from __future__ import annotations

import torch


def lorenz_rhs(
    x: torch.Tensor, *, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0
) -> torch.Tensor:
    dx = torch.zeros_like(x)
    dx[..., 0] = sigma * (x[..., 1] - x[..., 0])
    dx[..., 1] = x[..., 0] * (rho - x[..., 2]) - x[..., 1]
    dx[..., 2] = x[..., 0] * x[..., 1] - beta * x[..., 2]
    return dx


def lorenz_discrete_step(
    x: torch.Tensor, dt: float, *, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0
) -> torch.Tensor:
    return x + dt * lorenz_rhs(x, sigma=sigma, rho=rho, beta=beta)


def lorenz_discrete_jacobian(
    x: torch.Tensor, dt: float, *, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0
) -> torch.Tensor:
    J = torch.zeros((*x.shape[:-1], 3, 3), device=x.device, dtype=x.dtype)
    J[..., 0, 0] = -sigma
    J[..., 0, 1] = sigma
    J[..., 1, 0] = rho - x[..., 2]
    J[..., 1, 1] = -1.0
    J[..., 1, 2] = -x[..., 0]
    J[..., 2, 0] = x[..., 1]
    J[..., 2, 2] = -beta
    eye = torch.eye(3, device=x.device, dtype=x.dtype).expand_as(J)
    return eye + dt * J


__all__ = ["lorenz_rhs", "lorenz_discrete_step", "lorenz_discrete_jacobian"]
