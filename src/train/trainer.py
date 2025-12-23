"""NLL-only trainer for the GRU-augmented EKF."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from ..config import ExperimentConfig
from ..losses.innov_nll import innovation_nll
from ..models.ekf import DifferentiableEKF, EKFConfig
from ..models.measurement import LinearMeasurement, LinearMeasurementConfig
from ..models.psd import PSDConfig, PSDParameter
from ..models.residual_gru import DynamicsConfig, ResidualDynamics, build_residual_dynamics
from ..metrics.nis_nees import compute_nis
from ..utils.linear_algebra import safe_cholesky
from ..utils.masking import masked_mean
from ..data.lorenz import parse_noise_from_name
from .warmstart_q0 import set_psd_parameter_from_matrix
from ..utils.lorenz import lorenz_discrete_step, lorenz_discrete_jacobian


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0


def build_model_components(
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    train_path: Optional[Path] = None,
    obs_variance: Optional[torch.Tensor] = None,
) -> Tuple[ResidualDynamics, DifferentiableEKF, float, float]:
    """Create dynamics, EKF, and initialise Q/R."""
    q_from_name = parse_noise_from_name(train_path, "q2") if train_path else None
    r_from_name = parse_noise_from_name(train_path, "r2") if train_path else None
    q_diag_value = float(q_from_name if q_from_name is not None else cfg.model.q_init)
    r_diag_value = float(r_from_name if r_from_name is not None else cfg.model.r_init)

    dyn_cfg = DynamicsConfig(
        state_dim=cfg.model.state_dim,
        input_dim=cfg.model.state_dim + cfg.model.obs_dim,
        hidden_dim=cfg.model.dynamics_hidden,
        depth=cfg.model.dynamics_depth,
        use_gru=cfg.model.dynamics_use_gru,
        dt=cfg.model.dt,
        tanh_scale=cfg.model.dynamics_tanh_scale,
        residual_init_std=cfg.model.dynamics_residual_init_std,
        max_delta=cfg.model.max_delta,
        scale_a_min=cfg.model.dynamics_scale_a_min,
        scale_a_max=cfg.model.dynamics_scale_a_max,
    )

    def f_known(x: torch.Tensor) -> torch.Tensor:
        return lorenz_discrete_step(x, cfg.model.dt)

    dynamics = build_residual_dynamics(dyn_cfg, f_known=f_known, phys_derivative=None).to(device)
    dynamics.f_known_jacobian = lambda x: lorenz_discrete_jacobian(x, cfg.model.dt)  # type: ignore[attr-defined]

    measurement = LinearMeasurement(LinearMeasurementConfig(cfg.model.state_dim, cfg.model.obs_dim)).to(device)

    q_param = PSDParameter(PSDConfig(cfg.model.state_dim, init_scale=float(q_diag_value) ** 0.5)).to(device)
    r_param = PSDParameter(PSDConfig(cfg.model.obs_dim, init_scale=float(r_diag_value) ** 0.5)).to(device)
    q_matrix = torch.eye(cfg.model.state_dim, device=device) * q_diag_value
    r_matrix = torch.eye(cfg.model.obs_dim, device=device) * r_diag_value
    set_psd_parameter_from_matrix(q_param, q_matrix, jitter=1e-6)
    set_psd_parameter_from_matrix(r_param, r_matrix, jitter=1e-5)
    for p in q_param.parameters():
        p.requires_grad_(cfg.model.train_q)
    for p in r_param.parameters():
        p.requires_grad_(cfg.model.train_r)

    ekf_cfg = EKFConfig(
        state_dim=cfg.model.state_dim,
        obs_dim=cfg.model.obs_dim,
        dt=cfg.model.dt,
    )
    ekf = DifferentiableEKF(ekf_cfg, dynamics, measurement, q_param, r_param).to(device)
    return dynamics, ekf, q_diag_value, r_diag_value


class Trainer:
    def __init__(
        self,
        dynamics: ResidualDynamics,
        ekf: DifferentiableEKF,
        optimizer: optim.Optimizer,
        config: ExperimentConfig,
        device: torch.device,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> None:
        self.dynamics = dynamics
        self.ekf = ekf
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.scaler = scaler if config.train.amp else None
        self.state = TrainerState()

    def _initial_state(self, batch_size: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        state_dim = self.config.model.state_dim
        x0 = torch.zeros(batch_size, state_dim, device=self.device, dtype=dtype)
        sigma0_scale = float(self.config.train.sigma0_scale)
        Sigma0 = (
            torch.eye(state_dim, device=self.device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            * sigma0_scale
        )
        return x0, Sigma0

    def train_epoch(self, dataloader, writer: Optional[SummaryWriter] = None) -> dict[str, float]:
        self.ekf.train()
        self.dynamics.train()
        metrics = {"L_innov": 0.0, "nis_mean": 0.0}
        updates = 0

        iterator = iter(dataloader)
        for batch in iterator:
            obs, mask = batch
            obs = obs.to(self.device)
            mask = mask.to(self.device)
            batch_size = obs.size(0)
            total_steps = obs.size(1)
            chunk_size = int(self.config.train.tbptt_steps) if self.config.train.tbptt_steps else total_steps
            chunk_size = max(chunk_size, 1)

            x0, Sigma0 = self._initial_state(batch_size, obs.dtype)
            hidden = (
                self.dynamics.reset_hidden(batch_size, device=self.device, dtype=obs.dtype)
                if getattr(self.dynamics, "use_gru", False)
                else None
            )

            start_t = 0
            while start_t < total_steps:
                end_t = min(start_t + chunk_size, total_steps)
                obs_chunk = obs[:, start_t:end_t, :]
                mask_chunk = mask[:, start_t:end_t] if mask is not None else None
                outputs = self.ekf(obs_chunk, x0, Sigma0, mask=mask_chunk, hidden=hidden)
                delta_y = outputs["innovations"]
                S = outputs["S"]
                logdet_S = outputs["logdet_S"]
                whitened = outputs["whitened"]

                loss = innovation_nll(delta_y, S, logdet_S, mask_chunk)
                nis_vals, _ = compute_nis(delta_y, S, mask_chunk)
                nis_mean = masked_mean(nis_vals.unsqueeze(-1), mask_chunk).mean()

                self.optimizer.zero_grad()
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        list(self.dynamics.parameters())
                        + list(self.ekf.q_param.parameters())
                        + list(self.ekf.r_param.parameters()),
                        self.config.train.clip_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.dynamics.parameters())
                        + list(self.ekf.q_param.parameters())
                        + list(self.ekf.r_param.parameters()),
                        self.config.train.clip_norm,
                    )
                    self.optimizer.step()

                metrics["L_innov"] += loss.item()
                metrics["nis_mean"] += nis_mean.item()
                updates += 1
                self.state.global_step += 1

                x0 = outputs["x_filt"][:, -1].detach()
                Sigma0 = outputs["Sigma_filt"][:, -1].detach()
                hidden = outputs.get("hidden_last")
                if hidden is not None:
                    hidden = hidden.detach()
                start_t = end_t

        denom = max(updates, 1)
        metrics = {k: v / denom for k, v in metrics.items()}
        if writer is not None:
            for key, value in metrics.items():
                writer.add_scalar(f"train/{key}", value, self.state.epoch)
        self.state.epoch += 1
        return metrics


def evaluate_nll(
    ekf: DifferentiableEKF,
    dynamics: ResidualDynamics,
    dataloader,
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    ekf.eval()
    dynamics.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for obs, mask in dataloader:
            obs = obs.to(device)
            mask = mask.to(device)
            batch_size = obs.size(0)
            x0 = torch.zeros(batch_size, config.model.state_dim, device=device)
            Sigma0 = (
                torch.eye(config.model.state_dim, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                * float(config.train.sigma0_scale)
            )
            outputs = ekf(obs, x0, Sigma0, mask=mask)
            val_loss = innovation_nll(outputs["innovations"], outputs["S"], outputs.get("logdet_S"), mask)
            total += val_loss.item() * batch_size
            count += batch_size
    ekf.train()
    dynamics.train()
    return total / max(count, 1)


__all__ = ["Trainer", "TrainerState", "build_model_components", "evaluate_nll"]
