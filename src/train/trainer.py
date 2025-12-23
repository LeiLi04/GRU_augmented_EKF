"""NLL-only trainer for the GRU-augmented EKF."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from config import ExperimentConfig
from losses.innov_nll import innovation_nll
from models.gru_augmented_ekf.ekf import DifferentiableEKF, EKFConfig
from models.gru_augmented_ekf.measurement import LinearMeasurement, LinearMeasurementConfig, RangeMeasurement
from models.gru_augmented_ekf.psd import PSDConfig, PSDParameter
from models.gru_augmented_ekf.residual_gru import DynamicsConfig, ResidualDynamics, build_residual_dynamics
from metrics.nis_nees import compute_nis
from utils.linear_algebra import safe_cholesky
from utils.masking import masked_mean
from train.warmstart_q0 import set_psd_parameter_from_matrix


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
    """Create dynamics, EKF, and initialise Q/R for CV + range measurements."""
    q_diag_value = float(cfg.model.q_init)
    r_diag_value = float(cfg.model.r_init)

    dyn_cfg = DynamicsConfig(
        state_dim=cfg.model.state_dim,
        input_dim=cfg.model.state_dim + cfg.model.obs_dim,
        hidden_dim=cfg.model.dynamics_hidden,
        depth=cfg.model.dynamics_depth,
        cov_rank=cfg.model.cov_rank,
        cov_factor_scale=cfg.model.cov_factor_scale,
        use_gru=cfg.model.dynamics_use_gru,
        dt=cfg.model.dt,
        tanh_scale=cfg.model.dynamics_tanh_scale,
        residual_init_std=cfg.model.dynamics_residual_init_std,
        max_delta=cfg.model.max_delta,
        scale_a_min=cfg.model.dynamics_scale_a_min,
        scale_a_max=cfg.model.dynamics_scale_a_max,
    )

    def f_known(x: torch.Tensor) -> torch.Tensor:
        # Constant-velocity step: [px + vx*dt, py + vy*dt, vx, vy]
        dt = cfg.model.dt
        px, py, vx, vy = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        out = torch.stack([px + vx * dt, py + vy * dt, vx, vy], dim=-1)
        return out

    def f_known_jac(_: torch.Tensor) -> torch.Tensor:
        dt = cfg.model.dt
        F = torch.tensor(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        return F.expand(1, -1, -1)

    dynamics = build_residual_dynamics(dyn_cfg, f_known=f_known, phys_derivative=None).to(device)
    dynamics.f_known_jacobian = lambda x: f_known_jac(x)  # type: ignore[attr-defined]

    Q_base = None
    if train_path:
        payload = np.load(train_path, allow_pickle=True)
        anchors_np = payload["anchors"]
        Q_base = payload.get("Q", None)
        anchors = torch.as_tensor(anchors_np, dtype=torch.float32, device=device)
    else:
        anchors = torch.tensor([[-1.5, 1.0, -1.0, 1.5], [0.5, 1.0, -1.0, -0.5]], device=device, dtype=torch.float32)

    measurement = RangeMeasurement(anchors).to(device)

    if Q_base is not None:
        Q_base_t = torch.as_tensor(Q_base, dtype=torch.float32, device=device)
        q_init_scale = float(torch.mean(torch.diagonal(Q_base_t))) ** 0.5
    else:
        q_init_scale = float(q_diag_value) ** 0.5

    q_param = PSDParameter(PSDConfig(cfg.model.state_dim, init_scale=q_init_scale)).to(device)
    r_param = PSDParameter(PSDConfig(cfg.model.obs_dim, init_scale=float(r_diag_value) ** 0.5)).to(device)
    if Q_base is not None:
        q_matrix = Q_base_t
    else:
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

    def save(self, path: Path) -> None:
        """Save dynamics and covariance parameters to a checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dynamics": self.dynamics.state_dict(),
                "q_param": self.ekf.q_param.state_dict(),
                "r_param": self.ekf.r_param.state_dict(),
                "state": self.state,
            },
            path,
        )

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
        metrics = {"L_innov": 0.0, "nis_mean": 0.0, "reg_delta": 0.0, "reg_L": 0.0}
        updates = 0
        lambda_nis = float(self.config.train.lambda_nis)
        lambda_delta = float(self.config.train.lambda_delta)
        lambda_L = float(self.config.train.lambda_L)
        nis_target = float(self.config.model.obs_dim)

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
                if lambda_nis > 0.0:
                    nis_penalty = (nis_mean - nis_target) ** 2
                    loss = loss + lambda_nis * nis_penalty
                if lambda_delta > 0.0 and "delta" in outputs:
                    delta = outputs["delta"]
                    if mask_chunk is not None:
                        valid = (~mask_chunk).unsqueeze(-1).to(delta.dtype)
                        denom = valid.sum().clamp(min=1.0)
                        delta_reg = (delta.pow(2) * valid).sum() / denom
                    else:
                        delta_reg = delta.pow(2).mean()
                    loss = loss + lambda_delta * delta_reg
                    metrics["reg_delta"] += delta_reg.item()
                if lambda_L > 0.0 and "u_t" in outputs:
                    u_t_seq = outputs["u_t"]
                    if mask_chunk is not None:
                        valid_L = (~mask_chunk).unsqueeze(-1).unsqueeze(-1).to(u_t_seq.dtype)
                        denom_L = valid_L.sum().clamp(min=1.0)
                        L_reg = (u_t_seq.pow(2) * valid_L).sum() / denom_L
                    else:
                        L_reg = u_t_seq.pow(2).mean()
                    loss = loss + lambda_L * L_reg
                    metrics["reg_L"] += L_reg.item()

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
    lambda_nis = float(config.train.lambda_nis)
    nis_target = float(config.model.obs_dim)
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
            if lambda_nis > 0.0:
                nis_vals, _ = compute_nis(outputs["innovations"], outputs["S"], mask)
                nis_mean = masked_mean(nis_vals.unsqueeze(-1), mask).mean()
                val_loss = val_loss + lambda_nis * (nis_mean - nis_target) ** 2
            total += val_loss.item() * batch_size
            count += batch_size
    ekf.train()
    dynamics.train()
    return total / max(count, 1)


__all__ = ["Trainer", "TrainerState", "build_model_components", "evaluate_nll"]
