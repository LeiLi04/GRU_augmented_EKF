"""Evaluate short-horizon sequences with the trained GRU-augmented EKF."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from gru_augmented_ekf.config import load_config  # noqa: E402
from gru_augmented_ekf.data.lorenz import StateObservationDataset, collate_padded_state_obs  # noqa: E402
from gru_augmented_ekf.train.trainer import build_model_components  # noqa: E402
from gru_augmented_ekf.losses.innov_nll import innovation_nll  # noqa: E402
from gru_augmented_ekf.metrics.nis_nees import compute_nees  # noqa: E402
from gru_augmented_ekf.metrics.nmse import nmse  # noqa: E402
from gru_augmented_ekf.metrics.ljung_box import ljung_box_pvalues  # noqa: E402
from gru_augmented_ekf.utils.seed import seed_everything  # noqa: E402
from gru_augmented_ekf.utils.linear_algebra import chol_solve, safe_cholesky  # noqa: E402
from scipy.stats import chi2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Short-sequence evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to load.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed, cfg.deterministic)

    dataset_path = Path(cfg.data.test_path).resolve()
    dataset = StateObservationDataset(dataset_path)
    collate = lambda batch: collate_padded_state_obs(batch, cfg.model.state_dim, cfg.model.obs_dim)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate,
    )

    dynamics, ekf, _, _ = build_model_components(cfg, device, train_path=dataset_path, obs_variance=None)
    state = torch.load(Path(args.checkpoint), map_location=device)
    dynamics.load_state_dict(state["dynamics"])
    ekf.q_param.load_state_dict(state["q_param"])
    ekf.r_param.load_state_dict(state["r_param"])
    ekf.eval()
    dynamics.eval()

    total_nll = 0.0
    total_nmse = 0.0
    total_batches = 0
    nis_sum = 0.0
    nis_count = 0
    nis_pass = 0
    nees_sum = 0.0
    nees_count = 0
    nees_pass = 0
    lb_pvalues = []

    lower_nis = chi2.ppf(0.025, cfg.model.obs_dim)
    upper_nis = chi2.ppf(0.975, cfg.model.obs_dim)
    lower_nees = chi2.ppf(0.025, cfg.model.state_dim)
    upper_nees = chi2.ppf(0.975, cfg.model.state_dim)

    with torch.no_grad():
        for state_seq, obs_seq, mask in dataloader:
            obs = obs_seq.to(device)
            states = state_seq.to(device)
            mask = mask.to(device)
            batch_size, T, _ = obs.shape
            x0 = torch.zeros(batch_size, cfg.model.state_dim, device=device)
            Sigma0 = (
                torch.eye(cfg.model.state_dim, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                * float(cfg.train.sigma0_scale)
            )
            outputs = ekf(obs, x0, Sigma0, mask=mask)
            nll = innovation_nll(outputs["innovations"], outputs["S"], outputs.get("logdet_S"), mask)
            total_nll += nll.item() * batch_size
            total_batches += batch_size

            # Align state length (dataset stores T+1 states)
            if states.size(1) == obs.size(1) + 1:
                states_aligned = states[:, 1:]
            else:
                min_len = min(states.size(1), obs.size(1))
                states_aligned = states[:, :min_len]
                obs = obs[:, :min_len]
                mask = mask[:, :min_len]
                outputs["x_filt"] = outputs["x_filt"][:, :min_len]
                outputs["Sigma_filt"] = outputs["Sigma_filt"][:, :min_len]
                outputs["whitened"] = outputs["whitened"][:, :min_len]
                outputs["innovations"] = outputs["innovations"][:, :min_len]
                outputs["S"] = outputs["S"][:, :min_len]

            nmse_val = nmse(outputs["x_filt"], states_aligned, mask)
            total_nmse += nmse_val.item() * batch_size

            chol_S, _ = safe_cholesky(outputs["S"])
            S_inv_delta = chol_solve(chol_S, outputs["innovations"].unsqueeze(-1)).squeeze(-1)
            nis_vals = (outputs["innovations"] * S_inv_delta).sum(dim=-1)
            valid = (~mask)
            nis_sum += nis_vals[valid].sum().item()
            nis_count += valid.sum().item()
            nis_pass += ((nis_vals[valid] >= lower_nis) & (nis_vals[valid] <= upper_nis)).sum().item()

            nees_vals, _ = compute_nees(states_aligned - outputs["x_filt"], outputs["Sigma_filt"], mask)
            nees_sum += nees_vals[valid].sum().item()
            nees_count += valid.sum().item()
            nees_pass += ((nees_vals[valid] >= lower_nees) & (nees_vals[valid] <= upper_nees)).sum().item()

            lb_vals = ljung_box_pvalues(outputs["whitened"], mask=mask, lag=20)
            lb_pvalues.append(lb_vals.mean().item())

    mean_nll = total_nll / max(total_batches, 1)
    mean_nmse = total_nmse / max(total_batches, 1)
    nis_mean = nis_sum / max(nis_count, 1)
    nees_mean = nees_sum / max(nees_count, 1)
    nis_pass_rate = 100.0 * nis_pass / max(nis_count, 1)
    nees_pass_rate = 100.0 * nees_pass / max(nees_count, 1)
    lb_mean = float(sum(lb_pvalues) / max(len(lb_pvalues), 1))

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset:    {dataset_path}")
    print(f"NLL:        {mean_nll:.6f}")
    print(f"NMSE:       {mean_nmse:.6f}")
    print(f"NIS mean:   {nis_mean:.6f} | pass-rate: {nis_pass_rate:.2f}%")
    print(f"NEES mean:  {nees_mean:.6f} | pass-rate: {nees_pass_rate:.2f}%")
    print(f"Ljung-Box p (mean over dims): {lb_mean:.4f}")


if __name__ == "__main__":
    main()
