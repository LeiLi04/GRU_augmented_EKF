"""Run NLL-only training for the GRU-augmented EKF."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from gru_augmented_ekf.config import ExperimentConfig, load_config  # noqa: E402
from gru_augmented_ekf.data.lorenz import (  # noqa: E402
    MeasurementDataset,
    collate_padded_observations,
    windowed_dataset,
)
from gru_augmented_ekf.data.splits import (  # noqa: E402
    create_splits_file_name,
    load_splits_file,
    obtain_tr_val_test_idx,
    save_splits_file,
)
from gru_augmented_ekf.train.trainer import Trainer, build_model_components, evaluate_nll  # noqa: E402
from gru_augmented_ekf.train.warmstart_q0 import covariance_matching_warm_start  # noqa: E402
from gru_augmented_ekf.utils.seed import seed_everything  # noqa: E402


def compute_observation_statistics(dataset, indices, obs_dim: int):
    if not indices:
        return torch.zeros(obs_dim), torch.ones(obs_dim)
    total_sum = torch.zeros(obs_dim)
    total_sq = torch.zeros(obs_dim)
    total_count = 0
    for idx in indices:
        seq = dataset[idx]
        total_sum += seq.sum(dim=0)
        total_sq += (seq * seq).sum(dim=0)
        total_count += seq.size(0)
    if total_count == 0:
        return torch.zeros(obs_dim), torch.ones(obs_dim)
    mean = total_sum / total_count
    variance = torch.clamp(total_sq / total_count - mean * mean, min=1e-12)
    return mean, variance


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRU-augmented EKF with innovation NLL.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device

    seed_everything(cfg.seed, cfg.deterministic)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_path = Path(cfg.data.train_path).resolve()
    dataset = MeasurementDataset(train_path)
    collate = lambda batch: collate_padded_observations(batch, cfg.model.obs_dim)

    splits_path = (
        Path(cfg.data.split_path).resolve()
        if cfg.data.split_path
        else create_splits_file_name(train_path, cfg.data.splits_name)
    )
    if splits_path.exists():
        splits = load_splits_file(splits_path)
        train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    else:
        train_idx, val_idx, test_idx = obtain_tr_val_test_idx(
            dataset,
            cfg.data.tr_to_test_split,
            cfg.data.tr_to_val_split,
            cfg.seed,
        )
        save_splits_file(splits_path, {"train": train_idx, "val": val_idx, "test": test_idx})

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    _, obs_variance = compute_observation_statistics(dataset, train_idx, cfg.model.obs_dim)
    obs_variance = obs_variance.to(dtype=torch.float32)

    warm_split = cfg.train.warmup_split.lower() if cfg.train.warmup_split else "val"
    if warm_split == "test":
        warm_subset = test_subset
    elif warm_split == "train":
        warm_subset = train_subset
    else:
        warm_subset = val_subset

    warm_start_loader = DataLoader(
        warm_subset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate,
    )

    train_dataset = windowed_dataset(train_subset, cfg.data.window_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate,
    )

    dynamics, ekf, q_init, r_init = build_model_components(cfg, device, train_path=train_path, obs_variance=obs_variance)

    if cfg.train.use_warmup_q:
        for p in ekf.q_param.parameters():
            p.requires_grad_(True)
        tau_q = covariance_matching_warm_start(ekf, warm_start_loader, device=device)
        for p in ekf.q_param.parameters():
            p.requires_grad_(cfg.model.train_q)
        print(f"[warm start] tau_q={tau_q:.4f}")

    params = list(dynamics.parameters())
    if cfg.model.train_q:
        params += list(ekf.q_param.parameters())
    if cfg.model.train_r:
        params += list(ekf.r_param.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.train.lr, betas=cfg.train.betas)
    scaler = torch.amp.GradScaler("cuda") if cfg.train.amp and device.type == "cuda" else None
    trainer = Trainer(dynamics, ekf, optimizer, cfg, device, scaler=scaler)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_slug = f"{train_path.stem}/{timestamp}"
    tensorboard_dir = Path(cfg.logging.log_dir) / run_slug
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    log_file_dir = Path(cfg.logging.log_file_dir) / train_path.stem
    log_file_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_file_dir / f"{cfg.logging.run_name}_{timestamp}.log"

    model_dir = Path(cfg.logging.model_dir) / run_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = (
        Path(cfg.logging.best_model_path).resolve()
        if cfg.logging.best_model_path
        else model_dir / f"{cfg.logging.run_name}_best.pt"
    )
    checkpoint_path = (
        Path(cfg.logging.checkpoint_path).resolve()
        if cfg.logging.checkpoint_path
        else model_dir / f"{cfg.logging.run_name}_last.pt"
    )
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        "================ Run Configuration ================",
        f"run_name       : {cfg.logging.run_name}",
        f"dataset        : {train_path.stem}",
        f"train_path     : {cfg.data.train_path}",
        f"device         : {device}",
        "---------------- Model ----------------",
        f"n_states       : {cfg.model.state_dim}",
        f"n_obs          : {cfg.model.obs_dim}",
        f"hidden         : {cfg.model.dynamics_hidden}, depth: {cfg.model.dynamics_depth}",
        f"dt             : {cfg.model.dt}",
        "---------------- Noise ----------------",
        f"Q_diag_init    : {q_init:.6f}",
        f"R_diag_init    : {r_init:.6f}",
        "===================================================",
    ]
    with log_file_path.open("a") as log_file:
        log_file.write("\n".join(header_lines) + "\n")

    best_val = float("inf")
    epochs_no_improve = 0
    patience = int(cfg.earlystop.patience)
    min_delta = float(cfg.earlystop.min_delta)

    writer: SummaryWriter | None = SummaryWriter(log_dir=str(tensorboard_dir)) if cfg.logging.tensorboard else None
    try:
        for _ in range(cfg.train.epochs):
            epoch_start = time.time()
            metrics = trainer.train_epoch(train_loader, writer=writer)
            val_innov = evaluate_nll(ekf, dynamics, val_loader, cfg, device)
            elapsed = time.time() - epoch_start
            line = (
                f"Epoch {trainer.state.epoch}/{cfg.train.epochs} "
                f"Train NLL:{metrics['L_innov']:.6f} "
                f"Val NLL:{val_innov:.6f} "
                f"nis_mean:{metrics['nis_mean']:.6f} "
                f"time:{elapsed:.2f}s"
            )
            print(line)
            with log_file_path.open("a") as log_file:
                log_file.write(line + "\n")
            if writer is not None:
                writer.add_scalar("val/L_innov", val_innov, trainer.state.epoch)

            if val_innov < best_val - min_delta:
                best_val = val_innov
                epochs_no_improve = 0
                torch.save(
                    {
                        "dynamics": dynamics.state_dict(),
                        "q_param": ekf.q_param.state_dict(),
                        "r_param": ekf.r_param.state_dict(),
                        "metrics": {**metrics, "val_innov": val_innov},
                    },
                    best_model_path,
                )
            else:
                epochs_no_improve += 1

            if patience > 0 and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {trainer.state.epoch} (patience={patience})")
                break

        torch.save(
            {
                "dynamics": dynamics.state_dict(),
                "q_param": ekf.q_param.state_dict(),
                "r_param": ekf.r_param.state_dict(),
                "metrics": {"best_val_innov": best_val},
            },
            checkpoint_path,
        )
        print(f"Saved last checkpoint to {checkpoint_path}")
        print(f"Best validation innovation NLL: {best_val:.6f}")
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    main()
