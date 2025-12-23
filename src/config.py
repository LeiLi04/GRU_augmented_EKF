"""Configuration dataclasses for the GRU-augmented EKF project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass
class DataConfig:
    train_path: str = "src/data/trajectories/trajectories_m_3_n_3_LorenzSSM_data_T_100_N_1000_r2_-10.0dB_q2_0.0dB.pkl"
    test_path: str = "src/data/trajectories/trajectories_m_3_n_3_LorenzSSM_data_T_2000_N_100_r2_-10.0dB_q2_0.0dB.pkl"
    long_test_path: str = ""
    split_path: str = ""
    splits_name: str = ""
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = True
    window_length: int = 50
    tr_to_test_split: float = 0.9
    tr_to_val_split: float = 0.8333


@dataclass
class ModelConfig:
    state_dim: int = 3
    obs_dim: int = 3
    dt: float = 0.02
    dynamics_hidden: int = 128
    dynamics_depth: int = 3
    dynamics_use_gru: bool = True
    dynamics_tanh_scale: float = 0.1
    dynamics_residual_init_std: float = 1e-3
    dynamics_scale_a_min: float = 0.95
    dynamics_scale_a_max: float = 10.0
    max_delta: float | None = None
    q_init: float = 0.05
    r_init: float = 0.1
    train_q: bool = False
    train_r: bool = False


@dataclass
class TrainConfig:
    epochs: int = 200
    tbptt_steps: int = 50
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    clip_norm: float = 1.0
    amp: bool = False
    sigma0_scale: float = 1.0
    use_warmup_q: bool = False
    warmup_split: str = "val"


@dataclass
class EarlyStopConfig:
    patience: int = 15
    min_delta: float = 1e-4


@dataclass
class LoggingConfig:
    run_name: str = "gru_augmented_ekf"
    log_dir: str = "runs/gru_augmented_ekf"
    tensorboard: bool = True
    log_file_dir: str = "log/gru_augmented_ekf"
    model_dir: str = "models/gru_augmented_ekf"
    best_model_path: str = ""
    checkpoint_path: str = ""


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "cuda"
    deterministic: bool = False
    data: DataConfig = field(default_factory=DataConfig)  # type: ignore[arg-type]
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    earlystop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "ExperimentConfig":
        data_cfg = DataConfig(**raw.get("data", {}))
        model_cfg = ModelConfig(**raw.get("model", {}))
        train_raw = dict(raw.get("train", {}))
        if "betas" in train_raw and isinstance(train_raw["betas"], (list, tuple)):
            train_raw["betas"] = tuple(train_raw["betas"])
        train_cfg = TrainConfig(**train_raw)
        early_cfg = EarlyStopConfig(**raw.get("earlystop", {}))
        log_cfg = LoggingConfig(**raw.get("logging", {}))
        return ExperimentConfig(
            seed=raw.get("seed", 42),
            device=raw.get("device", "cuda"),
            deterministic=raw.get("deterministic", False),
            data=data_cfg,
            model=model_cfg,
            train=train_cfg,
            earlystop=early_cfg,
            logging=log_cfg,
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file."""
    with Path(path).open("r") as handle:
        raw = yaml.safe_load(handle)
    if "experiment" in raw:
        raw = raw["experiment"]
    return ExperimentConfig.from_dict(raw)


__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "EarlyStopConfig",
    "LoggingConfig",
    "ExperimentConfig",
    "load_config",
]
