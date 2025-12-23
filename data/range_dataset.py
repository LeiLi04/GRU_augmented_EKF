"""Range-only dataset generator for a 2D constant-velocity target.

Creates trajectories and range measurements to four fixed anchors following the
professor-provided model. Produces ``dataset_range_tbptt_professor_plus.npz``
with states ``X`` and measurements ``Z`` shaped (N_total, steps, 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DatasetConfig:
    """Static configuration for the dataset."""

    seed: int = 123
    dt: float = 0.01
    steps: int = 500  # time steps per trajectory
    N_total: int = 20  # number of trajectories
    qc: float = 0.1  # continuous-time acceleration PSD
    sd_r: float = 0.1  # measurement noise std (range)
    tbptt_L: int = 50
    tbptt_stride: int = 50
    anchors: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-1.5, 1.0, -1.0, 1.5],
                [0.5, 1.0, -1.0, -0.5],
            ],
            dtype=float,
        )
    )  # shape (2, 4), anchors[:, i] = (s_x^i, s_y^i)


def build_model_matrices(dt: float, qc: float) -> Tuple[np.ndarray, np.ndarray]:
    """Construct constant-velocity state transition A and process noise Q."""
    A = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    Q = np.array(
        [
            [qc * dt**3 / 3.0, 0.0, qc * dt**2 / 2.0, 0.0],
            [0.0, qc * dt**3 / 3.0, 0.0, qc * dt**2 / 2.0],
            [qc * dt**2 / 2.0, 0.0, qc * dt, 0.0],
            [0.0, qc * dt**2 / 2.0, 0.0, qc * dt],
        ],
        dtype=float,
    )
    # Ensure symmetry to guard against numerical asymmetry.
    Q = 0.5 * (Q + Q.T)
    return A, Q


def cholesky_with_jitter(Q: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """Return a Cholesky factor of Q, adding tiny jitter if needed."""
    try:
        return np.linalg.cholesky(Q)
    except np.linalg.LinAlgError:
        Q_jittered = Q + jitter * np.eye(Q.shape[0])
        return np.linalg.cholesky(Q_jittered)


def generate_one_trajectory(
    A: np.ndarray,
    Q: np.ndarray,
    sd_r: float,
    anchors: np.ndarray,
    steps: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a single trajectory.

    Returns:
        states: (steps, 4) array of [px, py, vx, vy]
        measurements: (steps, 4) array of range measurements
        process_noise: (steps-1, 4) w_k samples
        meas_noise: (steps, 4) r_k^i samples
    """
    L_Q = cholesky_with_jitter(Q)

    # Initial state sampling
    px0 = rng.uniform(-0.5, 0.5)
    py0 = rng.uniform(-0.5, 0.5)
    vx0 = rng.uniform(0.5, 1.5)
    vy0 = rng.uniform(-0.5, 0.5)
    x = np.array([px0, py0, vx0, vy0], dtype=float)

    states = np.zeros((steps, 4), dtype=float)
    measurements = np.zeros((steps, 4), dtype=float)
    process_noise = np.zeros((steps - 1, 4), dtype=float)
    meas_noise = np.zeros((steps, 4), dtype=float)
    states[0] = x

    # Propagate states
    for k in range(steps - 1):
        w_k = L_Q @ rng.standard_normal(4)
        process_noise[k] = w_k
        x = A @ x + w_k
        states[k + 1] = x

    # Generate measurements (range-only)
    for k in range(steps):
        px, py = states[k, 0], states[k, 1]
        for i in range(anchors.shape[1]):
            noise = sd_r * rng.standard_normal()
            meas_noise[k, i] = noise
            dx = anchors[0, i] + px
            dy = anchors[1, i] + py
            measurements[k, i] = np.sqrt(dx * dx + dy * dy) + noise

    return states, measurements, process_noise, meas_noise


def compute_tbptt_blocks(steps: int, L: int, stride: int) -> np.ndarray:
    """Precompute TBPTT block start/end indices (inclusive)."""
    blocks = []
    t = 0
    while t + L <= steps:
        start = t
        end = t + L - 1
        blocks.append((start, end))
        t += stride
    return np.array(blocks, dtype=int)


def generate_dataset(config: DatasetConfig) -> Dict[str, np.ndarray]:
    """Generate full dataset following the provided configuration."""
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    A, Q = build_model_matrices(config.dt, config.qc)

    X = np.zeros((config.N_total, config.steps, 4), dtype=float)  # (traj, time, state_dim)
    Z = np.zeros((config.N_total, config.steps, 4), dtype=float)  # (traj, time, num_anchors)
    W = np.zeros((config.N_total, config.steps - 1, 4), dtype=float)
    R = np.zeros((config.N_total, config.steps, 4), dtype=float)

    for n in range(config.N_total):
        x, z, w, r = generate_one_trajectory(
            A=A,
            Q=Q,
            sd_r=config.sd_r,
            anchors=config.anchors,
            steps=config.steps,
            rng=rng,
        )
        X[n] = x
        Z[n] = z
        W[n] = w
        R[n] = r

    tbptt_single = compute_tbptt_blocks(config.steps, config.tbptt_L, config.tbptt_stride)
    tbptt_blocks = np.repeat(tbptt_single[None, :, :], config.N_total, axis=0)

    train_ids = np.arange(0, 15, dtype=int)
    val_ids = np.arange(15, 18, dtype=int)
    test_ids = np.arange(18, 20, dtype=int)

    return {
        "dt": np.array(config.dt, dtype=float),
        "steps": np.array(config.steps, dtype=int),
        "N_total": np.array(config.N_total, dtype=int),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "A": A,
        "Q": Q,
        "qc": np.array(config.qc, dtype=float),
        "anchors": config.anchors,
        "sd_r": np.array(config.sd_r, dtype=float),
        "X": X,
        "Z": Z,
        "tbptt_blocks": tbptt_blocks,
        "process_noise": W,
        "meas_noise": R,
    }


def save_dataset(data: Dict[str, np.ndarray], path: Path) -> None:
    """Save dataset dictionary to a compressed NPZ file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def plot_trajectory_and_ranges(
    data: Dict[str, np.ndarray],
    anchors: np.ndarray,
    traj_idx: int = 0,
) -> None:
    """Visualize position path and range measurements for one trajectory."""
    states = data["X"][traj_idx]
    measurements = data["Z"][traj_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Position path with anchors
    axes[0].plot(states[:, 0], states[:, 1], label="trajectory")
    axes[0].scatter(anchors[0, :], anchors[1, :], c="red", marker="x", label="anchors")
    axes[0].set_xlabel("px")
    axes[0].set_ylabel("py")
    axes[0].set_title(f"Trajectory #{traj_idx + 1}")
    axes[0].axis("equal")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Range measurements
    for i in range(anchors.shape[1]):
        axes[1].plot(measurements[:, i], label=f"anchor {i + 1}")
    axes[1].set_xlabel("time step k")
    axes[1].set_ylabel("range measurement")
    axes[1].set_title("Range measurements")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def print_summary(data: Dict[str, np.ndarray]) -> None:
    """Print quick statistics to verify dataset contents."""
    X = data["X"]
    Z = data["Z"]
    tbptt_blocks = data["tbptt_blocks"]

    print(f"States X shape: {X.shape} (traj, time, state_dim)")
    print(f"Measurements Z shape: {Z.shape} (traj, time, num_anchors)")
    print(f"Mean range: {Z.mean():.4f}, Std range: {Z.std():.4f}")
    print(f"TBPTT blocks shape: {tbptt_blocks.shape} (traj, num_blocks, 2)")
    print(f"First trajectory blocks (start,end):\n{tbptt_blocks[0]}")


def main() -> None:
    config = DatasetConfig()
    data = generate_dataset(config)

    out_path = Path("dataset_range_tbptt_professor_plus.npz")
    save_dataset(data, out_path)

    print(f"Saved dataset to {out_path.resolve()}")
    print_summary(data)
    plot_trajectory_and_ranges(data, config.anchors, traj_idx=0)


if __name__ == "__main__":
    np.random.seed(123)  # align with requested seed call
    main()
