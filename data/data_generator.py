"""Range-only trajectory generator for matched and mismatched motion cases.

This script follows the experimental setup described in
docs/unsupervised_GRU_augmented_ekf/gru_augmented_ekf.tex:
- 2D constant-velocity state with process noise from the Wiener velocity model.
- Four fixed anchors with range-only measurements in the professor form.
- Three motion regimes: matched (omega=0), mild mismatch (omega=1),
  and strong mismatch (omega=5) active only inside the prescribed turn windows.

Running the script writes datasets to:
- data/trajectory       (matched)
- data/trajectory_w1    (mismatch, omega=1)
- data/trajectory_w5    (mismatch, omega=5)

Each dataset is stored as a single NPZ file named
``trajectory_w{w}_N{num}_T{steps}_qc{qc}_sigmar{sigma}.npz`` where w∈{0,1,5}
matches the mismatch setting. The NPZ contains states, measurements, splits,
and TBPTT blocks. Use --help to see configuration options.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


TURN_WINDOWS: Tuple[Tuple[int, int], ...] = ((50, 100), (200, 250), (350, 400))


@dataclass
class GeneratorConfig:
    """Static configuration for dataset generation."""

    dt: float = 0.01
    steps: int = 500
    trajectories: int = 20
    qc: float = 0.5
    sigma_r: float = 0.05
    seed: int = 0
    tbptt_L: int = 50
    tbptt_stride: int = 50
    turn_length: int = 50  # left-closed, right-open window length
    anchors: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-1.5, 1.0, -1.0, 1.5],
                [0.5, 1.0, -1.0, -0.5],
            ],
            dtype=float,
        )
    )
    turn_windows: Tuple[Tuple[int, int], ...] = field(default_factory=lambda: TURN_WINDOWS)


@dataclass(frozen=True)
class CaseSpec:
    key: str
    base_omega: float
    output_subdir: str

    @property
    def mismatch(self) -> bool:
        return self.base_omega != 0.0


def build_case_specs(turn_rate: float) -> Dict[str, CaseSpec]:
    """Define matched vs mismatch cases; mismatch uses turn_rate computed from L, dt."""
    return {
        "matched": CaseSpec(key="w0", base_omega=0.0, output_subdir="trajectory"),
        "mismatch": CaseSpec(key="wpi", base_omega=turn_rate, output_subdir="trajectory_wpi"),
    }


def build_transition(dt: float) -> np.ndarray:
    """Constant-velocity transition matrix."""
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def build_process_noise(qc: float, dt: float) -> np.ndarray:
    """Discrete-time process noise covariance from the Wiener velocity model."""
    Q = np.array(
        [
            [qc * dt**3 / 3.0, 0.0, qc * dt**2 / 2.0, 0.0],
            [0.0, qc * dt**3 / 3.0, 0.0, qc * dt**2 / 2.0],
            [qc * dt**2 / 2.0, 0.0, qc * dt, 0.0],
            [0.0, qc * dt**2 / 2.0, 0.0, qc * dt],
        ],
        dtype=float,
    )
    return 0.5 * (Q + Q.T)


def compute_tbptt_blocks(steps: int, L: int, stride: int) -> np.ndarray:
    """Compute TBPTT block start/end indices (inclusive)."""
    blocks = []
    t = 0
    while t + L <= steps:
        blocks.append((t, t + L - 1))
        t += stride
    return np.array(blocks, dtype=int)


def _fmt_float(val: float) -> str:
    """Format float for filenames with trimmed trailing zeros."""
    return np.format_float_positional(val, trim="k")


def omega_profile(base_omega: float, k: int, windows: Iterable[Tuple[int, int]]) -> float:
    """Angular rate profile that is nonzero only inside the given windows."""
    if base_omega == 0.0:
        return 0.0
    for start, end in windows:
        if start <= k < end:
            return base_omega
    return 0.0


def sample_initial_state(rng: np.random.Generator) -> np.ndarray:
    """Draw initial [x, y, vx, vy] from the specified uniforms."""
    px0 = rng.uniform(-0.5, 0.5)
    py0 = rng.uniform(-0.5, 0.5)
    vx0 = rng.uniform(0.5, 1.5)
    vy0 = rng.uniform(-0.5, 0.5)
    return np.array([px0, py0, vx0, vy0], dtype=float)


def simulate_trajectory(
    F: np.ndarray,
    Q: np.ndarray,
    cfg: GeneratorConfig,
    base_omega: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate one trajectory and its range measurements."""
    x = sample_initial_state(rng)
    states = np.zeros((cfg.steps, 4), dtype=float)
    states[0] = x

    for k in range(cfg.steps - 1):
        omega = omega_profile(base_omega, k, cfg.turn_windows)
        if omega != 0.0:
            c, s = np.cos(omega * cfg.dt), np.sin(omega * cfg.dt)
            vx, vy = x[2], x[3]
            x[2] = c * vx - s * vy
            x[3] = s * vx + c * vy
        w_k = rng.multivariate_normal(np.zeros(4, dtype=float), Q)
        x = F @ x + w_k
        states[k + 1] = x

    pos = states[:, :2]
    dx = pos[:, [0]] + cfg.anchors[0, :]
    dy = pos[:, [1]] + cfg.anchors[1, :]
    ranges = np.sqrt(dx * dx + dy * dy)
    noise = rng.normal(scale=cfg.sigma_r, size=ranges.shape)
    measurements = ranges + noise
    return states, measurements


def generate_dataset_for_case(spec: CaseSpec, cfg: GeneratorConfig) -> Dict[str, np.ndarray]:
    """Generate all trajectories for the specified motion case."""
    rng = np.random.default_rng(cfg.seed)
    F = build_transition(cfg.dt)
    Q = build_process_noise(cfg.qc, cfg.dt)

    X = np.zeros((cfg.trajectories, cfg.steps, 4), dtype=float)
    Y = np.zeros((cfg.trajectories, cfg.steps, cfg.anchors.shape[1]), dtype=float)

    for n in range(cfg.trajectories):
        states, meas = simulate_trajectory(F, Q, cfg, spec.base_omega, rng)
        X[n] = states
        Y[n] = meas

    tbptt_single = compute_tbptt_blocks(cfg.steps, cfg.tbptt_L, cfg.tbptt_stride)
    tbptt_blocks = np.repeat(tbptt_single[None, :, :], cfg.trajectories, axis=0)

    train_ids = np.arange(0, 15, dtype=int)
    val_ids = np.arange(15, 18, dtype=int)
    test_ids = np.arange(18, 20, dtype=int)

    return {
        "case": np.array(spec.key),
        "mismatch": np.array(spec.mismatch),
        "omega_base": np.array(spec.base_omega, dtype=float),
        "turn_length": np.array(cfg.turn_length, dtype=int),
        "dt": np.array(cfg.dt, dtype=float),
        "steps": np.array(cfg.steps, dtype=int),
        "trajectories": np.array(cfg.trajectories, dtype=int),
        "qc": np.array(cfg.qc, dtype=float),
        "sigma_r": np.array(cfg.sigma_r, dtype=float),
        "anchors": cfg.anchors,
        "turn_windows": np.array(cfg.turn_windows, dtype=int),
        "F": F,
        "Q": Q,
        "X": X,
        "Y": Y,
        "tbptt_blocks": tbptt_blocks,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }


def save_dataset(data: Dict[str, np.ndarray], root: Path, spec: CaseSpec) -> Path:
    """Save dataset to the appropriate case directory using the requested naming scheme."""
    out_dir = root / spec.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    num = int(data["trajectories"])
    steps = int(data["steps"])
    qc = _fmt_float(float(data["qc"]))
    sigma_r = _fmt_float(float(data["sigma_r"]))
    w_tag = spec.key
    filename = f"trajectory_{w_tag}_N{num}_T{steps}_qc{qc}_sigmar{sigma_r}.npz"
    out_path = out_dir / filename
    np.savez(out_path, **data)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate range-only datasets for matched/mismatched cases.")
    parser.add_argument(
        "--case",
        choices=["matched", "mismatch", "all"],
        default="all",
        help="Which motion regime to generate. 'all' produces both matched and mismatch.",
    )
    parser.add_argument("--trajectories", type=int, default=20, help="Number of trajectories per case.")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps per trajectory.")
    parser.add_argument("--dt", type=float, default=0.01, help="Sampling period.")
    parser.add_argument("--qc", type=float, default=0.5, help="Continuous-time acceleration PSD.")
    parser.add_argument("--sigma-r", type=float, default=0.05, dest="sigma_r", help="Range noise std.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--turn-length", type=int, default=50, help="Turn window length (steps).")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root output directory (defaults to the data/ folder).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GeneratorConfig(
        dt=args.dt,
        steps=args.steps,
        trajectories=args.trajectories,
        qc=args.qc,
        sigma_r=args.sigma_r,
        seed=args.seed,
        turn_length=args.turn_length,
    )

    # For mismatch: choose omega so that velocity rotates 90 deg over one window (Exercise 5.5 style).
    omega_turn = float(np.pi / (2.0 * cfg.turn_length * cfg.dt))
    case_specs = build_case_specs(omega_turn)
    case_keys = list(case_specs) if args.case == "all" else [args.case]

    for key in case_keys:
        spec = case_specs[key]
        data = generate_dataset_for_case(spec, cfg)
        out_path = save_dataset(data, args.out_root, spec)
        print(f"Saved {spec.key} dataset to {out_path}")


if __name__ == "__main__":
    main()
