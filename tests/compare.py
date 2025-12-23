"""Compare filters on the generated range-only dataset.

Features:
- Loads dataset (default: data/trajectory_wpi/trajectory_wpi_N20_T500_qc0.5_sigmar0.05.npz).
- Runs EKF (mandatory) and optional UKF/PF over all trajectories.
- Prints a small RMSE table and saves a trajectory plot for one sequence.

Notes:
- EKF assumes the known CV transition `F` and process noise `Q` stored in the NPZ.
- Measurement model is the professor range form using anchors from the dataset.
- UKF and PF are simple numpy implementations (no external dependencies).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def range_measurement(x: np.ndarray, anchors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Professor range measurement model."""
    dx = x[..., 0:1] + anchors[0, :]
    dy = x[..., 1:2] + anchors[1, :]
    dist = np.sqrt(np.maximum(dx * dx + dy * dy, eps))
    return dist


def range_jacobian(x: np.ndarray, anchors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Jacobian H for the range measurement w.r.t. state (px, py, vx, vy)."""
    px = x[..., 0:1]
    py = x[..., 1:2]
    dx = px + anchors[0, :]
    dy = py + anchors[1, :]
    dist = np.sqrt(np.maximum(dx * dx + dy * dy, eps))
    H = np.zeros(x.shape[:-1] + (anchors.shape[1], x.shape[-1]), dtype=float)
    H[..., :, 0] = dx / dist
    H[..., :, 1] = dy / dist
    # vx, vy columns stay zero
    return H


# ---------------- EKF ---------------- #
def run_ekf(
    measurements: np.ndarray, F: np.ndarray, Q: np.ndarray, R: np.ndarray, anchors: np.ndarray
) -> np.ndarray:
    """Run a basic EKF over all trajectories. Returns estimates shaped like measurements (N, T, 4)."""
    N, T, _ = measurements.shape
    state_dim = F.shape[0]

    x_hat = np.zeros((N, T, state_dim), dtype=float)
    P = np.tile(np.eye(state_dim) * 5.0, (N, 1, 1))  # broad prior
    x = np.zeros((N, state_dim), dtype=float)

    for k in range(T):
        # predict
        x = (F @ x.swapaxes(0, 1)).swapaxes(0, 1)
        P = F @ P @ F.T + Q
        # update
        z_pred = range_measurement(x, anchors)
        H = range_jacobian(x, anchors)
        S = H @ P @ np.swapaxes(H, -1, -2) + R
        K = np.empty((N, state_dim, anchors.shape[1]), dtype=float)
        for n in range(N):
            K[n] = P[n] @ H[n].T @ np.linalg.inv(S[n])
        innov = measurements[:, k] - z_pred
        x = x + np.einsum("nij,nj->ni", K, innov)
        I_KH = np.eye(state_dim) - np.einsum("nij,njk->nik", K, H)
        P = I_KH @ P @ I_KH.swapaxes(-1, -2) + np.einsum("nij,njk->nik", K, R @ K.swapaxes(-1, -2))

        x_hat[:, k] = x
    return x_hat


# ---------------- UKF ---------------- #
def _sigma_points(x: np.ndarray, P: np.ndarray, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
    n = x.shape[-1]
    lam = alpha * alpha * (n + kappa) - n
    U = np.linalg.cholesky((n + lam) * P)
    pts = [x]
    for i in range(n):
        pts.append(x + U[:, i])
        pts.append(x - U[:, i])
    pts = np.stack(pts, axis=0)
    wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)), dtype=float)
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1 - alpha * alpha + beta)
    return pts, wm, wc


def run_ukf(
    measurements: np.ndarray, F: np.ndarray, Q: np.ndarray, R: np.ndarray, anchors: np.ndarray
) -> np.ndarray:
    """Simple UKF assuming linear dynamics F and range measurement."""
    N, T, _ = measurements.shape
    n = F.shape[0]
    m = anchors.shape[1]
    x_hat = np.zeros((N, T, n), dtype=float)
    x = np.zeros((N, n), dtype=float)
    P = np.tile(np.eye(n) * 5.0, (N, 1, 1))

    for k in range(T):
        for i in range(N):
            pts, wm, wc = _sigma_points(x[i], P[i])
            # predict
            pts_pred = (F @ pts.swapaxes(0, 1)).swapaxes(0, 1)
            x_pred = np.sum(wm[:, None] * pts_pred, axis=0)
            P_pred = Q.copy()
            for j in range(pts_pred.shape[0]):
                diff = pts_pred[j] - x_pred
                P_pred += wc[j] * np.outer(diff, diff)

            # measurement transform
            z_sigma = range_measurement(pts_pred, anchors)
            z_pred = np.sum(wm[:, None] * z_sigma, axis=0)
            P_zz = R.copy()
            P_xz = np.zeros((n, m), dtype=float)
            for j in range(z_sigma.shape[0]):
                dz = z_sigma[j] - z_pred
                dx = pts_pred[j] - x_pred
                P_zz += wc[j] * np.outer(dz, dz)
                P_xz += wc[j] * np.outer(dx, dz)
            K = P_xz @ np.linalg.inv(P_zz)
            innov = measurements[i, k] - z_pred
            x[i] = x_pred + K @ innov
            P[i] = P_pred - K @ P_zz @ K.T
            x_hat[i, k] = x[i]
    return x_hat


# ---------------- PF ---------------- #
def systematic_resample(weights: np.ndarray) -> np.ndarray:
    N = weights.size
    positions = (np.arange(N) + np.random.rand()) / N
    cumulative_sum = np.cumsum(weights)
    indexes = np.zeros(N, dtype=int)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def run_pf(
    measurements: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    anchors: np.ndarray,
    num_particles: int,
) -> np.ndarray:
    """Bootstrap particle filter with Gaussian likelihood."""
    N, T, _ = measurements.shape
    n = F.shape[0]
    m = anchors.shape[1]
    x_hat = np.zeros((N, T, n), dtype=float)
    cov_R = R
    inv_R = np.linalg.inv(cov_R)
    det_R = np.linalg.det(cov_R)
    norm_const = np.sqrt((2 * np.pi) ** m * det_R)

    for i in range(N):
        particles = np.zeros((num_particles, n), dtype=float)
        weights = np.ones(num_particles, dtype=float) / num_particles
        for k in range(T):
            noise = np.random.multivariate_normal(np.zeros(n), Q, size=num_particles)
            particles = (F @ particles.swapaxes(0, 1)).swapaxes(0, 1) + noise
            z_pred = range_measurement(particles, anchors)
            innov = measurements[i, k] - z_pred
            # Gaussian likelihood
            exponents = -0.5 * np.einsum("ni,ij,nj->n", innov, inv_R, innov)
            w = np.exp(exponents) / norm_const
            weights *= w
            weights_sum = np.sum(weights)
            if weights_sum == 0:
                weights = np.ones(num_particles) / num_particles
            else:
                weights /= weights_sum
            if 1.0 / np.sum(weights * weights) < 0.5 * num_particles:
                idx = systematic_resample(weights)
                particles = particles[idx]
                weights = np.ones(num_particles) / num_particles
            x_hat[i, k] = np.average(particles, weights=weights, axis=0)
    return x_hat


# ---------------- Metrics & Viz ---------------- #
def position_rmse(truth: np.ndarray, est: np.ndarray) -> float:
    err = truth[..., :2] - est[..., :2]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def make_table(results: Dict[str, float]) -> str:
    header = f"{'Filter':<10} | {'Pos RMSE':>10}\n" + "-" * 24
    rows = [f"{name:<10} | {rmse:10.4f}" for name, rmse in results.items()]
    return "\n".join([header] + rows)


def plot_trajectories(
    truth: np.ndarray,
    estimates: Dict[str, np.ndarray],
    anchors: np.ndarray,
    idx: int,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    plt.plot(truth[:, 0], truth[:, 1], label="true", linewidth=2.0, color="black")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for c_idx, (name, est) in enumerate(estimates.items()):
        color = colors[c_idx % len(colors)]
        plt.plot(est[:, 0], est[:, 1], label=name, linewidth=1.5, color=color)
    plt.scatter(anchors[0, :], anchors[1, :], marker="x", color="red", label="anchors")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Trajectory #{idx}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_velocities(
    truth: np.ndarray,
    estimates: Dict[str, np.ndarray],
    idx: int,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 4))
    t = np.arange(truth.shape[0])
    components = [("vx", 2), ("vy", 3)]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for subplot_idx, (name, comp) in enumerate(components, start=1):
        plt.subplot(1, 2, subplot_idx)
        plt.plot(t, truth[:, comp], label="true", color="black", linewidth=2.0)
        for c_idx, (filt_name, est) in enumerate(estimates.items()):
            color = colors[c_idx % len(colors)]
            plt.plot(t, est[:, comp], label=filt_name, color=color, linewidth=1.2)
        plt.xlabel("time step")
        plt.ylabel(name)
        plt.title(f"{name} (traj #{idx})")
        plt.grid(True, linestyle="--", alpha=0.4)
        if subplot_idx == 1:
            plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------- Main ---------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare filters on range-only dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/trajectory_wpi/trajectory_wpi_N20_T500_qc0.5_sigmar0.05.npz"),
        help="Path to NPZ dataset.",
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        default=["ekf"],
        choices=["ekf", "ukf", "pf"],
        help="Filters to run (ekf mandatory, others optional).",
    )
    parser.add_argument("--num-particles", type=int, default=400, help="Particles for PF (if used).")
    parser.add_argument("--traj-idx", type=int, default=0, help="Trajectory index for plotting.")
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Plot save path. Defaults to figures/compare_{dataset_name}.png",
    )
    parser.add_argument("--max-traj", type=int, default=None, help="Optional limit on number of trajectories to run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filters: List[str] = list(dict.fromkeys(args.filters))  # unique, preserve order
    if "ekf" not in filters:
        filters.insert(0, "ekf")

    data = load_dataset(args.dataset)
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)  # measurements
    F = np.asarray(data["F"], dtype=float)
    Q = np.asarray(data["Q"], dtype=float)
    anchors = np.asarray(data["anchors"], dtype=float)
    sigma_r = float(data["sigma_r"])
    R = (sigma_r**2) * np.eye(anchors.shape[1])

    if args.max_traj is not None:
        X = X[: args.max_traj]
        Y = Y[: args.max_traj]

    results_estimates: Dict[str, np.ndarray] = {}
    metrics: Dict[str, float] = {}

    for name in filters:
        if name == "ekf":
            est = run_ekf(Y, F, Q, R, anchors)
        elif name == "ukf":
            est = run_ukf(Y, F, Q, R, anchors)
        elif name == "pf":
            est = run_pf(Y, F, Q, R, anchors, num_particles=args.num_particles)
        else:
            raise ValueError(f"Unknown filter: {name}")
        results_estimates[name] = est
        metrics[name] = position_rmse(X, est)

    print(make_table(metrics))

    plot_idx = min(args.traj_idx, X.shape[0] - 1)
    plot_est = {k: v[plot_idx] for k, v in results_estimates.items()}
    default_plot_pos = Path("figures") / f"compare_pos_{args.dataset.name}.png"
    default_plot_vel = Path("figures") / f"compare_vel_{args.dataset.name}.png"
    if args.plot_path is not None:
        plot_path_pos = args.plot_path
        plot_path_vel = args.plot_path.with_name(args.plot_path.stem + "_vel" + args.plot_path.suffix)
    else:
        plot_path_pos = default_plot_pos
        plot_path_vel = default_plot_vel
    plot_trajectories(X[plot_idx], plot_est, anchors, plot_idx, plot_path_pos)
    plot_velocities(X[plot_idx], plot_est, plot_idx, plot_path_vel)
    print(f"Saved position plot to {plot_path_pos}")
    print(f"Saved velocity plot to {plot_path_vel}")


if __name__ == "__main__":
    main()
