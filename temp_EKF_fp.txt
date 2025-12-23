"""FilterPy-based Extended Kalman Filter demo on Lorenz dataset."""

from __future__ import annotations

import argparse
from collections import namedtuple
from pathlib import Path
import pickle

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter


def dB_to_lin(x_dB: float) -> float:
    return 10.0 ** (x_dB / 10.0)


def lin_to_dB(x_lin: float, eps: float = 1e-12) -> float:
    return 10.0 * np.log10(max(x_lin, eps))


def lorenz_rhs(x: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    x1, x2, x3 = x
    return np.array([
        sigma * (x2 - x1),
        x1 * (rho - x3) - x2,
        x1 * x2 - beta * x3,
    ], dtype=float)


def lorenz_rk4(x: np.ndarray, dt: float) -> np.ndarray:
    k1 = lorenz_rhs(x)
    k2 = lorenz_rhs(x + 0.5 * dt * k1)
    k3 = lorenz_rhs(x + 0.5 * dt * k2)
    k4 = lorenz_rhs(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def lorenz_jacobian(x: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    x1, x2, x3 = x
    return np.array([
        [-sigma, sigma, 0.0],
        [rho - x3, -1.0, -x1],
        [x2, x1, -beta],
    ], dtype=float)


def _load_lorenz_dataset(path: Path):
    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)

    with path.open("rb") as handle:
        return _CompatUnpickler(handle).load()


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.cbook as cbook

    if not hasattr(cbook, "_is_pandas_dataframe"):
        def _is_pandas_dataframe(_obj):
            return False

        cbook._is_pandas_dataframe = _is_pandas_dataframe

    if not hasattr(cbook, "_Stack") and hasattr(cbook, "Stack"):
        cbook._Stack = cbook.Stack

    if not hasattr(cbook, "_ExceptionInfo"):
        cbook._ExceptionInfo = namedtuple("_ExceptionInfo", "exc_class exc traceback")

    if not hasattr(cbook, "_broadcast_with_masks"):
        def _broadcast_with_masks(*arrays):
            return tuple(np.broadcast_arrays(*[np.asarray(arr) for arr in arrays]))

        cbook._broadcast_with_masks = _broadcast_with_masks

    import matplotlib.pyplot as plt  # noqa: F401


def fx(x: np.ndarray, dt: float) -> np.ndarray:
    return lorenz_rk4(np.asarray(x, dtype=float), dt)


def hx(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def F_jacobian(x: np.ndarray, dt: float) -> np.ndarray:
    J = lorenz_jacobian(x)
    return np.eye(x.size) + dt * J


def H_jacobian(_: np.ndarray) -> np.ndarray:
    return np.eye(3)


def main():
    parser = argparse.ArgumentParser(description="FilterPy EKF demo on Lorenz dataset.")
    parser.add_argument(
        "--dataset",
        default="src/data/trajectories/trajectories_m_3_n_3_LorenzSSM_data_T_1000_N_500_r2_20.0dB_nu_-10.0dB.pkl",
        help="Path to Lorenz trajectory pickle.",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index to visualise (default: 0).")
    parser.add_argument("--delta_t", type=float, default=0.02, help="Integration time step for dynamics.")
    parser.add_argument("--inverse_r2_dB", type=float, default=20.0, help="Inverse measurement noise power in dB.")
    parser.add_argument("--nu_dB", type=float, default=-10.0, help="Process vs measurement noise ratio in dB.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    payload = _load_lorenz_dataset(dataset_path)
    num_samples = payload["num_samples"]
    if not (0 <= args.index < num_samples):
        raise IndexError(f"index {args.index} out of range (0 ≤ idx < {num_samples}).")

    sample_X, sample_Y = payload["data"][args.index]
    sample_X = np.asarray(sample_X, dtype=float)
    sample_Y = np.asarray(sample_Y, dtype=float)

    dt = float(args.delta_t)
    n_states = sample_X.shape[1]
    n_obs = sample_Y.shape[1]

    r2 = 1.0 / dB_to_lin(args.inverse_r2_dB)
    q2 = dB_to_lin(args.nu_dB - args.inverse_r2_dB)
    Q = q2 * np.eye(n_states)
    R = r2 * np.eye(n_obs)

    ekf = ExtendedKalmanFilter(dim_x=n_states, dim_z=n_obs)
    ekf.x = sample_Y[0].astype(float)
    ekf.P = np.eye(n_states) * 5.0
    ekf.R = R
    ekf.Q = Q

    estimates = np.zeros_like(sample_Y)
    covariances = np.zeros((sample_Y.shape[0], n_states, n_states))

    for t in range(sample_Y.shape[0]):
        F = F_jacobian(ekf.x, dt)
        ekf.x = fx(ekf.x, dt)
        ekf.P = F @ ekf.P @ F.T + Q
        ekf.update(
            z=sample_Y[t],
            HJacobian=H_jacobian,
            Hx=hx,
            R=R,
        )
        estimates[t] = ekf.x.copy()
        covariances[t] = ekf.P.copy()

    x_hat = np.zeros_like(sample_X)
    x_hat[0] = sample_X[0]
    x_hat[1:] = estimates

    mse_per_step = np.mean((x_hat[1:] - sample_X[1:]) ** 2, axis=1)
    mse_lin = float(np.mean(mse_per_step))
    mse_dB = lin_to_dB(mse_lin)

    time_axis = np.arange(sample_Y.shape[0])

    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(18, 5))
    ax_true = fig.add_subplot(1, 3, 1, projection="3d")
    ax_obs = fig.add_subplot(1, 3, 2, projection="3d")
    ax_est = fig.add_subplot(1, 3, 3, projection="3d")

    ax_true.plot(sample_X[1:, 0], sample_X[1:, 1], sample_X[1:, 2], label="x (true)", color="tab:blue", linewidth=1.5)
    ax_true.set_title("True state x")
    ax_true.set_xlabel("x1")
    ax_true.set_ylabel("x2")
    ax_true.set_zlabel("x3")
    ax_true.legend(fontsize="small")

    ax_obs.plot(sample_Y[:, 0], sample_Y[:, 1], sample_Y[:, 2], label="y (obs)", color="tab:green", linewidth=1.2)
    ax_obs.set_title("Observation y")
    ax_obs.set_xlabel("y1")
    ax_obs.set_ylabel("y2")
    ax_obs.set_zlabel("y3")
    ax_obs.legend(fontsize="small")

    ax_est.plot(x_hat[1:, 0], x_hat[1:, 1], x_hat[1:, 2], label="$\\hat{x}$ (estimate)", color="tab:orange", linewidth=1.5)
    ax_est.set_title("Estimate $\\hat{x}$")
    ax_est.set_xlabel("x1")
    ax_est.set_ylabel("x2")
    ax_est.set_zlabel("x3")
    ax_est.legend(fontsize="small")

    fig.tight_layout()
    fig.savefig("lorenz_ekf_fp_states.png", dpi=300)

    fig_mse = plt.figure(figsize=(10, 3))
    plt.plot(time_axis, mse_per_step, label="per-step MSE", color="tab:red")
    plt.axhline(0.0, color="black", linewidth=0.8, linestyle=":", label="zero baseline")
    plt.ylabel("MSE")
    plt.xlabel("Time step")
    plt.title("FilterPy EKF per-step state estimation MSE")
    plt.legend()
    plt.grid(True, linewidth=0.3, alpha=0.7)
    fig_mse.tight_layout()
    fig_mse.savefig("lorenz_ekf_fp_mse.png", dpi=300)

    print(f"Summary stats -> MSE_lin: {mse_lin:.3f}, MSE_dB: {mse_dB:.3f}")

    plt.show()


if __name__ == "__main__":
    main()
