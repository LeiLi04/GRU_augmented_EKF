# GRU_augmented_EKF

## Data generation
- Script: `py data/data_generator.py`
- Cases: `--case matched | mismatch | all` (default `all`)
- Default outputs (20 traj, 500 steps, dt=0.01, qc=0.5, sigma_r=0.05):
  - `data/trajectory/trajectory_w0_N20_T500_qc0.5_sigmar0.05.npz` (matched)
  - `data/trajectory_wpi/trajectory_wpi_N20_T500_qc0.5_sigmar0.05.npz` (mismatch: three 90° turn windows, ω=π rad/s, L=50, Δt=0.01)
- Adjustable args: `--trajectories`, `--steps`, `--dt`, `--qc`, `--sigma-r`, `--seed`, `--turn-length`, `--out-root`
- Internals: 2D CV model with WNA process noise; professor range measurements to 4 anchors; turn windows at k∈[50,100],[200,250],[350,400] (left-closed, right-open); TBPTT blocks (L=50, stride=50); fixed splits train 0–14, val 15–17, test 18–19 stored in NPZ.

## Filter comparison
- Script: `tests/compare.py`
- Default dataset: `data/trajectory_wpi/trajectory_wpi_N20_T500_qc0.5_sigmar0.05.npz`
- Example runs:
  - EKF only: `py tests/compare.py`
  - EKF + UKF + PF: `py tests/compare.py --filters ekf ukf pf`
  - Custom dataset: `py tests/compare.py --dataset data/trajectory/trajectory_w0_N20_T500_qc0.5_sigmar0.05.npz`
- Options: `--filters ekf [ukf] [pf]`, `--num-particles`, `--traj-idx`, `--plot-path`, `--max-traj`
- Outputs: RMSE table; figures for position (with anchors) and velocity traces saved to `figures/compare_pos_{dataset}.png` and `figures/compare_vel_{dataset}.png`.

## Warm-start of process covariance (q0)
- Purpose: before GRU training, select isotropic `Q = q0^2 I` minimizing innovation NLL on warm subset.
- How: nominal EKF (no GRU), log-spaced grid `torch.logspace(-3, 0, steps=7)` ≈ [0.001, 0.0032, 0.01, 0.032, 0.1, 0.32, 1.0]; compute mean (whitened_innov² + logdet S) and pick best q0.
- Implementation: `src/train/warmstart_q0.py` (`covariance_matching_warm_start`), invoked when `train.use_warmup_q=True` in `scripts/train.py`.
- Rationale: short, stable search covering low/medium/high process noise; Cholesky-based S for numerical stability.

## Suggested experiments (conference-ready)
- Grid: qc ∈ {0.2, 0.5, 1.0} × scenes ∈ {matched (w0), mismatch (wpi)} → 6 combos.
- Metrics: position RMSE, NIS mean/coverage; include 1–2 representative trajectory plots.
- Notes: covers both process-noise mis-tuning and motion-model mismatch; add an extreme qc (e.g., 0.05 or 2.0) as appendix if space allows.

## Notes on covariance adaptation
- Current GRU head scales the predicted covariance (prior Sigma) diagonally via α; assumes base shape is roughly correct.
- If you need to handle correlated/shape-mismatched process noise:
  - Enable `train_q=True` to learn a full PSD Q (initialized from dataset Q).
  - Add a GRU head to output a low-rank or full matrix correction to prior Sigma (e.g., Sigma_pred ← Sigma_pred + U_t U_tᵀ or a PD transform).
  - Keep diagonal scaling as a baseline, then stack a more flexible correction if budget allows.
## GRU low-rank covariance head
- GRU outputs diagonal scale _t and optional low-rank factor U_t (softplus, shape [B, state_dim, cov_rank]), applied as Sigma_pred <- A_t * Sigma_prior * A_t + U_t U_t^T (scheme B, PSD).
- cov_rank (config model.cov_rank) is a tunable hyperparameter; default 0 (off), typically 2�C8 and cov_rank << d_x.
- Regularizers 	rain.lambda_delta (GRU residuals) and 	rain.lambda_L (low-rank factor Frobenius) default to 1e-4 to discourage blowing up covariance.
- Still possible to set 	rain_q=True to learn full PSD Q; diagonal scaling remains the baseline.
- GRU head contains three FC outputs: fc_delta -> state residual delta, fc_scale -> diagonal scale a_t, fc_cov -> low-rank factor raw (softplus -> U_t); cov_rank=0 disables the third head.

