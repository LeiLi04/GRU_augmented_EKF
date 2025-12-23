Act as an expert ML/estimation engineer. Implement a dataset generator for 2D target tracking with a constant-velocity state model and range-only measurements to multiple anchors. Use MATLAB (preferred) and also provide a Python version if easy. Follow these specs exactly:

1) Random seed
- Fix RNG seed to 123 for reproducibility (MATLAB: rng(123); Python: np.random.seed(123)).

2) Time and dataset size
- Sampling period dt = 0.01
- Number of time steps per trajectory: steps = 500
- Number of trajectories: N_total = 20
- Split by trajectory (NOT by windows): train 75% (=15 traj), val 15% (=3 traj), test 10% (=2 traj).
- TBPTT configuration for later training (streaming TBPTT):
  - window length L = 50
  - stride = 50 (non-overlapping contiguous blocks)
  - For each trajectory, precompute TBPTT blocks indices: block j covers t = (j-1)*50 .. j*50-1, j=1..10.

3) State model
- State x_k = [px, py, vx, vy]^T
- Use constant velocity discretization:
  A = [[1,0,dt,0],
       [0,1,0,dt],
       [0,0,1,0],
       [0,0,0,1]]
- Process noise covariance Q from discretized Wiener velocity model with qc=0.1:
  Q = [[qc*dt^3/3, 0, qc*dt^2/2, 0],
       [0, qc*dt^3/3, 0, qc*dt^2/2],
       [qc*dt^2/2, 0, qc*dt, 0],
       [0, qc*dt^2/2, 0, qc*dt]]
- Generate ground-truth trajectories using the SAME linear model:
  x_{k+1} = A x_k + w_k, w_k ~ N(0,Q)
  (Cholesky sampling; ensure symmetry of Q)

4) Anchors (4 anchors)
- Use fixed anchor positions in 2D (units consistent with state). Use:
  S1 = [-1.5; 0.5]
  S2 = [ 1.0; 1.0]
  S3 = [-1.0;-1.0]
  S4 = [ 1.5;-0.5]
  Stack as S = [S1 S2 S3 S4] with shape (2,4), where S(1,i)=s_x^i and S(2,i)=s_y^i.

5) Range-only observation model (MUST MATCH PROFESSOR FORM EXACTLY)
- For each time step k and anchor i, measurement:
  z_k^i = sqrt((s_x^i + px_k)^2 + (s_y^i + py_k)^2) + r_k^i
  r_k^i ~ N(0, sd_r^2), independent across i,k.
- Choose sd_r = 0.1 (range noise standard deviation).
- Output measurements Z for each trajectory with shape (4, steps).

6) Initial state distribution across trajectories
- For each trajectory n:
  - Sample initial position px0, py0 from Uniform(-0.5, 0.5)
  - Sample initial velocity vx0, vy0 from Uniform(0.5, 1.5) and Uniform(-0.5, 0.5) respectively.
  - Set x0 = [px0, py0, vx0, vy0]^T

7) Outputs / file format
- Save a single dataset file:
  - MATLAB: dataset_range_tbptt_professor_plus.mat
  - Python: dataset_range_tbptt_professor_plus.npz
- Store:
  - dt, steps, N_total, split indices (train_ids, val_ids, test_ids)
  - A, Q, qc
  - anchors S (2x4)
  - sd_r
  - X: ground truth states with shape (N_total, steps, 4) OR (4, steps, N_total). Choose one and document clearly.
  - Z: measurements with shape (N_total, steps, 4) OR (4, steps, N_total). Choose one and document clearly.
  - tbptt_blocks: for each trajectory, a list/array of block start/end indices for L=50, stride=50 (10 blocks).
  - Optional: w process noise samples and r measurement noise samples for debugging.

8) Sanity checks / plots
- Implement a quick visualization for trajectory #1:
  - Plot true (px,py) path
  - Plot anchors
- Plot the 4 range measurements over time for trajectory #1 (4 curves).
- Print summary: shapes of X and Z, mean/std of ranges, and confirm number of TBPTT blocks is 10.

9) Code quality
- Write clean, self-contained code with functions:
  - generate_one_trajectory(...)
  - generate_dataset(...)
  - save_dataset(...)
- Add comments explaining dimensions.
- Ensure Q is positive semidefinite and cholesky works; if not, add small jitter 1e-12 to diagonal.

Deliver the final MATLAB script and (if possible) a Python script producing the same dataset structure and plots.
