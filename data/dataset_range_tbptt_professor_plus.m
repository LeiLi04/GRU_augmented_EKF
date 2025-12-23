function data = dataset_range_tbptt_professor_plus()
% DATASET_RANGE_TBPTT_PROFESSOR_PLUS
% Generate 2D constant-velocity trajectories with range-only measurements to
% four anchors following the professor-specified model.
%
% Outputs are saved to dataset_range_tbptt_professor_plus.mat with fields:
%   dt, steps, N_total, train_ids, val_ids, test_ids
%   A, Q, qc, anchors (2x4), sd_r
%   X: states shaped (N_total, steps, 4) -> [px, py, vx, vy]
%   Z: measurements shaped (N_total, steps, 4) -> ranges to anchors
%   tbptt_blocks: (N_total, 10, 2) inclusive [start, end] indices for L=50
%   process_noise: (N_total, steps-1, 4)
%   meas_noise: (N_total, steps, 4)
%
% The script also prints a summary and plots trajectory #1 with anchors plus
% its four range measurements.

    rng(123); % fixed seed for reproducibility
    cfg = get_config();
    [A, Q] = build_model_matrices(cfg.dt, cfg.qc);

    X = zeros(cfg.N_total, cfg.steps, 4);
    Z = zeros(cfg.N_total, cfg.steps, 4);
    W = zeros(cfg.N_total, cfg.steps - 1, 4);
    R = zeros(cfg.N_total, cfg.steps, 4);

    for n = 1:cfg.N_total
        [x, z, w, r] = generate_one_trajectory(A, Q, cfg.sd_r, cfg.anchors, cfg.steps);
        X(n, :, :) = x;
        Z(n, :, :) = z;
        W(n, :, :) = w;
        R(n, :, :) = r;
    end

    tbptt_single = compute_tbptt_blocks(cfg.steps, cfg.tbptt_L, cfg.tbptt_stride);
    tbptt_blocks = zeros(cfg.N_total, size(tbptt_single, 1), 2);
    for n = 1:cfg.N_total
        tbptt_blocks(n, :, :) = tbptt_single;
    end

    data = struct( ...
        "dt", cfg.dt, ...
        "steps", cfg.steps, ...
        "N_total", cfg.N_total, ...
        "train_ids", 0:14, ...
        "val_ids", 15:17, ...
        "test_ids", 18:19, ...
        "A", A, ...
        "Q", Q, ...
        "qc", cfg.qc, ...
        "anchors", cfg.anchors, ...
        "sd_r", cfg.sd_r, ...
        "X", X, ...
        "Z", Z, ...
        "tbptt_blocks", tbptt_blocks, ...
        "process_noise", W, ...
        "meas_noise", R);

    save_dataset(data, "dataset_range_tbptt_professor_plus.mat");
    print_summary(data);
    plot_trajectory_and_ranges(data, cfg.anchors, 1);
end

function cfg = get_config()
    cfg.seed = 123;
    cfg.dt = 0.01;
    cfg.steps = 500;
    cfg.N_total = 20;
    cfg.qc = 0.1;
    cfg.sd_r = 0.1;
    cfg.tbptt_L = 50;
    cfg.tbptt_stride = 50;
    cfg.anchors = [-1.5, 1.0, -1.0, 1.5; ...
                    0.5, 1.0, -1.0, -0.5];
end

function [A, Q] = build_model_matrices(dt, qc)
    A = [1, 0, dt, 0; ...
         0, 1, 0, dt; ...
         0, 0, 1, 0; ...
         0, 0, 0, 1];
    Q = [qc * dt^3 / 3, 0, qc * dt^2 / 2, 0; ...
         0, qc * dt^3 / 3, 0, qc * dt^2 / 2; ...
         qc * dt^2 / 2, 0, qc * dt, 0; ...
         0, qc * dt^2 / 2, 0, qc * dt];
    Q = 0.5 * (Q + Q'); % enforce symmetry
end

function L = chol_psd(Q)
    try
        L = chol(Q, "lower");
    catch
        L = chol(Q + 1e-12 * eye(size(Q)), "lower");
    end
end

function [states, measurements, process_noise, meas_noise] = generate_one_trajectory(A, Q, sd_r, anchors, steps)
% One trajectory with shapes:
%   states: (steps, 4)
%   measurements: (steps, 4)
%   process_noise: (steps-1, 4)
%   meas_noise: (steps, 4)
    LQ = chol_psd(Q);

    px0 = -0.5 + (0.5 - (-0.5)) * rand();
    py0 = -0.5 + (0.5 - (-0.5)) * rand();
    vx0 = 0.5 + (1.5 - 0.5) * rand();
    vy0 = -0.5 + (0.5 - (-0.5)) * rand();
    x = [px0; py0; vx0; vy0];

    states = zeros(steps, 4);
    measurements = zeros(steps, 4);
    process_noise = zeros(steps - 1, 4);
    meas_noise = zeros(steps, 4);
    states(1, :) = x';

    for k = 1:(steps - 1)
        w = LQ * randn(4, 1);
        process_noise(k, :) = w';
        x = A * x + w;
        states(k + 1, :) = x';
    end

    for k = 1:steps
        px = states(k, 1);
        py = states(k, 2);
        for i = 1:size(anchors, 2)
            noise = sd_r * randn();
            meas_noise(k, i) = noise;
            dx = anchors(1, i) + px;
            dy = anchors(2, i) + py;
            measurements(k, i) = sqrt(dx * dx + dy * dy) + noise;
        end
    end
end

function tbptt_blocks = compute_tbptt_blocks(steps, L, stride)
    num_blocks = floor((steps - L) / stride) + 1;
    tbptt_blocks = zeros(num_blocks, 2);
    for j = 1:num_blocks
        start_idx = (j - 1) * stride;
        tbptt_blocks(j, :) = [start_idx, start_idx + L - 1];
    end
end

function save_dataset(data, path)
    if nargin < 2
        path = "dataset_range_tbptt_professor_plus.mat";
    end
    save(path, "-struct", "data");
    fprintf("Saved dataset to %s\n", path);
end

function plot_trajectory_and_ranges(data, anchors, traj_idx)
    if nargin < 3
        traj_idx = 1;
    end
    states = squeeze(data.X(traj_idx, :, :));
    measurements = squeeze(data.Z(traj_idx, :, :));

    figure;
    subplot(1, 2, 1);
    plot(states(:, 1), states(:, 2), "LineWidth", 1.3); hold on;
    scatter(anchors(1, :), anchors(2, :), 50, "xr", "LineWidth", 1.5);
    xlabel("px"); ylabel("py");
    title(sprintf("Trajectory #%d", traj_idx));
    axis equal; grid on; legend("trajectory", "anchors");

    subplot(1, 2, 2);
    plot(measurements, "LineWidth", 1.1);
    xlabel("time step k"); ylabel("range");
    title("Range measurements");
    grid on; legend("anchor 1", "anchor 2", "anchor 3", "anchor 4");
end

function print_summary(data)
    X = data.X;
    Z = data.Z;
    tbptt_blocks = data.tbptt_blocks;
    fprintf("States X shape: [%d, %d, %d]\n", size(X, 1), size(X, 2), size(X, 3));
    fprintf("Measurements Z shape: [%d, %d, %d]\n", size(Z, 1), size(Z, 2), size(Z, 3));
    fprintf("Mean range: %.4f, Std range: %.4f\n", mean(Z, "all"), std(Z, 0, "all"));
    fprintf("TBPTT blocks shape: [%d, %d, %d]\n", size(tbptt_blocks, 1), size(tbptt_blocks, 2), size(tbptt_blocks, 3));
    disp("First trajectory blocks (start,end):");
    disp(squeeze(tbptt_blocks(1, :, :)));
end
