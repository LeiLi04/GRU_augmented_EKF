下面是更新后的 Codex Prompt。我已将新项目目录名从 `conference_nll/` 改为 **`GRU_augmented_EKF/`**，并同步更新了所有路径、验收条件与说明。

---

## Prompt to Codex

You are working inside an existing research codebase for “GAKF: residual-augmented EKF + GRU”, which currently includes two training signals:

1. innovation negative log-likelihood (NLL)
2. an adversarial WGAN-GP critic on whitened-innovation windows

Goal
Create a new, clean “conference” code project that keeps only the innovation NLL training pipeline. Remove the GAN critic and all WGAN-GP related code paths. The new project must live in a new folder and be self-contained for a fresh GitHub repository.

Hard constraints

* Do not modify or break the original project. Treat the original as read-only.
* All new work must be placed under a new top-level directory named `GRU_augmented_EKF/` in the current workspace.
* The new project must run end-to-end training and evaluation using only innovation NLL.
* The new project must not import, reference, or require any GAN, WGAN-GP, critic, gradient penalty, or innovation-window discriminator code.
* Keep the EKF generator with residual GRU and covariance scaling.
* Keep warm-start Q0 estimation if it exists in the original code, since it is part of the NLL-only pipeline. If warm-start is tightly coupled to GAN code, refactor it to be independent.
* Keep evaluation and diagnostics code if it is already implemented and not GAN-dependent. This includes NMSE, NIS, NEES, and Ljung–Box. These diagnostics are evaluation-only, not training losses.
* Training loss in the new project is NLL only:

  * `L_total = L_innov`
  * Remove `L_adv`, `L_nis`, `lambda_adv`, `lambda_nis`, `n_critic`, `L_window`, discriminator learning rate, gradient penalty, and all critic update loops.

What to produce inside `GRU_augmented_EKF/`

1. A clean repository layout, for example:

* `GRU_augmented_EKF/`

  * `README.md`
  * `LICENSE` (use MIT unless an existing license must be preserved)
  * `pyproject.toml` or `requirements.txt`
  * `src/`

    * `gru_augmented_ekf/`

      * `__init__.py`
      * `models/`

        * `ekf.py` (EKF recursion, Joseph update, covariance symmetrization if used)
        * `residual_gru.py` (GRU + heads for delta and alpha)
        * `gakf.py` (wrap generator forward pass; consider naming `gru_augmented_ekf.py`)
      * `data/`

        * `lorenz.py` (dataset generation, discretization, mismatch options if present)
        * `splits.py` (train, val, warm-start, long-horizon test)
      * `losses/`

        * `innov_nll.py` (Cholesky-based NLL computation)
      * `metrics/`

        * `nmse.py`
        * `nis_nees.py`
        * `ljung_box.py`
      * `train/`

        * `trainer.py`
        * `warmstart_q0.py`
      * `utils/`

        * `seed.py`
        * `linear_algebra.py` (Cholesky solve, logdet helpers)
  * `scripts/`

    * `train.py`
    * `eval_short.py`
    * `eval_long.py`
  * `configs/`

    * `default.yaml`
  * `.gitignore`

2. The training entrypoint

* `scripts/train.py` must run NLL-only training.
* It must support TBPTT if the original code uses TBPTT. Keep the current default TBPTT truncation length that the original code uses (commonly 50 in our experiments) unless the repo config says otherwise.
* Ensure it logs and saves:

  * model checkpoints
  * training and validation NLL curves
  * optional diagnostic metrics on validation

3. The evaluation entrypoints

* `scripts/eval_short.py` evaluates on short sequences.
* `scripts/eval_long.py` evaluates on long-horizon sequences.
* These scripts must not require any GAN components.
* If the original repo has baselines (EKF, UKF, DANSE, KalmanNet) that are easy to keep, you may keep them, but only if it does not pull in GAN code and does not bloat the conference version.
* At minimum, evaluation must work for:

  * EKF nominal
  * GRU-augmented EKF (NLL-only)

4. Configuration cleanup

* Create `configs/default.yaml` that contains only relevant hyperparameters:

  * data generation parameters
  * noise levels and sweep if present
  * train/val/warm splits
  * optimizer and learning rate
  * TBPTT length and stride
  * GRU hidden size and depth
  * alpha min and alpha max
  * residual clamp constant c
* Remove every GAN-related config field.

5. Documentation
   Write `README.md` with:

* one-paragraph description: “Residual-augmented EKF trained unsupervised by innovation NLL”
* how to install
* how to run training
* how to run evaluation short and long
* expected outputs and where logs and checkpoints are saved
* a minimal reproducibility section: seed handling, default dataset sizes

6. GitHub readiness
   Inside `GRU_augmented_EKF/`, add:

* `.gitignore` appropriate for Python
* a clean dependency spec
* optional `Makefile` or `justfile` if helpful

Implementation steps you must follow
A) First, scan the original repository and identify all files related to:

* critic network, discriminator, TCN critic
* WGAN-GP losses
* gradient penalty
* innovation window sampling and reference Gaussian windows
* any training loop that alternates critic and generator updates

B) Copy only the needed code into `GRU_augmented_EKF/` and refactor imports so everything is internal to the new project.

C) Remove GAN code in one of these ways:

* Preferred: do not copy GAN code at all into the new project
* If unavoidable due to tight coupling: copy but fully delete the GAN portions and rewrite the dependent parts so there are zero imports or calls to GAN functions and zero unused config fields

D) Ensure NLL-only correctness

* Implement innovation predictive distribution:

  * `y_t | Y_1:t-1 ~ N(yhat_t|t-1, S_t|t-1)`
* Implement per-step loss:

  * `dy_t^T S^{-1} dy_t + logdet(S)`
* Use Cholesky solves and Cholesky logdet. Avoid explicit matrix inverses.
* Ensure covariance matrices remain symmetric and PSD:

  * Joseph update if used
  * explicit symmetrization step if used

E) Provide a short “diff style” summary at the end
After you finish, output a concise summary:

* created files and directories
* removed modules
* how training loss changed
* how to run the new project

Acceptance tests

* Running `python scripts/train.py --config configs/default.yaml` inside `GRU_augmented_EKF/` must start training without errors.
* Running `python scripts/eval_long.py --checkpoint <path>` must run without importing any GAN module.
* A grep for keywords `wgan`, `critic`, `discriminator`, `gradient_penalty`, `gp`, `gan` inside `GRU_augmented_EKF/` should return no functional code dependencies. If any occurrence remains, it must be only in historical notes inside README, not in Python imports or execution paths.

Now implement all of the above.

---

如果你还想更贴合你的 conference 版本叙事（比如统一命名成 “GRU-Augmented EKF (NLL-only)” 而不出现 GAKF），我也建议你在 prompt 里再加一条：**所有代码与 README 中避免使用 “GAN / adversarial / critic” 术语**，只用 “NLL-only” 描述。这会让最终 repo 更干净、更容易投稿。
