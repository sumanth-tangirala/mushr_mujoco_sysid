# Data and Dataloading

## Data sources
- Directory: `sysid_trajs/`.
- Per trajectory files read via `load_traj_data` in `dataloader.py`:
  - `traj_vels_<id>.txt`: columns `[t, x, y, theta, xdot, ydot, thetadot]`.
  - `plan_<id>.txt`: commanded `[steering_u, velocity_u]` (constant per traj).
  - `circle_<id>.txt` or `cirlce_<id>.txt`: radius/error (used only for filtering validity; not required for new training).

## Trajectory selection
- `split_train_eval_ids(num_eval_trajectories)`: uses `shuffled_indices.txt` to pick train vs held-out trajectory IDs.
  - Train set: all IDs except the last `num_eval_trajectories`.
  - Held-out set: last `num_eval_trajectories`.

## Sample construction (per timestep)
- Implemented in `train.py::_collect_samples_for_id`:
  - For each `t` in a trajectory:
    - `xd0 = [vx_t, vy_t, w_t]` from `traj_vels`.
    - `ut = [velocity_cmd, steering_cmd]` from `plan` (constant across the trajectory).
    - `xd1 = [vx_{t+1}, vy_{t+1}, w_{t+1}]`.
  - Stores per-trajectory metadata: `{id, xd (T x 3), ut (T x 2)}` for rollout evaluation.

## Dataset assembly
- Training samples come from train trajectories only (timestep-level).
- Held-out trajectories remain untouched for trajectory-level eval.
- Within training samples, `split_indices` performs a shuffle + split into train/val by ratio `val_ratio` (config).

## Standardization
- `Standardizer` in `utils.py`:
  - `fit` computes mean/std over training inputs and targets separately.
  - `transform` / `inverse` for numpy; `transform_torch` / `inverse_torch` for tensors.
  - Applied to inputs `[vx, vy, w, vel_cmd, steer_cmd]` and targets `[vx_next, vy_next, w_next]`.

## PyTorch datasets/loaders
- `TimestepDataset`: stores standardized tensors for inputs/targets.
- DataLoaders in `train.py`:
  - Train loader: shuffled batches from train split.
  - Val loader: deterministic batches from val split.

## Time step
- `dt` is taken from the per-sample `dt` column in the sysid logs and passed to models during forward; there is no separate `dt` parameter in the configs.
