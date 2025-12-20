# Training and Evaluation

## Configuration
- Primary config: JSON file passed to `train.py --config`.
- Key fields:
  - `seed`: RNG seed for NumPy and PyTorch.
  - `data`: `data_dir`, `num_eval_trajectories`, `val_ratio`, `val_split_mode`.
  - `model`:
    - `type`: `"structured"` or `"direct"`.
    - `learn_friction`, `learn_residual`, `friction_use_dt/delta/vy` (structured).
    - `hidden_dims` (direct).
    - `control_adapter` flags for optional features and output scales.
    - `friction_param` (structured): friction parameterization options.
  - `training`: `batch_size`, `epochs`, `lr`, `weight_decay`, `device`, `run_root`, `run_name`, `ckpt_name`, `eval_every`.
    - `loss`: Loss configuration with one-step, rollout, and pose MSE options.
    - `regularization`: Regularization term weights.
    - `optim`: Optimization settings like gradient clipping.

## New Configuration Options (v2.0)

All new options default to values that reproduce the original training behavior.

### 1. Loss Configuration (`training.loss`)

```json
{
  "training": {
    "loss": {
      "one_step_mse": { "enabled": true, "weight": 1.0 },
      "rollout_mse": {
        "enabled": false,
        "horizon": 10,
        "weight": 1.0,
        "teacher_forcing_prob": 0.0,
        "detach_between_steps": false
      },
      "pose_mse": {
        "enabled": false,
        "weight": 0.1,
        "components": ["x", "y"]
      },
      "weights": { "vx": 1.0, "vy": 1.0, "w": 1.0 }
    }
  }
}
```

- **one_step_mse**: Single-step prediction loss (original behavior).
  - `enabled`: Whether to compute this loss (default: true).
  - `weight`: Multiplier for this loss term (default: 1.0).

- **rollout_mse**: Multi-step rollout loss on short horizons.
  - `enabled`: Whether to enable rollout loss (default: false).
  - `horizon`: Number of steps in each rollout (default: 10).
  - `weight`: Multiplier for rollout loss (default: 1.0).
  - `teacher_forcing_prob`: Probability of using ground truth state instead of prediction (default: 0.0).
  - `detach_between_steps`: Whether to detach gradients between rollout steps (default: false).

- **pose_mse**: Pose integration loss during rollout.
  - `enabled`: Whether to compute pose loss (default: false). Requires `rollout_mse.enabled=true`.
  - `weight`: Multiplier for pose loss (default: 0.1).
  - `components`: Which pose components to include: `["x", "y"]` or `["x", "y", "theta"]`.

- **weights**: Per-dimension weighting for velocity components in one-step MSE.
  - `vx`, `vy`, `w`: Weights for each velocity dimension (all default: 1.0).
  - When all weights are 1.0, standard MSE is used (no per-dimension weighting).

### 2. Regularization (`training.regularization`)

```json
{
  "training": {
    "regularization": {
      "adapter_identity_weight": 0.0,
      "residual_l2_weight": 0.0,
      "friction_prior_weight": 0.0
    }
  }
}
```

- **adapter_identity_weight**: Optional stabilizer for control adapter.
  - Loss term: `MSE(ut_eff, ut_raw)`.
  - **Default: 0.0 (disabled)** - let adapter learn freely.
  - **Emergency use only**: Set to very low values (1e-4 to 1e-3) if rollout loss causes pathological control remapping.
  - Higher values defeat the purpose of having an adapter.

- **residual_l2_weight**: L2 penalty on residual network output.
  - Loss term: `mean(residual^2)`.
  - Encourages small residual corrections.

- **friction_prior_weight**: Penalize friction coefficient deviation from 1.0.
  - Loss term: `mean((friction_k - 1)^2)`.
  - Encourages friction values close to the baseline.

### 3. Optimization (`training.optim`)

```json
{
  "training": {
    "optim": {
      "grad_clip_norm": 0.0
    }
  }
}
```

- **grad_clip_norm**: Maximum gradient norm for clipping (default: 0.0 = disabled).
  - When > 0, applies `torch.nn.utils.clip_grad_norm_` before optimizer step.

### 4. Validation Split Mode (`data.val_split_mode`)

```json
{
  "data": {
    "val_split_mode": "timestep"
  }
}
```

- **"timestep"** (default): Current behavior. Shuffles all samples from training trajectories and splits by `val_ratio` at the sample level.
- **"trajectory"**: Reserves a fraction of training trajectory IDs for validation. All samples from validation trajectories go to val set (no leakage between train/val at trajectory level).

### 5. Friction Parameterization (`model.friction_param`, structured model only)

```json
{
  "model": {
    "friction_param": {
      "mode": "softplus_offset_1",
      "k_min": 0.2,
      "k_max": 2.0
    }
  }
}
```

Supported modes:
- **"softplus_offset_1"** (default): `k = 1 + softplus(h)`. Original behavior, k >= 1.
- **"exp"**: `k = exp(h)`, clamped to `[k_min, k_max]`. Allows k < 1.
- **"sigmoid_range"**: `k = k_min + (k_max - k_min) * sigmoid(h)`. Smooth bounded output in `[k_min, k_max]`.

The `k_min` and `k_max` parameters are only used for "exp" and "sigmoid_range" modes.

## Training loop (per timestep)
- File: `scripts/train.py`.
- Steps per epoch:
  1) Iterate DataLoader batches (train split).
  2) Inputs are standardized; split into `xd0 = xb[:, :3]`, `ut = xb[:, 3:5]`, `dt = xb[:, 5]`.
  3) Forward pass with optional auxiliary output for regularization.
  4) Compute loss components:
     - One-step MSE (optional per-dimension weighting)
     - Rollout MSE (if enabled)
     - Pose MSE (if enabled)
     - Regularization terms (if weights > 0)
  5) Backprop with optional gradient clipping + Adam optimizer update.
  6) Track running average for all loss components.
- Validation each epoch on val split (no grad).
- Checkpoint best validation loss to `run_dir/ckpt_name`.

## Evaluation
- Per-timestep validation: identical to training loss computation, on val loader.
- Held-out trajectory rollouts (every `eval_every` epochs):
  - Uses untouched trajectories (IDs reserved by `num_eval_trajectories`).
  - For each step: standardize features, run model, de-standardize prediction, compare to true `xd_{t+1}`.
  - Reports per-trajectory MSE and averages.

## Data splits
- Trajectory split: train vs held-out by `split_train_eval_ids` using `shuffled_indices.txt`.
- Within-train split:
  - **timestep mode**: shuffle samples then split by `val_ratio` for train/val loaders.
  - **trajectory mode**: reserve `val_ratio` fraction of trajectory IDs for validation.
- Held-out trajectories are never used for fitting; they are only used for trajectory-level evaluation.

## Model choices and when to use them
- StructuredDynamicsModel:
  - Use when you want to keep kinematic constraints and only learn corrections (control effectiveness, friction, small residuals).
  - Enable `control_adapter` to learn steering/acc effectiveness from state-dependent features.
  - Enable `learn_friction` to adapt yaw dynamics; enable `learn_residual` for small unmodeled effects.
  - Use `friction_param.mode="sigmoid_range"` if you need k < 1 (e.g., to reduce yaw update).
- DirectDynamicsModel:
  - Use when plant fidelity is insufficient or rapid prototyping is needed; still can adapt controls with the adapter.

## Outputs and artifacts
- Best checkpoint: saved state dict + config at `run_dir/ckpt_name`.
- Standardizers: saved to `run_dir/standardizers.json`.
- Losses CSV: `run_dir/losses.csv` with columns for all loss components.
- Console logs: per-epoch train/val loss with all components; periodic held-out trajectory MSE average.
- Loss plots: `loss_train.png`, `loss_val.png`, `loss_eval_vel.png`, `loss_eval_pos.png`.

## Running
```
python scripts/train.py --config configs/config_structured.json
```

## Example Configurations

### Friction-only with new k < 1 capability
```json
{
  "model": {
    "type": "structured",
    "learn_friction": true,
    "learn_residual": false,
    "friction_param": {
      "mode": "sigmoid_range",
      "k_min": 0.2,
      "k_max": 2.0
    },
    "control_adapter": { "enabled": true }
  },
  "training": {
    "loss": {
      "one_step_mse": { "enabled": true, "weight": 1.0 }
    },
    "regularization": {
      "friction_prior_weight": 0.01
    }
  }
}
```

### Adapter-only
```json
{
  "model": {
    "type": "structured",
    "learn_friction": false,
    "learn_residual": false,
    "control_adapter": { "enabled": true }
  },
  "training": {
    "loss": {
      "one_step_mse": { "enabled": true, "weight": 1.0 }
    }
  }
}
```

### Full model with rollout loss
```json
{
  "model": {
    "type": "structured",
    "learn_friction": true,
    "learn_residual": true,
    "control_adapter": { "enabled": true }
  },
  "training": {
    "loss": {
      "one_step_mse": { "enabled": true, "weight": 1.0 },
      "rollout_mse": {
        "enabled": true,
        "horizon": 10,
        "weight": 0.5,
        "teacher_forcing_prob": 0.0,
        "detach_between_steps": false
      },
      "pose_mse": {
        "enabled": true,
        "weight": 0.1,
        "components": ["x", "y"]
      }
    },
    "optim": {
      "grad_clip_norm": 1.0
    }
  },
  "data": {
    "val_split_mode": "trajectory"
  }
}
```
