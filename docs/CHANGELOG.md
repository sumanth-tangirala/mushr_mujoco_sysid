# Changelog - MuSHR MuJoCo System Identification Training Pipeline

## Version 2.0 - Advanced Training Features

This major update introduces several significant improvements to the system identification training pipeline while maintaining full backward compatibility with existing configurations.

### Overview

The training pipeline has been enhanced with **9 new toggleable features**, each defaulting to OFF to preserve existing behavior. All features are controlled through JSON configuration files and require no code changes to enable/disable.

---

## Key Features

### 1. **Multi-Component Loss Functions**

The original single-step MSE loss has been extended with optional additional loss components:

#### One-Step MSE Loss (Original Behavior)
- **What**: Predict next velocity state from current state and controls
- **When to use**: Standard supervised learning on single timesteps
- **Config**: `training.loss.one_step_mse.enabled` (default: `true`)
- **Enhancement**: Now supports per-dimension weighting for vx, vy, w components

#### Rollout Loss (NEW)
- **What**: Multi-step trajectory prediction with recurrent rollout
- **When to use**: When you want the model to be accurate over longer horizons, not just single steps
- **Benefits**:
  - Reduces compounding errors in sequential predictions
  - Learns dynamics that are coherent over time
  - Prevents "one-step-optimal-but-unstable" solutions
- **Config**: `training.loss.rollout_mse.enabled` (default: `false`)
- **Parameters**:
  - `horizon`: Number of steps to roll out (default: 10)
  - `teacher_forcing_prob`: Probability of using ground truth vs prediction (default: 0.0)
  - `detach_between_steps`: Whether to stop gradients between steps (default: `false`)

#### Pose Loss (NEW)
- **What**: Integrate predicted velocities to poses and compare to ground truth poses
- **When to use**: When position accuracy matters more than velocity accuracy
- **Benefits**:
  - Directly optimizes for what matters in trajectory following
  - Catches systematic biases that don't show up in velocity MSE
  - Physics-based: uses proper SE(2) integration
- **Config**: `training.loss.pose_mse.enabled` (default: `false`)
- **Requirements**: Only works when `rollout_mse.enabled=true`
- **Parameters**:
  - `components`: Which pose elements to include: `["x", "y"]` or `["x", "y", "theta"]`
  - **Important**: Uses unstandardized velocities for physically accurate integration

#### Per-Dimension Velocity Weighting (NEW)
- **What**: Apply different weights to vx, vy, w components in one-step MSE
- **When to use**: When some velocity components are more important than others
- **Example**: Weight yaw rate (w) higher if turning accuracy is critical
- **Config**: `training.loss.weights` (default: all 1.0)

---

### 2. **Regularization Terms**

Optional penalties to prevent overfitting and encourage physically plausible solutions:

#### Residual L2 Penalty (NEW)
- **What**: Penalize large residual network outputs
- **When to use**: When using `learn_residual=true` to prevent residuals from dominating
- **Benefits**: Keeps residuals small, ensuring the plant model stays primary
- **Config**: `training.regularization.residual_l2_weight` (default: 0.0)
- **Formula**: `mean(residual²)`

#### Friction Prior (NEW)
- **What**: Penalize friction coefficients far from 1.0
- **When to use**: When using `learn_friction=true` to prevent extreme friction values
- **Benefits**: Encourages physically reasonable friction while allowing learning
- **Config**: `training.regularization.friction_prior_weight` (default: 0.0)
- **Formula**: `mean((friction_k - 1)²)`

**Note**: The `adapter_identity_weight` regularization has been removed as it unnecessarily constrains the control adapter from learning optimal mappings.

---

### 3. **Advanced Friction Parameterization**

The friction network can now use different output transformations:

#### Softplus Offset (Original)
- **Mode**: `"softplus_offset_1"`
- **Formula**: `k = 1 + softplus(h)`
- **Range**: k ≥ 1 (can only increase yaw dynamics)
- **When to use**: When friction should only increase resistance

#### Exponential with Clamping (NEW)
- **Mode**: `"exp"`
- **Formula**: `k = clamp(exp(h), k_min, k_max)`
- **Range**: `[k_min, k_max]`
- **When to use**: When you need k < 1 to reduce yaw dynamics
- **Benefits**: Unbounded network output, clamped to safe range

#### Sigmoid Range (NEW)
- **Mode**: `"sigmoid_range"`
- **Formula**: `k = k_min + (k_max - k_min) * sigmoid(h)`
- **Range**: `(k_min, k_max)` (smooth, always bounded)
- **When to use**: When you want guaranteed bounded output without harsh clamping
- **Benefits**: Smooth gradients throughout training

**Config**: `model.friction_param.mode`, `k_min`, `k_max`

---

### 4. **Validation Split Modes**

Two strategies for creating training/validation splits:

#### Timestep Split (Original)
- **Mode**: `"timestep"`
- **How**: Shuffle all samples from training trajectories, then split by ratio
- **When to use**: Standard ML practice, maximum data efficiency
- **Caveat**: Same trajectory appears in both train and val (at different timesteps)

#### Trajectory Split (NEW)
- **Mode**: `"trajectory"`
- **How**: Reserve entire trajectories for validation (no sample leakage)
- **When to use**: When you want to test generalization to unseen trajectories
- **Benefits**: More realistic evaluation of trajectory-level performance
- **Config**: `data.val_split_mode` (default: `"timestep"`)

---

### 5. **Gradient Clipping**

Optional gradient norm clipping for training stability:

- **What**: Clip gradients to maximum norm before optimizer step
- **When to use**: When training is unstable or diverging
- **Config**: `training.optim.grad_clip_norm` (default: 0.0 = disabled)
- **Typical values**: 1.0 to 10.0

---

## Implementation Details

### Backward Compatibility

All new features are **opt-in** through configuration:
- Default config values reproduce the original training behavior exactly
- Missing config blocks are automatically filled with backward-compatible defaults
- Existing config files work without modification

### Auxiliary Model Outputs

Both `StructuredDynamicsModel` and `DirectDynamicsModel` now support:
```python
xd1_pred, aux = model(xd0, ut, dt, return_aux=True)
```

Where `aux` contains:
- `"ut_eff"`: Effective controls (if control adapter enabled)
- `"residual"`: Residual correction (if `learn_residual=true`)
- `"friction_k"`: Friction coefficient (if `learn_friction=true`)

These auxiliaries enable regularization without breaking the default `return_aux=False` path.

### Snippet Dataset

A new `SnippetDataset` class provides contiguous trajectory windows for rollout loss:
- Automatically extracts all valid snippets of length `horizon+1` from training trajectories
- Handles standardization consistently with one-step dataset
- Returns batches with `xd`, `ut`, `dt`, and `pose` sequences

### Pose Integration

The pose loss uses **unstandardized velocities** for integration:
1. Model outputs standardized velocity predictions
2. Predictions are unstandardized using `target_std.inverse()`
3. Raw velocities are integrated via `plant.integrate_SE2()` (SE(2) manifold integration)
4. Predicted poses are compared to ground truth poses

This ensures physically accurate pose computation and prevents standardization artifacts.

---

## Configuration Schema

### Complete Example

```json
{
  "seed": 42,
  "data": {
    "data_dir": "data/sysid_trajs",
    "num_eval_trajectories": 50,
    "val_ratio": 0.1,
    "val_split_mode": "trajectory"
  },
  "model": {
    "type": "structured",
    "learn_friction": true,
    "learn_residual": true,
    "friction_param": {
      "mode": "sigmoid_range",
      "k_min": 0.2,
      "k_max": 2.0
    },
    "control_adapter": {
      "enabled": true,
      "include_speed_mag_steer": true,
      "include_speed_sign_steer": true
    }
  },
  "training": {
    "batch_size": 256,
    "epochs": 1000,
    "lr": 0.001,
    "weight_decay": 0.0,
    "device": "cuda",
    "loss": {
      "one_step_mse": {
        "enabled": true,
        "weight": 1.0
      },
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
      },
      "weights": {
        "vx": 1.0,
        "vy": 1.0,
        "w": 2.0
      }
    },
    "regularization": {
      "residual_l2_weight": 0.01,
      "friction_prior_weight": 0.01
    },
    "optim": {
      "grad_clip_norm": 1.0
    }
  }
}
```

---

## Use Cases & Recommendations

### Friction-Only Model
**Goal**: Learn velocity-dependent friction while keeping adapter and no residuals

```json
{
  "model": {
    "learn_friction": true,
    "learn_residual": false,
    "friction_param": {
      "mode": "sigmoid_range",
      "k_min": 0.2,
      "k_max": 2.0
    },
    "control_adapter": {"enabled": false}
  },
  "training": {
    "regularization": {
      "friction_prior_weight": 0.01
    }
  }
}
```

**Why**:
- Friction prior prevents extreme values
- Sigmoid range allows k < 1 if needed to reduce yaw
- No adapter keeps controls simple

### Adapter-Only Model
**Goal**: Learn control mapping without modifying plant dynamics

```json
{
  "model": {
    "learn_friction": false,
    "learn_residual": false,
    "control_adapter": {"enabled": true}
  },
  "training": {
    "loss": {
      "one_step_mse": {"enabled": true, "weight": 1.0}
    }
  }
}
```

**Why**:
- Pure control adaptation
- No regularization needed (adapter learns freely)
- Plant model stays unchanged

### Full Model with Trajectory Optimization
**Goal**: Best trajectory following performance

```json
{
  "model": {
    "learn_friction": true,
    "learn_residual": true,
    "friction_param": {"mode": "sigmoid_range"},
    "control_adapter": {"enabled": true}
  },
  "training": {
    "loss": {
      "one_step_mse": {"enabled": true, "weight": 1.0},
      "rollout_mse": {"enabled": true, "horizon": 10, "weight": 0.5},
      "pose_mse": {"enabled": true, "weight": 0.1}
    },
    "regularization": {
      "residual_l2_weight": 0.01,
      "friction_prior_weight": 0.01
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

**Why**:
- Rollout loss improves multi-step accuracy
- Pose loss directly optimizes trajectory following
- Trajectory split mode gives realistic validation
- Regularization prevents overfitting
- All model components work together

### Aggressive Yaw Correction
**Goal**: Fix systematic yaw rate errors

```json
{
  "model": {
    "learn_friction": true,
    "friction_param": {
      "mode": "sigmoid_range",
      "k_min": 0.1,
      "k_max": 5.0
    }
  },
  "training": {
    "loss": {
      "weights": {"vx": 1.0, "vy": 1.0, "w": 5.0}
    }
  }
}
```

**Why**:
- Wide k range allows aggressive correction
- High w weight prioritizes yaw accuracy
- Sigmoid mode prevents instability

---

## Logging and Monitoring

### Console Output

Training now logs all active loss components:
```
Epoch 001 | train 0.023456 | val 0.024567 | 1step 0.020000 | roll 0.003000 | pose 0.000456 | residual 0.000123 | friction_ 0.000234
```

### CSV Output

All metrics saved to `run_dir/losses.csv`:
- `epoch`
- `train_loss` (total)
- `val_loss`
- `train_one_step`
- `train_rollout`
- `train_pose`
- `eval_vel_mse` (held-out trajectories)
- `eval_pos_mse` (held-out trajectories)

### Config Summary

At training start, prints:
```
============================================================
Training Configuration Summary
============================================================
Validation split mode: trajectory
One-step MSE: enabled=True, weight=1.0
Rollout MSE: enabled=True, weight=0.5
Pose MSE: enabled=True, weight=0.1
Velocity weights: {'vx': 1.0, 'vy': 1.0, 'w': 2.0}
Regularization: residual_l2=0.01, friction_prior=0.01
Grad clip norm: 1.0
Friction param: mode=sigmoid_range, k_min=0.2, k_max=2.0
============================================================
```

---

## Migration Guide

### From v1.0 to v2.0

**No changes required** - v2.0 is fully backward compatible.

To enable new features:

1. **Add rollout loss**: Add `training.loss.rollout_mse` block to config
2. **Add pose loss**: Add `training.loss.pose_mse` block (requires rollout enabled)
3. **Change friction mode**: Update `model.friction_param.mode`
4. **Add regularization**: Add `training.regularization` with desired weights
5. **Enable trajectory split**: Set `data.val_split_mode: "trajectory"`
6. **Add gradient clipping**: Set `training.optim.grad_clip_norm > 0`

---

## Technical Notes

### Performance Impact

- **Rollout loss**: ~2-3x slower per epoch (depends on horizon and batch size)
- **Pose loss**: Negligible additional cost when rollout enabled
- **Regularization**: Negligible cost
- **Trajectory split**: No runtime impact, may reduce validation set size

### Memory Usage

- **Snippet dataset**: Stores ~H×N×D additional samples (H=horizon, N=num_trajectories, D=data_dim)
- **Rollout computation**: Requires storing H intermediate states per batch
- Typical increase: 10-20% for horizon=10

### Numerical Stability

- Pose integration uses SE(2) manifold operations (numerically stable)
- Gradient clipping prevents explosion
- Friction clamping prevents NaN/Inf values
- Teacher forcing can improve rollout stability

---

## Future Work

Potential extensions for v3.0:
- [ ] Action noise injection during rollout
- [ ] Adversarial trajectory samples
- [ ] Multi-horizon rollout (mix of short and long)
- [ ] Adaptive loss weighting
- [ ] Per-trajectory difficulty weighting
- [ ] Online trajectory evaluation during training

---

## Credits

System ID pipeline v2.0
- Multi-step rollout loss with SE(2) pose integration
- Flexible friction parameterization
- Trajectory-aware validation splitting
- Comprehensive regularization framework

For questions or issues, see `docs/training_and_evaluation.md`.
