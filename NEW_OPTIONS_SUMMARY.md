# New Configuration Options Summary

This document lists **all new configuration options** added to the system identification training pipeline, organized by category.

---

## Complete Configuration Schema (New Options)

```json
{
  "data": {
    "val_split_mode": "timestep"  // NEW: "timestep" or "trajectory"
  },
  "model": {
    "friction_param": {             // NEW: Structured models only
      "mode": "softplus_offset_1",  // NEW: 3 modes available
      "k_min": 0.2,                 // NEW: Min friction coefficient
      "k_max": 2.0                  // NEW: Max friction coefficient
    }
  },
  "training": {
    "loss": {                       // NEW: Entire loss block is new
      "one_step_mse": {
        "enabled": true,            // NEW: Can disable
        "weight": 1.0               // NEW: Configurable weight
      },
      "rollout_mse": {              // NEW: Multi-step rollout loss
        "enabled": false,
        "horizon": 10,
        "weight": 1.0,
        "teacher_forcing_prob": 0.0,
        "detach_between_steps": false
      },
      "pose_mse": {                 // NEW: Pose integration loss
        "enabled": false,
        "weight": 0.1,
        "components": ["x", "y"]
      },
      "weights": {                  // NEW: Per-dimension weighting
        "vx": 1.0,
        "vy": 1.0,
        "w": 1.0
      }
    },
    "regularization": {             // NEW: Entire regularization block
      "adapter_identity_weight": 0.0,
      "residual_l2_weight": 0.0,
      "friction_prior_weight": 0.0
    },
    "optim": {                      // NEW: Optimization config
      "grad_clip_norm": 0.0
    }
  }
}
```

---

## Option-by-Option Reference

### 1. DATA OPTIONS

#### `data.val_split_mode`
**Type**: string ("timestep" | "trajectory")
**Default**: "timestep"
**Added in**: v2.0

**What it does**:
- **timestep**: Shuffles all samples from training trajectories, splits by ratio
- **trajectory**: Reserves entire trajectories for validation (no sample leakage)

**When to use**:
- **timestep**: Standard ML practice, maximum data efficiency (default)
- **trajectory**: More realistic evaluation of trajectory-level generalization

**Example**:
```json
"data": {
  "val_split_mode": "trajectory"
}
```

**Used in experiments**: 1, 2, 3, 4, 5

---

### 2. MODEL OPTIONS

#### `model.friction_param.mode`
**Type**: string ("softplus_offset_1" | "exp" | "sigmoid_range")
**Default**: "softplus_offset_1"
**Applies to**: Structured models only
**Added in**: v2.0

**What it does**:
- Controls how friction network output is transformed to friction coefficient k

**Modes**:
1. **"softplus_offset_1"** (default, backward compatible)
   - Formula: `k = 1 + softplus(h)`
   - Range: k ≥ 1 (can only increase friction)
   - Use: When friction should only increase resistance

2. **"exp"** (NEW)
   - Formula: `k = clamp(exp(h), k_min, k_max)`
   - Range: [k_min, k_max]
   - Use: When you need k < 1 to reduce yaw dynamics

3. **"sigmoid_range"** (NEW)
   - Formula: `k = k_min + (k_max - k_min) * sigmoid(h)`
   - Range: (k_min, k_max) smooth bounded
   - Use: Guaranteed bounded output without harsh clamping

**Example**:
```json
"model": {
  "friction_param": {
    "mode": "sigmoid_range",
    "k_min": 0.25,
    "k_max": 3.0
  }
}
```

**Used in experiments**: 3 (friction rescue)

---

#### `model.friction_param.k_min`
**Type**: float
**Default**: 0.2
**Applies to**: "exp" and "sigmoid_range" modes only
**Added in**: v2.0

**What it does**: Minimum friction coefficient value

**Used in experiments**: 3

---

#### `model.friction_param.k_max`
**Type**: float
**Default**: 2.0
**Applies to**: "exp" and "sigmoid_range" modes only
**Added in**: v2.0

**What it does**: Maximum friction coefficient value

**Used in experiments**: 3

---

### 3. LOSS OPTIONS

#### `training.loss.one_step_mse.enabled`
**Type**: boolean
**Default**: true
**Added in**: v2.0

**What it does**: Enables/disables single-step prediction loss (original behavior)

**Example**:
```json
"loss": {
  "one_step_mse": {
    "enabled": true,
    "weight": 1.0
  }
}
```

**Used in experiments**: All (always enabled in experiments)

---

#### `training.loss.one_step_mse.weight`
**Type**: float
**Default**: 1.0
**Added in**: v2.0

**What it does**: Multiplier for one-step MSE loss in total loss

**Used in experiments**: All

---

#### `training.loss.rollout_mse.enabled`
**Type**: boolean
**Default**: false
**Added in**: v2.0

**What it does**: Enables multi-step trajectory rollout loss

**Why useful**: Prevents models that are good at 1-step but terrible at multi-step (compounding errors)

**Example**:
```json
"loss": {
  "rollout_mse": {
    "enabled": true,
    "horizon": 10,
    "weight": 0.5,
    "teacher_forcing_prob": 0.2,
    "detach_between_steps": false
  }
}
```

**Used in experiments**: 2, 3, 4, 5

---

#### `training.loss.rollout_mse.horizon`
**Type**: int
**Default**: 10
**Added in**: v2.0

**What it does**: Number of steps to roll out in sequence

**Typical values**:
- 5-10: Conservative, stable
- 10-20: Standard
- 20+: Aggressive, may be unstable

**Used in experiments**: 2, 3, 4, 5 (all use 10)

---

#### `training.loss.rollout_mse.weight`
**Type**: float
**Default**: 1.0
**Added in**: v2.0

**What it does**: Multiplier for rollout MSE in total loss

**Typical values**:
- 0.3-0.5: Conservative (keeps training anchored to one-step)
- 0.5-1.0: Standard
- 1.0+: Aggressive (rollout dominates)

**Used in experiments**: 2, 3, 4, 5 (all use 0.5)

---

#### `training.loss.rollout_mse.teacher_forcing_prob`
**Type**: float (0.0 to 1.0)
**Default**: 0.0
**Added in**: v2.0

**What it does**: Probability of using ground truth state instead of prediction during rollout

**Why useful**: Prevents early rollout blow-ups during training

**Typical values**:
- 0.0: No teacher forcing (pure prediction)
- 0.1-0.3: Conservative stabilization
- 0.5+: Heavy stabilization (may prevent learning)

**Used in experiments**: 2, 3, 4, 5 (all use 0.2)

---

#### `training.loss.rollout_mse.detach_between_steps`
**Type**: boolean
**Default**: false
**Added in**: v2.0

**What it does**: Whether to stop gradients between rollout steps

**Why useful**:
- true: Treats each step independently (less gradient flow)
- false: Full backprop through time (more gradient flow)

**Used in experiments**: 2, 3, 4, 5 (all use false)

---

#### `training.loss.pose_mse.enabled`
**Type**: boolean
**Default**: false
**Added in**: v2.0

**What it does**: Enables pose integration loss (integrates velocities to poses)

**Requirements**: Only works when `rollout_mse.enabled = true`

**Why useful**: Directly optimizes for position accuracy (what matters for trajectory following)

**Example**:
```json
"loss": {
  "pose_mse": {
    "enabled": true,
    "weight": 0.05,
    "components": ["x", "y"]
  }
}
```

**Used in experiments**: 2, 3, 4, 5

---

#### `training.loss.pose_mse.weight`
**Type**: float
**Default**: 0.1
**Added in**: v2.0

**What it does**: Multiplier for pose MSE in total loss

**Typical values**:
- 0.01-0.05: Conservative (small position correction)
- 0.05-0.1: Standard
- 0.1+: Aggressive (position dominates)

**Used in experiments**: 2, 3, 4, 5 (all use 0.05)

---

#### `training.loss.pose_mse.components`
**Type**: list of strings (["x", "y"] or ["x", "y", "theta"])
**Default**: ["x", "y"]
**Added in**: v2.0

**What it does**: Which pose components to include in loss

**Options**:
- `["x", "y"]`: Position only (standard)
- `["x", "y", "theta"]`: Position + orientation

**Used in experiments**: 2, 3, 4, 5 (all use ["x", "y"])

---

#### `training.loss.weights.vx`, `vy`, `w`
**Type**: float
**Default**: 1.0 for all
**Added in**: v2.0

**What it does**: Per-dimension weighting for velocity components in one-step MSE

**Why useful**: Emphasize accuracy in certain dimensions

**Example**:
```json
"loss": {
  "weights": {
    "vx": 1.0,
    "vy": 1.0,
    "w": 2.0  // Care 2x more about yaw rate
  }
}
```

**Used in experiments**: 2, 3, 4, 5 (all use w=2.0)

---

### 4. REGULARIZATION OPTIONS

#### `training.regularization.adapter_identity_weight`
**Type**: float
**Default**: 0.0
**Added in**: v2.0

**What it does**: Penalizes deviation of effective controls from raw controls

**Formula**: `adapter_identity_weight * MSE(ut_eff, ut_raw)`

**When to use**:
- **Default (0.0)**: Let adapter learn freely (99% of cases)
- **Very low (1e-4 to 1e-3)**: Emergency brake if rollout causes pathological remapping
- **DO NOT use high values**: Defeats the purpose of having an adapter

**Example**:
```json
"regularization": {
  "adapter_identity_weight": 0.0001  // Emergency brake only
}
```

**Used in experiments**: None (kept at 0.0 for all experiments)

---

#### `training.regularization.residual_l2_weight`
**Type**: float
**Default**: 0.0
**Added in**: v2.0

**What it does**: L2 penalty on residual network output

**Formula**: `residual_l2_weight * mean(residual²)`

**Why useful**: Keeps residuals small when `learn_residual=true`

**Typical values**:
- 0.0: No penalty (residual can grow freely)
- 0.001-0.01: Light penalty (residual stays moderate)
- 0.01-0.1: Strong penalty (residual stays tiny)

**Example**:
```json
"regularization": {
  "residual_l2_weight": 0.05
}
```

**Used in experiments**: 5 (adapter + tiny residual uses 0.05)

---

#### `training.regularization.friction_prior_weight`
**Type**: float
**Default**: 0.0
**Added in**: v2.0

**What it does**: Penalizes friction coefficients far from 1.0

**Formula**: `friction_prior_weight * mean((friction_k - 1)²)`

**Why useful**: Prevents extreme friction values when `learn_friction=true`

**Typical values**:
- 0.0: No prior (friction can be anything)
- 0.001-0.01: Light prior (friction can deviate)
- 0.01-0.1: Strong prior (friction stays near 1.0)

**Example**:
```json
"regularization": {
  "friction_prior_weight": 0.01
}
```

**Used in experiments**: 3 (friction rescue uses 0.01)

---

### 5. OPTIMIZATION OPTIONS

#### `training.optim.grad_clip_norm`
**Type**: float
**Default**: 0.0 (disabled)
**Added in**: v2.0

**What it does**: Clips gradients to maximum norm before optimizer step

**Why useful**: Prevents training instability/divergence from gradient explosions (especially with rollout loss)

**Typical values**:
- 0.0: No clipping (default)
- 1.0: Standard clipping
- 5.0-10.0: Lenient clipping

**Example**:
```json
"optim": {
  "grad_clip_norm": 1.0
}
```

**Used in experiments**: 2, 3, 4, 5 (all use 1.0)

---

## Quick Reference: Which Options Enable Which Features?

| Feature | Required Options | Optional Enhancements |
|---------|------------------|----------------------|
| **Trajectory validation** | `val_split_mode: "trajectory"` | - |
| **Rollout loss** | `rollout_mse.enabled: true` | `horizon`, `weight`, `teacher_forcing_prob`, `detach_between_steps` |
| **Pose loss** | `pose_mse.enabled: true`<br>`rollout_mse.enabled: true` | `weight`, `components` |
| **Per-dimension weighting** | `weights: {vx, vy, w}` with non-1.0 values | - |
| **Bounded friction** | `friction_param.mode: "exp"` or `"sigmoid_range"` | `k_min`, `k_max` |
| **Friction prior** | `friction_prior_weight > 0` | - |
| **Residual penalty** | `residual_l2_weight > 0` | - |
| **Adapter stabilizer** | `adapter_identity_weight > 0` | Use very low values |
| **Gradient clipping** | `grad_clip_norm > 0` | - |

---

## Backward Compatibility

**All new options default to OFF/disabled**:
- Old configs work without modification
- Missing config blocks are auto-filled with defaults
- Default behavior = original v1.0 behavior

Auto-fill handled by `mushr_mujoco_sysid/config_utils.py::populate_config_defaults()`

---

## Common Patterns

### Conservative Rollout + Pose (Recommended Starting Point)
```json
"training": {
  "loss": {
    "one_step_mse": { "enabled": true, "weight": 1.0 },
    "rollout_mse": {
      "enabled": true,
      "horizon": 10,
      "weight": 0.5,
      "teacher_forcing_prob": 0.2,
      "detach_between_steps": false
    },
    "pose_mse": {
      "enabled": true,
      "weight": 0.05,
      "components": ["x", "y"]
    },
    "weights": { "vx": 1.0, "vy": 1.0, "w": 2.0 }
  },
  "optim": {
    "grad_clip_norm": 1.0
  }
}
```

### Friction Rescue Pattern
```json
"model": {
  "friction_param": {
    "mode": "sigmoid_range",
    "k_min": 0.25,
    "k_max": 3.0
  }
},
"training": {
  "regularization": {
    "friction_prior_weight": 0.01
  }
}
```

### Tiny Residual Pattern
```json
"model": {
  "learn_residual": true
},
"training": {
  "regularization": {
    "residual_l2_weight": 0.05
  }
}
```

---

## Total New Options Count

- **Data options**: 1 (val_split_mode)
- **Model options**: 3 (friction_param.mode, k_min, k_max)
- **Loss options**: 13 (one_step 2, rollout 5, pose 3, weights 3)
- **Regularization options**: 3 (adapter_identity, residual_l2, friction_prior)
- **Optimization options**: 1 (grad_clip_norm)

**Total**: 21 new configuration options

All default to backward-compatible values!

---

## See Also

- **EXPERIMENT_GUIDE.md**: How to use these options in disciplined experiments
- **EXPERIMENT_SETUP_SUMMARY.md**: Overview of experiment sequence
- **docs/CHANGELOG.md**: Complete v2.0 feature guide
- **docs/training_and_evaluation.md**: Full configuration reference
