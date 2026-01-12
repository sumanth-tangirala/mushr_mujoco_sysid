# Git Commit Structure

This document describes the logical organization of commits in this repository.

## Commit History Overview

The repository is organized into **10 logical commits**, each representing a cohesive feature or component:

```
f7d0f7b (HEAD -> master) Add comprehensive documentation
f239309 Add example configuration files
4678d30 Add comprehensive training pipeline with multi-component losses
67f173e Add advanced dataset infrastructure
048e12c Add configuration utilities with backward compatibility
b3bca8b Add model factory and trajectory evaluation
7345917 Add data loading and standardization utilities
f52d6d7 Add dynamics models with auxiliary outputs
80a1375 Add plant dynamics and MLP building blocks
73e4f5a Initial commit: project structure and setup
```

---

## Commit Details

### 1. `73e4f5a` - Initial commit: project structure and setup
**Purpose**: Foundation for the project

**Files**:
- `.gitignore` - Python project ignore patterns
- `setup.py` - Package installation configuration

**Impact**: Establishes basic project structure

---

### 2. `80a1375` - Add plant dynamics and MLP building blocks
**Purpose**: Core mathematical and neural network components

**Files**:
- `mushr_mujoco_sysid/__init__.py` - Package initialization
- `mushr_mujoco_sysid/plant.py` - MushrPlant kinematics and dynamics
- `mushr_mujoco_sysid/models/mlp.py` - Multi-layer perceptron building block
- `mushr_mujoco_sysid/models/__init__.py` - Models subpackage

**Key Features**:
- SE(2) pose representation and integration
- Velocity-level dynamics with friction support
- Configurable MLP architecture

---

### 3. `f52d6d7` - Add dynamics models with auxiliary outputs
**Purpose**: Structured and direct dynamics models with advanced features

**Files**:
- `mushr_mujoco_sysid/models/system_models.py`

**Key Features**:
- **StructuredDynamicsModel**: Physics-based with learned corrections
  - Control adapter for state-dependent control mapping
  - Learnable friction network (3 parameterization modes)
  - Learnable residual network
- **DirectDynamicsModel**: Fully learned black-box
- **Auxiliary outputs**: `return_aux=True` exposes internal values for regularization
- **Friction modes**: `softplus_offset_1`, `exp`, `sigmoid_range`

**v2.0 Features**:
- Friction parameterization flexibility (allows k < 1)
- Auxiliary output support for regularization

---

### 4. `7345917` - Add data loading and standardization utilities
**Purpose**: Data handling and normalization infrastructure

**Files**:
- `mushr_mujoco_sysid/dataloader.py` - Trajectory data loading
- `mushr_mujoco_sysid/utils.py` - Standardization and utilities

**Key Features**:
- Load trajectory data from multiple file formats
- Parse velocities, poses, controls, timesteps
- Train/eval trajectory splitting
- Z-score standardization with inverse transforms
- JSON serialization for standardizers

---

### 5. `b3bca8b` - Add model factory and trajectory evaluation
**Purpose**: Model construction from config and evaluation utilities

**Files**:
- `mushr_mujoco_sysid/model_factory.py` - Build models from JSON config
- `mushr_mujoco_sysid/model.py` - LearnedDynamicsModel wrapper
- `mushr_mujoco_sysid/evaluation.py` - Trajectory evaluation

**Key Features**:
- Config-driven model instantiation
- Automatic standardization handling
- Trajectory rollout evaluation
- Per-component error metrics

---

### 6. `048e12c` - Add configuration utilities with backward compatibility
**Purpose**: Ensure new features don't break existing configs

**Files**:
- `mushr_mujoco_sysid/config_utils.py`

**Key Features**:
- `populate_config_defaults()` - Auto-fill missing fields
- `deep_merge()` - Recursive config merging
- **Default behaviors** (all new features OFF):
  - Loss: one_step_mse only
  - Regularization: all weights 0.0
  - Optim: no gradient clipping
  - Friction: softplus_offset_1 mode
  - Val split: timestep mode

**Critical v2.0 Feature**: Enables backward compatibility

---

### 7. `67f173e` - Add advanced dataset infrastructure
**Purpose**: Support for rollout loss and trajectory-aware validation

**Files**:
- `mushr_mujoco_sysid/data.py`

**Key Features**:
- **TimestepDataset**: Single-step prediction (original)
- **SnippetDataset** (NEW): Contiguous trajectory windows for rollout
- **Validation split modes**:
  - `timestep`: Shuffle samples, split by ratio (original)
  - `trajectory`: Reserve entire trajectories for validation (NEW)

**v2.0 Features**:
- Rollout loss infrastructure
- Trajectory-level validation (no leakage)

---

### 8. `4678d30` - Add comprehensive training pipeline with multi-component losses
**Purpose**: Full training loop with all v2.0 loss components

**Files**:
- `scripts/train.py`

**Key Features**:

#### Loss Components:
1. **One-Step MSE** (original):
   - Single-step prediction
   - Optional per-dimension weighting (vx, vy, w)

2. **Rollout Loss** (NEW):
   - Multi-step trajectory rollout
   - Configurable horizon
   - Teacher forcing support
   - Optional gradient detachment

3. **Pose Loss** (NEW):
   - SE(2) pose integration
   - **Uses unstandardized velocities** (physically accurate)
   - Selectable components (x, y, theta)

#### Regularization (NEW):
- `residual_l2_weight`: Penalize large residuals
- `friction_prior_weight`: Encourage friction near 1.0
- **Note**: `adapter_identity_weight` **removed** (unnecessary constraint)

#### Optimization (NEW):
- Gradient clipping support
- Per-component loss tracking
- Comprehensive logging

#### Evaluation:
- Per-timestep validation
- Held-out trajectory rollout
- Trajectory visualization

**Critical v2.0 Features**:
- Multi-component loss framework
- Pose loss with proper unstandardization
- Regularization (without adapter constraint)
- Gradient clipping

---

### 9. `f239309` - Add example configuration files
**Purpose**: Demonstrate all model configurations and features

**Files**: 10 JSON config files

**Configs**:

#### Baseline:
- `config_structured.json` - Full structured model
- `config_structured_minimal.json` - Minimal structured
- `config_direct.json` - Black-box direct model
- `config_direct_no_adapter.json` - Direct without adapter

#### Feature-specific:
- `config_control_adapter.json` - Adapter only
- `config_structured_adapter_only.json` - Structured + adapter
- `config_structured_friction_only.json` - Friction learning only
- `config_structured_residual_only.json` - Residual learning only

#### Advanced v2.0:
- `config_structured_friction_sigmoid.json` - Sigmoid friction mode
- `config_structured_full_features.json` - **ALL v2.0 features enabled**:
  - Trajectory validation split
  - Rollout + pose losses
  - Regularization (residual + friction)
  - Gradient clipping
  - Sigmoid friction mode

---

### 10. `f7d0f7b` - Add comprehensive documentation
**Purpose**: Complete documentation of features and usage

**Files**:
- `docs/CHANGELOG.md` - **v2.0 feature overview** (NEW)
- `docs/training_and_evaluation.md` - Training reference
- `docs/plant_and_models.md` - Model architecture docs
- `docs/data_and_dataloading.md` - Data pipeline docs
- `docs/system_and_configs.md` - System overview
- `docs/system_and_configs_with_results.md` - Results and analysis
- `docs/eval_runs_results.csv` - Evaluation metrics

**Key Documentation**:
- **CHANGELOG.md**: Complete v2.0 feature guide
  - When to use each feature
  - Configuration examples
  - Use case recommendations
  - Migration guide
  - Performance notes

---

## Version 2.0 Feature Summary

### What Changed (vs v1.0)

#### New Loss Components:
✅ Rollout loss for multi-step accuracy
✅ Pose loss with SE(2) integration
✅ Per-dimension velocity weighting

#### New Regularization:
✅ Residual L2 penalty
✅ Friction prior
❌ Removed: adapter_identity (unnecessary)

#### New Model Features:
✅ Friction parameterization modes (exp, sigmoid_range)
✅ Auxiliary output support

#### New Data Handling:
✅ SnippetDataset for rollout windows
✅ Trajectory-level validation split

#### New Training Features:
✅ Gradient clipping
✅ Comprehensive loss logging
✅ Per-component tracking

#### Infrastructure:
✅ Backward-compatible config system
✅ Unstandardized pose integration

---

## Key Implementation Decisions

### 1. Pose Integration Uses Unstandardized Velocities
**Why**: Physically accurate pose computation requires raw velocity values.

**Implementation**:
```python
x_next_raw = target_std.inverse(x_next.cpu().numpy())
pose_curr = plant.integrate_SE2(pose_curr, x_next_raw, dt)
```

### 2. Removed Adapter Identity Regularization
**Why**: The loss `MSE(ut_eff, ut_raw)` unnecessarily constrains the adapter from learning optimal control mappings.

**Impact**: Adapter now free to learn arbitrary control transformations.

### 3. All Features Default to OFF
**Why**: Backward compatibility - existing configs produce identical results.

**Implementation**: `config_utils.py` auto-fills missing fields with defaults.

### 4. Auxiliary Outputs Are Optional
**Why**: Avoid breaking existing code that doesn't use regularization.

**Implementation**: `return_aux=False` (default) preserves original behavior.

---

## Testing Backward Compatibility

To verify a config reproduces v1.0 behavior:

1. Ensure config has only original fields (or use defaults)
2. Check `training.loss.one_step_mse.enabled = true`
3. Check all other loss components disabled
4. Check all regularization weights = 0.0
5. Check `data.val_split_mode = "timestep"`
6. Check `model.friction_param.mode = "softplus_offset_1"`

Default config reproduces original training exactly.

---

## Recommended Commit Workflow

For future development:

1. **Feature branches**: Create branch for each new feature
2. **Atomic commits**: One logical change per commit
3. **Clear messages**: Explain what and why
4. **Tests first**: Add tests before implementation (when applicable)
5. **Docs last**: Update docs after feature is complete

---

## Quick Reference

| Commit | Focus | Lines Added |
|--------|-------|-------------|
| 73e4f5a | Project setup | 36 |
| 80a1375 | Plant & MLP | 335 |
| f52d6d7 | Dynamics models | 342 |
| 7345917 | Data loading | 667 |
| b3bca8b | Model factory | 276 |
| 048e12c | Config utils | 121 |
| 67f173e | Advanced datasets | 268 |
| 4678d30 | Training pipeline | 807 |
| f239309 | Config files | 435 |
| f7d0f7b | Documentation | 1524 |
| **Total** | | **4811** |

---

## Changelog Location

For user-facing changelog, see: **`docs/CHANGELOG.md`**

This document (`GIT_COMMIT_STRUCTURE.md`) is for developers understanding the repository structure.
