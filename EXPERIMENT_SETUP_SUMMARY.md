# Experiment Setup Summary

## What Was Done

This document summarizes the setup of a disciplined experiment sequence for validating and improving the system identification training pipeline.

---

## 1. Re-added Adapter Identity Weight

### Change
Re-introduced `adapter_identity_weight` to `training.regularization` config block as an **optional emergency brake**.

### Rationale
While adapter should generally learn freely (default: 0.0), rollout loss can occasionally cause pathological control remapping. Having a very low weight stabilizer (1e-4 to 1e-3) available provides a safety mechanism without constraining normal training.

### Usage
```json
"regularization": {
  "adapter_identity_weight": 0.0     // Default: OFF
  "adapter_identity_weight": 0.0001  // Emergency brake if needed
}
```

### When to Use
- **Default (0.0)**: Let adapter learn freely (99% of cases)
- **Low weight (1e-4 to 1e-3)**: Only if adapter outputs become unreasonable during rollout
- **DO NOT use high weights**: This defeats the purpose of having an adapter

---

## 2. Created Disciplined Experiment Sequence

### Philosophy
**NOT** "turn everything on everywhere" but:
1. Prove backward compatibility (nothing broke)
2. Improve evaluation quality (trajectory validation)
3. Add meaningful objectives (rollout + pose)
4. Rescue pathological configs (friction-only, adapter-only)

### Experiments Created

#### Experiment 0: Baseline Sanity Check
**File**: `configs/experiments/exp0_baseline_sanity_check.json`

**Purpose**: Reproduce previous best numbers with all new options at defaults.

**What it tests**:
- Backward compatibility
- No implementation bugs
- Training stability

**Success**: Match previous `eval_vel_mse` and `eval_pos_mse` within ±5%

---

#### Experiment 1: Trajectory Validation
**Files**:
- `configs/experiments/exp1_traj_val_residual.json`
- `configs/experiments/exp1_traj_val_direct.json`

**Purpose**: Fix validation selection to be less leaky.

**Change**: `val_split_mode: "trajectory"` (was `"timestep"`)

**Why**:
- Timestep mode: same trajectory in train and val (different timesteps)
- Trajectory mode: reserve entire trajectories for validation (no leakage)
- Better reflects real generalization

**Expected outcome**: "Best" checkpoint may change, eval metrics may improve

---

#### Experiment 2: Rollout + Pose (Strong Model)
**File**: `configs/experiments/exp2_rollout_pose_residual.json`

**Purpose**: Validate rollout pipeline, improve position accuracy.

**Key settings** (conservative):
```json
"rollout_mse": {
  "horizon": 10,
  "weight": 0.5,
  "teacher_forcing_prob": 0.2
},
"pose_mse": {
  "weight": 0.05,
  "components": ["x", "y"]
},
"weights": { "vx": 1.0, "vy": 1.0, "w": 2.0 },
"grad_clip_norm": 1.0
```

**Why conservative**:
- Rollout weight 0.5: keeps training anchored to stable one-step loss
- Teacher forcing 0.2: prevents early rollout blow-ups
- Pose weight 0.05: matters but doesn't dominate
- w=2.0: nudge yaw accuracy (main driver of position drift)

**Success**: `eval_pos_mse` decreases by 10-20%

---

#### Experiment 3: Friction Rescue
**File**: `configs/experiments/exp3_friction_rescue.json`

**Purpose**: Test if friction-only was failing due to **incapability** vs **poor training**.

**Key changes**:
```json
"friction_param": {
  "mode": "sigmoid_range",
  "k_min": 0.25,
  "k_max": 3.0
},
"regularization": {
  "friction_prior_weight": 0.01
}
// + rollout + pose
```

**Why sigmoid_range**:
- Original `softplus_offset_1`: forces k ≥ 1
- If yaw needs reduction (k < 1), old mode can't do it
- `sigmoid_range [0.25, 3.0]`: allows both increase and decrease

**CRITICAL**: Inspect friction_k values after training!

**Diagnostic guide**:
| Friction k behavior | Diagnosis | Action |
|---------------------|-----------|--------|
| Saturates at k_min=0.25 | Needs lower | Set k_min=0.15 |
| Saturates at k_max=3.0 | Needs higher | Set k_max=4-5 |
| Stays near 1.0, never moves | Prior too strong | Lower to 0.001 |
| Varies widely, unstable | No constraint | Increase to 0.05 |
| In [0.5, 1.5], improves | Working! | Success |

---

#### Experiment 4: Adapter-Only Strict Ablation
**File**: `configs/experiments/exp4_adapter_strict_ablation.json`

**Purpose**: Evidence that control remapping alone cannot fix plant mismatch.

**What it tests**:
- Pure adapter (no friction, no residual)
- With rollout + pose to expose compounding errors

**Expectation**: Will likely plateau or fail

**Why run it**:
- Ablation evidence: "Adapter alone is insufficient"
- Quantifies the gap: "Adapter gets X error, residual gets Y error"
- If it works: plant model is good, just needs control correction

---

#### Experiment 5: Adapter + Tiny Residual
**File**: `configs/experiments/exp5_adapter_tiny_residual.json`

**Purpose**: "Almost adapter-only" with emergency residual capability.

**Key change**:
```json
"model": {
  "learn_residual": true
},
"regularization": {
  "residual_l2_weight": 0.05  // STRONG penalty
}
```

**What this tests**:
- If Exp 4 fails but this succeeds: plant mismatch NOT representable by control remapping
- The residual is doing something adapter can't

**Diagnostic**:
- Residual tiny (~0.01), good performance: ideal
- Residual large (>0.1): this is just "residual model"
- Residual tiny, bad performance: neither helps

---

## 3. Complete Documentation

### EXPERIMENT_GUIDE.md
Comprehensive guide for running experiments:
- Purpose and rationale for each experiment
- Success criteria
- Diagnostic guides
- Failure mode handling
- Quick command reference

### Key Principles
1. **Run in order** - each builds on previous insights
2. **Don't skip experiments** - if one fails, fix before proceeding
3. **Understand why** - focus on learning, not just metrics
4. **No hyperparameter sweeps yet** - validate pipeline first

---

## 4. Experiment Folder Structure

Each experiment outputs to dedicated folder:
```
experiments/
├── exp0_baseline_sanity/          # Sanity check
├── exp1_traj_val/
│   ├── residual_only/             # Structured residual
│   └── direct/                    # Direct model
├── exp2_rollout_pose/
│   └── residual_conservative/     # Conservative settings
├── exp3_friction_rescue/
│   └── sigmoid_range_rollout/     # Bounded friction
├── exp4_adapter_strict/
│   └── adapter_only_ablation/     # Pure adapter
└── exp5_adapter_residual/
    └── adapter_tiny_residual/     # Adapter + small residual
```

---

## What You'll Learn

After running all 5 experiments, you'll know:

1. **Backward compatibility**: Does v2.0 reproduce v1.0 baseline? (Exp 0)
2. **Validation quality**: Does trajectory split improve checkpoint selection? (Exp 1)
3. **Rollout efficacy**: Does rollout+pose improve position accuracy? (Exp 2)
4. **Friction capability**: Can friction-only be rescued with better parameterization? (Exp 3)
5. **Adapter sufficiency**: Is adapter alone enough, or is dynamics correction needed? (Exp 4 vs 5)

---

## Quick Start

### Run Experiments in Order

```bash
# 0. Sanity check (MUST pass before proceeding)
python scripts/train.py --config configs/experiments/exp0_baseline_sanity_check.json

# 1. Trajectory validation (cheap improvement)
python scripts/train.py --config configs/experiments/exp1_traj_val_residual.json
python scripts/train.py --config configs/experiments/exp1_traj_val_direct.json

# 2. Rollout + pose (validate pipeline)
python scripts/train.py --config configs/experiments/exp2_rollout_pose_residual.json

# 3. Friction rescue (test new friction param)
python scripts/train.py --config configs/experiments/exp3_friction_rescue.json

# 4. Adapter strict (ablation evidence)
python scripts/train.py --config configs/experiments/exp4_adapter_strict_ablation.json

# 5. Adapter + tiny residual (practical compromise)
python scripts/train.py --config configs/experiments/exp5_adapter_tiny_residual.json
```

---

## Common Hyperparameters Across Experiments

### Rollout Loss (when enabled)
- `horizon`: 10 (long enough to expose drift)
- `weight`: 0.5 (anchored to one-step loss)
- `teacher_forcing_prob`: 0.2 (prevent early blow-ups)
- `detach_between_steps`: false (allow gradient flow)

### Pose Loss (when enabled)
- `weight`: 0.05 (matters but doesn't dominate)
- `components`: ["x", "y"] (position only, not theta)

### Velocity Weights
- `vx`: 1.0
- `vy`: 1.0
- `w`: 2.0 (emphasize yaw accuracy)

### Optimization
- `grad_clip_norm`: 1.0 (stabilize multi-step gradients)

### Validation
- `val_split_mode`: "trajectory" (after Exp 1)

---

## After Experiments: Next Steps

Once baseline experiments work:

### If Rollout+Pose Helps (Exp 2 success)
- Try more aggressive settings:
  - Increase pose weight (0.1)
  - Increase rollout weight (0.7)
  - Longer horizon (15-20)
  - Lower teacher forcing (0.1 or 0.0)

### If Friction Rescue Works (Exp 3 success)
- Fine-tune k_min and k_max based on saturation
- Try different friction prior weights
- Apply to full model (friction + residual)

### If Adapter Alone Fails (Exp 4 fails, Exp 5 succeeds)
- Confirms: dynamics correction is necessary
- Focus future work on plant model improvement
- Adapter is useful but not sufficient

---

## Files Changed/Created

### Modified
- `mushr_mujoco_sysid/config_utils.py`: Re-add adapter_identity_weight default
- `scripts/train.py`: Re-add adapter_identity regularization logic

### Created
- `configs/experiments/exp0_baseline_sanity_check.json`
- `configs/experiments/exp1_traj_val_residual.json`
- `configs/experiments/exp1_traj_val_direct.json`
- `configs/experiments/exp2_rollout_pose_residual.json`
- `configs/experiments/exp3_friction_rescue.json`
- `configs/experiments/exp4_adapter_strict_ablation.json`
- `configs/experiments/exp5_adapter_tiny_residual.json`
- `configs/experiments/EXPERIMENT_GUIDE.md`
- `EXPERIMENT_SETUP_SUMMARY.md` (this file)

---

## Git Commits

```
c503a45 Add disciplined experiment sequence configs
8fe4d20 Re-add adapter_identity_weight as optional emergency brake
acfd77f Add developer documentation for commit structure
f7d0f7b Add comprehensive documentation
...
```

---

## Key Takeaways

1. **Conservative by design**: All settings chosen to validate pipeline, not maximize performance
2. **Diagnostic-focused**: Each experiment has clear success criteria and failure diagnostics
3. **Evidence-driven**: Ablations (Exp 4) provide scientific evidence, not just metrics
4. **Incremental**: Build complexity gradually - don't jump to "everything on"
5. **Interpretable**: Focus on understanding why each works or fails

---

## Support

- See `configs/experiments/EXPERIMENT_GUIDE.md` for detailed experiment guide
- See `docs/CHANGELOG.md` for complete v2.0 feature documentation
- See `docs/training_and_evaluation.md` for configuration reference

Good luck with the experiments!
