# Disciplined Experiment Sequence

This directory contains a carefully designed sequence of experiments to validate and improve the system identification pipeline. **Run these in order** - each builds on insights from the previous.

---

## Overview

The goal is NOT to "turn everything on everywhere" but to:
1. **Prove backward compatibility** (nothing broke)
2. **Improve evaluation quality** (trajectory validation)
3. **Add meaningful objectives** (rollout + pose loss)
4. **Rescue pathological configs** (friction-only, adapter-only)

---

## Experiment 0: Baseline Sanity Check

**Config**: `exp0_baseline_sanity_check.json`

**Purpose**: Prove you didn't break anything - reproduce previous best numbers.

**What it does**:
- Runs `structured_residual_only` (a known-good config)
- All new options at defaults (rollout/pose OFF, regularization 0, timestep validation)
- Should match previous baseline within training noise

**Run**:
```bash
python scripts/train.py --config configs/experiments/exp0_baseline_sanity_check.json
```

**Success criteria**:
- Final `eval_vel_mse` and `eval_pos_mse` match previous runs (±5%)
- Training loss curves look normal
- No crashes, NaNs, or divergence

**If this fails**: STOP. Fix the implementation before proceeding.

---

## Experiment 1: Trajectory-Based Validation

**Configs**:
- `exp1_traj_val_residual.json` (structured_residual_only)
- `exp1_traj_val_direct.json` (direct model)

**Purpose**: Fix validation selection to be less leaky.

**What changes**:
- `val_split_mode: "trajectory"` instead of `"timestep"`
- No other changes (no rollout, no new losses)

**Why this matters**:
- Timestep validation allows same trajectory in train and val
- Trajectory validation reserves entire trajectories for validation
- Better reflects real generalization to unseen trajectories
- Often changes which checkpoint is "best" without changing training much

**Run**:
```bash
python scripts/train.py --config configs/experiments/exp1_traj_val_residual.json
python scripts/train.py --config configs/experiments/exp1_traj_val_direct.json
```

**What to check**:
- Does the "best" checkpoint change?
- Does `eval_pos_mse` on held-out trajectories improve?
- Validation loss may be slightly different (expected)

**Interpretation**:
- If validation loss increases but eval improves: timestep-val was overfitting
- If no change: your data has enough variety that it doesn't matter

---

## Experiment 2: Rollout + Pose Loss (Strong Model First)

**Config**: `exp2_rollout_pose_residual.json`

**Purpose**: Validate the rollout pipeline and see if position accuracy improves.

**What changes** (vs Exp 1):
```json
"loss": {
  "one_step_mse": { "enabled": true, "weight": 1.0 },
  "rollout_mse": {
    "enabled": true,
    "horizon": 10,
    "weight": 0.5,
    "teacher_forcing_prob": 0.2
  },
  "pose_mse": {
    "enabled": true,
    "weight": 0.05,
    "components": ["x", "y"]
  },
  "weights": { "vx": 1.0, "vy": 1.0, "w": 2.0 }
},
"optim": { "grad_clip_norm": 1.0 }
```

**Why these values**:
- `horizon=10`: Long enough to expose drift, not so long it's unstable
- `rollout weight=0.5`: Keeps training anchored to stable one-step loss
- `teacher_forcing_prob=0.2`: Prevents early rollout blow-ups
- `pose weight=0.05`: Enough to matter, not enough to dominate
- `w=2.0`: Nudge yaw accuracy (main driver of position drift)
- `grad_clip_norm=1.0`: Stabilize training with multi-step gradients

**Run**:
```bash
python scripts/train.py --config configs/experiments/exp2_rollout_pose_residual.json
```

**What to check**:
- Does `eval_pos_mse` (held-out trajectories) decrease?
- Training should be stable (no divergence)
- Check loss curves: `rollout_mse` and `pose_mse` should decrease
- Look at trajectory plots: are trajectories tighter?

**Success**:
- If `eval_pos_mse` improves by 10-20%: rollout loss is working!
- If training is stable: conservative weights are good
- If this helps, roll out to other configs

**Failure modes**:
- If `eval_pos_mse` doesn't improve: may need higher pose weight or longer horizon
- If training diverges: increase teacher forcing or lower rollout weight
- If `rollout_mse` doesn't decrease: check SnippetDataset is working

---

## Experiment 3: Rescue Friction-Only

**Config**: `exp3_friction_rescue.json`

**Purpose**: Test if friction-only was failing due to **incapability** vs **poor training**.

**What changes** (vs baseline friction-only):
```json
"model": {
  "learn_friction": true,
  "learn_residual": false,
  "friction_param": {
    "mode": "sigmoid_range",
    "k_min": 0.25,
    "k_max": 3.0
  }
},
"training": {
  "loss": { /* rollout + pose as in Exp 2 */ },
  "regularization": {
    "friction_prior_weight": 0.01
  }
}
```

**Why sigmoid_range**:
- Original `softplus_offset_1` forces k ≥ 1 (can only increase friction)
- If yaw dynamics need reduction (k < 1), old mode can't do it
- `sigmoid_range [0.25, 3.0]` allows both increase and decrease

**Why friction_prior_weight=0.01**:
- Prevents extreme friction values
- Encourages k near 1.0 but allows deviation when needed

**Run**:
```bash
python scripts/train.py --config configs/experiments/exp3_friction_rescue.json
```

**CRITICAL: Inspect friction_k values**

After training, check what friction coefficients were learned:
```python
# During or after training, log friction_k statistics
# Look for: min, max, mean, std of friction_k across batches
```

**Diagnostic guide**:

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| `friction_k` saturates at `k_min=0.25` | Needs to go lower | Set `k_min=0.15` and re-run |
| `friction_k` saturates at `k_max=3.0` | Needs to go higher | Set `k_max=4.0` or `5.0` and re-run |
| `friction_k` stays near 1.0, never moves | Prior too strong | Lower `friction_prior_weight` to `0.001` |
| `friction_k` varies widely, unstable | No prior constraint | Increase `friction_prior_weight` to `0.05` |
| `friction_k` in [0.5, 1.5], performance improves | Working as intended! | Success |

**Success criteria**:
- Performance improves over baseline friction-only
- `friction_k` values are reasonable (not saturated, not stuck)
- If still worse than residual model, at least you learned friction alone is insufficient

---

## Experiment 4: Adapter-Only Strict Ablation

**Config**: `exp4_adapter_strict_ablation.json`

**Purpose**: Evidence that control remapping alone cannot fix plant mismatch.

**What it does**:
- Pure adapter (no friction, no residual)
- Add rollout + pose to expose compounding errors
- Trajectory validation for realistic eval

**Expectation**: This will likely **plateau or fail**.

**Why run it anyway**:
- Ablation evidence: "Adapter alone is insufficient"
- Quantifies the gap: "Adapter gets X error, residual gets Y error"
- Guides future work: know what's missing

**Run**:
```bash
python scripts/train.py --config configs/experiments/exp4_adapter_strict_ablation.json
```

**What to check**:
- Compare `eval_pos_mse` to Exp 2 (residual model)
- If gap is large: control remapping can't fix dynamics mismatch
- If competitive: plant model is actually good, just needs control correction

---

## Experiment 5: Adapter + Tiny Residual

**Config**: `exp5_adapter_tiny_residual.json`

**Purpose**: "Almost adapter-only" but with an emergency residual escape hatch.

**What changes** (vs Exp 4):
```json
"model": {
  "learn_residual": true  // Enable residual
},
"training": {
  "regularization": {
    "residual_l2_weight": 0.05  // STRONG penalty keeps it tiny
  }
}
```

**Why this matters**:
- If adapter-only fails (Exp 4) but this succeeds: you've learned something
- It means: plant mismatch is NOT representable by control remapping
- The residual is doing something the adapter can't

**Run**:
```bash
python scripts/train.py --config configs/experiments/exp5_adapter_tiny_residual.json
```

**What to check**:
- Monitor residual magnitudes (should be small due to L2 penalty)
- Compare performance to Exp 4 (adapter-only)
- If big improvement: residual is critical, adapter is not enough

**Diagnostic**:
- If residual stays tiny (~0.01) and performance is good: ideal
- If residual grows large (>0.1): lower L2 weight isn't working, this is just "residual model"
- If residual stays tiny but performance bad: neither adapter nor small residual helps

---

## Summary: Minimal Experiment Set

Run these **in order**:

1. ✅ **Exp 0**: Baseline sanity check (prove nothing broke)
2. ✅ **Exp 1**: Trajectory validation (cheap improvement)
3. ✅ **Exp 2**: Rollout + pose on strong model (validate pipeline)
4. ✅ **Exp 3**: Rescue friction-only (test new friction param)
5. ✅ **Exp 4**: Adapter-only strict (ablation evidence)
6. ✅ **Exp 5**: Adapter + tiny residual (practical compromise)

**Do NOT**:
- Run hyperparameter sweeps before these work
- Turn on all features everywhere
- Skip experiments if earlier ones fail

---

## After These Experiments

Once you have results from all 5 experiments, you'll know:

1. **Did rollout+pose help?** (Compare Exp 2 vs Exp 1)
2. **Can friction-only be rescued?** (Exp 3 results)
3. **Is adapter alone sufficient?** (Exp 4 vs Exp 2)
4. **What's the role of residual?** (Exp 5 vs Exp 4)

Then you can:
- Tune hyperparameters (rollout weight, pose weight, horizon, etc.)
- Try more aggressive settings if conservative worked
- Apply successful patterns to other model types
- Write up findings with evidence

---

## Quick Commands Reference

```bash
# Experiment 0: Sanity check
python scripts/train.py --config configs/experiments/exp0_baseline_sanity_check.json

# Experiment 1: Trajectory validation
python scripts/train.py --config configs/experiments/exp1_traj_val_residual.json
python scripts/train.py --config configs/experiments/exp1_traj_val_direct.json

# Experiment 2: Rollout + pose
python scripts/train.py --config configs/experiments/exp2_rollout_pose_residual.json

# Experiment 3: Friction rescue
python scripts/train.py --config configs/experiments/exp3_friction_rescue.json

# Experiment 4: Adapter strict
python scripts/train.py --config configs/experiments/exp4_adapter_strict_ablation.json

# Experiment 5: Adapter + tiny residual
python scripts/train.py --config configs/experiments/exp5_adapter_tiny_residual.json
```

---

## Emergency: Adapter Identity Weight

If rollout loss causes the adapter to do pathological control remapping (e.g., wild oscillations, extreme values), you can add a **very low** adapter identity regularization:

```json
"regularization": {
  "adapter_identity_weight": 0.0001  // 1e-4 to 1e-3 range
}
```

This acts as a "soft identity prior" - adapter can still learn, but extreme deviations are penalized.

**Only use if**:
- Adapter outputs are clearly unreasonable (inspect `ut_eff` values)
- Training is unstable even with gradient clipping
- You've tried other fixes (lower rollout weight, higher teacher forcing)

Default: Leave at 0.0 (let adapter learn freely).

---

## Notes

- All experiments use `trajectory` validation mode (after Exp 1)
- All experiments use same conservative rollout settings (after Exp 2)
- All experiments use gradient clipping (after Exp 2)
- Results go to separate `experiments/expN_*/` folders

Good luck! Focus on understanding **why** each works or fails, not just chasing metrics.
