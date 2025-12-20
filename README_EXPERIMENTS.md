# System Identification Experiments - Quick Start

This README provides a quick overview of the experiment setup for the MuSHR system identification training pipeline v2.0.

## 🎯 Purpose

Validate and improve the training pipeline through a **disciplined sequence of experiments** that:
1. Proves backward compatibility
2. Improves evaluation quality  
3. Adds meaningful training objectives
4. Rescues pathological configurations

## 📋 What Was Added

### 21 New Configuration Options
All default to OFF for backward compatibility:
- **1** data option (trajectory validation)
- **3** model options (friction parameterization)
- **13** loss options (rollout, pose, per-dimension weights)
- **3** regularization options (adapter, residual, friction priors)
- **1** optimization option (gradient clipping)

### 7 Experiment Configurations
Located in `configs/experiments/`:
- **Exp 0**: Baseline sanity check
- **Exp 1**: Trajectory validation (2 configs)
- **Exp 2**: Rollout + pose loss
- **Exp 3**: Friction rescue
- **Exp 4**: Adapter-only strict
- **Exp 5**: Adapter + tiny residual

## 🚀 Quick Start

Run experiments **in order**:

```bash
# 0. Sanity check (MUST pass first!)
python scripts/train.py --config configs/experiments/exp0_baseline_sanity_check.json

# 1. Trajectory validation
python scripts/train.py --config configs/experiments/exp1_traj_val_residual.json
python scripts/train.py --config configs/experiments/exp1_traj_val_direct.json

# 2. Rollout + pose
python scripts/train.py --config configs/experiments/exp2_rollout_pose_residual.json

# 3. Friction rescue
python scripts/train.py --config configs/experiments/exp3_friction_rescue.json

# 4. Adapter strict
python scripts/train.py --config configs/experiments/exp4_adapter_strict_ablation.json

# 5. Adapter + tiny residual
python scripts/train.py --config configs/experiments/exp5_adapter_tiny_residual.json
```

## 📚 Documentation

### For Running Experiments
1. **configs/experiments/EXPERIMENT_GUIDE.md** - Detailed experiment guide
2. **EXPERIMENT_SETUP_SUMMARY.md** - Why each experiment matters
3. **NEW_OPTIONS_SUMMARY.md** - All 21 options explained

### For Understanding Features
1. **docs/CHANGELOG.md** - Complete v2.0 feature overview
2. **docs/training_and_evaluation.md** - Technical reference

### For Developers
1. **GIT_COMMIT_STRUCTURE.md** - Repository organization

## ✅ What You'll Learn

After running all experiments:
1. Does v2.0 reproduce v1.0 baseline? → **Backward compatibility**
2. Does trajectory split improve checkpoints? → **Validation quality**
3. Does rollout+pose improve position accuracy? → **Training objectives**
4. Can friction-only be rescued? → **Model capabilities**
5. Is adapter alone sufficient? → **Component contributions**

## 🔑 Key Principles

- **Backward Compatible**: All new options default to OFF
- **Conservative**: Validate pipeline before aggressive tuning
- **Diagnostic-Focused**: Clear success criteria
- **Evidence-Driven**: Ablations provide scientific evidence
- **Incremental**: Build complexity gradually

## 📊 Results Structure

Each experiment outputs to dedicated folder:
```
experiments/
├── exp0_baseline_sanity/
├── exp1_traj_val/
│   ├── residual_only/
│   └── direct/
├── exp2_rollout_pose/
├── exp3_friction_rescue/
├── exp4_adapter_strict/
└── exp5_adapter_residual/
```

## 🎓 Next Steps

1. Run Exp 0 to verify setup
2. Proceed through Exp 1-5 in order
3. Review results and diagnostics
4. Tune hyperparameters based on findings
5. Report successful patterns

---

**See EXPERIMENT_GUIDE.md for detailed instructions!**
