# V3 All-Stars Experiments: Controls Vary Within Trajectories

This directory contains 8 carefully designed experiment configurations to evaluate model performance on the new **v3 dataset** where controls can vary within a single trajectory (unlike the original dataset where controls were constant per trajectory).

## Dataset: sysid_trajs_v3

- **Total trajectories**: 3,000
  - 1,000 from v1 (constant controls per trajectory)
  - 2,000 from v2 (varying controls within trajectory)
- **Training**: 2,850 trajectories (95%)
- **Held-out eval**: 50 trajectories (reserved, split by `num_eval_trajectories`)
- **Validation**: 10% of training trajectories (trajectory-level split)
- **Format**: `x y theta xDot yDot thetaDot steering_angle velocity_desired`

## Debug Feature

All configs include `data.debug_control_variation: false`. Set to `true` to print control variation statistics for the first 5 training trajectories during data loading:
- Number of control changes per trajectory
- Percentage of timesteps where controls change
- Unique control values
- Control value ranges

## Experiment Configurations

### CONFIG A: v3A_struct_exp2_replay_seed4.json
**Baseline Replay of Previous Best (Exp2)**
- Model: Structured (plant + residual + adapter)
- Rollout: horizon=10, teacher_forcing=0.2
- Pose: components=[x, y], weight=0.05
- Seed: 4
- Purpose: Establish baseline under new dataset with proven config

### CONFIG B: v3B_struct_h10_tf0_seed4.json
**No Teacher Forcing (h=10)**
- Model: Structured (plant + residual + adapter)
- Rollout: horizon=10, teacher_forcing=0.0
- Pose: components=[x, y], weight=0.05
- Seed: 4
- Purpose: Test pure autoregressive rollout without teacher forcing

### CONFIG C: v3C_struct_h20_tf0_seed4.json
**Longer Rollout Horizon (h=20, MOST PROMISING)**
- Model: Structured (plant + residual + adapter)
- Rollout: horizon=20, teacher_forcing=0.0
- Pose: components=[x, y], weight=0.05
- Seed: 4
- Purpose: **Primary candidate** - longer horizon to capture control variations

### CONFIG D: v3D_struct_h20_tf0_seed2.json
**Seed Sensitivity Test**
- Model: Structured (plant + residual + adapter)
- Rollout: horizon=20, teacher_forcing=0.0
- Pose: components=[x, y], weight=0.05
- Seed: 2 (vs seed=4 in CONFIG C)
- Purpose: Verify robustness across different random initializations

### CONFIG E: v3E_struct_h20_tf0_resl2_0p01_seed4.json
**Residual Regularization**
- Model: Structured (plant + residual + adapter)
- Rollout: horizon=20, teacher_forcing=0.0
- Pose: components=[x, y], weight=0.05
- Regularization: residual_l2_weight=0.01
- Seed: 4
- Purpose: Prevent residual overfitting with mild L2 penalty

### CONFIG F: v3F_struct_h20_tf0_pose_xytheta_seed4.json
**Pose Supervision with Theta**
- Model: Structured (plant + residual + adapter)
- Rollout: horizon=20, teacher_forcing=0.0
- Pose: components=[x, y, theta], weight=0.02 (reduced from 0.05)
- Seed: 4
- Purpose: Improve orientation tracking with theta supervision

### CONFIG G: v3G_direct_no_adapter_h20_tf0_seed4.json
**Direct MLP Baseline (No Adapter)**
- Model: Direct MLP [256, 256, 128], no adapter
- Rollout: horizon=20, teacher_forcing=0.0
- Pose: components=[x, y], weight=0.05
- Seed: 4
- Purpose: Pure MLP baseline without control preprocessing

### CONFIG H: v3H_direct_with_adapter_h20_tf0_seed4.json
**Direct MLP with Adapter (Best Direct)**
- Model: Direct MLP [256, 256, 128] + control adapter
- Rollout: horizon=20, teacher_forcing=0.0
- Pose: components=[x, y], weight=0.05
- Seed: 4
- Purpose: Historically best direct model configuration

## Shared Configuration (All 8 Experiments)

### Data
- `data_dir`: `data/sysid_trajs_v3`
- `num_eval_trajectories`: 50
- `val_ratio`: 0.1
- `val_split_mode`: "trajectory" (no data leakage between train/val)

### Training
- `batch_size`: 256
- `epochs`: 1000
- `lr`: 0.001
- `lr_scheduler`: "cosine"
- `lr_t_max`: 1000
- `lr_eta_min`: 1e-6
- `weight_decay`: 0.0
- `eval_every`: 5
- `early_stop_patience`: 50
- `early_stop_min_delta`: 1e-6

### Loss Weights (Default)
- `one_step_mse.weight`: 1.0
- `rollout_mse.weight`: 0.5
- `pose_mse.weight`: 0.05 (except CONFIG F: 0.02)
- Per-dimension weights: `{vx: 1.0, vy: 1.0, w: 2.0}` (2x weight on angular velocity)

### Optimizer
- `grad_clip_norm`: 1.0

### Regularization (Default)
- `adapter_identity_weight`: 0.0
- `residual_l2_weight`: 0.0 (except CONFIG E: 0.01)
- `friction_prior_weight`: 0.0

## Running the Experiments

### Option 1A: Run All 8 in Parallel on 8 GPUs with tmuxp (RECOMMENDED)

```bash
# Training only (manually run eval afterward)
tmuxp load v3_allstars.yaml

# OR: Training + automatic eval (runs eval after each training completes)
tmuxp load v3_allstars_with_eval.yaml
```

This creates a tmux session with 8 windows, one experiment per GPU (0-7).

**Tmux Commands:**
- Attach: `tmux attach-session -t v3_allstars` (or `v3_allstars_eval`)
- Switch windows: `Ctrl+b` then `0-7` (or `n`/`p` for next/previous)
- Detach: `Ctrl+b` then `d`
- Kill session: `tmux kill-session -t v3_allstars`

**Window Layout:**
- `v3A_gpu0`: CONFIG A on GPU 0
- `v3B_gpu1`: CONFIG B on GPU 1
- `v3C_gpu2`: CONFIG C on GPU 2 ⭐ (most promising)
- `v3D_gpu3`: CONFIG D on GPU 3
- `v3E_gpu4`: CONFIG E on GPU 4
- `v3F_gpu5`: CONFIG F on GPU 5
- `v3G_gpu6`: CONFIG G on GPU 6
- `v3H_gpu7`: CONFIG H on GPU 7

### Option 1B: Run All 8 in Parallel with bash script

```bash
bash scripts/run_v3_allstars_tmux.sh
```

Alternative to tmuxp, creates a tmux session named `v3_allstars` with 8 windows.

### Option 2: Run Individual Experiments

```bash
# CONFIG A
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/v3_allstars_controls_vary/v3A_struct_exp2_replay_seed4.json

# CONFIG B
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/v3_allstars_controls_vary/v3B_struct_h10_tf0_seed4.json

# CONFIG C (MOST PROMISING)
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/v3_allstars_controls_vary/v3C_struct_h20_tf0_seed4.json

# CONFIG D
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/v3_allstars_controls_vary/v3D_struct_h20_tf0_seed2.json

# CONFIG E
CUDA_VISIBLE_DEVICES=4 python scripts/train.py --config configs/v3_allstars_controls_vary/v3E_struct_h20_tf0_resl2_0p01_seed4.json

# CONFIG F
CUDA_VISIBLE_DEVICES=5 python scripts/train.py --config configs/v3_allstars_controls_vary/v3F_struct_h20_tf0_pose_xytheta_seed4.json

# CONFIG G
CUDA_VISIBLE_DEVICES=6 python scripts/train.py --config configs/v3_allstars_controls_vary/v3G_direct_no_adapter_h20_tf0_seed4.json

# CONFIG H
CUDA_VISIBLE_DEVICES=7 python scripts/train.py --config configs/v3_allstars_controls_vary/v3H_direct_with_adapter_h20_tf0_seed4.json
```

### Option 3: Run Sequentially

```bash
for cfg in configs/v3_allstars_controls_vary/*.json; do
    python scripts/train.py --config $cfg
done
```

## Output Locations

Each experiment saves to its own directory:

```
experiments/v3_allstars_controls_vary/
├── v3A_struct_exp2_replay_seed4/
├── v3B_struct_h10_tf0_seed4/
├── v3C_struct_h20_tf0_seed4/
├── v3D_struct_h20_tf0_seed2/
├── v3E_struct_h20_tf0_resl2_0p01_seed4/
├── v3F_struct_h20_tf0_pose_xytheta_seed4/
├── v3G_direct_no_adapter_h20_tf0_seed4/
└── v3H_direct_with_adapter_h20_tf0_seed4/
```

Each experiment directory contains:
- `best.pt` - Best model checkpoint
- `config.json` - Full resolved config
- `standardizers.json` - Input/output normalizers
- `metrics.csv` - Training history
- `train.log` - Full training log

## Key Differences from Previous Experiments

1. **Control Variation**: The v3 dataset includes trajectories where controls change mid-trajectory, unlike the original dataset
2. **Longer Rollouts**: Focus on h=20 (vs h=10) to better capture control variation effects
3. **No Teacher Forcing**: Most configs use tf=0.0 to test pure autoregressive capability
4. **Trajectory-Level Validation**: Prevents data leakage when controls vary
5. **Larger Dataset**: 3,000 trajectories vs 1,000-2,000 previously

## Expected Outcomes

- **CONFIG C** is expected to perform best (h=20, tf=0.0, proven architecture)
- **CONFIG D** should match C if seed-robust
- **CONFIG E** may improve generalization via residual regularization
- **CONFIG F** could improve pose tracking with theta supervision
- **Configs G/H** provide MLP baselines for comparison
- **CONFIG A** establishes backward compatibility with Exp2 setup

## Notes

- All configs use `device: "cuda"` - ensure GPUs are available
- Training typically converges in 200-500 epochs with early stopping
- Monitor `eval_pos_mse` for pose prediction quality (primary metric for varied controls)
- Use `data.debug_control_variation: true` to verify control variation in your subset
