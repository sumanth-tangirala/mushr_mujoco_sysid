# Next Experiments - Configuration and Launch Guide

Generated from Exp2 baseline (best performing model: structured residual + rollout + pose)

## Quick Start

### Launch all experiments using tmuxp:
```bash
cd /common/home/st1122/Projects/mushr_mujoco_sysid
tmuxp load next_experiments.yaml
```

This will create a tmux session with 8 windows, distributing 12 experiments across 8 GPUs.

### Or run individually:
```bash
# Seed sweep
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/next_experiments/exp2_seed0.json
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/next_experiments/exp2_seed1.json
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/next_experiments/exp2_seed2.json
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/next_experiments/exp2_seed3.json
CUDA_VISIBLE_DEVICES=4 python scripts/train.py --config configs/next_experiments/exp2_seed4.json

# Rollout/TF sensitivity
CUDA_VISIBLE_DEVICES=5 python scripts/train.py --config configs/next_experiments/exp2_h20.json
CUDA_VISIBLE_DEVICES=6 python scripts/train.py --config configs/next_experiments/exp2_tf0.json
CUDA_VISIBLE_DEVICES=7 python scripts/train.py --config configs/next_experiments/exp2_tf0p5.json

# Pose + regularization
CUDA_VISIBLE_DEVICES=5 python scripts/train.py --config configs/next_experiments/exp2_xytheta.json
CUDA_VISIBLE_DEVICES=6 python scripts/train.py --config configs/next_experiments/exp2_resl2_0p01.json

# Direct model variants
CUDA_VISIBLE_DEVICES=7 python scripts/train.py --config configs/next_experiments/exp6_direct_rollout_pose.json
CUDA_VISIBLE_DEVICES=7 python scripts/train.py --config configs/next_experiments/exp7_direct_no_adapter_rollout_pose.json
```

## Experiment Overview

| File | Diff from Exp2 | Purpose | Expected Outcome |
|------|----------------|---------|------------------|
| **A) Seed Sweep** |
| `exp2_seed0.json` | seed=0 | Measure variance | Baseline for CI |
| `exp2_seed1.json` | seed=1 | Measure variance | ±σ estimate |
| `exp2_seed2.json` | seed=2 | Measure variance | ±σ estimate |
| `exp2_seed3.json` | seed=3 | Measure variance | ±σ estimate |
| `exp2_seed4.json` | seed=4 | Measure variance | ±σ estimate |
| **B) Rollout/TF Hyperparameters** |
| `exp2_h20.json` | `horizon=20` (vs 10) | Longer rollout | May improve generalization |
| `exp2_tf0.json` | `teacher_forcing=0.0` (vs 0.2) | Pure autoregressive | Harder training, more robust? |
| `exp2_tf0p5.json` | `teacher_forcing=0.5` (vs 0.2) | High teacher forcing | Easier training, may overfit |
| **C) Pose Loss** |
| `exp2_xytheta.json` | `pose=[x,y,theta]`, `weight=0.02` | Supervise orientation | Better theta tracking |
| **D) Regularization** |
| `exp2_resl2_0p01.json` | `residual_l2=0.01` (vs 0.0) | Light L2 penalty | Between Exp2 & Exp5 |
| **E) Direct Model** |
| `exp6_direct_rollout_pose.json` | `model.type=direct` + rollout | Pure NN multi-step | Match Exp2 performance? |
| `exp7_direct_no_adapter_rollout_pose.json` | Direct, `adapter=false` | Ablation study | Quantify adapter value |

## GPU Allocation (via tmuxp)

- **GPU 0:** exp2_seed0
- **GPU 1:** exp2_seed1
- **GPU 2:** exp2_seed2
- **GPU 3:** exp2_seed3
- **GPU 4:** exp2_seed4
- **GPU 5:** exp2_h20 → exp2_tf0 (sequential)
- **GPU 6:** exp2_tf0p5 → exp2_xytheta (sequential)
- **GPU 7:** exp2_resl2_0p01 → exp6_direct → exp7_direct (sequential)

Total: 12 experiments across 8 GPUs

## Baseline (Exp2) Configuration

```json
{
  "seed": 42,
  "model": {
    "type": "structured",
    "learn_friction": false,
    "learn_residual": true,
    "control_adapter": { "enabled": true, ... }
  },
  "training": {
    "loss": {
      "one_step_mse": { "weight": 1.0 },
      "rollout_mse": { "horizon": 10, "weight": 0.5, "teacher_forcing_prob": 0.2 },
      "pose_mse": { "weight": 0.05, "components": ["x", "y"] },
      "weights": { "vx": 1.0, "vy": 1.0, "w": 2.0 }
    },
    "regularization": {
      "residual_l2_weight": 0.0
    }
  }
}
```

**Exp2 Performance (baseline to beat):**
- Avg state MSE: 0.025
- Worst-case MSE: 0.505
- Median MSE: 0.001

## Monitoring Progress

### Attach to tmux session:
```bash
tmux attach-session -t mushr_next_exps
```

### Navigate between experiments:
- `Ctrl+b w` - List all windows
- `Ctrl+b n` - Next window
- `Ctrl+b p` - Previous window
- `Ctrl+b 0-7` - Jump to specific GPU

### Check logs:
```bash
# Example for seed0
tail -f experiment_logs/exp2_seed_sweep_*_seed0.log
```

## Post-Training Evaluation

After all experiments complete, run evaluation:

```bash
# Seed sweep
for i in 0 1 2 3 4; do
  python scripts/eval.py \
    --exp-dir experiments/exp2_seed_sweep/residual_seed${i} \
    --num-eval-trajs 50
done

# Other experiments
python scripts/eval.py --exp-dir experiments/exp2_h20/residual_h20 --num-eval-trajs 50
python scripts/eval.py --exp-dir experiments/exp2_tf0/residual_tf0 --num-eval-trajs 50
python scripts/eval.py --exp-dir experiments/exp2_tf0p5/residual_tf0p5 --num-eval-trajs 50
python scripts/eval.py --exp-dir experiments/exp2_xytheta/residual_xytheta --num-eval-trajs 50
python scripts/eval.py --exp-dir experiments/exp2_resl2_0p01/residual_resl2_0p01 --num-eval-trajs 50
python scripts/eval.py --exp-dir experiments/exp6_direct_rollout_pose/direct_rollout_pose --num-eval-trajs 50
python scripts/eval.py --exp-dir experiments/exp7_direct_no_adapter_rollout_pose/direct_no_adapter_rollout_pose --num-eval-trajs 50
```

## Analysis Questions

1. **Seed sweep:** What is the variance in Exp2's performance? (compute mean ± std)
2. **Horizon:** Does h=20 improve over h=10?
3. **Teacher forcing:** What's the optimal TF ratio? (0.0, 0.2, 0.5)
4. **Theta supervision:** Does pose loss on theta improve orientation tracking?
5. **Regularization:** Is residual_l2=0.01 a sweet spot between Exp2 (0.0) and Exp5 (0.05)?
6. **Direct model:** Can pure NN + rollout match structured model (Exp2)?
7. **Adapter value:** How much does adapter help direct models (Exp6 vs Exp7)?

## Expected Runtime

- Each experiment: ~2-4 hours (with early stopping)
- Total wall time: ~4-6 hours (with sequential execution on GPUs 5-7)
- Peak GPU utilization: 8/8 GPUs for first ~2-3 hours, then 3/8 GPUs

## File Locations

- **Configs:** `configs/next_experiments/*.json`
- **Checkpoints:** `experiments/exp*/*/best.pt`
- **Logs:** `experiment_logs/` (if logging is configured)
- **Evaluation results:** `experiments/exp*/*/eval_runs/*/metrics.json`
