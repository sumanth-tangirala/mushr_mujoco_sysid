# Experiments V2: CVaR Tail-Risk Loss Implementation

## Overview

This batch implements **CVaR (Conditional Value at Risk)** loss for rollout training to reduce worst-case trajectory errors by focusing gradient updates on the hardest rollout snippets.

**Total Experiments:** 24 (12 base + 12 CVaR variants)
**GPUs Used:** 8 (running 3 experiments in parallel per GPU)
**Estimated Runtime:** ~2-3 hours

---

## What's New: CVaR Loss

### Implementation Summary

**CVaR Configuration (in `training.loss.rollout_cvar`):**
```json
{
  "enabled": false,              // OFF by default (backward compatible)
  "alpha": 0.2,                  // Top 20% hardest samples
  "apply_to": "rollout_plus_pose", // CVaR on combined rollout+pose
  "min_k": 1                     // Minimum samples in top-k
}
```

**How It Works:**
1. Computes per-sample rollout losses [B] instead of mean
2. Optionally combines with pose loss per sample
3. Selects top-k% hardest samples via `torch.topk`
4. Backpropagates only through these hardest samples
5. Logs both mean (all samples) and CVaR (hard samples) for analysis

**Key Benefits:**
- Reduces worst-case trajectory errors
- Focuses learning on difficult scenarios
- Fully toggleable via config (no code changes)
- Backward compatible (default: OFF)

---

## Code Changes

### 1. Config Schema (`mushr_mujoco_sysid/config_utils.py`)
Added CVaR defaults to loss config:
```python
"rollout_cvar": {
    "enabled": False,
    "alpha": 0.2,
    "apply_to": "rollout_plus_pose",
    "min_k": 1,
}
```

### 2. Training Script (`scripts/train.py`)

**Modified Functions:**
- `_compute_rollout_loss`: Now computes per-sample losses when `return_per_sample=True`
- `train_one_epoch`: Applies CVaR reduction when enabled

**New Metrics Logged:**
- `rollout_mean`: Mean over all samples (for comparison)
- `rollout_cvar`: CVaR-reduced loss (actual backprop target)
- `k_used`: Number of samples selected
- `alpha`: CVaR alpha parameter

**Alpha Validation:**
- Raises `ValueError` if alpha not in (0, 1]
- Automatically disabled if rollout disabled

---

## Directory Structure

```
configs/
├── experiments-v1/          # Original 12 configs (baseline + hyperparameter sweeps)
└── experiments-v2/          # 24 configs (12 original + 12 CVaR variants)

experiments-v2/              # Output directory for all trained models
├── exp2_seed_sweep/         # Base models
├── exp2_seed_sweep_cvar0p2/ # CVaR variants
├── exp2_h20/
├── exp2_h20_cvar0p2/
... (24 total subdirectories)
```

---

## Experiments Matrix

All 24 experiments run simultaneously, distributed across 8 GPUs (3 per GPU):

| GPU | Experiments (3 per GPU) |
|-----|-------------------------|
| 0 | exp2_seed0, exp2_seed0_cvar0p2, exp2_xytheta |
| 1 | exp2_seed1, exp2_seed1_cvar0p2, exp2_xytheta_cvar0p2 |
| 2 | exp2_seed2, exp2_seed2_cvar0p2, exp2_resl2_0p01 |
| 3 | exp2_seed3, exp2_seed3_cvar0p2, exp2_resl2_0p01_cvar0p2 |
| 4 | exp2_seed4, exp2_seed4_cvar0p2, exp6_direct_rollout_pose |
| 5 | exp2_h20, exp2_h20_cvar0p2, exp6_direct_rollout_pose_cvar0p2 |
| 6 | exp2_tf0, exp2_tf0_cvar0p2, exp7_direct_no_adapter_rollout_pose |
| 7 | exp2_tf0p5, exp2_tf0p5_cvar0p2, exp7_direct_no_adapter_rollout_pose_cvar0p2 |

### Experiment Categories

| Category | Experiments | Purpose |
|----------|-------------|---------|
| **Seed Sweep** | exp2_seed0-4 + CVaR variants (10 total) | Measure variance + CVaR impact |
| **Rollout Hyperparameters** | exp2_h20, exp2_tf0, exp2_tf0p5 + CVaR variants (6 total) | Test horizon/TF sensitivity |
| **Loss Variants** | exp2_xytheta, exp2_resl2_0p01 + CVaR variants (4 total) | Theta supervision + regularization |
| **Direct Model** | exp6, exp7 + CVaR variants (4 total) | Pure NN vs structured model |

---

## Launch Instructions

### Option 1: Launch All Experiments (Recommended)
```bash
cd /common/home/st1122/Projects/mushr_mujoco_sysid
tmuxp load experiments-v2.yaml
```

This creates tmux session `mushr_exp_v2` with:
- **All 24 experiments:** Distributed across GPUs 0-7 (3 per GPU, all start simultaneously)
- **Auto-evaluation:** Each experiment runs eval immediately after training

### Option 2: Manual Launch (Single Experiment)
```bash
# Example: Run base config
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/experiments-v2/exp2_seed0.json

# Then evaluate
python scripts/eval.py --exp-dir experiments-v2/exp2_seed_sweep/residual_seed0 --num-eval-trajs 50

# Example: Run CVaR variant
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/experiments-v2/exp2_seed0_cvar0p2.json
python scripts/eval.py --exp-dir experiments-v2/exp2_seed_sweep_cvar0p2/residual_seed0_cvar0p2 --num-eval-trajs 50
```

---

## Monitoring

### Attach to tmux session:
```bash
tmux attach-session -t mushr_exp_v2
```

### Navigate:
- `Ctrl+b w` - List all windows
- `Ctrl+b 0-7` - Jump to GPU window
- `Ctrl+b d` - Detach (keeps running)

### Check progress:
```bash
# Watch GPU usage
nvidia-smi -l 1

# Check logs (example)
tail -f experiments-v2/exp2_seed_sweep/residual_seed0/*/losses.csv
```

---

## Expected Results

### Research Questions:

1. **Does CVaR reduce worst-case errors?**
   - Compare worst-case MSE: base vs CVaR variant
   - Hypothesis: CVaR should lower 95th percentile and max errors

2. **What's the trade-off on average performance?**
   - Compare avg MSE: base vs CVaR
   - Hypothesis: Slight degradation in mean, big gain in tail

3. **Seed variance with/without CVaR:**
   - Compute mean ± std across seeds 0-4
   - Does CVaR reduce variance?

4. **Hyperparameter sensitivity:**
   - h20 vs h10: Does longer horizon benefit CVaR?
   - TF sweep: Does CVaR help more with TF=0 (harder)?

5. **Direct model + CVaR:**
   - Can CVaR make pure NN competitive with structured models?

---

## Output Files (Per Experiment)

Each experiment produces:
```
experiments-v2/<exp_name>/<run_name>/
├── config.json              # Saved config
├── best.pt                  # Best checkpoint
├── standardizers.json       # Data normalization
├── losses.csv               # Training history (includes CVaR metrics)
└── eval_runs/
    └── <timestamp>/
        ├── metrics.json     # Evaluation results
        └── traj_*.png       # Trajectory plots
```

### CVaR-specific metrics in losses.csv:
- `rollout_mean`: Mean loss over all samples
- `rollout_cvar`: CVaR-reduced loss (top-k)
- `k_used`: Number of samples selected
- `alpha`: CVaR alpha (0.2)

---

## Analysis After Completion

### 1. Gather all results:
```bash
python3 << 'EOF'
import json
from pathlib import Path

results = []
for metrics_file in Path("experiments-v2").glob("*/*/eval_runs/*/metrics.json"):
    with open(metrics_file) as f:
        data = json.load(f)
    results.append({
        "exp": metrics_file.parts[1],
        "avg_mse": data["metrics"]["avg_traj_state_mse"],
        "worst_mse": max([t["traj_state_mse"] for t in data["per_trajectory"]]),
        "median_mse": sorted([t["traj_state_mse"] for t in data["per_trajectory"]])[25],
    })

# Sort by avg_mse
for r in sorted(results, key=lambda x: x["avg_mse"]):
    print(f"{r['exp']:50} | Avg: {r['avg_mse']:.4f} | Worst: {r['worst_mse']:.4f} | Median: {r['median_mse']:.6f}")
EOF
```

### 2. Compare base vs CVaR:
```python
# Group by base name
from collections import defaultdict
groups = defaultdict(list)
for r in results:
    base = r['exp'].replace('_cvar0p2', '')
    groups[base].append(r)

# Print comparison
for base, variants in sorted(groups.items()):
    if len(variants) == 2:
        v0, v1 = sorted(variants, key=lambda x: '_cvar' in x['exp'])
        delta_avg = ((v1['avg_mse'] - v0['avg_mse']) / v0['avg_mse']) * 100
        delta_worst = ((v1['worst_mse'] - v0['worst_mse']) / v0['worst_mse']) * 100
        print(f"{base:40} | Δ Avg: {delta_avg:+.1f}% | Δ Worst: {delta_worst:+.1f}%")
```

### 3. Seed variance analysis:
```python
# Compute seed sweep statistics
seed_base = [r for r in results if 'seed' in r['exp'] and 'cvar' not in r['exp']]
seed_cvar = [r for r in results if 'seed' in r['exp'] and 'cvar' in r['exp']]

import numpy as np
print(f"Base  - Mean: {np.mean([r['avg_mse'] for r in seed_base]):.4f} ± {np.std([r['avg_mse'] for r in seed_base]):.4f}")
print(f"CVaR  - Mean: {np.mean([r['avg_mse'] for r in seed_cvar]):.4f} ± {np.std([r['avg_mse'] for r in seed_cvar]):.4f}")
```

---

## Troubleshooting

### Config not found error:
- Ensure you're in `/common/home/st1122/Projects/mushr_mujoco_sysid`
- Check `configs/experiments-v2/` exists

### CVaR alpha validation error:
- Check `alpha` is in (0, 1]
- Default is 0.2 (top 20%)

### Per-sample loss shape mismatch:
- This is a bug - report immediately
- CVaR implementation handles [B] tensors correctly

---

## Next Steps

After experiments complete:
1. Run analysis scripts above
2. Generate comparison plots (base vs CVaR)
3. Identify best performing config
4. **Request full report:** "Create report for experiments-v2"

---

## Quick Reference

**Launch:**  `tmuxp load experiments-v2.yaml`
**Monitor:** `tmux attach -t mushr_exp_v2`
**Results:** `experiments-v2/*/eval_runs/*/metrics.json`
**Report:**  Ask Claude for `experiments-v2` analysis

---

**Total Runtime:** ~2-3 hours (all experiments run in parallel)
**Total Experiments:** 24 (3 per GPU across 8 GPUs)
**Total Metrics:** ~1200 trajectory evaluations (24 × 50)

Good luck! 🚀
