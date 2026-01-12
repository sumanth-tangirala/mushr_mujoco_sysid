# Experiments V2: CVaR Loss Analysis Report

**Date:** 2026-01-04
**Total Experiments:** 24 (12 base + 12 CVaR variants)
**Evaluation:** 50 trajectories per experiment

---

## Executive Summary

**CVaR Impact on Average Performance:**
- Trajectory State MSE: 0.084595 → 0.187298 (+121.4%)
- Trajectory Position MSE: 0.158541 → 0.314770 (+98.5%)

**CVaR Impact on Worst-Case Performance (95th percentile):**
- State MSE P95: 0.764134 → 1.010991 (+32.3%)
- Position MSE P95: 1.464801 → 1.854205 (+26.6%)

**Key Findings:**
- ⚠️ CVaR degrades average performance by 121.4%
- ⚠️ CVaR degrades worst-case performance by 32.3%

---

## Seed Sweep Analysis

### Individual Results

| Experiment | Traj State MSE | Traj Pos MSE | Traj Vel MSE | Worst State P95 |
|------------|----------------|--------------|--------------|-----------------|
| seed0 | 0.067611 | 0.127464 | 0.007758 | 0.258035 |
| seed1 | 0.004236 | 0.007834 | 0.000637 | 0.019605 |
| seed2 | 0.003266 | 0.005967 | 0.000565 | 0.016643 |
| seed3 | 0.320942 | 0.603694 | 0.038190 | 1.282014 |
| seed4 | 0.002512 | 0.004450 | 0.000574 | 0.012055 |
| seed0_cvar | 0.457225 | 0.862898 | 0.051553 | 1.266136 |
| seed1_cvar | 0.005032 | 0.009423 | 0.000641 | 0.022850 |
| seed2_cvar | 0.468321 | 0.687778 | 0.248865 | 1.301829 |
| seed3_cvar | 0.131587 | 0.248404 | 0.014770 | 0.538206 |
| seed4_cvar | 0.007352 | 0.013754 | 0.000951 | 0.023963 |

### Base vs CVaR Comparison

| Metric | Base (mean ± std) | CVaR (mean ± std) | Change |
|--------|-------------------|-------------------|--------|
| Avg Traj State MSE | 0.079713 ± 0.123158 | 0.213904 ± 0.208327 | +168.3% |
| Worst State MSE P95 | 0.707780 | 0.983215 | +38.9% |
| Worst State MSE Max | 1.736359 | 2.624893 | +51.2% |

---

## Rollout Hyperparameters Analysis

### Individual Results

| Experiment | Traj State MSE | Traj Pos MSE | Traj Vel MSE | Worst State P95 |
|------------|----------------|--------------|--------------|-----------------|
| h20 | 0.013525 | 0.025944 | 0.001107 | 0.050863 |
| tf0 | 0.005916 | 0.011181 | 0.000651 | 0.013086 |
| tf0p5 | 0.456591 | 0.851476 | 0.061706 | 1.415920 |
| h20_cvar | 0.014164 | 0.027188 | 0.001140 | 0.023981 |
| tf0_cvar | 0.718999 | 1.345985 | 0.092013 | 1.519315 |
| tf0p5_cvar | 0.189909 | 0.096305 | 0.283513 | 0.298833 |

### Base vs CVaR Comparison

| Metric | Base (mean ± std) | CVaR (mean ± std) | Change |
|--------|-------------------|-------------------|--------|
| Avg Traj State MSE | 0.158677 ± 0.210680 | 0.307691 ± 0.299558 | +93.9% |
| Worst State MSE P95 | 1.187484 | 1.179158 | -0.7% |
| Worst State MSE Max | 1.894065 | 7.316571 | +286.3% |

---

## Loss Variants Analysis

### Individual Results

| Experiment | Traj State MSE | Traj Pos MSE | Traj Vel MSE | Worst State P95 |
|------------|----------------|--------------|--------------|-----------------|
| xytheta | 0.053276 | 0.100123 | 0.006428 | 0.393219 |
| resl2 | 0.044873 | 0.083538 | 0.006208 | 0.252933 |
| xytheta_cvar | 0.004009 | 0.007389 | 0.000630 | 0.016629 |
| resl2_cvar | 0.196197 | 0.376321 | 0.016074 | 0.898083 |

### Base vs CVaR Comparison

| Metric | Base (mean ± std) | CVaR (mean ± std) | Change |
|--------|-------------------|-------------------|--------|
| Avg Traj State MSE | 0.049074 ± 0.004201 | 0.100103 ± 0.096094 | +104.0% |
| Worst State MSE P95 | 0.374215 | 0.306018 | -18.2% |
| Worst State MSE Max | 0.861674 | 3.161710 | +266.9% |

---

## Direct Models Analysis

### Individual Results

| Experiment | Traj State MSE | Traj Pos MSE | Traj Vel MSE | Worst State P95 |
|------------|----------------|--------------|--------------|-----------------|
| exp6 | 0.032935 | 0.062935 | 0.002935 | 0.013277 |
| exp7 | 0.009454 | 0.017889 | 0.001018 | 0.020754 |
| exp6_cvar | 0.051920 | 0.096648 | 0.007192 | 0.147834 |
| exp7_cvar | 0.002861 | 0.005142 | 0.000580 | 0.014177 |

### Base vs CVaR Comparison

| Metric | Base (mean ± std) | CVaR (mean ± std) | Change |
|--------|-------------------|-------------------|--------|
| Avg Traj State MSE | 0.021194 ± 0.011741 | 0.027391 ± 0.024530 | +29.2% |
| Worst State MSE P95 | 0.018456 | 0.082427 | +346.6% |
| Worst State MSE Max | 1.516770 | 0.811516 | -46.5% |

---

## Best Performing Models

### Top 10 by Average Performance

| Rank | Experiment | Category | Type | Avg State MSE | Avg Pos MSE | Worst P95 |
|------|------------|----------|------|---------------|-------------|-----------|
| 1 | seed4 | seed_sweep | base | 0.002512 | 0.004450 | 0.012055 |
| 2 | exp7_cvar | direct_models | cvar | 0.002861 | 0.005142 | 0.014177 |
| 3 | seed2 | seed_sweep | base | 0.003266 | 0.005967 | 0.016643 |
| 4 | xytheta_cvar | loss_variants | cvar | 0.004009 | 0.007389 | 0.016629 |
| 5 | seed1 | seed_sweep | base | 0.004236 | 0.007834 | 0.019605 |
| 6 | seed1_cvar | seed_sweep | cvar | 0.005032 | 0.009423 | 0.022850 |
| 7 | tf0 | rollout_hyperparams | base | 0.005916 | 0.011181 | 0.013086 |
| 8 | seed4_cvar | seed_sweep | cvar | 0.007352 | 0.013754 | 0.023963 |
| 9 | exp7 | direct_models | base | 0.009454 | 0.017889 | 0.020754 |
| 10 | h20 | rollout_hyperparams | base | 0.013525 | 0.025944 | 0.050863 |

### Top 10 by Worst-Case Performance (P95)

| Rank | Experiment | Category | Type | Worst P95 | Avg State MSE | Avg Pos MSE |
|------|------------|----------|------|-----------|---------------|-------------|
| 1 | seed4 | seed_sweep | base | 0.012055 | 0.002512 | 0.004450 |
| 2 | tf0 | rollout_hyperparams | base | 0.013086 | 0.005916 | 0.011181 |
| 3 | exp6 | direct_models | base | 0.013277 | 0.032935 | 0.062935 |
| 4 | exp7_cvar | direct_models | cvar | 0.014177 | 0.002861 | 0.005142 |
| 5 | xytheta_cvar | loss_variants | cvar | 0.016629 | 0.004009 | 0.007389 |
| 6 | seed2 | seed_sweep | base | 0.016643 | 0.003266 | 0.005967 |
| 7 | seed1 | seed_sweep | base | 0.019605 | 0.004236 | 0.007834 |
| 8 | exp7 | direct_models | base | 0.020754 | 0.009454 | 0.017889 |
| 9 | seed1_cvar | seed_sweep | cvar | 0.022850 | 0.005032 | 0.009423 |
| 10 | seed4_cvar | seed_sweep | cvar | 0.023963 | 0.007352 | 0.013754 |

---

## Recommendations

**Best Overall Model:** `seed4` (seed_sweep, base)
- Average Trajectory State MSE: 0.002512
- Worst-Case P95: 0.012055

**CVaR Loss Recommendation:** ❌ **REJECT**
- CVaR degrades both average (121.4%) and worst-case (32.3%) performance
- Recommend sticking with standard loss formulation

---

## Appendix: Experiment Configurations

### CVaR Configuration
```json
"rollout_cvar": {
  "enabled": true,
  "alpha": 0.2,
  "apply_to": "rollout_plus_pose",
  "min_k": 1
}
```

### Experiment Categories

1. **Seed Sweep:** 5 random seeds (0-4) to measure variance
2. **Rollout Hyperparameters:**
   - h20: horizon=20 (vs baseline h=10)
   - tf0: teacher_forcing=0.0 (vs baseline TF=0.0)
   - tf0p5: teacher_forcing=0.5 (vs baseline TF=0.0)
3. **Loss Variants:**
   - xytheta: pose loss on [x, y, theta] (vs baseline [x, y])
   - resl2: residual L2 weight=0.01 (vs baseline 0.0)
4. **Direct Models:**
   - exp6: Direct model with adapter
   - exp7: Direct model without adapter
