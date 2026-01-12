# Experiments V2 - Quick Summary

**Date:** 2026-01-04
**Experiments:** 24 total (12 base + 12 CVaR variants)
**Status:** ✅ Complete

---

## Key Findings

### ❌ CVaR Loss (alpha=0.2) **NOT RECOMMENDED**

**Average Performance:**
- CVaR **degrades** average performance by **121.4%**
- Base: 0.0846 → CVaR: 0.1873 trajectory state MSE

**Worst-Case Performance (P95):**
- CVaR **degrades** worst-case by **32.3%**
- Base: 0.7641 → CVaR: 1.0110 state MSE P95

**Conclusion:** CVaR loss with alpha=0.2 applied to rollout+pose significantly hurts both average and worst-case performance. The technique may require different hyperparameters or may not be suitable for this problem.

---

## Best Performing Models

### Top 3 Overall (by avg trajectory state MSE):

| Rank | Model | Category | Type | Avg MSE | Worst P95 | Notes |
|------|-------|----------|------|---------|-----------|-------|
| 1 | **seed4** | seed_sweep | **base** | 0.002512 | 0.012055 | Best overall |
| 2 | exp7_cvar | direct_models | cvar | 0.002861 | 0.014177 | Direct, no adapter |
| 3 | seed2 | seed_sweep | base | 0.003266 | 0.016643 | Baseline config |

### Notable Results:

**Seed Variance:** High variance across seeds (0.0025 to 0.32), suggesting training instability
- seed4 (base): 0.0025 ← **BEST**
- seed2 (base): 0.0033
- seed1 (base): 0.0042
- seed0 (base): 0.0676
- seed3 (base): 0.3209 ← unstable

**CVaR Impact by Category:**
- Seed Sweep: +168.3% (much worse)
- Rollout Hyperparams: +93.9% (worse)
- Loss Variants: +104.0% (worse)
- Direct Models: +29.2% (slightly worse)

**Hyperparameter Insights:**
- Horizon=20: 0.0135 (good)
- Teacher Forcing=0.0: 0.0059 (very good)
- Teacher Forcing=0.5: 0.4566 (bad - high TF hurts)
- XYTheta supervision: 0.0533 (decent)
- Residual L2=0.01: 0.0449 (good)

**Model Architecture:**
- Direct with adapter (exp6): 0.0329
- Direct without adapter (exp7): 0.0095 ← **BEST DIRECT MODEL**

---

## Recommendations

### 1. **Use seed4 configuration** (best overall)
- Seed: 4
- Structured model with residual learning
- Rollout loss (h=10) + Pose loss (x,y)
- No CVaR, no regularization

### 2. **Avoid CVaR loss** with current settings
- CVaR (alpha=0.2) consistently degrades performance
- May need different alpha or different application strategy
- Current implementation may have bugs or training issues

### 3. **Teacher forcing=0.0 works best**
- TF=0.0: 0.0059 MSE
- TF=0.5: 0.4566 MSE (77x worse!)
- Use autoregressive rollout without teacher forcing

### 4. **Seed sensitivity is HIGH**
- Run multiple seeds for any new configuration
- seed4 and seed2 consistently outperform
- seed3 appears to have training instability

### 5. **Direct models are viable**
- exp7 (direct, no adapter): 0.0095 MSE
- Competitive with best structured models
- Simpler architecture, fewer components

---

## Files Generated

- `EXPERIMENTS_V2_ANALYSIS_REPORT.md` - Full detailed analysis
- `QUICK_SUMMARY.md` - This file
- 24 experiment directories with:
  - `config.json` - Experiment configuration
  - `best.pt` - Best model checkpoint
  - `losses.csv` - Training history
  - `eval_runs/*/metrics.json` - Evaluation results

---

## Next Steps

1. **Investigate seed variance:** Why does seed3 fail while seed4 succeeds?
2. **Debug CVaR implementation:** Check if CVaR is computing per-sample losses correctly
3. **Try lower CVaR alpha:** Test alpha=0.05 or 0.1 instead of 0.2
4. **Test direct models further:** exp7 shows promise, explore this architecture
5. **Analyze training curves:** Look for overfitting or convergence issues in failed runs
