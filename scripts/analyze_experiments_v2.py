#!/usr/bin/env python3
"""
Analyze experiments-v2 results and generate comprehensive report.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def find_metrics_file(exp_dir: Path) -> Path | None:
    """Find the metrics.json file in eval_runs subdirectory."""
    eval_runs = exp_dir / "eval_runs"
    if not eval_runs.exists():
        return None

    # Find the most recent eval run
    metrics_files = list(eval_runs.glob("*/metrics.json"))
    if not metrics_files:
        return None

    # Return the most recent one
    return sorted(metrics_files)[-1]

def load_experiment_results(experiments_root: Path) -> Dict:
    """Load all experiment results."""
    results = {
        "seed_sweep": {"base": [], "cvar": []},
        "rollout_hyperparams": {"base": [], "cvar": []},
        "loss_variants": {"base": [], "cvar": []},
        "direct_models": {"base": [], "cvar": []},
    }

    # Seed sweep experiments
    for seed in range(5):
        # Base
        base_dir = experiments_root / f"exp2_seed_sweep/residual_seed{seed}"
        metrics_file = find_metrics_file(base_dir)
        if metrics_file:
            with open(metrics_file) as f:
                data = json.load(f)
                results["seed_sweep"]["base"].append({
                    "name": f"seed{seed}",
                    "metrics": data["metrics"],
                    "per_traj": data["per_trajectory"],
                })

        # CVaR
        cvar_dir = experiments_root / f"exp2_seed_sweep_cvar0p2/residual_seed{seed}_cvar0p2"
        metrics_file = find_metrics_file(cvar_dir)
        if metrics_file:
            with open(metrics_file) as f:
                data = json.load(f)
                results["seed_sweep"]["cvar"].append({
                    "name": f"seed{seed}_cvar",
                    "metrics": data["metrics"],
                    "per_traj": data["per_trajectory"],
                })

    # Rollout hyperparameters
    rollout_exps = [
        ("h20", "exp2_h20/residual_h20", "exp2_h20_cvar0p2/residual_h20_cvar0p2"),
        ("tf0", "exp2_tf0/residual_tf0", "exp2_tf0_cvar0p2/residual_tf0_cvar0p2"),
        ("tf0p5", "exp2_tf0p5/residual_tf0p5", "exp2_tf0p5_cvar0p2/residual_tf0p5_cvar0p2"),
    ]
    for name, base_path, cvar_path in rollout_exps:
        # Base
        metrics_file = find_metrics_file(experiments_root / base_path)
        if metrics_file:
            with open(metrics_file) as f:
                data = json.load(f)
                results["rollout_hyperparams"]["base"].append({
                    "name": name,
                    "metrics": data["metrics"],
                    "per_traj": data["per_trajectory"],
                })

        # CVaR
        metrics_file = find_metrics_file(experiments_root / cvar_path)
        if metrics_file:
            with open(metrics_file) as f:
                data = json.load(f)
                results["rollout_hyperparams"]["cvar"].append({
                    "name": f"{name}_cvar",
                    "metrics": data["metrics"],
                    "per_traj": data["per_trajectory"],
                })

    # Loss variants
    loss_exps = [
        ("xytheta", "exp2_xytheta/residual_xytheta", "exp2_xytheta_cvar0p2/residual_xytheta_cvar0p2"),
        ("resl2", "exp2_resl2_0p01/residual_resl2_0p01", "exp2_resl2_0p01_cvar0p2/residual_resl2_0p01_cvar0p2"),
    ]
    for name, base_path, cvar_path in loss_exps:
        # Base
        metrics_file = find_metrics_file(experiments_root / base_path)
        if metrics_file:
            with open(metrics_file) as f:
                data = json.load(f)
                results["loss_variants"]["base"].append({
                    "name": name,
                    "metrics": data["metrics"],
                    "per_traj": data["per_trajectory"],
                })

        # CVaR
        metrics_file = find_metrics_file(experiments_root / cvar_path)
        if metrics_file:
            with open(metrics_file) as f:
                data = json.load(f)
                results["loss_variants"]["cvar"].append({
                    "name": f"{name}_cvar",
                    "metrics": data["metrics"],
                    "per_traj": data["per_trajectory"],
                })

    # Direct models
    direct_exps = [
        ("exp6", "exp6_direct_rollout_pose/direct_rollout_pose", "exp6_direct_rollout_pose_cvar0p2/direct_rollout_pose_cvar0p2"),
        ("exp7", "exp7_direct_no_adapter_rollout_pose/direct_no_adapter_rollout_pose", "exp7_direct_no_adapter_rollout_pose_cvar0p2/direct_no_adapter_rollout_pose_cvar0p2"),
    ]
    for name, base_path, cvar_path in direct_exps:
        # Base
        metrics_file = find_metrics_file(experiments_root / base_path)
        if metrics_file:
            with open(metrics_file) as f:
                data = json.load(f)
                results["direct_models"]["base"].append({
                    "name": name,
                    "metrics": data["metrics"],
                    "per_traj": data["per_trajectory"],
                })

        # CVaR
        metrics_file = find_metrics_file(experiments_root / cvar_path)
        if metrics_file:
            with open(metrics_file) as f:
                data = json.load(f)
                results["direct_models"]["cvar"].append({
                    "name": f"{name}_cvar",
                    "metrics": data["metrics"],
                    "per_traj": data["per_trajectory"],
                })

    return results

def compute_statistics(experiments: List[Dict]) -> Dict:
    """Compute mean, std, worst-case statistics across experiments."""
    if not experiments:
        return {}

    # Extract metrics
    traj_state_mse = [e["metrics"]["avg_traj_state_mse"] for e in experiments]
    traj_pos_mse = [e["metrics"]["avg_traj_pos_mse"] for e in experiments]
    traj_vel_mse = [e["metrics"]["avg_traj_vel_mse"] for e in experiments]

    # Compute worst-case (95th percentile across all trajectories)
    all_traj_state_mse = []
    all_traj_pos_mse = []
    for e in experiments:
        for traj in e["per_traj"]:
            all_traj_state_mse.append(traj["traj_state_mse"])
            all_traj_pos_mse.append(traj["traj_pos_mse"])

    worst_state_p95 = np.percentile(all_traj_state_mse, 95) if all_traj_state_mse else 0
    worst_pos_p95 = np.percentile(all_traj_pos_mse, 95) if all_traj_pos_mse else 0
    worst_state_max = np.max(all_traj_state_mse) if all_traj_state_mse else 0
    worst_pos_max = np.max(all_traj_pos_mse) if all_traj_pos_mse else 0

    return {
        "mean_traj_state_mse": np.mean(traj_state_mse),
        "std_traj_state_mse": np.std(traj_state_mse),
        "mean_traj_pos_mse": np.mean(traj_pos_mse),
        "std_traj_pos_mse": np.std(traj_pos_mse),
        "mean_traj_vel_mse": np.mean(traj_vel_mse),
        "std_traj_vel_mse": np.std(traj_vel_mse),
        "worst_state_p95": worst_state_p95,
        "worst_pos_p95": worst_pos_p95,
        "worst_state_max": worst_state_max,
        "worst_pos_max": worst_pos_max,
    }

def generate_report(results: Dict, output_path: Path):
    """Generate comprehensive markdown report."""

    report = []
    report.append("# Experiments V2: CVaR Loss Analysis Report")
    report.append("")
    report.append("**Date:** 2026-01-04")
    report.append("**Total Experiments:** 24 (12 base + 12 CVaR variants)")
    report.append("**Evaluation:** 50 trajectories per experiment")
    report.append("")
    report.append("---")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")

    # Analyze overall CVaR impact across all categories
    all_base = []
    all_cvar = []
    for category in ["seed_sweep", "rollout_hyperparams", "loss_variants", "direct_models"]:
        all_base.extend(results[category]["base"])
        all_cvar.extend(results[category]["cvar"])

    base_stats = compute_statistics(all_base)
    cvar_stats = compute_statistics(all_cvar)

    report.append("**CVaR Impact on Average Performance:**")
    report.append(f"- Trajectory State MSE: {base_stats['mean_traj_state_mse']:.6f} → {cvar_stats['mean_traj_state_mse']:.6f} ({((cvar_stats['mean_traj_state_mse']/base_stats['mean_traj_state_mse']-1)*100):+.1f}%)")
    report.append(f"- Trajectory Position MSE: {base_stats['mean_traj_pos_mse']:.6f} → {cvar_stats['mean_traj_pos_mse']:.6f} ({((cvar_stats['mean_traj_pos_mse']/base_stats['mean_traj_pos_mse']-1)*100):+.1f}%)")
    report.append("")
    report.append("**CVaR Impact on Worst-Case Performance (95th percentile):**")
    report.append(f"- State MSE P95: {base_stats['worst_state_p95']:.6f} → {cvar_stats['worst_state_p95']:.6f} ({((cvar_stats['worst_state_p95']/base_stats['worst_state_p95']-1)*100):+.1f}%)")
    report.append(f"- Position MSE P95: {base_stats['worst_pos_p95']:.6f} → {cvar_stats['worst_pos_p95']:.6f} ({((cvar_stats['worst_pos_p95']/base_stats['worst_pos_p95']-1)*100):+.1f}%)")
    report.append("")
    report.append("**Key Findings:**")

    # Determine if CVaR helps
    avg_improvement = (cvar_stats['mean_traj_state_mse']/base_stats['mean_traj_state_mse']-1)*100
    worst_improvement = (cvar_stats['worst_state_p95']/base_stats['worst_state_p95']-1)*100

    if avg_improvement < -1:
        report.append(f"- ✅ CVaR improves average performance by {-avg_improvement:.1f}%")
    elif avg_improvement > 1:
        report.append(f"- ⚠️ CVaR degrades average performance by {avg_improvement:.1f}%")
    else:
        report.append("- ➖ CVaR has minimal impact on average performance")

    if worst_improvement < -1:
        report.append(f"- ✅ CVaR improves worst-case performance by {-worst_improvement:.1f}%")
    elif worst_improvement > 1:
        report.append(f"- ⚠️ CVaR degrades worst-case performance by {worst_improvement:.1f}%")
    else:
        report.append("- ➖ CVaR has minimal impact on worst-case performance")

    report.append("")
    report.append("---")
    report.append("")

    # Detailed Analysis by Category
    categories = {
        "seed_sweep": "Seed Sweep Analysis",
        "rollout_hyperparams": "Rollout Hyperparameters Analysis",
        "loss_variants": "Loss Variants Analysis",
        "direct_models": "Direct Models Analysis",
    }

    for cat_key, cat_title in categories.items():
        report.append(f"## {cat_title}")
        report.append("")

        base_exps = results[cat_key]["base"]
        cvar_exps = results[cat_key]["cvar"]

        if not base_exps:
            report.append("No experiments found in this category.")
            report.append("")
            continue

        # Individual results table
        report.append("### Individual Results")
        report.append("")
        report.append("| Experiment | Traj State MSE | Traj Pos MSE | Traj Vel MSE | Worst State P95 |")
        report.append("|------------|----------------|--------------|--------------|-----------------|")

        # Base experiments
        for exp in base_exps:
            name = exp["name"]
            metrics = exp["metrics"]

            # Compute worst-case for this experiment
            traj_state_mses = [t["traj_state_mse"] for t in exp["per_traj"]]
            worst_p95 = np.percentile(traj_state_mses, 95) if traj_state_mses else 0

            report.append(f"| {name} | {metrics['avg_traj_state_mse']:.6f} | {metrics['avg_traj_pos_mse']:.6f} | {metrics['avg_traj_vel_mse']:.6f} | {worst_p95:.6f} |")

        # CVaR experiments
        for exp in cvar_exps:
            name = exp["name"]
            metrics = exp["metrics"]

            # Compute worst-case for this experiment
            traj_state_mses = [t["traj_state_mse"] for t in exp["per_traj"]]
            worst_p95 = np.percentile(traj_state_mses, 95) if traj_state_mses else 0

            report.append(f"| {name} | {metrics['avg_traj_state_mse']:.6f} | {metrics['avg_traj_pos_mse']:.6f} | {metrics['avg_traj_vel_mse']:.6f} | {worst_p95:.6f} |")

        report.append("")

        # Statistics comparison
        if base_exps and cvar_exps:
            base_stats = compute_statistics(base_exps)
            cvar_stats = compute_statistics(cvar_exps)

            report.append("### Base vs CVaR Comparison")
            report.append("")
            report.append("| Metric | Base (mean ± std) | CVaR (mean ± std) | Change |")
            report.append("|--------|-------------------|-------------------|--------|")

            # Average trajectory state MSE
            base_mean = base_stats['mean_traj_state_mse']
            base_std = base_stats['std_traj_state_mse']
            cvar_mean = cvar_stats['mean_traj_state_mse']
            cvar_std = cvar_stats['std_traj_state_mse']
            change_pct = ((cvar_mean / base_mean - 1) * 100) if base_mean > 0 else 0
            report.append(f"| Avg Traj State MSE | {base_mean:.6f} ± {base_std:.6f} | {cvar_mean:.6f} ± {cvar_std:.6f} | {change_pct:+.1f}% |")

            # Worst-case P95
            base_worst = base_stats['worst_state_p95']
            cvar_worst = cvar_stats['worst_state_p95']
            worst_change_pct = ((cvar_worst / base_worst - 1) * 100) if base_worst > 0 else 0
            report.append(f"| Worst State MSE P95 | {base_worst:.6f} | {cvar_worst:.6f} | {worst_change_pct:+.1f}% |")

            # Worst-case max
            base_max = base_stats['worst_state_max']
            cvar_max = cvar_stats['worst_state_max']
            max_change_pct = ((cvar_max / base_max - 1) * 100) if base_max > 0 else 0
            report.append(f"| Worst State MSE Max | {base_max:.6f} | {cvar_max:.6f} | {max_change_pct:+.1f}% |")

            report.append("")

        report.append("---")
        report.append("")

    # Best performing models
    report.append("## Best Performing Models")
    report.append("")

    # Rank all experiments by average trajectory state MSE
    all_experiments = []
    for category in ["seed_sweep", "rollout_hyperparams", "loss_variants", "direct_models"]:
        for exp_type in ["base", "cvar"]:
            for exp in results[category][exp_type]:
                all_experiments.append({
                    "category": category,
                    "type": exp_type,
                    "name": exp["name"],
                    "avg_mse": exp["metrics"]["avg_traj_state_mse"],
                    "avg_pos_mse": exp["metrics"]["avg_traj_pos_mse"],
                    "worst_p95": np.percentile([t["traj_state_mse"] for t in exp["per_traj"]], 95),
                })

    # Sort by average MSE
    all_experiments.sort(key=lambda x: x["avg_mse"])

    report.append("### Top 10 by Average Performance")
    report.append("")
    report.append("| Rank | Experiment | Category | Type | Avg State MSE | Avg Pos MSE | Worst P95 |")
    report.append("|------|------------|----------|------|---------------|-------------|-----------|")

    for i, exp in enumerate(all_experiments[:10], 1):
        report.append(f"| {i} | {exp['name']} | {exp['category']} | {exp['type']} | {exp['avg_mse']:.6f} | {exp['avg_pos_mse']:.6f} | {exp['worst_p95']:.6f} |")

    report.append("")

    # Sort by worst-case P95
    all_experiments.sort(key=lambda x: x["worst_p95"])

    report.append("### Top 10 by Worst-Case Performance (P95)")
    report.append("")
    report.append("| Rank | Experiment | Category | Type | Worst P95 | Avg State MSE | Avg Pos MSE |")
    report.append("|------|------------|----------|------|-----------|---------------|-------------|")

    for i, exp in enumerate(all_experiments[:10], 1):
        report.append(f"| {i} | {exp['name']} | {exp['category']} | {exp['type']} | {exp['worst_p95']:.6f} | {exp['avg_mse']:.6f} | {exp['avg_pos_mse']:.6f} |")

    report.append("")
    report.append("---")
    report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")

    # Find best overall model
    all_experiments.sort(key=lambda x: x["avg_mse"])
    best_model = all_experiments[0]

    report.append(f"**Best Overall Model:** `{best_model['name']}` ({best_model['category']}, {best_model['type']})")
    report.append(f"- Average Trajectory State MSE: {best_model['avg_mse']:.6f}")
    report.append(f"- Worst-Case P95: {best_model['worst_p95']:.6f}")
    report.append("")

    # CVaR effectiveness analysis
    if avg_improvement < -1 and worst_improvement < -1:
        report.append("**CVaR Loss Recommendation:** ✅ **ADOPT**")
        report.append(f"- CVaR improves both average ({-avg_improvement:.1f}%) and worst-case ({-worst_improvement:.1f}%) performance")
        report.append("- Recommend using CVaR loss (alpha=0.2) for all future training")
    elif avg_improvement > 1 and worst_improvement < -5:
        report.append("**CVaR Loss Recommendation:** ⚠️ **CONDITIONAL**")
        report.append(f"- CVaR degrades average performance ({avg_improvement:.1f}%) but significantly improves worst-case ({-worst_improvement:.1f}%)")
        report.append("- Recommend CVaR for safety-critical applications where worst-case matters")
    elif avg_improvement > 1 and worst_improvement > 1:
        report.append("**CVaR Loss Recommendation:** ❌ **REJECT**")
        report.append(f"- CVaR degrades both average ({avg_improvement:.1f}%) and worst-case ({worst_improvement:.1f}%) performance")
        report.append("- Recommend sticking with standard loss formulation")
    else:
        report.append("**CVaR Loss Recommendation:** ➖ **NEUTRAL**")
        report.append("- CVaR has minimal impact on performance")
        report.append("- Use standard loss unless worst-case performance is critical")

    report.append("")
    report.append("---")
    report.append("")

    # Appendix
    report.append("## Appendix: Experiment Configurations")
    report.append("")
    report.append("### CVaR Configuration")
    report.append("```json")
    report.append('"rollout_cvar": {')
    report.append('  "enabled": true,')
    report.append('  "alpha": 0.2,')
    report.append('  "apply_to": "rollout_plus_pose",')
    report.append('  "min_k": 1')
    report.append('}')
    report.append("```")
    report.append("")
    report.append("### Experiment Categories")
    report.append("")
    report.append("1. **Seed Sweep:** 5 random seeds (0-4) to measure variance")
    report.append("2. **Rollout Hyperparameters:**")
    report.append("   - h20: horizon=20 (vs baseline h=10)")
    report.append("   - tf0: teacher_forcing=0.0 (vs baseline TF=0.0)")
    report.append("   - tf0p5: teacher_forcing=0.5 (vs baseline TF=0.0)")
    report.append("3. **Loss Variants:**")
    report.append("   - xytheta: pose loss on [x, y, theta] (vs baseline [x, y])")
    report.append("   - resl2: residual L2 weight=0.01 (vs baseline 0.0)")
    report.append("4. **Direct Models:**")
    report.append("   - exp6: Direct model with adapter")
    report.append("   - exp7: Direct model without adapter")
    report.append("")

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    print(f"Report written to: {output_path}")
    print(f"Total experiments analyzed: {len(all_base) + len(all_cvar)}")

if __name__ == "__main__":
    experiments_root = Path("/common/home/st1122/Projects/mushr_mujoco_sysid/experiments-v2")
    output_path = experiments_root / "EXPERIMENTS_V2_ANALYSIS_REPORT.md"

    print("Loading experiment results...")
    results = load_experiment_results(experiments_root)

    print("Generating report...")
    generate_report(results, output_path)

    print("Done!")
