"""
Comprehensive comparison of Polynomial vs Neural Network dynamics models.
Generates detailed metrics and visualizations.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mushr_mujoco_sysid.poly_sysid import MushrPlant
from mushr_mujoco_sysid.dataloader import load_shuffled_filenames_v3
from mushr_mujoco_sysid.data import _collect_samples_for_v3_file, load_datasets
from mushr_mujoco_sysid.evaluation import TrajectoryEvaluator
from mushr_mujoco_sysid.model import LearnedDynamicsModel
from mushr_mujoco_sysid.model_factory import build_model
from mushr_mujoco_sysid.utils import load_standardizers_json


class PolyDynamicsModel:
    """Wrapper around MushrPlant for trajectory rollout."""

    def __init__(self):
        self.mushr = MushrPlant()

    def rollout(self, x0, xd0, ut, dt):
        """Rollout trajectory given initial state and control sequence."""
        N = ut.shape[0]

        # Convert to torch
        x0_torch = self.mushr.SE2(
            torch.tensor(x0[0], dtype=torch.float32),
            torch.tensor(x0[1], dtype=torch.float32),
            torch.tensor(x0[2], dtype=torch.float32)
        )
        xd_torch = torch.tensor(xd0, dtype=torch.float32)

        # Storage for predictions
        pose_pred = np.zeros((N + 1, 3))
        xd_pred = np.zeros((N + 1, 3))

        # Initial state
        pose_pred[0] = x0
        xd_pred[0] = xd0

        x_curr = x0_torch
        xd_curr = xd_torch

        for i in range(N):
            ut_torch = torch.tensor(ut[i], dtype=torch.float32)
            dt_val = float(dt[i])

            # Predict next velocity state
            xd_next = self.mushr.xdot(xd_curr, ut_torch, dt_val)

            # Integrate pose
            x_next = self.mushr.integrate_SE2(x_curr, xd_next, dt_val)

            # Extract pose from SE2 matrix
            r21 = x_next[1, 0]
            r11 = x_next[0, 0]
            theta = torch.atan2(r21, r11)

            pose_pred[i + 1] = [
                float(x_next[0, 2]),
                float(x_next[1, 2]),
                float(theta)
            ]
            xd_pred[i + 1] = xd_next.numpy()

            # Update current state
            x_curr = x_next
            xd_curr = xd_next

        return {
            'pose': pose_pred,
            'xd': xd_pred
        }


def compute_detailed_metrics(pose_gt, xd_gt, pose_pred, xd_pred):
    """
    Compute detailed error metrics for trajectory.

    Returns:
        Dictionary with avg/worst position/velocity errors
    """
    min_len = min(pose_gt.shape[0], pose_pred.shape[0])
    pose_gt = pose_gt[:min_len]
    pose_pred = pose_pred[:min_len]
    xd_gt = xd_gt[:min_len]
    xd_pred = xd_pred[:min_len]

    # Position errors (x, y only)
    pos_errors = np.sqrt(np.sum((pose_gt[:, :2] - pose_pred[:, :2]) ** 2, axis=1))

    # Velocity errors (vx, vy, w)
    vel_errors = np.sqrt(np.sum((xd_gt - xd_pred) ** 2, axis=1))

    return {
        'avg_pos_error': float(np.mean(pos_errors)),
        'worst_pos_error': float(np.max(pos_errors)),
        'avg_vel_error': float(np.mean(vel_errors)),
        'worst_vel_error': float(np.max(vel_errors)),
        'final_pos_error': float(pos_errors[-1]),
        'final_vel_error': float(vel_errors[-1]),
        'pos_errors_full': pos_errors,  # For plotting
        'vel_errors_full': vel_errors,  # For plotting
    }


def load_nn_model(exp_dir: str, device: torch.device):
    """Load trained neural network model."""
    config_path = os.path.join(exp_dir, "config.json")
    checkpoint_path = os.path.join(exp_dir, "best.pt")
    std_path = os.path.join(exp_dir, "standardizers.json")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Load standardizers
    if os.path.isfile(std_path):
        input_std, target_std = load_standardizers_json(std_path)
    else:
        raise FileNotFoundError(f"Standardizers not found at {std_path}")

    # Build and load model
    model = build_model(cfg, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    dyn_model = LearnedDynamicsModel(
        model=model,
        input_std=input_std,
        target_std=target_std,
        device=device,
    )

    traj_evaluator = TrajectoryEvaluator(dynamics_model=dyn_model, device=device)

    return traj_evaluator, cfg


def compare_models(
    nn_exp_dir: str,
    data_dir: str,
    num_eval_trajs: int,
    random_select: bool,
    seed: int | None,
    output_dir: str,
):
    """
    Compare polynomial and neural network models on same trajectories.
    """
    # Load trajectory IDs
    all_ids = load_shuffled_filenames_v3()
    if not all_ids:
        raise RuntimeError("No trajectory filenames found.")

    if num_eval_trajs > len(all_ids):
        raise ValueError(f"num_eval_trajs ({num_eval_trajs}) > total IDs ({len(all_ids)})")

    if random_select:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(all_ids))
        selected_ids = [all_ids[i] for i in perm[:num_eval_trajs]]
    else:
        selected_ids = all_ids[-num_eval_trajs:]

    # Create output directory
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(output_dir, "poly_vs_nn_comparison", run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    poly_model = PolyDynamicsModel()
    nn_evaluator, nn_cfg = load_nn_model(nn_exp_dir, device)

    print(f"Loaded NN model from: {nn_exp_dir}")
    print(f"Evaluating on {len(selected_ids)} trajectories...")
    print()

    # Storage for results
    poly_metrics = []
    nn_metrics = []
    comparison_data = []

    # Collect all trajectory data for aggregate plotting
    all_gt_paths = []
    all_poly_paths = []
    all_nn_paths = []

    for i, tid in enumerate(selected_ids):
        samples, traj = _collect_samples_for_v3_file(tid, data_dir)
        _ = samples
        if not traj:
            continue

        xd = traj["xd"]
        pose = traj["pose"]
        ut = traj["ut"]
        dt = traj["dt"]

        if xd.shape[0] < 2:
            continue

        x0 = pose[0]
        xd0 = xd[0]
        ut_seq = ut[:-1]
        dt_seq = dt[:-1]

        # Rollout both models
        poly_rollout = poly_model.rollout(x0=x0, xd0=xd0, ut=ut_seq, dt=dt_seq)
        nn_rollout = nn_evaluator.rollout(x0=x0, xd0=xd0, ut=ut_seq, dt=dt_seq)

        pose_poly = poly_rollout["pose"]
        xd_poly = poly_rollout["xd"]
        pose_nn = nn_rollout["pose"]
        xd_nn = nn_rollout["xd"]

        # Compute metrics
        poly_metrics_traj = compute_detailed_metrics(pose, xd, pose_poly, xd_poly)
        nn_metrics_traj = compute_detailed_metrics(pose, xd, pose_nn, xd_nn)

        poly_metrics_traj["id"] = tid
        nn_metrics_traj["id"] = tid

        poly_metrics.append(poly_metrics_traj)
        nn_metrics.append(nn_metrics_traj)

        # Store comparison data
        comparison_data.append({
            "id": tid,
            "traj_len": pose.shape[0],
            "poly": poly_metrics_traj,
            "nn": nn_metrics_traj,
        })

        # Store paths for aggregate plot
        all_gt_paths.append(pose[:, :2])
        all_poly_paths.append(pose_poly[:, :2])
        all_nn_paths.append(pose_nn[:, :2])

        # Individual trajectory plot
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 3, figure=fig)

        # Plot 1: Trajectories
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(pose[:, 0], pose[:, 1], 'k-', label='Ground Truth', linewidth=2, alpha=0.7)
        ax1.plot(pose_poly[:, 0], pose_poly[:, 1], 'r--', label='Polynomial', linewidth=2)
        ax1.plot(pose_nn[:, 0], pose_nn[:, 1], 'b:', label='Neural Net', linewidth=2)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title(f'Trajectory Comparison - {tid}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # Plot 2: Position errors over time
        ax2 = fig.add_subplot(gs[1])
        timesteps = np.arange(len(poly_metrics_traj['pos_errors_full']))
        ax2.plot(timesteps, poly_metrics_traj['pos_errors_full'], 'r-', label='Polynomial', linewidth=2)
        ax2.plot(timesteps, nn_metrics_traj['pos_errors_full'], 'b-', label='Neural Net', linewidth=2)
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Position Error Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Velocity errors over time
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(timesteps, poly_metrics_traj['vel_errors_full'], 'r-', label='Polynomial', linewidth=2)
        ax3.plot(timesteps, nn_metrics_traj['vel_errors_full'], 'b-', label='Neural Net', linewidth=2)
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Velocity Error (m/s)')
        ax3.set_title('Velocity Error Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"comparison_{tid}.png"), dpi=150, bbox_inches='tight')
        plt.close()

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(selected_ids)} trajectories...")

    print(f"\nProcessed all {len(comparison_data)} trajectories")
    print()

    # Create aggregate plot with all trajectories
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # Plot 1: All trajectories
    ax1 = fig.add_subplot(gs[0])
    for i, (gt_xy, poly_xy, nn_xy) in enumerate(zip(all_gt_paths, all_poly_paths, all_nn_paths)):
        if i == 0:
            ax1.plot(gt_xy[:, 0], gt_xy[:, 1], 'k-', alpha=0.3, linewidth=1, label='Ground Truth')
            ax1.plot(poly_xy[:, 0], poly_xy[:, 1], 'r-', alpha=0.3, linewidth=1, label='Polynomial')
            ax1.plot(nn_xy[:, 0], nn_xy[:, 1], 'b-', alpha=0.3, linewidth=1, label='Neural Net')
        else:
            ax1.plot(gt_xy[:, 0], gt_xy[:, 1], 'k-', alpha=0.3, linewidth=1)
            ax1.plot(poly_xy[:, 0], poly_xy[:, 1], 'r-', alpha=0.3, linewidth=1)
            ax1.plot(nn_xy[:, 0], nn_xy[:, 1], 'b-', alpha=0.3, linewidth=1)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title(f'All {len(all_gt_paths)} Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot 2: Distribution of average position errors
    ax2 = fig.add_subplot(gs[1])
    poly_avg_pos_errors = [m['avg_pos_error'] for m in poly_metrics]
    nn_avg_pos_errors = [m['avg_pos_error'] for m in nn_metrics]

    bins = np.linspace(0, max(max(poly_avg_pos_errors), max(nn_avg_pos_errors)), 30)
    ax2.hist(poly_avg_pos_errors, bins=bins, alpha=0.5, color='red', label='Polynomial', edgecolor='black')
    ax2.hist(nn_avg_pos_errors, bins=bins, alpha=0.5, color='blue', label='Neural Net', edgecolor='black')
    ax2.set_xlabel('Average Position Error (m)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Avg Position Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Distribution of average velocity errors
    ax3 = fig.add_subplot(gs[2])
    poly_avg_vel_errors = [m['avg_vel_error'] for m in poly_metrics]
    nn_avg_vel_errors = [m['avg_vel_error'] for m in nn_metrics]

    bins = np.linspace(0, max(max(poly_avg_vel_errors), max(nn_avg_vel_errors)), 30)
    ax3.hist(poly_avg_vel_errors, bins=bins, alpha=0.5, color='red', label='Polynomial', edgecolor='black')
    ax3.hist(nn_avg_vel_errors, bins=bins, alpha=0.5, color='blue', label='Neural Net', edgecolor='black')
    ax3.set_xlabel('Average Velocity Error (m/s)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Avg Velocity Errors')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_trajectories_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Compute aggregate statistics
    poly_stats = {
        'avg_pos_error_mean': float(np.mean(poly_avg_pos_errors)),
        'avg_pos_error_std': float(np.std(poly_avg_pos_errors)),
        'worst_pos_error_mean': float(np.mean([m['worst_pos_error'] for m in poly_metrics])),
        'worst_pos_error_max': float(np.max([m['worst_pos_error'] for m in poly_metrics])),
        'avg_vel_error_mean': float(np.mean(poly_avg_vel_errors)),
        'avg_vel_error_std': float(np.std(poly_avg_vel_errors)),
        'worst_vel_error_mean': float(np.mean([m['worst_vel_error'] for m in poly_metrics])),
        'worst_vel_error_max': float(np.max([m['worst_vel_error'] for m in poly_metrics])),
    }

    nn_stats = {
        'avg_pos_error_mean': float(np.mean(nn_avg_pos_errors)),
        'avg_pos_error_std': float(np.std(nn_avg_pos_errors)),
        'worst_pos_error_mean': float(np.mean([m['worst_pos_error'] for m in nn_metrics])),
        'worst_pos_error_max': float(np.max([m['worst_pos_error'] for m in nn_metrics])),
        'avg_vel_error_mean': float(np.mean(nn_avg_vel_errors)),
        'avg_vel_error_std': float(np.std(nn_avg_vel_errors)),
        'worst_vel_error_mean': float(np.mean([m['worst_vel_error'] for m in nn_metrics])),
        'worst_vel_error_max': float(np.max([m['worst_vel_error'] for m in nn_metrics])),
    }

    # Print summary
    print("=" * 100)
    print("POLYNOMIAL vs NEURAL NETWORK COMPARISON RESULTS")
    print("=" * 100)
    print()
    print(f"Neural Network Model: {nn_exp_dir}")
    print(f"Trajectories Evaluated: {len(comparison_data)}")
    print()
    print("-" * 100)
    print(f"{'Metric':<40} {'Polynomial':>25} {'Neural Net':>25}")
    print("-" * 100)
    print(f"{'Avg Position Error (mean ± std)':<40} {poly_stats['avg_pos_error_mean']:>10.6f} ± {poly_stats['avg_pos_error_std']:>8.6f} {nn_stats['avg_pos_error_mean']:>10.6f} ± {nn_stats['avg_pos_error_std']:>8.6f}")
    print(f"{'Worst Position Error (mean)':<40} {poly_stats['worst_pos_error_mean']:>25.6f} {nn_stats['worst_pos_error_mean']:>25.6f}")
    print(f"{'Worst Position Error (max)':<40} {poly_stats['worst_pos_error_max']:>25.6f} {nn_stats['worst_pos_error_max']:>25.6f}")
    print(f"{'Avg Velocity Error (mean ± std)':<40} {poly_stats['avg_vel_error_mean']:>10.6f} ± {poly_stats['avg_vel_error_std']:>8.6f} {nn_stats['avg_vel_error_mean']:>10.6f} ± {nn_stats['avg_vel_error_std']:>8.6f}")
    print(f"{'Worst Velocity Error (mean)':<40} {poly_stats['worst_vel_error_mean']:>25.6f} {nn_stats['worst_vel_error_mean']:>25.6f}")
    print(f"{'Worst Velocity Error (max)':<40} {poly_stats['worst_vel_error_max']:>25.6f} {nn_stats['worst_vel_error_max']:>25.6f}")
    print("-" * 100)
    print()
    print(f"IMPROVEMENT FACTOR (Polynomial / Neural Net):")
    print(f"  Avg Position Error:   {poly_stats['avg_pos_error_mean'] / nn_stats['avg_pos_error_mean']:.2f}x")
    print(f"  Worst Position Error: {poly_stats['worst_pos_error_mean'] / nn_stats['worst_pos_error_mean']:.2f}x")
    print(f"  Avg Velocity Error:   {poly_stats['avg_vel_error_mean'] / nn_stats['avg_vel_error_mean']:.2f}x")
    print(f"  Worst Velocity Error: {poly_stats['worst_vel_error_mean'] / nn_stats['worst_vel_error_mean']:.2f}x")
    print()
    print("=" * 100)

    # Save results
    results = {
        "nn_model": nn_exp_dir,
        "num_trajectories": len(comparison_data),
        "polynomial_stats": poly_stats,
        "neural_net_stats": nn_stats,
        "improvement_factors": {
            "avg_pos_error": poly_stats['avg_pos_error_mean'] / nn_stats['avg_pos_error_mean'],
            "worst_pos_error": poly_stats['worst_pos_error_mean'] / nn_stats['worst_pos_error_mean'],
            "avg_vel_error": poly_stats['avg_vel_error_mean'] / nn_stats['avg_vel_error_mean'],
            "worst_vel_error": poly_stats['worst_vel_error_mean'] / nn_stats['worst_vel_error_mean'],
        },
        "per_trajectory": comparison_data,
    }

    results_path = os.path.join(out_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        # Remove full error arrays for JSON
        for traj in results["per_trajectory"]:
            for model in ["poly", "nn"]:
                traj[model].pop("pos_errors_full", None)
                traj[model].pop("vel_errors_full", None)
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")
    print(f"Plots saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nn-exp-dir",
        type=str,
        required=True,
        help="Path to neural network experiment directory (e.g., experiments/.../v3B_struct_h10_tf0_seed4/v3B_struct_h10_tf0_seed4)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sysid_trajs_v3",
        help="Path to v3 trajectory data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/v3_allstars_controls_vary",
        help="Directory to save comparison results.",
    )
    parser.add_argument(
        "--num-eval-trajs",
        type=int,
        default=100,
        help="Number of trajectories to evaluate.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly select trajectories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    compare_models(
        nn_exp_dir=args.nn_exp_dir,
        data_dir=args.data_dir,
        num_eval_trajs=args.num_eval_trajs,
        random_select=args.random,
        seed=args.seed,
        output_dir=args.output_dir,
    )
