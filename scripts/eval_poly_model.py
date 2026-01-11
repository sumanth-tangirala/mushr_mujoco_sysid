"""
Evaluate the polynomial (physics-based) dynamics model on v3 trajectories.
This script mirrors eval.py but uses the MushrPlant model from poly_sysid.py.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict

import numpy as np
import torch

from mushr_mujoco_sysid.poly_sysid import MushrPlant
from mushr_mujoco_sysid.dataloader import load_shuffled_filenames_v3
from mushr_mujoco_sysid.data import _collect_samples_for_v3_file


def compute_traj_errors(pose_gt, xd_gt, pose_pred, xd_pred):
    """
    Compute trajectory errors matching the neural network evaluation.
    """
    min_len = min(pose_gt.shape[0], pose_pred.shape[0])
    pose_gt = pose_gt[:min_len]
    pose_pred = pose_pred[:min_len]
    xd_gt = xd_gt[:min_len]
    xd_pred = xd_pred[:min_len]

    # Trajectory mean errors
    pose_errors = np.mean((pose_gt - pose_pred) ** 2, axis=0)
    xd_errors = np.mean((xd_gt - xd_pred) ** 2, axis=0)

    traj_pos_mse = float(np.mean(pose_errors[:2]))  # x, y
    traj_vel_mse = float(np.mean(xd_errors))  # vx, vy, w
    traj_state_mse = float(np.mean([traj_pos_mse, traj_vel_mse]))

    # Final state errors
    final_pose_error = (pose_gt[-1] - pose_pred[-1]) ** 2
    final_xd_error = (xd_gt[-1] - xd_pred[-1]) ** 2

    final_pos_mse = float(np.mean(final_pose_error[:2]))
    final_vel_mse = float(np.mean(final_xd_error))
    final_state_mse = float(np.mean([final_pos_mse, final_vel_mse]))

    return {
        'traj_state_mse': traj_state_mse,
        'traj_pos_mse': traj_pos_mse,
        'traj_vel_mse': traj_vel_mse,
        'final_state_mse': final_state_mse,
        'final_pos_mse': final_pos_mse,
        'final_vel_mse': final_vel_mse,
    }


class PolyDynamicsModel:
    """
    Wrapper around MushrPlant that provides a rollout interface matching
    the neural network evaluation.
    """
    def __init__(self):
        self.mushr = MushrPlant()

    def rollout(self, x0, xd0, ut, dt):
        """
        Rollout trajectory given initial state and control sequence.

        Args:
            x0: Initial pose [x, y, theta] (numpy array)
            xd0: Initial velocity [vx, vy, w] (numpy array)
            ut: Control sequence [N, 2] with [velocity_cmd, steering_cmd]
            dt: Timestep sequence [N]

        Returns:
            Dictionary with 'pose' and 'xd' predictions
        """
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


def evaluate_poly_model(
    data_dir: str,
    num_eval_trajs: int,
    random_select: bool,
    seed: int | None,
    output_dir: str,
):
    """
    Evaluate polynomial dynamics model on v3 trajectories.
    """
    # Load trajectory IDs
    all_ids = load_shuffled_filenames_v3()
    if not all_ids:
        raise RuntimeError("No trajectory filenames found in v3 shuffled indices file.")

    if num_eval_trajs <= 0:
        raise ValueError("num_eval_trajs must be positive.")

    if num_eval_trajs > len(all_ids):
        raise ValueError(
            f"num_eval_trajs ({num_eval_trajs}) cannot be greater than total IDs ({len(all_ids)})."
        )

    if random_select:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(all_ids))
        selected_ids = [all_ids[i] for i in perm[:num_eval_trajs]]
    else:
        selected_ids = all_ids[-num_eval_trajs:]

    # Create output directory
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(output_dir, "poly_model_eval", run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Initialize polynomial dynamics model
    poly_model = PolyDynamicsModel()

    traj_metrics = []
    per_traj_results = []
    gt_paths = []
    pred_paths = []

    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    for tid in selected_ids:
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

        # Rollout using polynomial model
        rollout = poly_model.rollout(x0=x0, xd0=xd0, ut=ut_seq, dt=dt_seq)
        pose_pred = rollout["pose"]
        xd_pred = rollout["xd"]

        metrics = compute_traj_errors(
            pose_gt=pose, xd_gt=xd, pose_pred=pose_pred, xd_pred=xd_pred
        )
        if not metrics:
            continue

        metrics["id"] = tid
        traj_metrics.append(metrics)

        gt_paths.append(pose[:, :2])
        pred_paths.append(pose_pred[:, :2])

        per_traj_results.append(
            {
                "id": tid,
                "traj_len": int(min(pose.shape[0], pose_pred.shape[0])),
                **metrics,
            }
        )

        if plt is not None:
            plt.figure()
            plt.plot(pose[:, 0], pose[:, 1], label="gt", color="tab:blue")
            plt.plot(pose_pred[:, 0], pose_pred[:, 1], label="pred", color="tab:orange")
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Trajectory {tid}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"traj_{tid}.png"), dpi=150)
            plt.close()

    if traj_metrics and plt is not None:
        plt.figure()
        for i, (gt_xy, pred_xy) in enumerate(zip(gt_paths, pred_paths)):
            if i == 0:
                plt.plot(
                    gt_xy[:, 0],
                    gt_xy[:, 1],
                    color="tab:blue",
                    alpha=0.4,
                    label="gt",
                )
                plt.plot(
                    pred_xy[:, 0],
                    pred_xy[:, 1],
                    color="tab:orange",
                    alpha=0.4,
                    label="pred",
                )
            else:
                plt.plot(gt_xy[:, 0], gt_xy[:, 1], color="tab:blue", alpha=0.4)
                plt.plot(pred_xy[:, 0], pred_xy[:, 1], color="tab:orange", alpha=0.4)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("All evaluation trajectories - Polynomial Model")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "traj_all.png"), dpi=150)
        plt.close()

    if not traj_metrics:
        print("No valid trajectories were evaluated.")
        return

    evaluated_ids = [m["id"] for m in traj_metrics]

    avg_traj_state_mse = float(np.mean([m["traj_state_mse"] for m in traj_metrics]))
    avg_traj_pos_mse = float(np.mean([m["traj_pos_mse"] for m in traj_metrics]))
    avg_traj_vel_mse = float(np.mean([m["traj_vel_mse"] for m in traj_metrics]))

    avg_final_state_mse = float(np.mean([m["final_state_mse"] for m in traj_metrics]))
    avg_final_pos_mse = float(np.mean([m["final_pos_mse"] for m in traj_metrics]))
    avg_final_vel_mse = float(np.mean([m["final_vel_mse"] for m in traj_metrics]))

    print(f"Evaluated {len(traj_metrics)} trajectories with polynomial model.")
    print(f"Average trajectory-mean state MSE: {avg_traj_state_mse:.6f}")
    print(f"  positions (x, y, theta): {avg_traj_pos_mse:.6f}")
    print(f"  velocities (vx, vy, w): {avg_traj_vel_mse:.6f}")
    print(f"Average final-state MSE: {avg_final_state_mse:.6f}")
    print(f"  final position: {avg_final_pos_mse:.6f}")
    print(f"  final velocity: {avg_final_vel_mse:.6f}")

    summary_path = os.path.join(out_dir, "metrics.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model_type": "polynomial_physics_based",
                "data_dir": data_dir,
                "eval_run_name": run_name,
                "eval_config": {
                    "num_eval_trajs_requested": num_eval_trajs,
                    "num_eval_trajs_evaluated": len(evaluated_ids),
                    "random_select": random_select,
                    "seed": seed,
                    "selected_ids": selected_ids,
                    "evaluated_ids": evaluated_ids,
                },
                "metrics": {
                    "avg_traj_state_mse": avg_traj_state_mse,
                    "avg_traj_pos_mse": avg_traj_pos_mse,
                    "avg_traj_vel_mse": avg_traj_vel_mse,
                    "avg_final_state_mse": avg_final_state_mse,
                    "avg_final_pos_mse": avg_final_pos_mse,
                    "avg_final_vel_mse": avg_final_vel_mse,
                },
                "per_trajectory": per_traj_results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--num-eval-trajs",
        type=int,
        default=100,
        help="Number of trajectories to use for evaluation.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="If set, pick num-eval-trajs randomly from shuffled indices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selection when using --random.",
    )
    args = parser.parse_args()

    evaluate_poly_model(
        data_dir=args.data_dir,
        num_eval_trajs=args.num_eval_trajs,
        random_select=args.random,
        seed=args.seed,
        output_dir=args.output_dir,
    )
