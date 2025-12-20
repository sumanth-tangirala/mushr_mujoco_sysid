import argparse
import json
import os
from datetime import datetime
from time import perf_counter
from typing import Dict, List

import numpy as np
import torch

from mushr_mujoco_sysid.dataloader import load_shuffled_indices
from mushr_mujoco_sysid.data import _collect_samples_for_id, load_datasets
from mushr_mujoco_sysid.evaluation import TrajectoryEvaluator, compute_traj_errors
from mushr_mujoco_sysid.model import LearnedDynamicsModel
from mushr_mujoco_sysid.model_factory import build_model
from mushr_mujoco_sysid.utils import load_standardizers_json, save_standardizers_json


def _benchmark_model_speed(
    dyn_model: LearnedDynamicsModel, steps: int = 1000
) -> Dict[str, float]:
    xd = np.zeros(3, dtype=np.float32)
    ut = np.array([1.0, 0.1], dtype=np.float32)
    dt = 0.01

    for _ in range(10):
        xd = dyn_model.predict_xd_next(xd, ut, dt)

    start = perf_counter()
    for _ in range(steps):
        xd = dyn_model.predict_xd_next(xd, ut, dt)
    elapsed = perf_counter() - start
    time_per_step = elapsed / steps if steps > 0 else float("nan")
    steps_per_sec = steps / elapsed if elapsed > 0 else float("inf")

    return {
        "steps": steps,
        "total_time_sec": float(elapsed),
        "time_per_step_sec": float(time_per_step),
        "steps_per_sec": float(steps_per_sec),
    }


def evaluate_model(
    exp_dir: str,
    num_eval_trajs: int,
    random_select: bool,
    seed: int | None,
) -> None:
    config_path = os.path.join(exp_dir, "config.json")
    checkpoint_path = os.path.join(exp_dir, "best.pt")
    out_root = os.path.join(exp_dir, "eval_runs")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    with open(config_path, "r") as f:
        cfg: Dict = json.load(f)

    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    device = torch.device(cfg.get("training", {}).get("device", "cpu"))

    # Ensure data_dir points to an existing directory; if not, fall back to repo data/sysid_trajs.
    data_cfg = cfg.get("data", {})
    data_dir_cfg = data_cfg.get("data_dir")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback_data_dir = os.path.join(repo_root, "data", "sysid_trajs")
    if not data_dir_cfg or not os.path.isdir(data_dir_cfg):
        if os.path.isdir(fallback_data_dir):
            cfg.setdefault("data", {})["data_dir"] = fallback_data_dir
        else:
            raise FileNotFoundError(
                f"Configured data_dir '{data_dir_cfg}' does not exist and "
                f"fallback '{fallback_data_dir}' is also missing."
            )

    # Standardizers: prefer experiment-local JSON; fall back to recomputing once.
    std_path = os.path.join(exp_dir, "standardizers.json")
    if os.path.isfile(std_path):
        input_std, target_std = load_standardizers_json(std_path)
    else:
        _, _, meta = load_datasets(cfg)
        input_std = meta["input_std"]
        target_std = meta["target_std"]
        save_standardizers_json(std_path, input_std, target_std)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(cfg, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
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

    speed_metrics = _benchmark_model_speed(dyn_model)

    all_ids = load_shuffled_indices()
    if not all_ids:
        raise RuntimeError("No trajectory IDs found in shuffled indices file.")

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

    data_dir = cfg["data"]["data_dir"]

    traj_metrics = []
    per_traj_results = []
    gt_paths = []
    pred_paths = []

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None

    for tid in selected_ids:
        samples, traj = _collect_samples_for_id(tid, data_dir)
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

        rollout = traj_evaluator.rollout(x0=x0, xd0=xd0, ut=ut_seq, dt=dt_seq)
        pose_pred = rollout["pose"]
        xd_pred = rollout["xd"]

        metrics = compute_traj_errors(
            pose_gt=pose, xd_gt=xd, pose_pred=pose_pred, xd_pred=xd_pred
        )
        if not metrics:
            continue
        metrics["id"] = int(tid)
        traj_metrics.append(metrics)

        gt_paths.append(pose[:, :2])
        pred_paths.append(pose_pred[:, :2])

        per_traj_results.append(
            {
                "id": int(tid),
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
        plt.title("All evaluation trajectories")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "traj_all.png"), dpi=150)
        plt.close()

    if not traj_metrics:
        print("No valid trajectories were evaluated.")
        return

    evaluated_ids = [int(m["id"]) for m in traj_metrics]

    avg_traj_state_mse = float(np.mean([m["traj_state_mse"] for m in traj_metrics]))
    avg_traj_pos_mse = float(np.mean([m["traj_pos_mse"] for m in traj_metrics]))
    avg_traj_vel_mse = float(np.mean([m["traj_vel_mse"] for m in traj_metrics]))

    avg_final_state_mse = float(np.mean([m["final_state_mse"] for m in traj_metrics]))
    avg_final_pos_mse = float(np.mean([m["final_pos_mse"] for m in traj_metrics]))
    avg_final_vel_mse = float(np.mean([m["final_vel_mse"] for m in traj_metrics]))

    print(f"Evaluated {len(traj_metrics)} trajectories.")
    print(f"Average trajectory-mean state MSE: {avg_traj_state_mse:.6f}")
    print(f"  positions (x, y, theta): {avg_traj_pos_mse:.6f}")
    print(f"  velocities (vx, vy, w): {avg_traj_vel_mse:.6f}")
    print(f"Average final-state MSE: {avg_final_state_mse:.6f}")
    print(f"  final position: {avg_final_pos_mse:.6f}")
    print(f"  final velocity: {avg_final_vel_mse:.6f}")
    print(
        f"Model speed: {speed_metrics['steps_per_sec']:.1f} steps/s "
        f"(time per step {speed_metrics['time_per_step_sec']*1e3:.3f} ms)"
    )

    summary_path = os.path.join(out_dir, "metrics.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "experiment_dir": exp_dir,
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "device": str(device),
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
                "speed": speed_metrics,
                "per_trajectory": per_traj_results,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to experiment directory containing config.json and best.pt.",
    )
    parser.add_argument(
        "--num-eval-trajs",
        type=int,
        default=20,
        help="Number of trajectories to use for evaluation.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help=(
            "If set, pick num-eval-trajs randomly from shuffled indices. "
            "Otherwise, use the last num-eval-trajs from the shuffled indices."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for selection when using --random.",
    )
    args = parser.parse_args()
    evaluate_model(
        exp_dir=args.exp_dir,
        num_eval_trajs=args.num_eval_trajs,
        random_select=args.random,
        seed=args.seed,
    )
