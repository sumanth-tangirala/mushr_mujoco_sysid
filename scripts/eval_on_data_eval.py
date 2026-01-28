"""
Evaluate a trained dynamics model (structured or direct) on the data_eval
trajectories and produce:

  1. Predicted trajectory text files  (same format as sim_analytical_trajs/)
  2. Per-trajectory 4-panel figures   (Plan | Trajectory | X vs Time | Y vs Time)
  3. A comparison overlay figure      (ground truth, polynomial sysid, learned sysid)
  4. Per-trajectory and aggregate error metrics (position L2 + angle error)
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mushr_mujoco_sysid.model_factory import build_model
from mushr_mujoco_sysid.plant import MushrPlant
from mushr_mujoco_sysid.utils import Standardizer, load_standardizers_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angular difference wrapped to [-pi, pi]."""
    d = a - b
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def read_plan(plan_path: str) -> List[Dict]:
    """
    Parse a plan file.

    Returns a list of segments, each a dict with keys:
        steering, velocity, duration, start_time
    """
    segments: List[Dict] = []
    with open(plan_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            seg = {
                "steering": float(parts[0]),
                "velocity": float(parts[1]),
                "duration": float(parts[2]),
                "start_time": float(parts[3]) if len(parts) > 3 else 0.0,
            }
            segments.append(seg)
    return segments


def build_control_sequence(
    segments: List[Dict], total_time: float, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-step control arrays from plan segments.

    Returns
    -------
    ut : (N, 2)  [velocity_cmd, steering_cmd]  -- matches LearnedDynamicsModel convention
    times : (N+1,)  cumulative time for each state point (0, dt, 2*dt, ...)
    """
    N = int(round(total_time / dt))
    ut = np.zeros((N, 2), dtype=np.float64)
    times = np.arange(N + 1) * dt

    for i in range(N):
        t = i * dt
        # Find active segment
        ctrl_steer = 0.0
        ctrl_vel = 0.0
        for seg in segments:
            seg_start = seg["start_time"]
            seg_end = seg_start + seg["duration"]
            if seg_start <= t < seg_end:
                ctrl_steer = seg["steering"]
                ctrl_vel = seg["velocity"]
                break
        ut[i, 0] = ctrl_vel
        ut[i, 1] = ctrl_steer

    return ut, times


def read_mj_traj(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a MuJoCo ground-truth trajectory file.

    Format: dt  x  y  col3  cos(theta)  col5  col6  sin(theta)
    Columns 4 and 7 are cos(theta) and sin(theta) respectively.

    Returns
    -------
    times : (T,) cumulative time
    xy    : (T, 2) positions
    theta : (T,) heading angle
    """
    data = np.loadtxt(path)
    dts = data[:, 0]
    times = np.cumsum(dts)
    xy = data[:, 1:3]
    theta = np.arctan2(data[:, 7], data[:, 4])
    return times, xy, theta


def read_sim_traj(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a sim / sim_analytical trajectory file.

    Format: time  x  y  theta  vx  vy  w

    Returns
    -------
    times  : (T,)
    xy     : (T, 2)
    theta  : (T,)
    """
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:3], data[:, 3]


def load_nn_model(exp_dir: str, device: torch.device):
    """Load trained NN model and return (model, input_std, target_std, cfg)."""
    config_path = os.path.join(exp_dir, "config.json")
    checkpoint_path = os.path.join(exp_dir, "best.pt")
    std_path = os.path.join(exp_dir, "standardizers.json")

    for p, label in [
        (config_path, "config.json"),
        (checkpoint_path, "best.pt"),
        (std_path, "standardizers.json"),
    ]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{label} not found at {p}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    input_std, target_std = load_standardizers_json(std_path)
    model = build_model(cfg, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    return model, input_std, target_std, cfg


@torch.no_grad()
def batched_rollout(
    model: torch.nn.Module,
    input_std: Standardizer,
    target_std: Standardizer,
    plant: MushrPlant,
    ut_batch: np.ndarray,
    dt: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model rollout for all trajectories in parallel.

    Args:
        model: Trained dynamics model (StructuredDynamicsModel or DirectDynamicsModel)
        input_std: Input standardizer
        target_std: Target standardizer
        plant: MushrPlant instance for SE2 integration
        ut_batch: Control sequences, shape (B, N, 2) [velocity_cmd, steering_cmd]
        dt: Timestep (scalar, same for all)
        device: Torch device

    Returns:
        pose_all: (B, N+1, 3) poses  [x, y, theta]
        xd_all:   (B, N+1, 3) velocities [vx, vy, w]
    """
    from mushr_mujoco_sysid.models.system_models import StructuredDynamicsModel

    B, N, _ = ut_batch.shape

    # Pre-compute standardizer tensors on device
    inp_mean = torch.as_tensor(input_std.mean, device=device, dtype=torch.float32)
    inp_std = torch.as_tensor(input_std.std, device=device, dtype=torch.float32)
    tgt_mean = torch.as_tensor(target_std.mean, device=device, dtype=torch.float32)
    tgt_std = torch.as_tensor(target_std.std, device=device, dtype=torch.float32)

    # Convert controls to tensor: (B, N, 2)
    ut_t = torch.as_tensor(ut_batch, device=device, dtype=torch.float32)
    dt_scalar = torch.tensor(dt, device=device, dtype=torch.float32)

    # Initial states: all zeros
    xd_curr = torch.zeros(B, 3, device=device, dtype=torch.float32)

    # SE2 pose matrices (B, 3, 3) – identity
    pose_mat = plant.SE2(
        torch.zeros(B, device=device), torch.zeros(B, device=device),
        torch.zeros(B, device=device),
    )

    # Storage
    pose_list = [torch.zeros(B, 3, device=device)]  # initial [0,0,0]
    xd_list = [xd_curr.clone()]

    is_structured = isinstance(model, StructuredDynamicsModel)

    for step in range(N):
        ut_step = ut_t[:, step, :]  # (B, 2)

        # Standardize inputs: [xd(3), ut(2)] -> normalized
        feats = torch.cat([xd_curr, ut_step], dim=-1)  # (B, 5)
        feats_norm = (feats - inp_mean) / inp_std

        dt_col = dt_scalar.expand(B)

        if is_structured:
            xd_next_norm = model(feats_norm[:, :3], feats_norm[:, 3:5], dt_col)
        else:
            dt_feat = dt_col.unsqueeze(-1)  # (B, 1)
            xd_next_norm = model(feats_norm[:, :3], feats_norm[:, 3:5], dt_feat)

        # Inverse standardize
        xd_next = xd_next_norm * tgt_std + tgt_mean

        # Integrate SE2 pose
        pose_mat = plant.integrate_SE2(pose_mat, xd_next, dt_scalar)

        # Extract x, y, theta from SE2 matrix
        x = pose_mat[:, 0, 2]
        y = pose_mat[:, 1, 2]
        th = torch.atan2(pose_mat[:, 1, 0], pose_mat[:, 0, 0])
        pose_list.append(torch.stack([x, y, th], dim=-1))
        xd_list.append(xd_next)
        xd_curr = xd_next

    pose_all = torch.stack(pose_list, dim=1).cpu().numpy()  # (B, N+1, 3)
    xd_all = torch.stack(xd_list, dim=1).cpu().numpy()      # (B, N+1, 3)
    return pose_all, xd_all


def compute_errors(
    gt_times: np.ndarray,
    gt_xy: np.ndarray,
    pred_times: np.ndarray,
    pred_xy: np.ndarray,
    pred_theta: Optional[np.ndarray] = None,
    gt_theta: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute trajectory errors by interpolating predictions onto GT time grid.

    Returns dict with avg_pos_err, max_pos_err, final_pos_err,
    and if theta is available: avg_angle_err, max_angle_err, final_angle_err.
    """
    # Interpolate predicted x,y onto ground-truth time points
    pred_x_interp = np.interp(gt_times, pred_times, pred_xy[:, 0])
    pred_y_interp = np.interp(gt_times, pred_times, pred_xy[:, 1])

    pos_errors = np.sqrt(
        (gt_xy[:, 0] - pred_x_interp) ** 2 + (gt_xy[:, 1] - pred_y_interp) ** 2
    )

    result: Dict[str, float] = {
        "avg_pos_err": float(np.mean(pos_errors)),
        "max_pos_err": float(np.max(pos_errors)),
        "final_pos_err": float(pos_errors[-1]),
    }

    # Error at t=1s: find the GT time index closest to 1.0s
    t1s_idx = int(np.argmin(np.abs(gt_times - 1.0)))
    result["t1s_pos_err"] = float(pos_errors[t1s_idx])

    if pred_theta is not None and gt_theta is not None:
        pred_th_interp = np.interp(gt_times, pred_times, pred_theta)
        gt_th_interp = gt_theta  # already on gt_times grid
        angle_errors = np.abs(angle_diff(pred_th_interp, gt_th_interp))
        result["avg_angle_err"] = float(np.mean(angle_errors))
        result["max_angle_err"] = float(np.max(angle_errors))
        result["final_angle_err"] = float(angle_errors[-1])
        result["t1s_angle_err"] = float(angle_errors[t1s_idx])

    return result


def write_sim_traj(
    path: str,
    times: np.ndarray,
    pose: np.ndarray,
    xd: np.ndarray,
) -> None:
    """
    Write trajectory in sim format: time x y theta vx vy w
    """
    with open(path, "w") as f:
        for i in range(len(times)):
            f.write(
                f"{times[i]:.5f} {pose[i, 0]:.5f} {pose[i, 1]:.5f} "
                f"{pose[i, 2]:.5f} {xd[i, 0]:.5f} {xd[i, 1]:.5f} {xd[i, 2]:.5f} \n"
            )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectory_comparison(
    idx: int,
    plan_segments: List[Dict],
    total_time: float,
    dt: float,
    mj_times: np.ndarray,
    mj_xy: np.ndarray,
    poly_times: np.ndarray,
    poly_xy: np.ndarray,
    learned_times: np.ndarray,
    learned_xy: np.ndarray,
    save_path: str,
) -> None:
    """Generate a 4-panel figure matching the reference image style."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Top-left: Plan ---
    ax = axes[0, 0]
    ax.set_title("Plan")
    # Draw control segments as step functions
    for seg in plan_segments:
        t0 = seg["start_time"]
        t1 = t0 + seg["duration"]
        ax.plot([t0, t1], [seg["steering"], seg["steering"]], "k-", linewidth=1.5,
                label="Steering" if seg is plan_segments[0] else "")
        ax.plot([t0, t1], [seg["velocity"], seg["velocity"]], "r-", linewidth=1.5,
                label="Velocity Desired" if seg is plan_segments[0] else "")
    ax.set_xlim(0, total_time)
    ax.set_ylim(-1.2, 1.2)
    ax.legend(loc="upper right")

    # --- Top-right: Trajectory (X-Y) ---
    ax = axes[0, 1]
    ax.set_title("Trajectory")
    ax.plot(mj_xy[:, 0], mj_xy[:, 1], "k-", linewidth=1.5, label="Mujoco")
    ax.plot(poly_xy[:, 0], poly_xy[:, 1], "rx", markersize=3, label="Poly SysId")
    ax.plot(learned_xy[:, 0], learned_xy[:, 1], "b+", markersize=3, label="Learned SysId")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right")

    # --- Bottom-left: X vs Time ---
    ax = axes[1, 0]
    ax.set_title("X vs Time")
    ax.plot(mj_times, mj_xy[:, 0], "k-", linewidth=1.5, label="Mujoco")
    ax.plot(poly_times, poly_xy[:, 0], "rx", markersize=3, label="Poly SysId")
    ax.plot(learned_times, learned_xy[:, 0], "b+", markersize=3, label="Learned SysId")
    ax.set_xlabel("time")
    ax.set_ylabel("X")
    ax.set_xlim(0, total_time)
    ax.legend(loc="upper right")

    # --- Bottom-right: Y vs Time ---
    ax = axes[1, 1]
    ax.set_title("Y vs Time")
    ax.plot(mj_times, mj_xy[:, 1], "k-", linewidth=1.5, label="Mujoco")
    ax.plot(poly_times, poly_xy[:, 1], "rx", markersize=3, label="Poly SysId")
    ax.plot(learned_times, learned_xy[:, 1], "b+", markersize=3, label="Learned SysId")
    ax.set_xlabel("time")
    ax.set_ylabel("Y")
    ax.set_xlim(0, total_time)
    ax.legend(loc="upper right")

    fig.suptitle(f"Trajectory {idx:03d}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_overview(
    all_mj: List[np.ndarray],
    all_poly: List[np.ndarray],
    all_learned: List[np.ndarray],
    save_path: str,
) -> None:
    """Single overlay plot with all trajectories from all three models."""
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, (mj, poly, lr) in enumerate(zip(all_mj, all_poly, all_learned)):
        kw_mj = dict(color="black", alpha=0.35, linewidth=1)
        kw_poly = dict(color="red", alpha=0.35, linewidth=1)
        kw_lr = dict(color="blue", alpha=0.35, linewidth=1)
        if i == 0:
            kw_mj["label"] = "Mujoco"
            kw_poly["label"] = "Poly SysId"
            kw_lr["label"] = "Learned SysId"
        ax.plot(mj[:, 0], mj[:, 1], **kw_mj)
        ax.plot(poly[:, 0], poly[:, 1], **kw_poly)
        ax.plot(lr[:, 0], lr[:, 1], **kw_lr)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("All Evaluation Trajectories")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on data_eval trajectories."
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to experiment directory (config.json + best.pt + standardizers.json).",
    )
    parser.add_argument(
        "--data-eval-dir",
        type=str,
        default="data_eval",
        help="Path to data_eval/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <data_eval_dir>/<exp-dir-name>/).",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1, help="Simulation timestep (default 0.1)."
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=10.0,
        help="Total trajectory duration in seconds (default 10.0).",
    )
    args = parser.parse_args()

    data_eval_dir = args.data_eval_dir
    mj_dir = os.path.join(data_eval_dir, "mj_trajs")
    poly_dir = os.path.join(data_eval_dir, "sim_analytical_trajs")
    exp_name = os.path.basename(os.path.normpath(args.exp_dir))
    output_dir = args.output_dir or os.path.join(data_eval_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    dt = args.dt
    total_time = args.total_time

    # Discover trajectory indices by plan files
    plan_files = sorted(
        f
        for f in os.listdir(mj_dir)
        if f.startswith("plan_") and f.endswith(".txt")
    )
    indices = []
    for pf in plan_files:
        idx_str = pf.replace("plan_", "").replace(".txt", "")
        try:
            indices.append(int(idx_str))
        except ValueError:
            continue
    indices.sort()
    print(f"Found {len(indices)} plan files (indices {indices[0]:03d}–{indices[-1]:03d})")

    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {args.exp_dir}")
    print(f"Device: {device}")
    model, input_std, target_std, cfg = load_nn_model(args.exp_dir, device)
    plant = MushrPlant()
    print(f"Model type: {cfg.get('model', {}).get('type', 'unknown')}")

    # --- Phase 1: Build all control sequences and filter valid indices ---
    valid_indices: List[int] = []
    all_segments: List[List[Dict]] = []
    all_ut: List[np.ndarray] = []
    N = int(round(total_time / dt))
    learned_times = np.arange(N + 1) * dt

    for idx in indices:
        plan_path = os.path.join(mj_dir, f"plan_{idx:03d}.txt")
        mj_path = os.path.join(mj_dir, f"mj_traj_e{idx:03d}.txt")
        if not os.path.isfile(mj_path):
            print(f"  [{idx:03d}] SKIP – MJ trajectory missing")
            continue
        segments = read_plan(plan_path)
        ut, _ = build_control_sequence(segments, total_time, dt)
        valid_indices.append(idx)
        all_segments.append(segments)
        all_ut.append(ut)

    B = len(valid_indices)
    print(f"Running batched rollout for {B} trajectories ({N} steps each)...")

    # --- Phase 2: Batched model rollout ---
    ut_batch = np.stack(all_ut, axis=0)  # (B, N, 2)
    pose_all, xd_all = batched_rollout(
        model=model,
        input_std=input_std,
        target_std=target_std,
        plant=plant,
        ut_batch=ut_batch,
        dt=dt,
        device=device,
    )
    print("Rollout complete. Generating outputs...")

    # --- Phase 3: Per-trajectory outputs, errors, plots ---
    all_mj_xy: List[np.ndarray] = []
    all_poly_xy: List[np.ndarray] = []
    all_learned_xy: List[np.ndarray] = []
    error_rows: List[Dict] = []

    for i, idx in enumerate(valid_indices):
        mj_path = os.path.join(mj_dir, f"mj_traj_e{idx:03d}.txt")
        poly_path = os.path.join(poly_dir, f"traj_{idx:03d}.txt")
        segments = all_segments[i]

        learned_pose = pose_all[i]  # (N+1, 3)
        learned_xd = xd_all[i]     # (N+1, 3)

        # --- Write trajectory file ---
        traj_out_path = os.path.join(output_dir, f"traj_{idx:03d}.txt")
        write_sim_traj(traj_out_path, learned_times, learned_pose, learned_xd)

        # --- Read ground truth ---
        mj_times, mj_xy, mj_theta = read_mj_traj(mj_path)

        # --- Read polynomial prediction (if available) ---
        have_poly = os.path.isfile(poly_path)
        if have_poly:
            poly_times, poly_xy, poly_theta = read_sim_traj(poly_path)
        else:
            poly_times = learned_times
            poly_xy = np.full_like(learned_pose[:, :2], np.nan)
            poly_theta = np.full(len(learned_times), np.nan)

        # --- Compute errors vs MJ ground truth ---
        poly_errs = compute_errors(
            mj_times, mj_xy, poly_times, poly_xy,
            pred_theta=poly_theta, gt_theta=mj_theta,
        )
        learned_errs = compute_errors(
            mj_times, mj_xy, learned_times, learned_pose[:, :2],
            pred_theta=learned_pose[:, 2], gt_theta=mj_theta,
        )

        row = {"idx": idx}
        for key in ["avg_pos_err", "max_pos_err", "final_pos_err", "t1s_pos_err",
                     "avg_angle_err", "max_angle_err", "final_angle_err", "t1s_angle_err"]:
            row[f"poly_{key}"] = poly_errs.get(key, float("nan"))
            row[f"learned_{key}"] = learned_errs.get(key, float("nan"))
        error_rows.append(row)

        # --- Store for overview ---
        all_mj_xy.append(mj_xy)
        all_poly_xy.append(poly_xy)
        all_learned_xy.append(learned_pose[:, :2])

        # --- Per-trajectory plot ---
        plot_path = os.path.join(output_dir, f"traj_{idx:03d}.png")
        plot_trajectory_comparison(
            idx=idx,
            plan_segments=segments,
            total_time=total_time,
            dt=dt,
            mj_times=mj_times,
            mj_xy=mj_xy,
            poly_times=poly_times,
            poly_xy=poly_xy,
            learned_times=learned_times,
            learned_xy=learned_pose[:, :2],
            save_path=plot_path,
        )

        if (i + 1) % 20 == 0 or idx == valid_indices[-1]:
            poly_ang = poly_errs.get('avg_angle_err', float('nan'))
            learned_ang = learned_errs.get('avg_angle_err', float('nan'))
            print(
                f"  [{idx:03d}] poly pos={poly_errs['avg_pos_err']:.4f} ang={poly_ang:.4f}  "
                f"learned pos={learned_errs['avg_pos_err']:.4f} ang={learned_ang:.4f}"
            )

    # --- Overview plot ---
    if all_mj_xy:
        overview_path = os.path.join(output_dir, "comparison_all.png")
        plot_comparison_overview(all_mj_xy, all_poly_xy, all_learned_xy, overview_path)
        print(f"\nSaved overview plot to: {overview_path}")

    # --- Write CSV ---
    if error_rows:
        csv_path = os.path.join(output_dir, "errors.csv")
        fieldnames = list(error_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(error_rows)
        print(f"Saved per-trajectory errors to: {csv_path}")

    # --- Aggregate summary ---
    if error_rows:
        n = len(error_rows)
        summary: Dict = {"n_trajectories": n, "exp_dir": args.exp_dir}

        for prefix in ["poly", "learned"]:
            for metric in ["avg_pos_err", "max_pos_err", "final_pos_err", "t1s_pos_err",
                           "avg_angle_err", "max_angle_err", "final_angle_err", "t1s_angle_err"]:
                key = f"{prefix}_{metric}"
                vals = [r[key] for r in error_rows if not np.isnan(r.get(key, float("nan")))]
                if vals:
                    summary[f"{key}_mean"] = float(np.mean(vals))
                    summary[f"{key}_std"] = float(np.std(vals))

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to: {summary_path}")

        # --- Print table ---
        print()
        print("=" * 90)
        print("EVALUATION RESULTS: Polynomial vs Learned SysId  (errors vs MuJoCo GT)")
        print("=" * 90)
        print(f"Trajectories evaluated: {n}")
        print()
        print(f"{'Metric':<35} {'Polynomial':>22} {'Learned':>22}")
        print("-" * 90)

        def _fmt(key_prefix, metric_name, label):
            p_key = f"poly_{key_prefix}_mean"
            p_std = f"poly_{key_prefix}_std"
            l_key = f"learned_{key_prefix}_mean"
            l_std = f"learned_{key_prefix}_std"
            pv = summary.get(p_key, float("nan"))
            ps = summary.get(p_std, float("nan"))
            lv = summary.get(l_key, float("nan"))
            ls = summary.get(l_std, float("nan"))
            print(f"{label:<35} {pv:>9.5f} +/- {ps:<9.5f} {lv:>9.5f} +/- {ls:<9.5f}")

        _fmt("avg_pos_err", "avg_pos_err", "Avg Position Error (m)")
        _fmt("max_pos_err", "max_pos_err", "Max Position Error (m)")
        _fmt("final_pos_err", "final_pos_err", "Final Position Error (m)")
        _fmt("t1s_pos_err", "t1s_pos_err", "Position Error @ t=1s (m)")
        _fmt("avg_angle_err", "avg_angle_err", "Avg Angle Error (rad)")
        _fmt("max_angle_err", "max_angle_err", "Max Angle Error (rad)")
        _fmt("final_angle_err", "final_angle_err", "Final Angle Error (rad)")
        _fmt("t1s_angle_err", "t1s_angle_err", "Angle Error @ t=1s (rad)")
        print("=" * 90)

        # Improvement ratios
        p_avg = summary.get("poly_avg_pos_err_mean", float("nan"))
        l_avg = summary.get("learned_avg_pos_err_mean", float("nan"))
        if l_avg > 0 and p_avg > 0:
            print(f"\nPosition error ratio (poly/learned): {p_avg / l_avg:.2f}x")
        p_ang = summary.get("poly_avg_angle_err_mean", float("nan"))
        l_ang = summary.get("learned_avg_angle_err_mean", float("nan"))
        if l_ang > 0 and p_ang > 0:
            print(f"Angle error ratio    (poly/learned): {p_ang / l_ang:.2f}x")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
