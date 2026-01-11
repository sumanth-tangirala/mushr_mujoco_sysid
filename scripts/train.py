import argparse
import json
import os
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mushr_mujoco_sysid.config_utils import populate_config_defaults
from mushr_mujoco_sysid.data import load_datasets, SnippetDataset
from mushr_mujoco_sysid.model import LearnedDynamicsModel
from mushr_mujoco_sysid.model_factory import build_model
from mushr_mujoco_sysid.models.system_models import StructuredDynamicsModel
from mushr_mujoco_sysid.plant import MushrPlant
from mushr_mujoco_sysid.utils import Standardizer, save_standardizers_json
from mushr_mujoco_sysid.evaluation import TrajectoryEvaluator, compute_traj_errors


def _get_vel_weights(loss_cfg: Dict, device: torch.device) -> torch.Tensor:
    """Get velocity dimension weights as a tensor."""
    weights_dict = loss_cfg.get("weights", {"vx": 1.0, "vy": 1.0, "w": 1.0})
    weights = torch.tensor(
        [weights_dict.get("vx", 1.0), weights_dict.get("vy", 1.0), weights_dict.get("w", 1.0)],
        device=device,
        dtype=torch.float32,
    )
    return weights


def _use_weighted_loss(loss_cfg: Dict) -> bool:
    """Check if per-dimension weighting should be applied."""
    weights_dict = loss_cfg.get("weights", {"vx": 1.0, "vy": 1.0, "w": 1.0})
    vx = weights_dict.get("vx", 1.0)
    vy = weights_dict.get("vy", 1.0)
    w = weights_dict.get("w", 1.0)
    # Use weighted loss only if weights differ from all-ones
    return not (vx == 1.0 and vy == 1.0 and w == 1.0)


def _compute_weighted_mse(
    preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Compute weighted MSE loss on velocity components."""
    # preds, targets: (batch, 3)
    # weights: (3,)
    squared_diff = (preds - targets) ** 2  # (batch, 3)
    weighted_sq_diff = squared_diff * weights.unsqueeze(0)  # (batch, 3)
    return weighted_sq_diff.mean()


def _compute_regularization_loss(
    aux: Dict[str, torch.Tensor],
    ut_raw: torch.Tensor,
    reg_cfg: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute regularization losses from model auxiliaries.

    Returns:
        total_reg_loss: Sum of all regularization terms
        reg_components: Dict mapping reg name to scalar loss value
    """
    device = ut_raw.device
    total_reg = torch.tensor(0.0, device=device)
    components = {}

    # Adapter identity: optional emergency brake for pathological control remapping
    # Typically used at very low weights (1e-4 to 1e-3) if rollout causes instability
    adapter_weight = reg_cfg.get("adapter_identity_weight", 0.0)
    if adapter_weight > 0.0 and "ut_eff" in aux:
        ut_eff = aux["ut_eff"]
        adapter_loss = torch.nn.functional.mse_loss(ut_eff, ut_raw)
        total_reg = total_reg + adapter_weight * adapter_loss
        components["adapter_identity"] = adapter_loss.item()

    residual_weight = reg_cfg.get("residual_l2_weight", 0.0)
    if residual_weight > 0.0 and "residual" in aux:
        residual = aux["residual"]
        residual_loss = (residual ** 2).mean()
        total_reg = total_reg + residual_weight * residual_loss
        components["residual_l2"] = residual_loss.item()

    friction_weight = reg_cfg.get("friction_prior_weight", 0.0)
    if friction_weight > 0.0 and "friction_k" in aux:
        friction_k = aux["friction_k"]
        friction_loss = ((friction_k - 1.0) ** 2).mean()
        total_reg = total_reg + friction_weight * friction_loss
        components["friction_prior"] = friction_loss.item()

    return total_reg, components


def _compute_rollout_loss(
    model: nn.Module,
    snippet_batch: Dict[str, torch.Tensor],
    rollout_cfg: Dict,
    plant: MushrPlant,
    device: torch.device,
    pose_cfg: Optional[Dict] = None,
    target_std = None,
    return_per_sample: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute rollout loss over a batch of snippets.

    Args:
        model: Dynamics model
        snippet_batch: Dict with 'xd', 'ut', 'dt', 'pose' tensors
        rollout_cfg: Rollout loss configuration
        plant: MushrPlant for pose integration
        device: Torch device
        pose_cfg: Optional pose loss configuration
        target_std: Standardizer for unstandardizing velocities (required for pose loss)
        return_per_sample: If True, return per-sample losses for CVaR

    Returns:
        rollout_loss: Scalar tensor for rollout velocity MSE
        pose_loss: Scalar tensor for pose MSE (0 if pose_cfg is None or disabled)
        metrics: Dict with loss components
        rollout_per_sample: Per-sample rollout losses [B] (if return_per_sample=True, else None)
        pose_per_sample: Per-sample pose losses [B] (if return_per_sample=True, else None)
    """
    xd_seq = snippet_batch["xd"].to(device)  # (B, H+1, 3)
    ut_seq = snippet_batch["ut"].to(device)  # (B, H, 2)
    dt_seq = snippet_batch["dt"].to(device)  # (B, H)
    pose_seq = snippet_batch["pose"].to(device)  # (B, H+1, 3)

    B, Hp1, _ = xd_seq.shape
    H = Hp1 - 1

    horizon = rollout_cfg.get("horizon", 10)
    H = min(H, horizon)

    teacher_forcing_prob = rollout_cfg.get("teacher_forcing_prob", 0.0)
    detach_between_steps = rollout_cfg.get("detach_between_steps", False)

    is_structured = isinstance(model, StructuredDynamicsModel)

    # Initialize with first state
    x_curr = xd_seq[:, 0]  # (B, 3)

    vel_losses = []
    pose_losses = []

    # For per-sample tracking (CVaR)
    if return_per_sample:
        vel_losses_per_sample = []  # List of [B] tensors
        pose_losses_per_sample = []  # List of [B] tensors

    # For pose integration, we need to track the pose through SE(2)
    # Start from the ground truth initial pose
    if pose_cfg is not None and pose_cfg.get("enabled", False):
        pose_curr = plant.SE2(
            pose_seq[:, 0, 0], pose_seq[:, 0, 1], pose_seq[:, 0, 2]
        )  # (B, 3, 3)

    for k in range(H):
        # Teacher forcing: with some probability use ground truth instead of prediction
        if teacher_forcing_prob > 0.0 and torch.rand(1).item() < teacher_forcing_prob:
            x_in = xd_seq[:, k]
        else:
            x_in = x_curr

        u_k = ut_seq[:, k]
        dt_k = dt_seq[:, k]

        # Forward pass
        dt_arg = dt_k if is_structured else dt_k.unsqueeze(-1)
        x_next = model(x_in, u_k, dt_arg)

        # Velocity loss: compare to ground truth xd_seq[:, k+1]
        if return_per_sample:
            # Per-sample MSE: (B, 3) -> (B,)
            vel_loss_k_per_sample = ((x_next - xd_seq[:, k + 1]) ** 2).mean(dim=1)
            vel_losses_per_sample.append(vel_loss_k_per_sample)
            vel_loss_k = vel_loss_k_per_sample.mean()
        else:
            vel_loss_k = torch.nn.functional.mse_loss(x_next, xd_seq[:, k + 1])
        vel_losses.append(vel_loss_k)

        # Pose loss: integrate and compare
        if pose_cfg is not None and pose_cfg.get("enabled", False):
            # Unstandardize predicted velocities for physically accurate pose integration
            if target_std is not None:
                x_next_raw = torch.tensor(
                    target_std.inverse(x_next.detach().cpu().numpy()),
                    device=device,
                    dtype=x_next.dtype,
                )
            else:
                x_next_raw = x_next

            pose_curr = plant.integrate_SE2(
                pose_curr,
                x_next_raw,
                dt_k.unsqueeze(-1),
            )

            # Extract pose (x, y, theta) from SE(2) matrix
            pred_x = pose_curr[:, 0, 2]
            pred_y = pose_curr[:, 1, 2]
            pred_theta = torch.atan2(pose_curr[:, 1, 0], pose_curr[:, 0, 0])
            pred_pose = torch.stack([pred_x, pred_y, pred_theta], dim=-1)

            gt_pose = pose_seq[:, k + 1]

            # Compute pose loss on selected components
            components = pose_cfg.get("components", ["x", "y"])

            if return_per_sample:
                # Per-sample pose loss
                pose_err_per_sample = torch.zeros(B, device=device)
                for i, comp in enumerate(["x", "y", "theta"]):
                    if comp in components:
                        if comp == "theta":
                            # Angle difference handling
                            diff = pred_pose[:, i] - gt_pose[:, i]
                            diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
                            pose_err_per_sample = pose_err_per_sample + (diff ** 2)
                        else:
                            pose_err_per_sample = pose_err_per_sample + ((pred_pose[:, i] - gt_pose[:, i]) ** 2)
                pose_err_per_sample = pose_err_per_sample / len(components)
                pose_losses_per_sample.append(pose_err_per_sample)
                pose_err = pose_err_per_sample.mean()
            else:
                pose_err = torch.tensor(0.0, device=device)
                for i, comp in enumerate(["x", "y", "theta"]):
                    if comp in components:
                        if comp == "theta":
                            # Angle difference handling
                            diff = pred_pose[:, i] - gt_pose[:, i]
                            diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
                            pose_err = pose_err + (diff ** 2).mean()
                        else:
                            pose_err = pose_err + ((pred_pose[:, i] - gt_pose[:, i]) ** 2).mean()
                pose_err = pose_err / len(components)

            pose_losses.append(pose_err)

        # Prepare for next step
        if detach_between_steps:
            x_curr = x_next.detach()
        else:
            x_curr = x_next

    rollout_loss = torch.stack(vel_losses).mean()
    pose_loss = torch.tensor(0.0, device=device)
    if pose_losses:
        pose_loss = torch.stack(pose_losses).mean()

    metrics = {
        "rollout_vel_mse": rollout_loss.item(),
    }
    if pose_losses:
        metrics["rollout_pose_mse"] = pose_loss.item()

    # Compute per-sample losses if requested
    rollout_per_sample = None
    pose_per_sample = None
    if return_per_sample:
        # Average over horizon: [H, B] -> [B]
        rollout_per_sample = torch.stack(vel_losses_per_sample).mean(dim=0)
        if pose_losses_per_sample:
            pose_per_sample = torch.stack(pose_losses_per_sample).mean(dim=0)
        else:
            pose_per_sample = torch.zeros(B, device=device)

    return rollout_loss, pose_loss, metrics, rollout_per_sample, pose_per_sample


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Dict,
    snippet_loader: Optional[DataLoader] = None,
    plant: Optional[MushrPlant] = None,
    target_std = None,
) -> Dict[str, float]:
    """
    Train for one epoch with all loss components.

    Returns:
        Dict of loss component names to their average values
    """
    model.train()
    loss_cfg = cfg["training"]["loss"]
    reg_cfg = cfg["training"]["regularization"]
    optim_cfg = cfg["training"]["optim"]

    is_structured = isinstance(model, StructuredDynamicsModel)

    # Determine if we need auxiliaries for regularization
    need_aux = (
        reg_cfg.get("adapter_identity_weight", 0.0) > 0.0
        or reg_cfg.get("residual_l2_weight", 0.0) > 0.0
        or reg_cfg.get("friction_prior_weight", 0.0) > 0.0
    )

    # Per-dimension weighting
    use_weighted = _use_weighted_loss(loss_cfg)
    vel_weights = _get_vel_weights(loss_cfg, device) if use_weighted else None

    # Loss weights
    one_step_cfg = loss_cfg.get("one_step_mse", {})
    one_step_enabled = one_step_cfg.get("enabled", True)
    one_step_weight = one_step_cfg.get("weight", 1.0)

    rollout_cfg = loss_cfg.get("rollout_mse", {})
    rollout_enabled = rollout_cfg.get("enabled", False)
    rollout_weight = rollout_cfg.get("weight", 1.0)

    pose_cfg = loss_cfg.get("pose_mse", {})
    pose_enabled = pose_cfg.get("enabled", False)
    pose_weight = pose_cfg.get("weight", 0.1)

    # CVaR configuration
    cvar_cfg = loss_cfg.get("rollout_cvar", {})
    cvar_enabled = cvar_cfg.get("enabled", False) and rollout_enabled
    if cvar_enabled:
        cvar_alpha = cvar_cfg.get("alpha", 0.2)
        cvar_apply_to = cvar_cfg.get("apply_to", "rollout_plus_pose")
        cvar_min_k = cvar_cfg.get("min_k", 1)
        # Validate alpha
        if not (0.0 < cvar_alpha <= 1.0):
            raise ValueError(f"rollout_cvar.alpha must be in (0, 1], got {cvar_alpha}")

    grad_clip_norm = optim_cfg.get("grad_clip_norm", 0.0)

    # Accumulators
    running_total = 0.0
    running_one_step = 0.0
    running_rollout = 0.0
    running_rollout_mean = 0.0  # Mean over batch (for logging)
    running_rollout_cvar = 0.0  # CVaR reduced (for logging)
    running_pose = 0.0
    running_reg = {k: 0.0 for k in ["adapter_identity", "residual_l2", "friction_prior"]}
    running_k_used = 0.0
    count = 0

    # Create snippet iterator if needed
    snippet_iter = None
    if rollout_enabled and snippet_loader is not None:
        snippet_iter = iter(snippet_loader)

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        xd0 = xb[:, :3]
        ut = xb[:, 3:5]
        dt_raw = xb[:, 5]

        optimizer.zero_grad()

        dt_arg = dt_raw if is_structured else dt_raw.unsqueeze(-1)

        total_loss = torch.tensor(0.0, device=device)

        # One-step MSE loss
        one_step_loss = torch.tensor(0.0, device=device)
        if one_step_enabled:
            if need_aux:
                preds, aux = model(xd0, ut, dt_arg, return_aux=True)
            else:
                preds = model(xd0, ut, dt_arg)
                aux = {}

            if use_weighted and vel_weights is not None:
                one_step_loss = _compute_weighted_mse(preds, yb, vel_weights)
            else:
                one_step_loss = torch.nn.functional.mse_loss(preds, yb)

            total_loss = total_loss + one_step_weight * one_step_loss

            # Regularization losses
            if need_aux:
                reg_loss, reg_components = _compute_regularization_loss(aux, ut, reg_cfg)
                total_loss = total_loss + reg_loss
                for k, v in reg_components.items():
                    running_reg[k] += v * xb.size(0)
        else:
            # If one_step disabled but we need aux for regularization
            if need_aux:
                preds, aux = model(xd0, ut, dt_arg, return_aux=True)
                reg_loss, reg_components = _compute_regularization_loss(aux, ut, reg_cfg)
                total_loss = total_loss + reg_loss
                for k, v in reg_components.items():
                    running_reg[k] += v * xb.size(0)

        # Rollout loss
        rollout_loss = torch.tensor(0.0, device=device)
        rollout_loss_mean = torch.tensor(0.0, device=device)
        rollout_loss_cvar = torch.tensor(0.0, device=device)
        pose_loss = torch.tensor(0.0, device=device)
        k_used = 0
        if rollout_enabled and snippet_iter is not None:
            try:
                snippet_batch = next(snippet_iter)
            except StopIteration:
                snippet_iter = iter(snippet_loader)
                snippet_batch = next(snippet_iter)

            if cvar_enabled:
                # Compute per-sample losses for CVaR
                _, _, _, rollout_per_sample, pose_per_sample = _compute_rollout_loss(
                    model, snippet_batch, rollout_cfg, plant, device,
                    pose_cfg if pose_enabled else None,
                    target_std=target_std,
                    return_per_sample=True
                )

                # Compute per-sample total loss for CVaR selection
                if cvar_apply_to == "rollout_plus_pose" and pose_enabled and pose_per_sample is not None:
                    per_sample_total = rollout_weight * rollout_per_sample + pose_weight * pose_per_sample
                else:
                    per_sample_total = rollout_weight * rollout_per_sample

                # CVaR reduction: select top-k hardest samples
                B = per_sample_total.size(0)
                k = max(cvar_min_k, int(np.ceil(cvar_alpha * B)))
                k = min(k, B)  # Clamp to batch size

                topk_vals, topk_indices = torch.topk(per_sample_total, k, largest=True)
                rollout_loss_cvar = topk_vals.mean()
                rollout_loss_mean = per_sample_total.mean()

                # Use CVaR-selected loss for backprop
                rollout_loss = rollout_loss_mean  # Store mean for logging
                total_loss = total_loss + rollout_loss_cvar

                # Pose loss: we already included it in CVaR if apply_to='rollout_plus_pose'
                # But we still want to log it separately
                if pose_enabled and pose_per_sample is not None:
                    pose_loss = pose_per_sample.mean()
                    # Don't add pose_loss again to total_loss if it was already included in CVaR
                    if cvar_apply_to != "rollout_plus_pose":
                        total_loss = total_loss + pose_weight * pose_loss

                k_used = k

            else:
                # Standard mean reduction (original behavior)
                rollout_loss, pose_loss, _, _, _ = _compute_rollout_loss(
                    model, snippet_batch, rollout_cfg, plant, device,
                    pose_cfg if pose_enabled else None,
                    target_std=target_std,
                    return_per_sample=False
                )
                rollout_loss_mean = rollout_loss
                rollout_loss_cvar = rollout_loss
                total_loss = total_loss + rollout_weight * rollout_loss

                if pose_enabled:
                    total_loss = total_loss + pose_weight * pose_loss

        total_loss.backward()

        # Gradient clipping
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        batch_size = xb.size(0)
        running_total += total_loss.item() * batch_size
        running_one_step += one_step_loss.item() * batch_size
        running_rollout += rollout_loss.item() * batch_size
        running_rollout_mean += rollout_loss_mean.item() * batch_size
        running_rollout_cvar += rollout_loss_cvar.item() * batch_size
        running_pose += pose_loss.item() * batch_size
        running_k_used += k_used
        count += batch_size

    # Compute averages
    results = {
        "total": running_total / max(count, 1),
        "one_step_mse": running_one_step / max(count, 1),
    }

    if rollout_enabled:
        results["rollout_mse"] = running_rollout / max(count, 1)
        if cvar_enabled:
            results["rollout_mean"] = running_rollout_mean / max(count, 1)
            results["rollout_cvar"] = running_rollout_cvar / max(count, 1)
            results["k_used"] = running_k_used / max(len(loader), 1)
            results["alpha"] = cvar_alpha
    if pose_enabled:
        results["pose_mse"] = running_pose / max(count, 1)

    for k, v in running_reg.items():
        if v > 0:
            results[f"reg_{k}"] = v / max(count, 1)

    return results


def evaluate_step(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    loss_cfg = cfg["training"]["loss"]

    is_structured = isinstance(model, StructuredDynamicsModel)

    one_step_cfg = loss_cfg.get("one_step_mse", {})
    one_step_enabled = one_step_cfg.get("enabled", True)

    use_weighted = _use_weighted_loss(loss_cfg)
    vel_weights = _get_vel_weights(loss_cfg, device) if use_weighted else None

    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xd0 = xb[:, :3]
            ut = xb[:, 3:5]
            dt_raw = xb[:, 5]

            dt_arg = dt_raw if is_structured else dt_raw.unsqueeze(-1)
            preds = model(xd0, ut, dt_arg)

            if one_step_enabled:
                if use_weighted and vel_weights is not None:
                    loss = _compute_weighted_mse(preds, yb, vel_weights)
                else:
                    loss = torch.nn.functional.mse_loss(preds, yb)
            else:
                loss = torch.nn.functional.mse_loss(preds, yb)

            running_loss += loss.item() * xb.size(0)
            count += xb.size(0)

    return {"val_loss": running_loss / max(count, 1)}


def evaluate_trajectories(
    model,
    trajectories,
    input_std,
    target_std,
    device,
    plant,
    run_dir,
    make_plots: bool = True,
):
    model.eval()
    dyn_model = LearnedDynamicsModel(
        model=model,
        input_std=input_std,
        target_std=target_std,
        device=device,
    )
    traj_evaluator = TrajectoryEvaluator(
        dynamics_model=dyn_model, plant=plant, device=device
    )

    results = []
    all_pred_xy = []
    all_gt_xy = []

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None
        make_plots = False

    with torch.no_grad():
        for traj in trajectories:
            xd = traj["xd"]
            pose = traj["pose"]
            ut = traj["ut"]
            traj_dt = traj.get("dt")
            if xd.shape[0] < 2 or traj_dt is None:
                continue

            rollout = traj_evaluator.rollout(
                x0=pose[0],
                xd0=xd[0],
                ut=ut[:-1],
                dt=traj_dt[:-1],
            )
            pose_pred = rollout["pose"]
            xd_pred = rollout["xd"]

            metrics = compute_traj_errors(
                pose_gt=pose,
                xd_gt=xd,
                pose_pred=pose_pred,
                xd_pred=xd_pred,
            )
            if not metrics:
                continue

            T = int(min(pose.shape[0], pose_pred.shape[0]))
            results.append(
                {
                    "id": traj["id"],
                    "traj_len": T,
                    **metrics,
                }
            )

            gt_xy = pose[:T, :2]
            pred_xy = pose_pred[:T, :2]
            all_pred_xy.append(pred_xy)
            all_gt_xy.append(gt_xy)

            if make_plots and plt is not None:
                os.makedirs(run_dir, exist_ok=True)
                plt.figure()
                plt.plot(gt_xy[:, 0], gt_xy[:, 1], label="gt")
                plt.plot(pred_xy[:, 0], pred_xy[:, 1], label="pred")
                plt.legend()
                plt.xlabel("x")
                plt.ylabel("y")
                plt.title(f"Trajectory {traj['id']}")
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, f"traj_{traj['id']}.png"), dpi=150)
                plt.close()

    if make_plots and plt is not None and all_pred_xy:
        os.makedirs(run_dir, exist_ok=True)
        plt.figure()
        first = True
        for gt_xy_i, pred_xy_i in zip(all_gt_xy, all_pred_xy):
            if first:
                plt.plot(
                    gt_xy_i[:, 0],
                    gt_xy_i[:, 1],
                    label="gt",
                    alpha=0.4,
                    color="tab:blue",
                )
                plt.plot(
                    pred_xy_i[:, 0],
                    pred_xy_i[:, 1],
                    label="pred",
                    alpha=0.4,
                    color="tab:orange",
                )
                first = False
            else:
                plt.plot(gt_xy_i[:, 0], gt_xy_i[:, 1], alpha=0.4, color="tab:blue")
                plt.plot(
                    pred_xy_i[:, 0], pred_xy_i[:, 1], alpha=0.4, color="tab:orange"
                )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("All trajectories")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "traj_all.png"), dpi=150)
        plt.close()

    return results


def _plot_loss_curve(epochs, values, title, ylabel, filename, run_dir):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    os.makedirs(run_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, values)
    axes[0].set_title(title + " (linear)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel(ylabel)
    axes[1].plot(epochs, values)
    axes[1].set_yscale("log")
    axes[1].set_title(title + " (log)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, filename), dpi=150)
    plt.close(fig)


def train_pipeline(config_path: str):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Populate defaults for backward compatibility
    cfg = populate_config_defaults(cfg)

    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    # experiment directory
    train_cfg = cfg.get("training", {})
    run_root = train_cfg.get("run_root", "experiments")
    run_name = train_cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    train_dataset, val_dataset, meta = load_datasets(cfg)

    std_path = os.path.join(run_dir, "standardizers.json")
    save_standardizers_json(std_path, meta["input_std"], meta["target_std"])

    device = torch.device(cfg["training"].get("device", "cpu"))
    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create snippet loader for rollout loss if enabled
    loss_cfg = cfg["training"]["loss"]
    rollout_cfg = loss_cfg.get("rollout_mse", {})
    snippet_loader = None
    if rollout_cfg.get("enabled", False):
        horizon = rollout_cfg.get("horizon", 10)
        snippet_dataset = SnippetDataset(
            trajectories=meta["train_trajs"],
            horizon=horizon,
            input_std=meta["input_std"],
            target_std=meta["target_std"],
        )
        if len(snippet_dataset) > 0:
            snippet_loader = DataLoader(
                snippet_dataset, batch_size=batch_size, shuffle=True
            )
            print(f"Created snippet dataset with {len(snippet_dataset)} snippets (horizon={horizon})")
        else:
            print("Warning: No valid snippets for rollout loss (trajectories too short?)")

    plant = MushrPlant()
    model = build_model(cfg, device)
    lr = cfg["training"]["lr"]
    weight_decay = cfg["training"].get("weight_decay", 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if cfg["training"].get("lr_scheduler", "").lower() == "cosine":
        t_max = cfg["training"].get("lr_t_max", cfg["training"]["epochs"])
        eta_min = cfg["training"].get("lr_eta_min", 0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )

    best_val = float("inf")
    ckpt_dir = run_dir
    ckpt_path = os.path.join(ckpt_dir, cfg["training"].get("ckpt_name", "best.pt"))

    epochs_list: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []
    eval_vel_mses: List[float] = []
    eval_pos_mses: List[float] = []

    # Additional loss tracking
    train_one_step_losses: List[float] = []
    train_rollout_losses: List[float] = []
    train_pose_losses: List[float] = []

    # Early stopping setup
    early_stop_patience = cfg["training"].get("early_stop_patience", 0)
    early_stop_min_delta = cfg["training"].get("early_stop_min_delta", 1e-6)
    epochs_without_improvement = 0

    # Print training config summary
    print("=" * 60)
    print("Training Configuration Summary")
    print("=" * 60)
    print(f"Validation split mode: {meta.get('val_split_mode', 'timestep')}")
    print(f"One-step MSE: enabled={loss_cfg['one_step_mse']['enabled']}, weight={loss_cfg['one_step_mse']['weight']}")
    print(f"Rollout MSE: enabled={rollout_cfg.get('enabled', False)}, weight={rollout_cfg.get('weight', 1.0)}")
    pose_cfg = loss_cfg.get("pose_mse", {})
    print(f"Pose MSE: enabled={pose_cfg.get('enabled', False)}, weight={pose_cfg.get('weight', 0.1)}")
    print(f"Velocity weights: {loss_cfg.get('weights', {'vx': 1.0, 'vy': 1.0, 'w': 1.0})}")
    reg_cfg = cfg["training"]["regularization"]
    print(f"Regularization: adapter_identity={reg_cfg['adapter_identity_weight']}, "
          f"residual_l2={reg_cfg['residual_l2_weight']}, friction_prior={reg_cfg['friction_prior_weight']}")
    optim_cfg = cfg["training"]["optim"]
    print(f"Grad clip norm: {optim_cfg['grad_clip_norm']}")
    if cfg["model"].get("type", "structured") == "structured":
        fp = cfg["model"].get("friction_param", {})
        print(f"Friction param: mode={fp.get('mode', 'softplus_offset_1')}, k_min={fp.get('k_min', 0.2)}, k_max={fp.get('k_max', 2.0)}")
    if early_stop_patience > 0:
        print(f"Early stopping: enabled (patience={early_stop_patience}, min_delta={early_stop_min_delta})")
    else:
        print("Early stopping: disabled")
    print("=" * 60)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, cfg,
            snippet_loader=snippet_loader, plant=plant,
            target_std=meta["target_std"]
        )
        val_metrics = evaluate_step(model, val_loader, device, cfg)

        train_loss = train_metrics["total"]
        val_loss = val_metrics["val_loss"]

        # Build log message with all loss components
        log_parts = [f"Epoch {epoch:03d}", f"train {train_loss:.6f}", f"val {val_loss:.6f}"]
        if "one_step_mse" in train_metrics:
            log_parts.append(f"1step {train_metrics['one_step_mse']:.6f}")
        if "rollout_mse" in train_metrics:
            log_parts.append(f"roll {train_metrics['rollout_mse']:.6f}")
        if "pose_mse" in train_metrics:
            log_parts.append(f"pose {train_metrics['pose_mse']:.6f}")

        # Log regularization terms
        for k in ["reg_adapter_identity", "reg_residual_l2", "reg_friction_prior"]:
            if k in train_metrics and train_metrics[k] > 0:
                short_name = k.replace("reg_", "")[:8]
                log_parts.append(f"{short_name} {train_metrics[k]:.6f}")

        print(" | ".join(log_parts))

        if scheduler is not None:
            scheduler.step()

        # Check for improvement
        if val_loss < best_val - early_stop_min_delta:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {early_stop_patience} epochs)")
            print(f"Best validation loss: {best_val:.6f}")
            break

        eval_vel = float("nan")
        eval_pos = float("nan")

        if epoch % cfg["training"].get("eval_every", 10) == 0:
            traj_metrics = evaluate_trajectories(
                model,
                meta["heldout_trajs"],
                meta["input_std"],
                meta["target_std"],
                device,
                plant=plant,
                run_dir=run_dir,
            )
            if traj_metrics:
                avg_state = float(np.mean([m["traj_state_mse"] for m in traj_metrics]))
                avg_pos = float(np.mean([m["traj_pos_mse"] for m in traj_metrics]))
                avg_vel = float(np.mean([m["traj_vel_mse"] for m in traj_metrics]))

                eval_vel = avg_vel
                eval_pos = avg_pos

                print(f"  Held-out trajectory MSE avg (state): {avg_state:.6f}")
                print(f"  Held-out trajectory MSE avg (pos):   {avg_pos:.6f}")
                print(f"  Held-out trajectory MSE avg (vel):   {avg_vel:.6f}")

                avg_vel_mse_per_dim = np.mean(
                    [np.asarray(m["traj_vel_mse_per_dim"]) for m in traj_metrics],
                    axis=0,
                )
                print(
                    "  Held-out vel MSE per-dim (vx, vy, w): "
                    + " ".join(f"{v:.6f}" for v in avg_vel_mse_per_dim)
                )

                avg_pos_mse_per_dim = np.mean(
                    [np.asarray(m["traj_pos_mse_per_dim"]) for m in traj_metrics],
                    axis=0,
                )
                print(
                    "  Held-out pos MSE per-dim (x, y): "
                    + " ".join(f"{v:.6f}" for v in avg_pos_mse_per_dim)
                )

                worst = max(traj_metrics, key=lambda m: m["traj_state_mse"])
                worst_vel_mse_per_dim = np.asarray(worst["traj_vel_mse_per_dim"])
                worst_pos_mse_per_dim = np.asarray(worst["traj_pos_mse_per_dim"])
                print(
                    f"  Worst trajectory id {worst['id']} | total state MSE {worst['traj_state_mse']:.6f}"
                )
                print(
                    "    Worst traj vel MSE per-dim (vx, vy, w): "
                    + " ".join(f"{v:.6f}" for v in worst_vel_mse_per_dim)
                )
                print(
                    "    Worst traj pos MSE per-dim (x, y): "
                    + " ".join(f"{v:.6f}" for v in worst_pos_mse_per_dim)
                )

        epochs_list.append(epoch)
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        eval_vel_mses.append(eval_vel)
        eval_pos_mses.append(eval_pos)
        train_one_step_losses.append(train_metrics.get("one_step_mse", float("nan")))
        train_rollout_losses.append(train_metrics.get("rollout_mse", float("nan")))
        train_pose_losses.append(train_metrics.get("pose_mse", float("nan")))

    # Save detailed CSV
    csv_path = os.path.join(run_dir, "losses.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "train_one_step", "train_rollout",
             "train_pose", "eval_vel_mse", "eval_pos_mse"]
        )
        for i, e in enumerate(epochs_list):
            writer.writerow([
                e, train_losses[i], val_losses[i], train_one_step_losses[i],
                train_rollout_losses[i], train_pose_losses[i],
                eval_vel_mses[i], eval_pos_mses[i]
            ])

    _plot_loss_curve(
        epochs_list, train_losses, "Train loss", "MSE", "loss_train.png", run_dir
    )
    _plot_loss_curve(
        epochs_list, val_losses, "Val loss", "MSE", "loss_val.png", run_dir
    )
    _plot_loss_curve(
        epochs_list, eval_vel_mses, "Eval vel MSE", "MSE", "loss_eval_vel.png", run_dir
    )
    _plot_loss_curve(
        epochs_list, eval_pos_mses, "Eval pos MSE", "MSE", "loss_eval_pos.png", run_dir
    )

    return {"best_val": best_val, "ckpt_path": ckpt_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config JSON.",
    )
    args = parser.parse_args()
    train_pipeline(args.config)
