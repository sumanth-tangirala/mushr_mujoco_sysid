from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np
import torch

from .plant import MushrPlant
from .model import LearnedDynamicsModel


class TrajectoryEvaluator:
    def __init__(
        self,
        dynamics_model: LearnedDynamicsModel,
        plant: MushrPlant | None = None,
        device: torch.device | str = "cpu",
    ):
        self.dynamics_model = dynamics_model
        self.device = torch.device(device)
        self.plant = plant if plant is not None else MushrPlant()

    def rollout(
        self,
        x0: np.ndarray,
        xd0: np.ndarray,
        ut: np.ndarray | Iterable[np.ndarray],
        dt: np.ndarray | Iterable[float],
    ) -> Dict[str, np.ndarray]:
        x0_np = np.asarray(x0, dtype=np.float32).reshape(-1)
        xd0_np = np.asarray(xd0, dtype=np.float32).reshape(-1)
        if x0_np.shape[0] != 3:
            raise ValueError("x0 must have shape (3,).")
        if xd0_np.shape[0] != 3:
            raise ValueError("xd0 must have shape (3,).")

        ut_arr = np.asarray(ut, dtype=np.float32)
        dt_arr = np.asarray(dt, dtype=np.float32)

        if dt_arr.ndim != 1:
            dt_arr = dt_arr.reshape(-1)

        if ut_arr.ndim == 1:
            ut_arr = np.repeat(ut_arr.reshape(1, -1), dt_arr.shape[0], axis=0)

        if ut_arr.shape[0] != dt_arr.shape[0]:
            raise ValueError("ut and dt must have the same length along time.")

        pose = self.plant.SE2(
            torch.tensor(x0_np[0], device=self.device, dtype=torch.float32),
            torch.tensor(x0_np[1], device=self.device, dtype=torch.float32),
            torch.tensor(x0_np[2], device=self.device, dtype=torch.float32),
        )

        pose_list = [x0_np]
        xd_list = [xd0_np]
        xd_curr = xd0_np

        for u_t, dt_t in zip(ut_arr, dt_arr):
            xd_next = self.dynamics_model.predict_xd_next(xd_curr, u_t, float(dt_t))
            pose = self.plant.integrate_SE2(
                pose,
                torch.tensor(xd_next, device=self.device, dtype=torch.float32),
                torch.tensor(float(dt_t), device=self.device, dtype=torch.float32),
            )
            pose_np = _pose_from_se2(pose)
            pose_list.append(pose_np)
            xd_list.append(xd_next)
            xd_curr = xd_next

        pose_arr = np.stack(pose_list, axis=0)
        xd_arr = np.stack(xd_list, axis=0)
        return {"pose": pose_arr, "xd": xd_arr}


def _pose_from_se2(mat: torch.Tensor) -> np.ndarray:
    if mat.dim() != 2 or mat.shape != (3, 3):
        raise ValueError("SE2 matrix must have shape (3, 3).")
    x = mat[0, 2].item()
    y = mat[1, 2].item()
    th = math.atan2(mat[1, 0].item(), mat[0, 0].item())
    return np.array([x, y, th], dtype=np.float32)


def compute_traj_errors(
    pose_gt: np.ndarray,
    xd_gt: np.ndarray,
    pose_pred: np.ndarray,
    xd_pred: np.ndarray,
) -> Dict[str, float | List[float]]:
    """
    Compute position/velocity MSEs for a full rollout.
    """
    T = min(pose_gt.shape[0], pose_pred.shape[0], xd_gt.shape[0], xd_pred.shape[0])
    if T <= 1:
        return {}

    gt_pos = pose_gt[1:T]
    pred_pos = pose_pred[1:T]
    gt_vel = xd_gt[1:T]
    pred_vel = xd_pred[1:T]

    pos_err_xy = pred_pos[:, :2] - gt_pos[:, :2]
    theta_err = _angle_diff(pred_pos[:, 2], gt_pos[:, 2])
    vel_err = pred_vel - gt_vel

    pos_err_sq = np.concatenate(
        [pos_err_xy**2, theta_err[:, None] ** 2], axis=1
    )  # x, y, theta
    vel_err_sq = vel_err**2
    state_err_sq = np.concatenate([pos_err_sq, vel_err_sq], axis=1)

    traj_state_mse = float(state_err_sq.mean())
    traj_pos_mse = float(pos_err_sq.mean())
    traj_vel_mse = float(vel_err_sq.mean())

    pos_mse_per_dim = pos_err_sq.mean(axis=0)
    vel_mse_per_dim = vel_err_sq.mean(axis=0)

    pos_err_xy_final = pos_err_xy[-1]
    theta_err_final = theta_err[-1]
    vel_err_final = vel_err[-1]

    pos_err_sq_final = np.array(
        [pos_err_xy_final[0] ** 2, pos_err_xy_final[1] ** 2, theta_err_final**2]
    )
    vel_err_sq_final = vel_err_final**2
    state_err_sq_final = np.concatenate([pos_err_sq_final, vel_err_sq_final])

    state_mse_final = float(state_err_sq_final.mean())
    pos_mse_final = float(pos_err_sq_final.mean())
    vel_mse_final = float(vel_err_sq_final.mean())

    return {
        "traj_state_mse": traj_state_mse,
        "traj_pos_mse": traj_pos_mse,
        "traj_vel_mse": traj_vel_mse,
        "traj_pos_mse_per_dim": pos_mse_per_dim.tolist(),
        "traj_vel_mse_per_dim": vel_mse_per_dim.tolist(),
        "final_state_mse": state_mse_final,
        "final_pos_mse": pos_mse_final,
        "final_vel_mse": vel_mse_final,
        "final_pos_mse_per_dim": pos_err_sq_final.tolist(),
        "final_vel_mse_per_dim": vel_err_sq_final.tolist(),
    }


def _angle_diff(theta_pred: np.ndarray, theta_gt: np.ndarray) -> np.ndarray:
    diff = theta_pred - theta_gt
    return (diff + np.pi) % (2.0 * np.pi) - np.pi


__all__ = ["TrajectoryEvaluator", "compute_traj_errors"]
