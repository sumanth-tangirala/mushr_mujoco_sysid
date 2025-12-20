import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

from .dataloader import load_traj_data, split_train_eval_ids
from .utils import Standardizer, split_indices


@dataclass
class Sample:
    xd0: np.ndarray
    ut: np.ndarray
    xd1: np.ndarray
    dt: float


class TimestepDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        input_std: Standardizer,
        target_std: Standardizer,
    ):
        inputs_xu = input_std.transform(inputs[:, :5])
        inputs_proc = np.concatenate([inputs_xu, inputs[:, 5:6]], axis=1)
        self.inputs = torch.tensor(inputs_proc, dtype=torch.float32)
        self.targets = torch.tensor(target_std.transform(targets), dtype=torch.float32)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


class SnippetDataset(Dataset):
    """
    Dataset that returns contiguous windows (snippets) from trajectories.

    Each snippet contains:
    - xd_seq: velocity states of shape (horizon+1, 3)
    - ut_seq: controls of shape (horizon, 2)
    - dt_seq: timesteps of shape (horizon,)
    - pose_seq: poses of shape (horizon+1, 3)

    Used for rollout loss computation.
    """

    def __init__(
        self,
        trajectories: List[Dict[str, Any]],
        horizon: int,
        input_std: Standardizer,
        target_std: Standardizer,
    ):
        """
        Args:
            trajectories: List of trajectory dicts with keys 'xd', 'pose', 'ut', 'dt'
            horizon: Number of steps in each snippet (snippet length = horizon + 1)
            input_std: Standardizer for inputs (xd, ut)
            target_std: Standardizer for targets (xd)
        """
        self.horizon = horizon
        self.input_std = input_std
        self.target_std = target_std
        self.snippets: List[Dict[str, np.ndarray]] = []

        for traj in trajectories:
            xd = traj["xd"]  # (T, 3)
            pose = traj["pose"]  # (T, 3)
            ut = traj["ut"]  # (T, 2)
            dt = traj["dt"]  # (T,)

            T = xd.shape[0]
            # We need at least horizon+1 timesteps for a valid snippet
            if T < horizon + 1:
                continue

            # Extract all valid snippets from this trajectory
            for start_idx in range(T - horizon):
                end_idx = start_idx + horizon + 1
                snippet = {
                    "xd": xd[start_idx:end_idx],  # (H+1, 3)
                    "pose": pose[start_idx:end_idx],  # (H+1, 3)
                    "ut": ut[start_idx : start_idx + horizon],  # (H, 2)
                    "dt": dt[start_idx : start_idx + horizon],  # (H,)
                }
                self.snippets.append(snippet)

    def __len__(self) -> int:
        return len(self.snippets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        snippet = self.snippets[idx]

        # Standardize xd using target standardizer (same as training targets)
        xd_norm = self.target_std.transform(snippet["xd"])

        # Standardize ut using input standardizer (positions 3:5)
        ut_mean = self.input_std.mean[3:5]
        ut_std = self.input_std.std[3:5]
        ut_norm = (snippet["ut"] - ut_mean) / ut_std

        return {
            "xd": torch.tensor(xd_norm, dtype=torch.float32),
            "ut": torch.tensor(ut_norm, dtype=torch.float32),
            "dt": torch.tensor(snippet["dt"], dtype=torch.float32),
            "pose": torch.tensor(snippet["pose"], dtype=torch.float32),
        }


def _collect_samples_for_id(id_: int, data_dir: str) -> Tuple[List[Sample], dict]:
    traj_vels, plan, _, dts = load_traj_data(id_, data_dir=data_dir)
    if (
        traj_vels is None
        or plan is None
        or dts is None
        or traj_vels.shape[0] < 2
        or dts.shape[0] < traj_vels.shape[0]
    ):
        return [], {}
    x = traj_vels[:, 0]
    y = traj_vels[:, 1]
    theta = traj_vels[:, 2]
    vx = traj_vels[:, 3]
    vy = traj_vels[:, 4]
    w = traj_vels[:, 5]
    ut = np.stack([np.full_like(vx, plan[1]), np.full_like(vx, plan[0])], axis=1)

    samples: List[Sample] = []
    for t in range(traj_vels.shape[0] - 1):
        xd0 = np.array([vx[t], vy[t], w[t]], dtype=float)
        xd1 = np.array([vx[t + 1], vy[t + 1], w[t + 1]], dtype=float)
        samples.append(Sample(xd0=xd0, ut=ut[t], xd1=xd1, dt=float(dts[t])))

    trajectory = {
        "xd": np.stack([vx, vy, w], axis=1),
        "pose": np.stack([x, y, theta], axis=1),
        "ut": ut,
        "dt": dts,
        "id": id_,
    }
    return samples, trajectory


def load_datasets(cfg: Dict):
    """
    Load and split datasets for training and evaluation.

    Supports two validation split modes (controlled by data.val_split_mode):
    - "timestep" (default): Shuffle all samples from training trajectories,
      then split into train/val at the sample level. This is the original behavior.
    - "trajectory": Reserve a fraction of training trajectory IDs for validation.
      All samples from validation trajectories go to val set (no leakage).

    Returns:
        train_dataset: TimestepDataset for training
        val_dataset: TimestepDataset for validation
        meta: Dict with standardizers, trajectories, and training trajectory info
    """
    data_cfg = cfg["data"]
    data_dir = data_cfg["data_dir"]
    num_eval = data_cfg["num_eval_trajectories"]
    val_ratio = data_cfg["val_ratio"]
    seed = cfg.get("seed", 42)
    val_split_mode = data_cfg.get("val_split_mode", "timestep")

    train_ids, eval_ids = split_train_eval_ids(num_eval)

    # Collect all data from train and eval trajectory IDs
    all_train_samples: List[Sample] = []
    all_train_trajs: List[Dict] = []
    heldout_trajs: List[Dict] = []

    # Track which samples belong to which trajectory (for trajectory split)
    traj_to_sample_indices: Dict[int, List[int]] = {}
    sample_idx = 0

    for tid in train_ids:
        samples, traj = _collect_samples_for_id(tid, data_dir)
        if samples:
            traj_to_sample_indices[tid] = list(
                range(sample_idx, sample_idx + len(samples))
            )
            sample_idx += len(samples)
            all_train_samples.extend(samples)
            all_train_trajs.append(traj)

    for eid in eval_ids:
        _, traj = _collect_samples_for_id(eid, data_dir)
        if traj:
            heldout_trajs.append(traj)

    if not all_train_samples:
        raise RuntimeError("No training samples could be loaded.")

    # Build full input/target arrays
    inputs = np.stack(
        [np.concatenate([s.xd0, s.ut, np.array([s.dt])]) for s in all_train_samples],
        axis=0,
    )
    targets = np.stack([s.xd1 for s in all_train_samples], axis=0)

    # Fit standardizers on ALL training samples (before split)
    input_std = Standardizer.fit(inputs[:, :5])
    target_std = Standardizer.fit(targets)

    if val_split_mode == "trajectory":
        # Split at the trajectory level
        valid_train_ids = [tid for tid in train_ids if tid in traj_to_sample_indices]
        rng = np.random.default_rng(seed)
        rng.shuffle(valid_train_ids)

        num_val_trajs = max(1, int(len(valid_train_ids) * val_ratio))
        val_traj_ids = set(valid_train_ids[:num_val_trajs])
        train_traj_ids = set(valid_train_ids[num_val_trajs:])

        train_sample_indices = []
        val_sample_indices = []
        train_trajs_final = []
        val_trajs_final = []

        for traj in all_train_trajs:
            tid = traj["id"]
            if tid in val_traj_ids:
                val_sample_indices.extend(traj_to_sample_indices[tid])
                val_trajs_final.append(traj)
            elif tid in train_traj_ids:
                train_sample_indices.extend(traj_to_sample_indices[tid])
                train_trajs_final.append(traj)

        train_idx = np.array(train_sample_indices)
        val_idx = np.array(val_sample_indices)
    else:
        # Default: timestep-level split (original behavior)
        train_idx, val_idx = split_indices(inputs.shape[0], val_ratio, seed)
        train_trajs_final = all_train_trajs
        val_trajs_final = []  # Not meaningful for timestep split

    train_dataset = TimestepDataset(
        inputs[train_idx], targets[train_idx], input_std, target_std
    )
    val_dataset = TimestepDataset(
        inputs[val_idx], targets[val_idx], input_std, target_std
    )

    meta = {
        "input_std": input_std,
        "target_std": target_std,
        "train_trajs": train_trajs_final,
        "val_trajs": val_trajs_final,  # Only populated for trajectory split
        "heldout_trajs": heldout_trajs,
        "val_split_mode": val_split_mode,
    }
    return train_dataset, val_dataset, meta


__all__ = [
    "Sample",
    "TimestepDataset",
    "SnippetDataset",
    "_collect_samples_for_id",
    "load_datasets",
]
