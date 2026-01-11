import json
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    _torch_cache: Dict[Tuple[str, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = (
        field(default_factory=dict, init=False, repr=False)
    )

    def _get_torch_mean_std(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), dtype)
        cached = self._torch_cache.get(key)
        if cached is not None:
            return cached

        mean = torch.as_tensor(self.mean, device=device, dtype=dtype)
        std = torch.as_tensor(self.std, device=device, dtype=dtype)
        self._torch_cache[key] = (mean, std)
        return mean, std

    @classmethod
    def fit(cls, data: np.ndarray, eps: float = 1e-8) -> "Standardizer":
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std < eps, eps, std)
        return cls(mean=mean, std=std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def transform_torch(self, data: torch.Tensor) -> torch.Tensor:
        mean, std = self._get_torch_mean_std(data.device, data.dtype)
        return (data - mean) / std

    def inverse_torch(self, data: torch.Tensor) -> torch.Tensor:
        mean, std = self._get_torch_mean_std(data.device, data.dtype)
        return data * std + mean


def split_indices(
    num_items: int, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_items)
    rng.shuffle(indices)
    split = int(num_items * (1.0 - val_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]
    return train_idx, val_idx


def save_standardizers_json(
    path: str, input_std: Standardizer, target_std: Standardizer
) -> None:
    data = {
        "input": {
            "mean": input_std.mean.tolist(),
            "std": input_std.std.tolist(),
        },
        "target": {
            "mean": target_std.mean.tolist(),
            "std": target_std.std.tolist(),
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_standardizers_json(path: str) -> Tuple[Standardizer, Standardizer]:
    with open(path, "r") as f:
        data = json.load(f)
    input_data = data["input"]
    target_data = data["target"]
    input_std = Standardizer(
        mean=np.asarray(input_data["mean"], dtype=float),
        std=np.asarray(input_data["std"], dtype=float),
    )
    target_std = Standardizer(
        mean=np.asarray(target_data["mean"], dtype=float),
        std=np.asarray(target_data["std"], dtype=float),
    )
    return input_std, target_std
