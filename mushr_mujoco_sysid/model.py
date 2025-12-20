from __future__ import annotations

import numpy as np
import torch

from .utils import Standardizer
from .models.system_models import DirectDynamicsModel, StructuredDynamicsModel


class LearnedDynamicsModel:
    def __init__(
        self,
        model: torch.nn.Module,
        input_std: Standardizer,
        target_std: Standardizer,
        device: torch.device | str = "cpu",
    ):
        self.model = model
        self.input_std = input_std
        self.target_std = target_std
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict_xd_next(self, xd: np.ndarray, ut: np.ndarray, dt: float) -> np.ndarray:
        xd_np = np.asarray(xd, dtype=np.float32).reshape(-1)
        ut_np = np.asarray(ut, dtype=np.float32).reshape(-1)
        if xd_np.shape[0] != 3:
            raise ValueError("xd must have shape (3,).")
        if ut_np.shape[0] != 2:
            raise ValueError("ut must have shape (2,) [velocity_cmd, steering_cmd].")

        feats_first5 = np.concatenate([xd_np, ut_np], axis=0)[None, :]
        xb = torch.from_numpy(feats_first5).to(self.device, dtype=torch.float32)
        xb_norm = self.input_std.transform_torch(xb)

        dt_tensor = torch.tensor([[float(dt)]], device=self.device, dtype=torch.float32)
        xb_full = torch.cat([xb_norm, dt_tensor], dim=1)

        xd0_n = xb_full[:, :3]
        ut_n = xb_full[:, 3:5]
        dt_col = xb_full[:, 5]

        if isinstance(self.model, StructuredDynamicsModel):
            dt_arg = dt_col
        elif isinstance(self.model, DirectDynamicsModel):
            dt_arg = xb_full[:, 5:]
        else:
            dt_arg = xb_full[:, 5:]

        with torch.no_grad():
            out = self.model(xd0_n, ut_n, dt_arg)
            twist = self.target_std.inverse_torch(out).squeeze(0)

        return twist.cpu().numpy()


__all__ = ["LearnedDynamicsModel"]
