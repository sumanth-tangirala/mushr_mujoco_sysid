from typing import Dict

import torch

from .plant import MushrPlant
from .models.system_models import (
    ControlAdapter,
    DirectDynamicsModel,
    StructuredDynamicsModel,
)


def build_model(cfg: Dict, device: torch.device | str):
    model_cfg = cfg["model"]
    device = torch.device(device)

    plant = MushrPlant()

    control_adapter = None
    ctrl_cfg = model_cfg.get("control_adapter", {})
    if ctrl_cfg.get("enabled", False):
        control_adapter = ControlAdapter(
            include_speed_mag_steer=ctrl_cfg.get("include_speed_mag_steer", False),
            include_speed_sign_steer=ctrl_cfg.get("include_speed_sign_steer", False),
            include_speed_mag_acc=ctrl_cfg.get("include_speed_mag_acc", False),
            include_speed_sign_acc=ctrl_cfg.get("include_speed_sign_acc", False),
            include_vy_acc=ctrl_cfg.get("include_vy_acc", False),
            include_w_acc=ctrl_cfg.get("include_w_acc", False),
            steer_output_scale=ctrl_cfg.get("steer_output_scale"),
            acc_output_scale=ctrl_cfg.get("acc_output_scale"),
        )

    model_type = model_cfg.get("type", "structured").lower()
    if model_type == "structured":
        # Get friction parameterization config with defaults
        friction_param_cfg = model_cfg.get("friction_param", {})
        friction_param_mode = friction_param_cfg.get("mode", "softplus_offset_1")
        friction_k_min = friction_param_cfg.get("k_min", 0.2)
        friction_k_max = friction_param_cfg.get("k_max", 2.0)

        model = StructuredDynamicsModel(
            plant=plant,
            control_adapter=control_adapter,
            learn_friction=model_cfg.get("learn_friction", False),
            learn_residual=model_cfg.get("learn_residual", False),
            friction_use_dt=model_cfg.get("friction_use_dt", True),
            friction_use_delta=model_cfg.get("friction_use_delta", True),
            friction_use_vy=model_cfg.get("friction_use_vy", True),
            friction_param_mode=friction_param_mode,
            friction_k_min=friction_k_min,
            friction_k_max=friction_k_max,
        )
    elif model_type == "direct":
        model = DirectDynamicsModel(
            control_adapter=control_adapter,
            hidden_dims=tuple(model_cfg.get("hidden_dims", [128, 128])),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


__all__ = ["build_model"]
