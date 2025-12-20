"""Configuration utilities with default values for backward compatibility."""

from typing import Any, Dict


def get_default_loss_config() -> Dict[str, Any]:
    """Return default loss configuration (backward compatible with one-step MSE only)."""
    return {
        "one_step_mse": {"enabled": True, "weight": 1.0},
        "rollout_mse": {
            "enabled": False,
            "horizon": 10,
            "weight": 1.0,
            "teacher_forcing_prob": 0.0,
            "detach_between_steps": False,
        },
        "pose_mse": {
            "enabled": False,
            "weight": 0.1,
            "components": ["x", "y"],
        },
        "weights": {"vx": 1.0, "vy": 1.0, "w": 1.0},
    }


def get_default_regularization_config() -> Dict[str, float]:
    """Return default regularization configuration (all disabled)."""
    return {
        "adapter_identity_weight": 0.0,
        "residual_l2_weight": 0.0,
        "friction_prior_weight": 0.0,
    }


def get_default_optim_config() -> Dict[str, float]:
    """Return default optimization configuration."""
    return {
        "grad_clip_norm": 0.0,
    }


def get_default_friction_param_config() -> Dict[str, Any]:
    """Return default friction parameterization configuration."""
    return {
        "mode": "softplus_offset_1",
        "k_min": 0.2,
        "k_max": 2.0,
    }


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def populate_config_defaults(cfg: Dict) -> Dict:
    """
    Populate configuration with default values for all new fields.

    This ensures backward compatibility - missing fields get defaults that
    reproduce the original behavior.
    """
    cfg = cfg.copy()

    # Ensure training block exists
    if "training" not in cfg:
        cfg["training"] = {}

    # Ensure data block exists
    if "data" not in cfg:
        cfg["data"] = {}

    # Ensure model block exists
    if "model" not in cfg:
        cfg["model"] = {}

    # Populate training.loss with defaults
    default_loss = get_default_loss_config()
    cfg["training"]["loss"] = deep_merge(
        default_loss, cfg["training"].get("loss", {})
    )

    # Populate training.regularization with defaults
    default_reg = get_default_regularization_config()
    cfg["training"]["regularization"] = deep_merge(
        default_reg, cfg["training"].get("regularization", {})
    )

    # Populate training.optim with defaults
    default_optim = get_default_optim_config()
    cfg["training"]["optim"] = deep_merge(
        default_optim, cfg["training"].get("optim", {})
    )

    # Populate data.val_split_mode with default
    if "val_split_mode" not in cfg["data"]:
        cfg["data"]["val_split_mode"] = "timestep"

    # Populate model.friction_param with defaults (structured model only)
    if cfg["model"].get("type", "structured").lower() == "structured":
        default_friction = get_default_friction_param_config()
        cfg["model"]["friction_param"] = deep_merge(
            default_friction, cfg["model"].get("friction_param", {})
        )

    return cfg


__all__ = [
    "get_default_loss_config",
    "get_default_regularization_config",
    "get_default_optim_config",
    "get_default_friction_param_config",
    "deep_merge",
    "populate_config_defaults",
]
