from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .mlp import MLP
from ..plant import MushrPlant


def _maybe_append(features, value, enabled):
    if enabled:
        features.append(value)


def _compute_friction_k(
    raw_output: torch.Tensor,
    mode: str = "softplus_offset_1",
    k_min: float = 0.2,
    k_max: float = 2.0,
) -> torch.Tensor:
    """
    Compute friction coefficient k from raw network output.

    Args:
        raw_output: Raw output from friction network (unbounded)
        mode: Parameterization mode:
            - "softplus_offset_1": k = 1 + softplus(h) (backward compatible, k >= 1)
            - "exp": k = exp(h), clamped to [k_min, k_max]
            - "sigmoid_range": k = k_min + (k_max - k_min) * sigmoid(h)
        k_min: Minimum friction coefficient (used by exp and sigmoid_range)
        k_max: Maximum friction coefficient (used by exp and sigmoid_range)

    Returns:
        Friction coefficient k
    """
    if mode == "softplus_offset_1":
        # Original behavior: k = 1 + softplus(h), always >= 1
        return 1.0 + torch.nn.functional.softplus(raw_output)
    elif mode == "exp":
        # Exponential with clamping: allows k < 1
        k = torch.exp(raw_output)
        return torch.clamp(k, min=k_min, max=k_max)
    elif mode == "sigmoid_range":
        # Sigmoid mapping to [k_min, k_max]: smooth bounded output
        return k_min + (k_max - k_min) * torch.sigmoid(raw_output)
    else:
        raise ValueError(f"Unknown friction parameterization mode: {mode}")


class ControlAdapter(nn.Module):
    def __init__(
        self,
        steer_hidden=(64, 64),
        acc_hidden=(64, 64),
        include_speed_mag_steer=False,
        include_speed_sign_steer=False,
        include_speed_mag_acc=False,
        include_speed_sign_acc=False,
        include_vy_acc=False,
        include_w_acc=False,
        steer_output_scale=None,
        acc_output_scale=None,
    ):
        super().__init__()
        self.include_speed_mag_steer = include_speed_mag_steer
        self.include_speed_sign_steer = include_speed_sign_steer
        self.include_speed_mag_acc = include_speed_mag_acc
        self.include_speed_sign_acc = include_speed_sign_acc
        self.include_vy_acc = include_vy_acc
        self.include_w_acc = include_w_acc
        self.steer_output_scale = steer_output_scale
        self.acc_output_scale = acc_output_scale

        steer_in_dim = 5  # delta_raw, velocity_raw, vx, vy, w
        if include_speed_mag_steer:
            steer_in_dim += 1
        if include_speed_sign_steer:
            steer_in_dim += 1

        acc_in_dim = 3  # velocity_raw, delta_raw, vx
        if include_speed_mag_acc:
            acc_in_dim += 1
        if include_speed_sign_acc:
            acc_in_dim += 1
        if include_vy_acc:
            acc_in_dim += 1
        if include_w_acc:
            acc_in_dim += 1

        self.steer_net = MLP(
            input_dim=steer_in_dim,
            output_dim=1,
            hidden_dims=list(steer_hidden),
            activation="tanh",
        )
        self.acc_net = MLP(
            input_dim=acc_in_dim,
            output_dim=1,
            hidden_dims=list(acc_hidden),
            activation="tanh",
        )

    def forward(self, xd0, ut_raw):
        vx = xd0[..., 0]
        vy = xd0[..., 1]
        w = xd0[..., 2]
        delta_raw = ut_raw[..., 1]
        vel_raw = ut_raw[..., 0]
        speed_mag = torch.sqrt(vx * vx + vy * vy)
        speed_sign = torch.sign(vx)

        steer_features = [
            delta_raw,
            vel_raw,
            vx,
            vy,
            w,
        ]
        _maybe_append(steer_features, speed_mag, self.include_speed_mag_steer)
        _maybe_append(steer_features, speed_sign, self.include_speed_sign_steer)
        steer_in = torch.stack(steer_features, dim=-1)
        delta_eff = self.steer_net(steer_in).squeeze(-1)
        if self.steer_output_scale is not None:
            delta_eff = torch.tanh(delta_eff) * self.steer_output_scale

        acc_features = [vel_raw, delta_raw, vx]
        _maybe_append(acc_features, speed_mag, self.include_speed_mag_acc)
        _maybe_append(acc_features, speed_sign, self.include_speed_sign_acc)
        _maybe_append(acc_features, vy, self.include_vy_acc)
        _maybe_append(acc_features, w, self.include_w_acc)
        acc_in = torch.stack(acc_features, dim=-1)
        acc_eff = self.acc_net(acc_in).squeeze(-1)
        if self.acc_output_scale is not None:
            acc_eff = torch.tanh(acc_eff) * self.acc_output_scale

        ut_eff = ut_raw.clone()
        ut_eff[..., 1] = delta_eff
        ut_eff[..., 0] = acc_eff
        return ut_eff


class StructuredDynamicsModel(nn.Module):
    def __init__(
        self,
        plant: MushrPlant,
        control_adapter: ControlAdapter | None = None,
        learn_friction: bool = False,
        learn_residual: bool = False,
        friction_hidden=(32, 32),
        residual_hidden=(32, 32),
        friction_use_dt: bool = True,
        friction_use_delta: bool = True,
        friction_use_vy: bool = True,
        friction_param_mode: str = "softplus_offset_1",
        friction_k_min: float = 0.2,
        friction_k_max: float = 2.0,
    ):
        super().__init__()
        self.plant = plant
        self.control_adapter = control_adapter
        self.learn_friction = learn_friction
        self.learn_residual = learn_residual
        self.friction_use_dt = friction_use_dt
        self.friction_use_delta = friction_use_delta
        self.friction_use_vy = friction_use_vy
        self.friction_param_mode = friction_param_mode
        self.friction_k_min = friction_k_min
        self.friction_k_max = friction_k_max

        if learn_friction:
            # Inputs: omega_prev, optional delta, optional vy, optional dt
            friction_in = 1
            if friction_use_delta:
                friction_in += 1
            if friction_use_vy:
                friction_in += 1
            if friction_use_dt:
                friction_in += 1
            self.friction_net = MLP(
                input_dim=friction_in,
                output_dim=1,
                hidden_dims=list(friction_hidden),
                activation="tanh",
            )
        else:
            self.friction_net = None

        if learn_residual:
            self.residual_net = MLP(
                input_dim=5,  # xd0 (3) + ut_eff (2)
                output_dim=3,
                hidden_dims=list(residual_hidden),
                activation="tanh",
            )
        else:
            self.residual_net = None

    def forward(
        self,
        xd0: torch.Tensor,
        ut_raw: torch.Tensor,
        dt: Union[float, torch.Tensor],
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass for structured dynamics model.

        Args:
            xd0: Initial velocity state [vx, vy, w], shape (batch, 3)
            ut_raw: Raw control input [vel_cmd, steer_cmd], shape (batch, 2)
            dt: Timestep duration (scalar or tensor)
            return_aux: If True, return auxiliary outputs for regularization

        Returns:
            If return_aux=False: xd1_pred tensor of shape (batch, 3)
            If return_aux=True: (xd1_pred, aux_dict) where aux_dict contains:
                - "ut_eff": Effective controls if control adapter enabled
                - "residual": Residual correction if learn_residual enabled
                - "friction_k": Friction coefficient if learn_friction enabled
        """
        aux = {}

        ut_eff = ut_raw
        if self.control_adapter is not None:
            ut_eff = self.control_adapter(xd0, ut_raw)
            if return_aux:
                aux["ut_eff"] = ut_eff

        delta_eff = ut_eff[..., 1]
        vx = xd0[..., 0]
        vy = xd0[..., 1]
        beta_prev = torch.atan2(vy, vx)
        omega_prev = 2.0 * torch.sin(beta_prev) / self.plant.L

        dt_tensor = torch.as_tensor(dt, device=xd0.device, dtype=xd0.dtype)
        if dt_tensor.dim() == 0:
            dt_tensor = dt_tensor.expand_as(omega_prev)
        else:
            dt_tensor = dt_tensor.squeeze()
            if dt_tensor.dim() == 0:
                dt_tensor = dt_tensor.expand_as(omega_prev)
            elif dt_tensor.shape != omega_prev.shape:
                dt_tensor = dt_tensor.expand_as(omega_prev)

        friction_val = None
        if self.friction_net is not None:
            friction_feats = [omega_prev]
            _maybe_append(friction_feats, delta_eff, self.friction_use_delta)
            _maybe_append(friction_feats, vy, self.friction_use_vy)
            if self.friction_use_dt:
                friction_feats.append(dt_tensor)
            friction_in = torch.stack(friction_feats, dim=-1)
            friction_raw = self.friction_net(friction_in).squeeze(-1)
            friction_val = _compute_friction_k(
                friction_raw,
                mode=self.friction_param_mode,
                k_min=self.friction_k_min,
                k_max=self.friction_k_max,
            )
            if return_aux:
                aux["friction_k"] = friction_val

        residual = None
        if self.residual_net is not None:
            residual_in = torch.cat([xd0, ut_eff], dim=-1)
            residual = self.residual_net(residual_in)
            if return_aux:
                aux["residual"] = residual

        xd1 = self.plant.xdot(
            xd0,
            ut_eff,
            dt_tensor,
            friction=friction_val,
            residual=residual,
            delta_override=delta_eff,
            acc_override=ut_eff[..., 0],
        )

        if return_aux:
            return xd1, aux
        return xd1


class DirectDynamicsModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,  # xd0 (3) + ut_raw (2) + dt (1)
        hidden_dims=(128, 128),
        control_adapter: ControlAdapter | None = None,
    ):
        super().__init__()
        self.control_adapter = control_adapter
        self.net = MLP(
            input_dim=input_dim,
            output_dim=3,
            hidden_dims=list(hidden_dims),
            activation="tanh",
        )

    def forward(
        self,
        xd0: torch.Tensor,
        ut_raw: torch.Tensor,
        dt: Union[float, torch.Tensor],
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass for direct dynamics model.

        Args:
            xd0: Initial velocity state [vx, vy, w], shape (batch, 3)
            ut_raw: Raw control input [vel_cmd, steer_cmd], shape (batch, 2)
            dt: Timestep duration (scalar or tensor)
            return_aux: If True, return auxiliary outputs for regularization

        Returns:
            If return_aux=False: xd1_pred tensor of shape (batch, 3)
            If return_aux=True: (xd1_pred, aux_dict) where aux_dict contains:
                - "ut_eff": Effective controls if control adapter enabled
        """
        aux = {}

        ut_eff = ut_raw
        if self.control_adapter is not None:
            ut_eff = self.control_adapter(xd0, ut_raw)
            if return_aux:
                aux["ut_eff"] = ut_eff

        dt_tensor = torch.as_tensor(dt, device=xd0.device, dtype=xd0.dtype)
        if dt_tensor.dim() == 0:
            dt_tensor = dt_tensor.expand_as(ut_eff[..., :1])
        elif dt_tensor.dim() == 1:
            dt_tensor = dt_tensor.unsqueeze(-1)
        if dt_tensor.shape != ut_eff[..., :1].shape:
            dt_tensor = dt_tensor.expand_as(ut_eff[..., :1])
        input_vec = torch.cat([xd0, ut_eff, dt_tensor], dim=-1)
        xd1 = self.net(input_vec)

        if return_aux:
            return xd1, aux
        return xd1
