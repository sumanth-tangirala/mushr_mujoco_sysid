"""
Fast model variants for GPU-optimized inference.

These are opt-in alternatives to the reference implementations in models/system_models.py.
They make stronger assumptions (fixed shapes, batch=1, GPU) to minimize allocations
and branching overhead, while preserving mathematical equivalence within tolerance.

Use these via FastInferenceSession or manually if you know your constraints.
Keep the original models as the reference implementation for training/research.
"""

from typing import Union

import torch
import torch.nn as nn

from ..models.system_models import _maybe_append, _compute_friction_k
from ..models.mlp import MLP
from ..plant import MushrPlant


class ControlAdapterFast(nn.Module):
    """
    Fast variant of ControlAdapter with minimized allocations.
    
    Assumes: batch=1, float32, GPU tensors.
    """

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

        steer_in_dim = 5
        if include_speed_mag_steer:
            steer_in_dim += 1
        if include_speed_sign_steer:
            steer_in_dim += 1

        acc_in_dim = 3
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

        # Construct ut_eff directly instead of cloning then overwriting
        ut_eff = torch.stack([acc_eff, delta_eff], dim=-1)
        return ut_eff


class StructuredDynamicsModelFast(nn.Module):
    """
    Fast variant of StructuredDynamicsModel.
    
    Assumptions:
    - xd0 shape (1, 3), float32, GPU
    - ut_raw shape (1, 2), float32, GPU
    - dt is a scalar tensor (no complex broadcasting logic needed)
    - return_aux=False (value-only inference)
    
    Optimizations:
    - Minimized intermediate allocations
    - Removed dt broadcasting branches (assumes canonical shape)
    - Reuses reference implementations for control adapter / friction / residual nets
    """

    def __init__(
        self,
        plant: MushrPlant,
        control_adapter: Union[ControlAdapterFast, None] = None,
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
                input_dim=5,
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
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fast forward pass (value-only).
        
        Args:
            xd0: shape (1, 3)
            ut_raw: shape (1, 2)
            dt: scalar tensor (no broadcasting)
        
        Returns:
            xd1: shape (1, 3)
        """
        ut_eff = ut_raw
        if self.control_adapter is not None:
            ut_eff = self.control_adapter(xd0, ut_raw)

        delta_eff = ut_eff[..., 1]
        vx = xd0[..., 0]
        vy = xd0[..., 1]
        beta_prev = torch.atan2(vy, vx)
        omega_prev = 2.0 * torch.sin(beta_prev) / self.plant.L

        # Assume dt is already a scalar tensor or (1,) - no complex branching
        if dt.dim() == 0:
            dt_expanded = dt.expand_as(omega_prev)
        else:
            dt_expanded = dt

        friction_val = None
        if self.friction_net is not None:
            friction_feats = [omega_prev]
            _maybe_append(friction_feats, delta_eff, self.friction_use_delta)
            _maybe_append(friction_feats, vy, self.friction_use_vy)
            if self.friction_use_dt:
                friction_feats.append(dt_expanded)
            friction_in = torch.stack(friction_feats, dim=-1)
            friction_raw = self.friction_net(friction_in).squeeze(-1)
            friction_val = _compute_friction_k(
                friction_raw,
                mode=self.friction_param_mode,
                k_min=self.friction_k_min,
                k_max=self.friction_k_max,
            )

        residual = None
        if self.residual_net is not None:
            residual_in = torch.cat([xd0, ut_eff], dim=-1)
            residual = self.residual_net(residual_in)

        xd1 = self.plant.xdot(
            xd0,
            ut_eff,
            dt_expanded,
            friction=friction_val,
            residual=residual,
            delta_override=delta_eff,
            acc_override=ut_eff[..., 0],
        )

        return xd1


class DirectDynamicsModelFast(nn.Module):
    """
    Fast variant of DirectDynamicsModel.
    
    Assumptions:
    - xd0 shape (1, 3), float32, GPU
    - ut_raw shape (1, 2), float32, GPU
    - dt is a (1, 1) tensor (no complex broadcasting)
    - return_aux=False (value-only inference)
    
    Optimizations:
    - Minimized intermediate allocations
    - Removed dt broadcasting branches
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims=(128, 128),
        control_adapter: Union[ControlAdapterFast, None] = None,
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
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fast forward pass (value-only).
        
        Args:
            xd0: shape (1, 3)
            ut_raw: shape (1, 2)
            dt: shape (1, 1) or scalar
        
        Returns:
            xd1: shape (1, 3)
        """
        ut_eff = ut_raw
        if self.control_adapter is not None:
            ut_eff = self.control_adapter(xd0, ut_raw)

        # Assume dt is (1, 1) or scalar - minimal shaping
        if dt.dim() == 0:
            dt_shaped = dt.unsqueeze(0).unsqueeze(0)
        elif dt.dim() == 1:
            dt_shaped = dt.unsqueeze(-1)
        else:
            dt_shaped = dt

        input_vec = torch.cat([xd0, ut_eff, dt_shaped], dim=-1)
        xd1 = self.net(input_vec)

        return xd1
