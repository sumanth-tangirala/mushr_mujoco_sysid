"""
TorchScript deploy modules for C++ inference.

These modules embed standardizers as buffers and provide raw-in/raw-out interface
with manual Jacobian computation (no runtime autograd).
"""

from typing import Tuple

import torch
import torch.nn as nn

from .mlp_manual_jac import MLPManualJac


class DirectDeployModule(nn.Module):
    """
    TorchScript-exportable direct dynamics model.

    Takes raw (unstandardized) inputs and returns raw outputs.
    Embeds standardizers as buffers for self-contained deployment.
    """

    def __init__(
        self,
        net: MLPManualJac,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        dt: float,
        has_adapter: bool = False,
        adapter_steer_net: MLPManualJac = None,
        adapter_acc_net: MLPManualJac = None,
        adapter_config: dict = None,
    ):
        super().__init__()
        self.net = net
        self.has_adapter = has_adapter
        self.adapter_steer_net = adapter_steer_net
        self.adapter_acc_net = adapter_acc_net

        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.register_buffer("dt_buf", torch.tensor([dt], dtype=input_mean.dtype))

        # Precompute per-dimension scalings used in Jacobian unstandardization.
        self.register_buffer("_inv_input_std_x", 1.0 / input_std[:3])
        self.register_buffer("_inv_input_std_u", 1.0 / input_std[3:5])
        self.register_buffer("_zeros_2x3", torch.zeros(2, 3, dtype=input_mean.dtype))
        self.register_buffer("_eye_2", torch.eye(2, dtype=input_mean.dtype))

        if adapter_config is not None:
            self.include_speed_mag_steer = adapter_config.get(
                "include_speed_mag_steer", False
            )
            self.include_speed_sign_steer = adapter_config.get(
                "include_speed_sign_steer", False
            )
            self.include_speed_mag_acc = adapter_config.get(
                "include_speed_mag_acc", False
            )
            self.include_speed_sign_acc = adapter_config.get(
                "include_speed_sign_acc", False
            )
            self.include_vy_acc = adapter_config.get("include_vy_acc", False)
            self.include_w_acc = adapter_config.get("include_w_acc", False)
            steer_scale = adapter_config.get("steer_output_scale", None)
            acc_scale = adapter_config.get("acc_output_scale", None)
            self.steer_output_scale = steer_scale if steer_scale is not None else -1.0
            self.acc_output_scale = acc_scale if acc_scale is not None else -1.0
        else:
            self.include_speed_mag_steer = False
            self.include_speed_sign_steer = False
            self.include_speed_mag_acc = False
            self.include_speed_sign_acc = False
            self.include_vy_acc = False
            self.include_w_acc = False
            self.steer_output_scale = -1.0
            self.acc_output_scale = -1.0

    def _normalize_input(
        self, xd0_raw: torch.Tensor, ut_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_raw = torch.cat([xd0_raw, ut_raw], dim=-1)
        x_norm = (x_raw - self.input_mean) / self.input_std
        return x_norm[:3], x_norm[3:5]

    def _denormalize_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.target_std + self.target_mean

    def _adapter_forward(
        self, xd0_norm: torch.Tensor, ut_norm: torch.Tensor
    ) -> torch.Tensor:
        if not self.has_adapter:
            return ut_norm

        assert self.adapter_steer_net is not None, "adapter_steer_net must not be None"
        assert self.adapter_acc_net is not None, "adapter_acc_net must not be None"

        vx = xd0_norm[0]
        vy = xd0_norm[1]
        w = xd0_norm[2]
        vel_raw = ut_norm[0]
        delta_raw = ut_norm[1]

        eps = 1e-8
        speed_mag = torch.sqrt(vx * vx + vy * vy + eps)
        speed_sign = torch.sign(vx)

        steer_features = [delta_raw, vel_raw, vx, vy, w]
        if self.include_speed_mag_steer:
            steer_features.append(speed_mag)
        if self.include_speed_sign_steer:
            steer_features.append(speed_sign)
        steer_in = torch.stack(steer_features)
        delta_eff = self.adapter_steer_net(steer_in)[0]
        if self.steer_output_scale > 0:
            delta_eff = torch.tanh(delta_eff) * self.steer_output_scale

        acc_features = [vel_raw, delta_raw, vx]
        if self.include_speed_mag_acc:
            acc_features.append(speed_mag)
        if self.include_speed_sign_acc:
            acc_features.append(speed_sign)
        if self.include_vy_acc:
            acc_features.append(vy)
        if self.include_w_acc:
            acc_features.append(w)
        acc_in = torch.stack(acc_features)
        acc_eff = self.adapter_acc_net(acc_in)[0]
        if self.acc_output_scale > 0:
            acc_eff = torch.tanh(acc_eff) * self.acc_output_scale

        return torch.stack([acc_eff, delta_eff])

    def forward(self, xd0_raw: torch.Tensor, ut_raw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: raw inputs -> raw outputs.

        Args:
            xd0_raw: Raw velocity state [vx, vy, w], shape (3,)
            ut_raw: Raw control [vel_cmd, steer_cmd], shape (2,)

        Returns:
            xd_next_raw: Raw next velocity state, shape (3,)
        """
        xd0_norm, ut_norm = self._normalize_input(xd0_raw, ut_raw)
        ut_eff = self._adapter_forward(xd0_norm, ut_norm)

        dt_val = self.dt_buf[0]
        net_in = torch.cat([xd0_norm, ut_eff, dt_val.unsqueeze(0)])
        y_norm = self.net(net_in)

        return self._denormalize_output(y_norm)

    @torch.jit.export
    def forward_with_jacobian(
        self, xd0_raw: torch.Tensor, ut_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with Jacobians w.r.t. raw inputs.

        Args:
            xd0_raw: Raw velocity state [vx, vy, w], shape (3,)
            ut_raw: Raw control [vel_cmd, steer_cmd], shape (2,)

        Returns:
            xd_next_raw: Raw next velocity state, shape (3,)
            Jx: d(xd_next_raw)/d(xd0_raw), shape (3, 3)
            Ju: d(xd_next_raw)/d(ut_raw), shape (3, 2)
        """
        xd0_norm, ut_norm = self._normalize_input(xd0_raw, ut_raw)

        if self.has_adapter:
            ut_eff, J_adapter_x, J_adapter_u = self._adapter_forward_with_jac(
                xd0_norm, ut_norm
            )
        else:
            ut_eff = ut_norm
            J_adapter_x = self._zeros_2x3
            J_adapter_u = self._eye_2

        dt_val = self.dt_buf[0]
        net_in = torch.cat([xd0_norm, ut_eff, dt_val.unsqueeze(0)])
        y_norm, J_net = self.net.forward_with_jacobian(net_in)

        J_net_xd0 = J_net[:, :3]
        J_net_ueff = J_net[:, 3:5]

        J_norm_x = J_net_xd0 + J_net_ueff @ J_adapter_x
        J_norm_u = J_net_ueff @ J_adapter_u

        # Unstandardize Jacobians with broadcasted row/col scaling (faster than diag/matmul for tiny matrices).
        Jx_raw = (self.target_std.unsqueeze(1) * J_norm_x) * self._inv_input_std_x.unsqueeze(0)
        Ju_raw = (self.target_std.unsqueeze(1) * J_norm_u) * self._inv_input_std_u.unsqueeze(0)

        y_raw = self._denormalize_output(y_norm)

        return y_raw, Jx_raw, Ju_raw

    def _adapter_forward_with_jac(
        self, xd0_norm: torch.Tensor, ut_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.adapter_steer_net is not None, "adapter_steer_net must not be None"
        assert self.adapter_acc_net is not None, "adapter_acc_net must not be None"

        vx = xd0_norm[0]
        vy = xd0_norm[1]
        w = xd0_norm[2]
        vel_raw = ut_norm[0]
        delta_raw = ut_norm[1]

        eps = 1e-8
        speed_sq = vx * vx + vy * vy
        speed_mag = torch.sqrt(speed_sq + eps)
        speed_sign = torch.sign(vx)

        d_speed_dvx = vx / speed_mag
        d_speed_dvy = vy / speed_mag

        steer_features = [delta_raw, vel_raw, vx, vy, w]
        J_steer_feat_x = torch.zeros(5, 3, dtype=xd0_norm.dtype, device=xd0_norm.device)
        J_steer_feat_u = torch.zeros(5, 2, dtype=xd0_norm.dtype, device=xd0_norm.device)
        J_steer_feat_x[2, 0] = 1.0
        J_steer_feat_x[3, 1] = 1.0
        J_steer_feat_x[4, 2] = 1.0
        J_steer_feat_u[0, 1] = 1.0
        J_steer_feat_u[1, 0] = 1.0

        feat_idx = 5
        if self.include_speed_mag_steer:
            steer_features.append(speed_mag)
            extra_row_x = torch.stack([d_speed_dvx, d_speed_dvy, torch.zeros_like(d_speed_dvx)]).unsqueeze(0)
            extra_row_u = torch.zeros(
                1, 2, dtype=xd0_norm.dtype, device=xd0_norm.device
            )
            J_steer_feat_x = torch.cat([J_steer_feat_x, extra_row_x], dim=0)
            J_steer_feat_u = torch.cat([J_steer_feat_u, extra_row_u], dim=0)
            feat_idx += 1
        if self.include_speed_sign_steer:
            steer_features.append(speed_sign)
            extra_row_x = torch.zeros(
                1, 3, dtype=xd0_norm.dtype, device=xd0_norm.device
            )
            extra_row_u = torch.zeros(
                1, 2, dtype=xd0_norm.dtype, device=xd0_norm.device
            )
            J_steer_feat_x = torch.cat([J_steer_feat_x, extra_row_x], dim=0)
            J_steer_feat_u = torch.cat([J_steer_feat_u, extra_row_u], dim=0)

        steer_in = torch.stack(steer_features)
        delta_eff_raw, J_steer_net = self.adapter_steer_net.forward_with_jacobian(
            steer_in
        )
        delta_eff = delta_eff_raw[0]

        if self.steer_output_scale > 0:
            t = torch.tanh(delta_eff)
            dt_dh = (1.0 - t * t) * self.steer_output_scale
            delta_eff = t * self.steer_output_scale
            J_steer_net = dt_dh * J_steer_net

        J_delta_x = J_steer_net @ J_steer_feat_x
        J_delta_u = J_steer_net @ J_steer_feat_u

        acc_features = [vel_raw, delta_raw, vx]
        J_acc_feat_x = torch.zeros(3, 3, dtype=xd0_norm.dtype, device=xd0_norm.device)
        J_acc_feat_u = torch.zeros(3, 2, dtype=xd0_norm.dtype, device=xd0_norm.device)
        J_acc_feat_x[2, 0] = 1.0
        J_acc_feat_u[0, 0] = 1.0
        J_acc_feat_u[1, 1] = 1.0

        if self.include_speed_mag_acc:
            acc_features.append(speed_mag)
            extra_row_x = torch.stack([d_speed_dvx, d_speed_dvy, torch.zeros_like(d_speed_dvx)]).unsqueeze(0)
            extra_row_u = torch.zeros(
                1, 2, dtype=xd0_norm.dtype, device=xd0_norm.device
            )
            J_acc_feat_x = torch.cat([J_acc_feat_x, extra_row_x], dim=0)
            J_acc_feat_u = torch.cat([J_acc_feat_u, extra_row_u], dim=0)
        if self.include_speed_sign_acc:
            acc_features.append(speed_sign)
            extra_row_x = torch.zeros(
                1, 3, dtype=xd0_norm.dtype, device=xd0_norm.device
            )
            extra_row_u = torch.zeros(
                1, 2, dtype=xd0_norm.dtype, device=xd0_norm.device
            )
            J_acc_feat_x = torch.cat([J_acc_feat_x, extra_row_x], dim=0)
            J_acc_feat_u = torch.cat([J_acc_feat_u, extra_row_u], dim=0)
        if self.include_vy_acc:
            acc_features.append(vy)
            extra_row_x = torch.tensor(
                [[0.0, 1.0, 0.0]], dtype=xd0_norm.dtype, device=xd0_norm.device
            )
            extra_row_u = torch.zeros(
                1, 2, dtype=xd0_norm.dtype, device=xd0_norm.device
            )
            J_acc_feat_x = torch.cat([J_acc_feat_x, extra_row_x], dim=0)
            J_acc_feat_u = torch.cat([J_acc_feat_u, extra_row_u], dim=0)
        if self.include_w_acc:
            acc_features.append(w)
            extra_row_x = torch.tensor(
                [[0.0, 0.0, 1.0]], dtype=xd0_norm.dtype, device=xd0_norm.device
            )
            extra_row_u = torch.zeros(
                1, 2, dtype=xd0_norm.dtype, device=xd0_norm.device
            )
            J_acc_feat_x = torch.cat([J_acc_feat_x, extra_row_x], dim=0)
            J_acc_feat_u = torch.cat([J_acc_feat_u, extra_row_u], dim=0)

        acc_in = torch.stack(acc_features)
        acc_eff_raw, J_acc_net = self.adapter_acc_net.forward_with_jacobian(acc_in)
        acc_eff = acc_eff_raw[0]

        if self.acc_output_scale > 0:
            t = torch.tanh(acc_eff)
            dt_dh = (1.0 - t * t) * self.acc_output_scale
            acc_eff = t * self.acc_output_scale
            J_acc_net = dt_dh * J_acc_net

        J_acc_x = J_acc_net @ J_acc_feat_x
        J_acc_u = J_acc_net @ J_acc_feat_u

        ut_eff = torch.stack([acc_eff, delta_eff])
        J_adapter_x = torch.cat([J_acc_x, J_delta_x], dim=0)
        J_adapter_u = torch.cat([J_acc_u, J_delta_u], dim=0)

        return ut_eff, J_adapter_x, J_adapter_u


class StructuredAuxDeployModule(nn.Module):
    """
    TorchScript-exportable structured dynamics aux module.

    Returns u_eff, friction_k, residual and their Jacobians.
    The C++ side combines these with the analytic plant.
    """

    def __init__(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        dt: float,
        L: float = 0.31,
        has_adapter: bool = False,
        adapter_steer_net: MLPManualJac = None,
        adapter_acc_net: MLPManualJac = None,
        adapter_config: dict = None,
        has_friction: bool = False,
        friction_net: MLPManualJac = None,
        friction_config: dict = None,
        has_residual: bool = False,
        residual_net: MLPManualJac = None,
    ):
        super().__init__()

        self.has_adapter = has_adapter
        self.adapter_steer_net = adapter_steer_net
        self.adapter_acc_net = adapter_acc_net
        self.has_friction = has_friction
        self.friction_net = friction_net
        self.has_residual = has_residual
        self.residual_net = residual_net
        self.L = L

        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.register_buffer("dt_buf", torch.tensor([dt], dtype=input_mean.dtype))

        if adapter_config is not None:
            self.include_speed_mag_steer = adapter_config.get(
                "include_speed_mag_steer", False
            )
            self.include_speed_sign_steer = adapter_config.get(
                "include_speed_sign_steer", False
            )
            self.include_speed_mag_acc = adapter_config.get(
                "include_speed_mag_acc", False
            )
            self.include_speed_sign_acc = adapter_config.get(
                "include_speed_sign_acc", False
            )
            self.include_vy_acc = adapter_config.get("include_vy_acc", False)
            self.include_w_acc = adapter_config.get("include_w_acc", False)
            steer_scale = adapter_config.get("steer_output_scale", None)
            acc_scale = adapter_config.get("acc_output_scale", None)
            self.steer_output_scale = steer_scale if steer_scale is not None else -1.0
            self.acc_output_scale = acc_scale if acc_scale is not None else -1.0
        else:
            self.include_speed_mag_steer = False
            self.include_speed_sign_steer = False
            self.include_speed_mag_acc = False
            self.include_speed_sign_acc = False
            self.include_vy_acc = False
            self.include_w_acc = False
            self.steer_output_scale = -1.0
            self.acc_output_scale = -1.0

        if friction_config is not None:
            self.friction_use_dt = friction_config.get("friction_use_dt", True)
            self.friction_use_delta = friction_config.get("friction_use_delta", True)
            self.friction_use_vy = friction_config.get("friction_use_vy", True)
            self.friction_param_mode = friction_config.get(
                "friction_param_mode", "softplus_offset_1"
            )
            self.friction_k_min = friction_config.get("friction_k_min", 0.2)
            self.friction_k_max = friction_config.get("friction_k_max", 2.0)
            mode_map = {"softplus_offset_1": 0, "exp": 1, "sigmoid_range": 2}
            self.friction_mode_int = mode_map.get(self.friction_param_mode, 0)
        else:
            self.friction_use_dt = True
            self.friction_use_delta = True
            self.friction_use_vy = True
            self.friction_mode_int = 0
            self.friction_k_min = 0.2
            self.friction_k_max = 2.0

        # Pre-allocate diagonal matrices for Jacobian standardization (fused)
        # These avoid creating torch.diag() at runtime
        self.register_buffer("_diag_inv_std_x", torch.diag(1.0 / input_std[:3]))
        self.register_buffer("_diag_inv_std_u", torch.diag(1.0 / input_std[3:5]))
        self.register_buffer("_diag_std_u", torch.diag(input_std[3:5]))
        self.register_buffer("_diag_std_out", torch.diag(target_std))

        # Vector forms for faster broadcasted scaling (avoid small matmuls in forward_with_jacobian).
        self.register_buffer("_inv_input_std_x", 1.0 / input_std[:3])
        self.register_buffer("_inv_input_std_u", 1.0 / input_std[3:5])
        self.register_buffer("_std_u", input_std[3:5])
        self.register_buffer("_std_out", target_std)

        # Pre-allocate common zero/identity matrices
        self.register_buffer("_zeros_2x3", torch.zeros(2, 3, dtype=input_mean.dtype))
        self.register_buffer("_eye_2", torch.eye(2, dtype=input_mean.dtype))
        self.register_buffer("_ones_1", torch.ones(1, dtype=input_mean.dtype))
        self.register_buffer("_zeros_1x3", torch.zeros(1, 3, dtype=input_mean.dtype))
        self.register_buffer("_zeros_1x2", torch.zeros(1, 2, dtype=input_mean.dtype))
        self.register_buffer("_zeros_3", torch.zeros(3, dtype=input_mean.dtype))
        self.register_buffer("_zeros_3x3", torch.zeros(3, 3, dtype=input_mean.dtype))
        self.register_buffer("_zeros_3x2", torch.zeros(3, 2, dtype=input_mean.dtype))

        # Scratch buffers to avoid per-call cat/stack allocations in hot paths.
        # These are especially important for single-sample Jacobian queries.
        self.register_buffer("_residual_in_buf", torch.empty(5, dtype=input_mean.dtype))

        # Fast-path buffers for the common "all adapter features enabled" configuration.
        # steer features: [delta, vel, vx, vy, w, speed_mag, speed_sign] (7)
        # acc features:   [vel, delta, vx, speed_mag, speed_sign, vy, w] (7)
        self._adapter_all_features = (
            bool(self.include_speed_mag_steer)
            and bool(self.include_speed_sign_steer)
            and bool(self.include_speed_mag_acc)
            and bool(self.include_speed_sign_acc)
            and bool(self.include_vy_acc)
            and bool(self.include_w_acc)
        )
        if self._adapter_all_features:
            self.register_buffer("_steer_in_buf", torch.empty(7, dtype=input_mean.dtype))
            self.register_buffer("_acc_in_buf", torch.empty(7, dtype=input_mean.dtype))
            self.register_buffer("_ut_eff_buf", torch.empty(2, dtype=input_mean.dtype))
            self.register_buffer("_J_adapter_x_buf", torch.empty(2, 3, dtype=input_mean.dtype))
            self.register_buffer("_J_adapter_u_buf", torch.empty(2, 2, dtype=input_mean.dtype))

            J_steer_feat_x = torch.zeros(7, 3, dtype=input_mean.dtype)
            J_steer_feat_u = torch.zeros(7, 2, dtype=input_mean.dtype)
            # [delta, vel, vx, vy, w, speed_mag, speed_sign]
            J_steer_feat_u[0, 1] = 1.0  # d(delta)/d(steer)
            J_steer_feat_u[1, 0] = 1.0  # d(vel)/d(vel)
            J_steer_feat_x[2, 0] = 1.0  # d(vx)/d(vx)
            J_steer_feat_x[3, 1] = 1.0  # d(vy)/d(vy)
            J_steer_feat_x[4, 2] = 1.0  # d(w)/d(w)
            # speed_mag row (5) is filled per-call
            # speed_sign row (6) treated as constant zero derivative
            self.register_buffer("_J_steer_feat_x", J_steer_feat_x)
            self.register_buffer("_J_steer_feat_u", J_steer_feat_u)

            J_acc_feat_x = torch.zeros(7, 3, dtype=input_mean.dtype)
            J_acc_feat_u = torch.zeros(7, 2, dtype=input_mean.dtype)
            # [vel, delta, vx, speed_mag, speed_sign, vy, w]
            J_acc_feat_u[0, 0] = 1.0  # d(vel)/d(vel)
            J_acc_feat_u[1, 1] = 1.0  # d(delta)/d(steer)
            J_acc_feat_x[2, 0] = 1.0  # d(vx)/d(vx)
            J_acc_feat_x[5, 1] = 1.0  # d(vy)/d(vy)
            J_acc_feat_x[6, 2] = 1.0  # d(w)/d(w)
            # speed_mag row (3) is filled per-call
            # speed_sign row (4) treated as constant zero derivative
            self.register_buffer("_J_acc_feat_x", J_acc_feat_x)
            self.register_buffer("_J_acc_feat_u", J_acc_feat_u)

    def _normalize_input(
        self, xd0_raw: torch.Tensor, ut_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_raw = torch.cat([xd0_raw, ut_raw], dim=-1)
        x_norm = (x_raw - self.input_mean) / self.input_std
        return x_norm[:3], x_norm[3:5]

    def _compute_friction_k(self, raw_output: torch.Tensor) -> torch.Tensor:
        if self.friction_mode_int == 0:
            return 1.0 + torch.nn.functional.softplus(raw_output)
        elif self.friction_mode_int == 1:
            k = torch.exp(raw_output)
            return torch.clamp(k, min=self.friction_k_min, max=self.friction_k_max)
        else:
            return self.friction_k_min + (
                self.friction_k_max - self.friction_k_min
            ) * torch.sigmoid(raw_output)

    def _friction_k_derivative(self, raw_output: torch.Tensor) -> torch.Tensor:
        if self.friction_mode_int == 0:
            return torch.sigmoid(raw_output)
        elif self.friction_mode_int == 1:
            k = torch.exp(raw_output)
            in_range = (k >= self.friction_k_min) & (k <= self.friction_k_max)
            return torch.where(in_range, k, torch.zeros_like(k))
        else:
            s = torch.sigmoid(raw_output)
            return (self.friction_k_max - self.friction_k_min) * s * (1.0 - s)

    def forward(
        self, xd0_raw: torch.Tensor, ut_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning aux outputs.

        Args:
            xd0_raw: Raw velocity state [vx, vy, w], shape (3,)
            ut_raw: Raw control [vel_cmd, steer_cmd], shape (2,)

        Returns:
            ut_eff: Effective control (raw-space), shape (2,)
            friction_k: Friction coefficient (scalar as shape (1,)), or zeros if disabled
            residual: Residual correction (raw-space), shape (3,), or zeros if disabled
        """
        xd0_norm, ut_norm = self._normalize_input(xd0_raw, ut_raw)

        if self.has_adapter:
            ut_eff = self._adapter_forward(xd0_norm, ut_norm)
        else:
            ut_eff = ut_norm

        if self.has_friction:
            friction_k = self._friction_forward(xd0_norm, ut_eff)
        else:
            friction_k = torch.ones(1, dtype=xd0_raw.dtype, device=xd0_raw.device)

        if self.has_residual:
            residual_norm = self._residual_forward(xd0_norm, ut_eff)
            residual = residual_norm * self.target_std + self.target_mean
        else:
            residual = torch.zeros(3, dtype=xd0_raw.dtype, device=xd0_raw.device)

        ut_eff_raw = ut_eff * self.input_std[3:5] + self.input_mean[3:5]

        return ut_eff_raw, friction_k, residual

    def _adapter_forward(
        self, xd0_norm: torch.Tensor, ut_norm: torch.Tensor
    ) -> torch.Tensor:
        assert self.adapter_steer_net is not None, "adapter_steer_net must not be None"
        assert self.adapter_acc_net is not None, "adapter_acc_net must not be None"

        vx = xd0_norm[0]
        vy = xd0_norm[1]
        w = xd0_norm[2]
        vel_raw = ut_norm[0]
        delta_raw = ut_norm[1]

        eps = 1e-8
        speed_mag = torch.sqrt(vx * vx + vy * vy + eps)
        speed_sign = torch.sign(vx)

        steer_features = [delta_raw, vel_raw, vx, vy, w]
        if self.include_speed_mag_steer:
            steer_features.append(speed_mag)
        if self.include_speed_sign_steer:
            steer_features.append(speed_sign)
        steer_in = torch.stack(steer_features)
        delta_eff = self.adapter_steer_net(steer_in)[0]
        if self.steer_output_scale > 0:
            delta_eff = torch.tanh(delta_eff) * self.steer_output_scale

        acc_features = [vel_raw, delta_raw, vx]
        if self.include_speed_mag_acc:
            acc_features.append(speed_mag)
        if self.include_speed_sign_acc:
            acc_features.append(speed_sign)
        if self.include_vy_acc:
            acc_features.append(vy)
        if self.include_w_acc:
            acc_features.append(w)
        acc_in = torch.stack(acc_features)
        acc_eff = self.adapter_acc_net(acc_in)[0]
        if self.acc_output_scale > 0:
            acc_eff = torch.tanh(acc_eff) * self.acc_output_scale

        return torch.stack([acc_eff, delta_eff])

    def _friction_forward(
        self, xd0_norm: torch.Tensor, ut_eff: torch.Tensor
    ) -> torch.Tensor:
        assert self.friction_net is not None, "friction_net must not be None"

        vx = xd0_norm[0]
        vy = xd0_norm[1]
        delta_eff = ut_eff[1]
        dt_val = self.dt_buf[0]

        eps = 1e-8
        beta_prev = torch.atan2(vy, vx + eps)
        omega_prev = 2.0 * torch.sin(beta_prev) / self.L

        friction_feats = [omega_prev]
        if self.friction_use_delta:
            friction_feats.append(delta_eff)
        if self.friction_use_vy:
            friction_feats.append(vy)
        if self.friction_use_dt:
            friction_feats.append(dt_val)

        friction_in = torch.stack(friction_feats)
        friction_raw = self.friction_net(friction_in)[0]
        friction_k = self._compute_friction_k(friction_raw)

        return friction_k.unsqueeze(0)

    def _residual_forward(
        self, xd0_norm: torch.Tensor, ut_eff: torch.Tensor
    ) -> torch.Tensor:
        assert self.residual_net is not None, "residual_net must not be None"
        # Avoid torch.cat allocation on single-sample path.
        residual_in = self._residual_in_buf
        residual_in[:3] = xd0_norm
        residual_in[3] = ut_eff[0]
        residual_in[4] = ut_eff[1]
        return self.residual_net(residual_in)

    @torch.jit.export
    def forward_with_jacobian(
        self, xd0_raw: torch.Tensor, ut_raw: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Forward pass with Jacobians for all aux outputs.

        Returns:
            ut_eff_raw: shape (2,)
            friction_k: shape (1,)
            residual_raw: shape (3,)
            J_ueff_x: shape (2, 3) - d(ut_eff)/d(xd0_raw)
            J_ueff_u: shape (2, 2) - d(ut_eff)/d(ut_raw)
            J_k_x: shape (1, 3) - d(friction_k)/d(xd0_raw)
            J_k_u: shape (1, 2) - d(friction_k)/d(ut_raw)
            J_r_x: shape (3, 3) - d(residual)/d(xd0_raw)
            J_r_u: shape (3, 2) - d(residual)/d(ut_raw)
        """
        xd0_norm, ut_norm = self._normalize_input(xd0_raw, ut_raw)

        if self.has_adapter:
            ut_eff, J_adapter_x_norm, J_adapter_u_norm = self._adapter_forward_with_jac(
                xd0_norm, ut_norm
            )
        else:
            ut_eff = ut_norm
            J_adapter_x_norm = self._zeros_2x3
            J_adapter_u_norm = self._eye_2

        # Standardize Jacobians with broadcasted row/col scaling (faster than diag/matmul for tiny matrices).
        J_ueff_x = (self._std_u.unsqueeze(1) * J_adapter_x_norm) * self._inv_input_std_x.unsqueeze(0)
        J_ueff_u = (self._std_u.unsqueeze(1) * J_adapter_u_norm) * self._inv_input_std_u.unsqueeze(0)
        ut_eff_raw = ut_eff * self.input_std[3:5] + self.input_mean[3:5]

        # Sequential execution for friction and residual (CPU-optimal)
        if self.has_friction and self.has_residual:
            # Friction computation
            friction_k, J_k_xd_norm, J_k_ueff_norm = self._friction_forward_with_jac(
                xd0_norm, ut_eff
            )
            J_k_x_norm = J_k_xd_norm + J_k_ueff_norm @ J_adapter_x_norm
            J_k_u_norm = J_k_ueff_norm @ J_adapter_u_norm
            J_k_x = J_k_x_norm * self._inv_input_std_x.unsqueeze(0)
            J_k_u = J_k_u_norm * self._inv_input_std_u.unsqueeze(0)

            # Residual computation
            residual_norm, J_r_in = self._residual_forward_with_jac(xd0_norm, ut_eff)
            J_r_xd_norm = J_r_in[:, :3]
            J_r_ueff_norm = J_r_in[:, 3:5]
            J_r_x_norm = J_r_xd_norm + J_r_ueff_norm @ J_adapter_x_norm
            J_r_u_norm = J_r_ueff_norm @ J_adapter_u_norm
            J_r_x = (self._std_out.unsqueeze(1) * J_r_x_norm) * self._inv_input_std_x.unsqueeze(0)
            J_r_u = (self._std_out.unsqueeze(1) * J_r_u_norm) * self._inv_input_std_u.unsqueeze(0)
            residual_raw = residual_norm * self.target_std + self.target_mean
        elif self.has_friction:
            friction_k, J_k_xd_norm, J_k_ueff_norm = self._friction_forward_with_jac(
                xd0_norm, ut_eff
            )
            J_k_x_norm = J_k_xd_norm + J_k_ueff_norm @ J_adapter_x_norm
            J_k_u_norm = J_k_ueff_norm @ J_adapter_u_norm
            J_k_x = J_k_x_norm * self._inv_input_std_x.unsqueeze(0)
            J_k_u = J_k_u_norm * self._inv_input_std_u.unsqueeze(0)
            residual_raw = self._zeros_3
            J_r_x = self._zeros_3x3
            J_r_u = self._zeros_3x2
        elif self.has_residual:
            residual_norm, J_r_in = self._residual_forward_with_jac(xd0_norm, ut_eff)
            J_r_xd_norm = J_r_in[:, :3]
            J_r_ueff_norm = J_r_in[:, 3:5]
            J_r_x_norm = J_r_xd_norm + J_r_ueff_norm @ J_adapter_x_norm
            J_r_u_norm = J_r_ueff_norm @ J_adapter_u_norm
            J_r_x = (self._std_out.unsqueeze(1) * J_r_x_norm) * self._inv_input_std_x.unsqueeze(0)
            J_r_u = (self._std_out.unsqueeze(1) * J_r_u_norm) * self._inv_input_std_u.unsqueeze(0)
            residual_raw = residual_norm * self.target_std + self.target_mean
            friction_k = self._ones_1
            J_k_x = self._zeros_1x3
            J_k_u = self._zeros_1x2
        else:
            friction_k = self._ones_1
            J_k_x = self._zeros_1x3
            J_k_u = self._zeros_1x2
            residual_raw = self._zeros_3
            J_r_x = self._zeros_3x3
            J_r_u = self._zeros_3x2

        return (
            ut_eff_raw,
            friction_k,
            residual_raw,
            J_ueff_x,
            J_ueff_u,
            J_k_x,
            J_k_u,
            J_r_x,
            J_r_u,
        )

    def _steer_adapter_jac(
        self, steer_in: torch.Tensor, J_steer_feat_x: torch.Tensor, J_steer_feat_u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute steer adapter output and jacobians (for parallel execution)."""
        delta_eff_raw, J_steer_net = self.adapter_steer_net.forward_with_jacobian(steer_in)
        delta_eff = delta_eff_raw[0]

        if self.steer_output_scale > 0:
            t = torch.tanh(delta_eff)
            dt_dh = (1.0 - t * t) * self.steer_output_scale
            delta_eff = t * self.steer_output_scale
            J_steer_net = dt_dh * J_steer_net

        J_delta_x = J_steer_net @ J_steer_feat_x
        J_delta_u = J_steer_net @ J_steer_feat_u
        return delta_eff, J_delta_x, J_delta_u

    def _acc_adapter_jac(
        self, acc_in: torch.Tensor, J_acc_feat_x: torch.Tensor, J_acc_feat_u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute acc adapter output and jacobians (for parallel execution)."""
        acc_eff_raw, J_acc_net = self.adapter_acc_net.forward_with_jacobian(acc_in)
        acc_eff = acc_eff_raw[0]

        if self.acc_output_scale > 0:
            t = torch.tanh(acc_eff)
            dt_dh = (1.0 - t * t) * self.acc_output_scale
            acc_eff = t * self.acc_output_scale
            J_acc_net = dt_dh * J_acc_net

        J_acc_x = J_acc_net @ J_acc_feat_x
        J_acc_u = J_acc_net @ J_acc_feat_u
        return acc_eff, J_acc_x, J_acc_u

    def _adapter_forward_with_jac(
        self, xd0_norm: torch.Tensor, ut_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.adapter_steer_net is not None, "adapter_steer_net must not be None"
        assert self.adapter_acc_net is not None, "adapter_acc_net must not be None"

        vx = xd0_norm[0]
        vy = xd0_norm[1]
        w = xd0_norm[2]
        vel_raw = ut_norm[0]
        delta_raw = ut_norm[1]

        eps = 1e-8
        speed_sq = vx * vx + vy * vy
        speed_mag = torch.sqrt(speed_sq + eps)
        speed_sign = torch.sign(vx)

        d_speed_dvx = vx / speed_mag
        d_speed_dvy = vy / speed_mag

        # Fast path: all adapter features enabled -> avoid cat/stack and repeated tensor allocations.
        if self._adapter_all_features:
            steer_in = self._steer_in_buf
            steer_in[0] = delta_raw
            steer_in[1] = vel_raw
            steer_in[2] = vx
            steer_in[3] = vy
            steer_in[4] = w
            steer_in[5] = speed_mag
            steer_in[6] = speed_sign

            # Update only the dynamic row for speed_mag derivatives.
            J_steer_feat_x = self._J_steer_feat_x
            J_steer_feat_u = self._J_steer_feat_u
            J_steer_feat_x[5, 0] = d_speed_dvx
            J_steer_feat_x[5, 1] = d_speed_dvy
            J_steer_feat_x[5, 2] = 0.0

            delta_eff_raw, J_steer_net = self.adapter_steer_net.forward_with_jacobian(steer_in)
            delta_eff = delta_eff_raw[0]

            if self.steer_output_scale > 0:
                t = torch.tanh(delta_eff)
                dt_dh = (1.0 - t * t) * self.steer_output_scale
                delta_eff = t * self.steer_output_scale
                J_steer_net = dt_dh * J_steer_net

            J_delta_x = J_steer_net @ J_steer_feat_x
            J_delta_u = J_steer_net @ J_steer_feat_u

            acc_in = self._acc_in_buf
            acc_in[0] = vel_raw
            acc_in[1] = delta_raw
            acc_in[2] = vx
            acc_in[3] = speed_mag
            acc_in[4] = speed_sign
            acc_in[5] = vy
            acc_in[6] = w

            J_acc_feat_x = self._J_acc_feat_x
            J_acc_feat_u = self._J_acc_feat_u
            J_acc_feat_x[3, 0] = d_speed_dvx
            J_acc_feat_x[3, 1] = d_speed_dvy
            J_acc_feat_x[3, 2] = 0.0

            acc_eff_raw, J_acc_net = self.adapter_acc_net.forward_with_jacobian(acc_in)
            acc_eff = acc_eff_raw[0]

            if self.acc_output_scale > 0:
                t = torch.tanh(acc_eff)
                dt_dh = (1.0 - t * t) * self.acc_output_scale
                acc_eff = t * self.acc_output_scale
                J_acc_net = dt_dh * J_acc_net

            J_acc_x = J_acc_net @ J_acc_feat_x
            J_acc_u = J_acc_net @ J_acc_feat_u

            # Assemble outputs into preallocated buffers to avoid stack/cat allocations.
            ut_eff = self._ut_eff_buf
            ut_eff[0] = acc_eff
            ut_eff[1] = delta_eff

            J_adapter_x = self._J_adapter_x_buf
            J_adapter_x[0, :] = J_acc_x[0, :]
            J_adapter_x[1, :] = J_delta_x[0, :]

            J_adapter_u = self._J_adapter_u_buf
            J_adapter_u[0, :] = J_acc_u[0, :]
            J_adapter_u[1, :] = J_delta_u[0, :]

            return ut_eff, J_adapter_x, J_adapter_u

        # Build steer features and jacobians
        steer_features = [delta_raw, vel_raw, vx, vy, w]
        J_steer_feat_x = torch.zeros(5, 3, dtype=xd0_norm.dtype, device=xd0_norm.device)
        J_steer_feat_u = torch.zeros(5, 2, dtype=xd0_norm.dtype, device=xd0_norm.device)
        J_steer_feat_x[2, 0] = 1.0
        J_steer_feat_x[3, 1] = 1.0
        J_steer_feat_x[4, 2] = 1.0
        J_steer_feat_u[0, 1] = 1.0
        J_steer_feat_u[1, 0] = 1.0

        if self.include_speed_mag_steer:
            steer_features.append(speed_mag)
            extra_row_x = torch.stack([d_speed_dvx, d_speed_dvy, torch.zeros_like(d_speed_dvx)]).unsqueeze(0)
            extra_row_u = torch.zeros(1, 2, dtype=xd0_norm.dtype, device=xd0_norm.device)
            J_steer_feat_x = torch.cat([J_steer_feat_x, extra_row_x], dim=0)
            J_steer_feat_u = torch.cat([J_steer_feat_u, extra_row_u], dim=0)
        if self.include_speed_sign_steer:
            steer_features.append(speed_sign)
            extra_row_x = torch.zeros(1, 3, dtype=xd0_norm.dtype, device=xd0_norm.device)
            extra_row_u = torch.zeros(1, 2, dtype=xd0_norm.dtype, device=xd0_norm.device)
            J_steer_feat_x = torch.cat([J_steer_feat_x, extra_row_x], dim=0)
            J_steer_feat_u = torch.cat([J_steer_feat_u, extra_row_u], dim=0)

        steer_in = torch.stack(steer_features)

        # Build acc features and jacobians
        acc_features = [vel_raw, delta_raw, vx]
        J_acc_feat_x = torch.zeros(3, 3, dtype=xd0_norm.dtype, device=xd0_norm.device)
        J_acc_feat_u = torch.zeros(3, 2, dtype=xd0_norm.dtype, device=xd0_norm.device)
        J_acc_feat_x[2, 0] = 1.0
        J_acc_feat_u[0, 0] = 1.0
        J_acc_feat_u[1, 1] = 1.0

        if self.include_speed_mag_acc:
            acc_features.append(speed_mag)
            extra_row_x = torch.stack([d_speed_dvx, d_speed_dvy, torch.zeros_like(d_speed_dvx)]).unsqueeze(0)
            extra_row_u = torch.zeros(1, 2, dtype=xd0_norm.dtype, device=xd0_norm.device)
            J_acc_feat_x = torch.cat([J_acc_feat_x, extra_row_x], dim=0)
            J_acc_feat_u = torch.cat([J_acc_feat_u, extra_row_u], dim=0)
        if self.include_speed_sign_acc:
            acc_features.append(speed_sign)
            extra_row_x = torch.zeros(1, 3, dtype=xd0_norm.dtype, device=xd0_norm.device)
            extra_row_u = torch.zeros(1, 2, dtype=xd0_norm.dtype, device=xd0_norm.device)
            J_acc_feat_x = torch.cat([J_acc_feat_x, extra_row_x], dim=0)
            J_acc_feat_u = torch.cat([J_acc_feat_u, extra_row_u], dim=0)
        if self.include_vy_acc:
            acc_features.append(vy)
            extra_row_x = torch.tensor([[0.0, 1.0, 0.0]], dtype=xd0_norm.dtype, device=xd0_norm.device)
            extra_row_u = torch.zeros(1, 2, dtype=xd0_norm.dtype, device=xd0_norm.device)
            J_acc_feat_x = torch.cat([J_acc_feat_x, extra_row_x], dim=0)
            J_acc_feat_u = torch.cat([J_acc_feat_u, extra_row_u], dim=0)
        if self.include_w_acc:
            acc_features.append(w)
            extra_row_x = torch.tensor([[0.0, 0.0, 1.0]], dtype=xd0_norm.dtype, device=xd0_norm.device)
            extra_row_u = torch.zeros(1, 2, dtype=xd0_norm.dtype, device=xd0_norm.device)
            J_acc_feat_x = torch.cat([J_acc_feat_x, extra_row_x], dim=0)
            J_acc_feat_u = torch.cat([J_acc_feat_u, extra_row_u], dim=0)

        acc_in = torch.stack(acc_features)

        # Sequential execution of steer and acc networks (CPU-optimal)
        delta_eff_raw, J_steer_net = self.adapter_steer_net.forward_with_jacobian(steer_in)
        delta_eff = delta_eff_raw[0]

        if self.steer_output_scale > 0:
            t = torch.tanh(delta_eff)
            dt_dh = (1.0 - t * t) * self.steer_output_scale
            delta_eff = t * self.steer_output_scale
            J_steer_net = dt_dh * J_steer_net

        J_delta_x = J_steer_net @ J_steer_feat_x
        J_delta_u = J_steer_net @ J_steer_feat_u

        acc_eff_raw, J_acc_net = self.adapter_acc_net.forward_with_jacobian(acc_in)
        acc_eff = acc_eff_raw[0]

        if self.acc_output_scale > 0:
            t = torch.tanh(acc_eff)
            dt_dh = (1.0 - t * t) * self.acc_output_scale
            acc_eff = t * self.acc_output_scale
            J_acc_net = dt_dh * J_acc_net

        J_acc_x = J_acc_net @ J_acc_feat_x
        J_acc_u = J_acc_net @ J_acc_feat_u

        ut_eff = torch.stack([acc_eff, delta_eff])
        J_adapter_x = torch.cat([J_acc_x, J_delta_x], dim=0)
        J_adapter_u = torch.cat([J_acc_u, J_delta_u], dim=0)

        return ut_eff, J_adapter_x, J_adapter_u

    def _friction_forward_with_jac(
        self, xd0_norm: torch.Tensor, ut_eff: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.friction_net is not None, "friction_net must not be None"

        dtype = xd0_norm.dtype
        device = xd0_norm.device

        vx = xd0_norm[0]
        vy = xd0_norm[1]
        delta_eff = ut_eff[1]
        dt_val = self.dt_buf[0]

        eps = 1e-8
        denom = vx * vx + vy * vy + eps
        beta_prev = torch.atan2(vy, vx + eps)

        d_beta_dvx = -vy / denom
        d_beta_dvy = vx / denom

        cos_beta = torch.cos(beta_prev)
        omega_prev = 2.0 * torch.sin(beta_prev) / self.L
        d_omega_dbeta = 2.0 * cos_beta / self.L

        friction_feats = [omega_prev]
        J_feat_xd = torch.zeros(1, 3, dtype=dtype, device=device)
        J_feat_ueff = torch.zeros(1, 2, dtype=dtype, device=device)
        J_feat_xd[0, 0] = d_omega_dbeta * d_beta_dvx
        J_feat_xd[0, 1] = d_omega_dbeta * d_beta_dvy

        feat_idx = 1
        if self.friction_use_delta:
            friction_feats.append(delta_eff)
            extra_row_xd = torch.zeros(1, 3, dtype=dtype, device=device)
            extra_row_ueff = torch.zeros(1, 2, dtype=dtype, device=device)
            extra_row_ueff[0, 1] = 1.0
            J_feat_xd = torch.cat([J_feat_xd, extra_row_xd], dim=0)
            J_feat_ueff = torch.cat([J_feat_ueff, extra_row_ueff], dim=0)
            feat_idx += 1
        if self.friction_use_vy:
            friction_feats.append(vy)
            extra_row_xd = torch.zeros(1, 3, dtype=dtype, device=device)
            extra_row_xd[0, 1] = 1.0
            extra_row_ueff = torch.zeros(1, 2, dtype=dtype, device=device)
            J_feat_xd = torch.cat([J_feat_xd, extra_row_xd], dim=0)
            J_feat_ueff = torch.cat([J_feat_ueff, extra_row_ueff], dim=0)
            feat_idx += 1
        if self.friction_use_dt:
            friction_feats.append(dt_val)
            extra_row_xd = torch.zeros(1, 3, dtype=dtype, device=device)
            extra_row_ueff = torch.zeros(1, 2, dtype=dtype, device=device)
            J_feat_xd = torch.cat([J_feat_xd, extra_row_xd], dim=0)
            J_feat_ueff = torch.cat([J_feat_ueff, extra_row_ueff], dim=0)

        friction_in = torch.stack(friction_feats)
        friction_raw, J_net = self.friction_net.forward_with_jacobian(friction_in)
        friction_raw_val = friction_raw[0]

        friction_k = self._compute_friction_k(friction_raw_val)
        dk_dh = self._friction_k_derivative(friction_raw_val)

        J_k_feat = dk_dh * J_net
        J_k_xd = J_k_feat @ J_feat_xd
        J_k_ueff = J_k_feat @ J_feat_ueff

        return friction_k.unsqueeze(0), J_k_xd, J_k_ueff

    def _residual_forward_with_jac(
        self, xd0_norm: torch.Tensor, ut_eff: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.residual_net is not None, "residual_net must not be None"
        # Avoid torch.cat allocation on single-sample path.
        residual_in = self._residual_in_buf
        residual_in[:3] = xd0_norm
        residual_in[3] = ut_eff[0]
        residual_in[4] = ut_eff[1]
        residual_norm, J_r = self.residual_net.forward_with_jacobian(residual_in)
        return residual_norm, J_r
