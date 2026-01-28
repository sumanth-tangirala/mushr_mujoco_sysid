"""
MLP with manual Jacobian computation for TorchScript export.

This module provides a TorchScript-compatible MLP that can compute
Jacobians without runtime autograd.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class MLPManualJac(nn.Module):
    """
    TorchScript-compatible MLP with manual Jacobian computation.

    Supports activations: tanh, relu, leaky_relu, elu.
    Must be in eval() mode for correct Jacobian computation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "tanh",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        act_map = {
            "tanh": 0,
            "relu": 1,
            "leaky_relu": 2,
            "elu": 3,
        }
        if activation.lower() not in act_map:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation_type: int = act_map[activation.lower()]

        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h))
            prev_dim = h
        self.layers.append(nn.Linear(prev_dim, output_dim))

        # Pre-allocate identity matrix for Jacobian computation
        # Will be moved to correct device/dtype when model is moved
        self.register_buffer("_eye", torch.eye(input_dim))

    def _activation(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation_type == 0:  # tanh
            return torch.tanh(z)
        elif self.activation_type == 1:  # relu
            return torch.relu(z)
        elif self.activation_type == 2:  # leaky_relu
            return torch.nn.functional.leaky_relu(z, 0.01)
        else:  # elu
            return torch.nn.functional.elu(z)

    def _activation_derivative(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation_type == 0:  # tanh
            t = torch.tanh(z)
            return 1.0 - t * t
        elif self.activation_type == 1:  # relu
            return (z > 0).to(z.dtype)
        elif self.activation_type == 2:  # leaky_relu
            return torch.where(z > 0, torch.ones_like(z), 0.01 * torch.ones_like(z))
        else:  # elu
            return torch.where(z > 0, torch.ones_like(z), torch.exp(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self._activation(x)
        return x

    @torch.jit.export
    def forward_with_jacobian(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with manual Jacobian computation.

        Args:
            x: Input tensor of shape (in_dim,) - single sample only

        Returns:
            y: Output tensor of shape (out_dim,)
            J: Jacobian matrix of shape (out_dim, in_dim)
        """
        # Use pre-allocated identity matrix (clone to avoid in-place modification)
        J = self._eye.clone()

        h = x
        for i, layer in enumerate(self.layers):
            W = layer.weight
            b = layer.bias
            z = h @ W.t() + b

            J = W @ J

            if i < len(self.layers) - 1:
                h = self._activation(z)
                dphi = self._activation_derivative(z)
                # Element-wise row scaling instead of diag matrix multiply
                # dphi.unsqueeze(1) * J is O(H*N) vs torch.diag(dphi) @ J which is O(H²*N)
                J = dphi.unsqueeze(1) * J
            else:
                h = z

        return h, J

    @classmethod
    def from_mlp(cls, mlp: nn.Module) -> "MLPManualJac":
        """
        Create MLPManualJac from an existing MLP module.

        Args:
            mlp: Source MLP module with .network as nn.Sequential

        Returns:
            New MLPManualJac with copied weights
        """
        act_class = mlp.activation_fn
        act_name_map = {
            nn.Tanh: "tanh",
            nn.ReLU: "relu",
            nn.LeakyReLU: "leaky_relu",
            nn.ELU: "elu",
        }
        if act_class not in act_name_map:
            raise ValueError(f"Unsupported activation class: {act_class}")
        act_name = act_name_map[act_class]

        new_mlp = cls(
            input_dim=mlp.input_dim,
            output_dim=mlp.output_dim,
            hidden_dims=mlp.hidden_dims,
            activation=act_name,
        )

        layer_idx = 0
        for module in mlp.network:
            if isinstance(module, nn.Linear):
                new_mlp.layers[layer_idx].weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    new_mlp.layers[layer_idx].bias.data.copy_(module.bias.data)
                layer_idx += 1

        new_mlp.eval()
        return new_mlp
