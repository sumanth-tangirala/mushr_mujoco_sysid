import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for system identification.

    Flexible architecture that can be configured via config file.
    """

    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 4,
        hidden_dims: List[int] = [64, 64],
        activation: str = 'relu',
        dropout: float = 0.0,
        use_batch_norm: bool = False
    ):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu', 'gelu')
            dropout: Dropout probability (0.0 = no dropout)
            use_batch_norm: Whether to use batch normalization
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Select activation function
        activation_dict = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU,
            'gelu': nn.GELU,
            'leaky_relu': nn.LeakyReLU
        }
        if activation.lower() not in activation_dict:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activation_dict.keys())}")
        self.activation_fn = activation_dict[activation.lower()]

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: dict) -> nn.Module:
    """
    Factory function to create models based on config.

    This allows easy extension to other model types in the future.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        Initialized model
    """
    model_type = config.get('model_type', 'mlp').lower()

    if model_type == 'mlp':
        return MLP(
            input_dim=config.get('input_dim', 5),
            output_dim=config.get('output_dim', 4),
            hidden_dims=config.get('hidden_dims', [64, 64]),
            activation=config.get('activation', 'relu'),
            dropout=config.get('dropout', 0.0),
            use_batch_norm=config.get('use_batch_norm', False)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
