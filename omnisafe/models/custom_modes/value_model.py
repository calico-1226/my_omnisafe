"""Value Model"""

import torch
from torch import nn

from omnisafe.utils.model import build_mlp_network


class MLPValueModel(nn.Module):
    """Value Model predicts a state-related value given a state."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        activation: str,
        output_activation: str = "identity",
        weight_initialization_mode: str = "kaiming_uniform",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        # *save network configuration
        self._in_dim: int = obs_dim
        self._out_dim: int = 1
        self._hidden_sizes: list = hidden_sizes
        self._activation: str = activation
        self._output_activation: str = output_activation
        self._weight_initialization_mode: str = weight_initialization_mode
        self._dtype: torch.dtype = dtype

        # *build network
        assert isinstance(hidden_sizes, list), "hidden_sizes must be a list"
        self._layers = [self._in_dim] + hidden_sizes + [self._out_dim]
        self.net = build_mlp_network(
            sizes=self._layers,
            activation=self._activation,
            output_activation=self._output_activation,
            weight_initialization_mode=self._weight_initialization_mode,
            # dtype=self._dtype,
        ).to(self._dtype)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward of the network.

        Args:
            obs(torch.Tensor): The state of the environment. Its shape is (batch_size, state_dim).

        Returns:
            torch.Tensor: The value of the state. Its shape is (batch_size,).
        """
        return self.net(obs).squeeze(-1)
