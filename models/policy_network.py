from __future__ import annotations

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        hidden_layers: list[int] | None = None,
        activation: str = "relu",
        weight_init: str = "he",
        normalization: str | None = None,
        normalization_position: str = "before_activation",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layer_widths = hidden_layers or [hidden_dim, hidden_dim]
        self.network = self._build_network(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=layer_widths,
            activation=activation,
            normalization=normalization,
            normalization_position=normalization_position,
            dropout=dropout,
        )
        self._initialize_weights(weight_init)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def hidden_activations(self, inputs: torch.Tensor) -> torch.Tensor:
        activations = inputs
        seen_linear = False
        for layer in self.network:
            activations = layer(activations)
            if isinstance(layer, nn.Linear):
                seen_linear = True
            elif seen_linear and _is_activation(layer):
                return activations
        return activations

    @staticmethod
    def _build_network(
        *,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        activation: str,
        normalization: str | None,
        normalization_position: str,
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            norm_layer = _build_normalization(normalization, hidden_dim)
            activation_layer = _build_activation(activation)

            if norm_layer is not None and normalization_position == "before_activation":
                layers.append(norm_layer)
            layers.append(activation_layer)
            if norm_layer is not None and normalization_position == "after_activation":
                layers.append(norm_layer)
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))
        return nn.Sequential(*layers)

    def _initialize_weights(self, weight_init: str) -> None:
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            if weight_init == "he":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif weight_init == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif weight_init == "orthogonal":
                nn.init.orthogonal_(module.weight)
            else:
                msg = f"Unsupported weight initialization '{weight_init}'."
                raise ValueError(msg)
            nn.init.zeros_(module.bias)


def _build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "tanh":
        return nn.Tanh()
    msg = f"Unsupported activation '{name}'."
    raise ValueError(msg)


def _build_normalization(name: str | None, features: int) -> nn.Module | None:
    if name is None:
        return None
    if name == "batch":
        return nn.BatchNorm1d(features)
    if name == "layer":
        return nn.LayerNorm(features)
    msg = f"Unsupported normalization '{name}'."
    raise ValueError(msg)


def _is_activation(layer: nn.Module) -> bool:
    return isinstance(
        layer,
        nn.ReLU | nn.LeakyReLU | nn.ELU | nn.SELU | nn.Tanh,
    )
