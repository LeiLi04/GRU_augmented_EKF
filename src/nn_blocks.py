"""Reusable neural network blocks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import torch
from torch import nn


@dataclass
class MLPConfig:
    in_features: int
    hidden_sizes: Sequence[int]
    out_features: int
    activation: Callable[[], nn.Module] = nn.SiLU
    dropout: Optional[float] = None
    layer_norm: bool = False


class MLP(nn.Module):
    """Simple configurable multilayer perceptron."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        input_size = config.in_features
        for hidden_size in config.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if config.layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(config.activation())
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, config.out_features))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, features: int, activation: Optional[nn.Module] = None, gated: bool = False) -> None:
        super().__init__()
        self.linear1 = nn.Linear(features, features)
        self.linear2 = nn.Linear(features, features)
        self.activation = activation or nn.SiLU()
        self.gated = gated
        if gated:
            self.gate = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        if self.gated:
            gate = torch.sigmoid(self.gate(residual))
            out = gate * out
        return residual + out


class ResidualMLP(nn.Module):
    """Stack of residual blocks with an input/output projection."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int = 2,
        activation: Optional[nn.Module] = None,
        gated: bool = False,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_features)
        blocks = [ResidualBlock(hidden_features, activation=activation, gated=gated) for _ in range(depth)]
        self.blocks = nn.Sequential(*blocks)
        self.activation = activation or nn.SiLU()
        self.output_proj = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.input_proj(x))
        hidden = self.blocks(hidden)
        return self.output_proj(hidden)


__all__ = ["MLP", "MLPConfig", "ResidualBlock", "ResidualMLP"]
