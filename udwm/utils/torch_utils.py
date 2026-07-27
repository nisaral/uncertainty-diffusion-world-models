from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


def get_device(name: str = "cpu") -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: Sequence[int] = (256, 256),
    activation: Optional[nn.Module] = None,
    output_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    if activation is None:
        activation = nn.ReLU()
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(last, h), type(activation)()])
        last = h
    layers.append(nn.Linear(last, out_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def to_tensor(x, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)
