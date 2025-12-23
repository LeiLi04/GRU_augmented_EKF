"""Per-sample Jacobian helpers."""
from __future__ import annotations

from typing import Callable

import torch

try:
    from torch.func import jacrev
except ImportError:  # pragma: no cover
    jacrev = None  # type: ignore


TensorFn = Callable[[torch.Tensor], torch.Tensor]


def batch_jacobian(fn: TensorFn, x: torch.Tensor) -> torch.Tensor:
    if jacrev is not None:
        def single_fn(single_x: torch.Tensor) -> torch.Tensor:
            return fn(single_x.unsqueeze(0)).squeeze(0)

        jac = jacrev(single_fn)
        return torch.stack([jac(x_i) for x_i in x], dim=0)

    jac_list = []
    for x_i in x:
        x_i = x_i.detach().requires_grad_(True)
        jac_i = torch.autograd.functional.jacobian(
            lambda inp: fn(inp.unsqueeze(0)).squeeze(0),
            x_i,
            create_graph=True,
            vectorize=True,
        )
        jac_list.append(jac_i)
    return torch.stack(jac_list, dim=0)


__all__ = ["batch_jacobian"]
