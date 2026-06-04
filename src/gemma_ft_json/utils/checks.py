"""Tensor guards: shape checks and NaN/Inf detection.

These implement the "layers must check dimensions so training never throws NaN"
requirement. We call them at the boundaries between modules
(encoder -> projector -> decoder) and around the loss, so a wiring/shape bug
fails LOUDLY at the exact site instead of silently producing a NaN ten steps
later. They are cheap and can be globally disabled with GEMMA_FT_DISABLE_CHECKS=1.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import torch

from ..exceptions import ShapeMismatchError, NumericalError

_DISABLED = os.environ.get("GEMMA_FT_DISABLE_CHECKS", "0") == "1"


def check_shape(t: torch.Tensor, expected: Sequence[Optional[int]], name: str) -> None:
    """Assert `t` has the expected shape. `None` means "any size" on that axis,
    e.g. check_shape(x, (None, None, 640), "x") -> rank-3, last dim 640."""
    if _DISABLED:
        return
    if t.dim() != len(expected):
        raise ShapeMismatchError(
            f"[{name}] expected rank {len(expected)} but got shape {tuple(t.shape)}"
        )
    for axis, (got, want) in enumerate(zip(t.shape, expected)):
        if want is not None and got != want:
            raise ShapeMismatchError(
                f"[{name}] axis {axis}: expected {want}, got {got} "
                f"(full shape {tuple(t.shape)})"
            )


def assert_finite(t: torch.Tensor, name: str) -> None:
    """Raise if the tensor contains NaN or Inf (covers both via torch.isfinite)."""
    if _DISABLED:
        return
    if not torch.isfinite(t).all():
        n_nan = int(torch.isnan(t).sum())
        n_inf = int(torch.isinf(t).sum())
        raise NumericalError(
            f"[{name}] non-finite values: {n_nan} NaN, {n_inf} Inf (shape {tuple(t.shape)})"
        )


def safe_softmax_dim(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax: subtract the per-row max before exp().

    softmax is shift-invariant, so this does not change the result but prevents
    exp() overflow -> Inf -> NaN, a classic training blow-up.
    """
    z = logits - logits.amax(dim=dim, keepdim=True)
    return torch.softmax(z, dim=dim)
