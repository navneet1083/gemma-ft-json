"""Dimension + numerical guards used at EVERY module boundary.

Rationale (project requirement: "should not throw any exception while
training like NaN etc."): the only thing worse than a crash at step 12,000
is a *silent* NaN that poisons the optimizer state. These guards (a) catch
shape bugs at module boundaries with named, actionable errors, and
(b) detect/neutralize non-finite values before they reach the optimizer.
"""
from __future__ import annotations

from typing import Sequence

import torch

from ..exceptions import NumericalError, ShapeError


def assert_shape(t: torch.Tensor, shape: Sequence, name: str) -> None:
    """Validate tensor shape; `None` entries are wildcards.

    Example: assert_shape(x, (None, 196, 640), "visual_embeds")
    """
    if t.dim() != len(shape):
        raise ShapeError(f"{name}: expected {len(shape)}D {tuple(shape)}, "
                         f"got {t.dim()}D {tuple(t.shape)}")
    for i, (got, want) in enumerate(zip(t.shape, shape)):
        if want is not None and got != want:
            raise ShapeError(f"{name}: dim {i} expected {want}, got {got} "
                             f"(full shape {tuple(t.shape)})")


def check_finite(t: torch.Tensor, name: str, sanitize: bool = False) -> torch.Tensor:
    """Detect NaN/Inf. If `sanitize`, replace with zeros (and clamp) instead
    of raising — used in *forward* paths where one bad activation should not
    abort the run; raising is reserved for the loss/grad checks in the
    trainer, where a skip-step policy handles it gracefully.
    """
    if torch.isfinite(t).all():
        return t
    if sanitize:
        # nan_to_num: NaN->0, +Inf-> large finite, -Inf-> -large finite.
        return torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)
    bad = (~torch.isfinite(t)).sum().item()
    raise NumericalError(f"{name}: {bad} non-finite values detected "
                         f"(shape {tuple(t.shape)})")
