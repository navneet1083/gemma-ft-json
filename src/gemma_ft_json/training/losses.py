"""Structure-weighted causal-LM loss with NaN containment.

Token-weighting rationale: in a JSON target roughly 30-40% of tokens are
scaffolding ({, }, ", :, ,) that the model learns in minutes. With uniform
CE the loss curve looks great while the model still misreads every cell.
Down-weighting structure / up-weighting content makes the loss a faithful
proxy for the metric we actually care about (cell-level accuracy).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from ..exceptions import NumericalError
from ..models.guards import assert_shape


def weighted_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor,
                            weights: torch.Tensor | None = None,
                            label_smoothing: float = 0.0,
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute next-token CE.

    Shapes:
        logits  [B, T, V]   (T = fused length incl. visual prefix)
        labels  [B, T]      (-100 = ignored: prompt, visual, padding)
        weights [B, T] or None  (already aligned to labels by caller)

    Causal shift: position t predicts token t+1, hence logits[:, :-1] vs
    labels[:, 1:]. The CE itself is `-log softmax(logits) [gathered at the
    gold ids]`; the softmax normalizer is a matmul-free reduction but the
    logits feeding it came from hidden @ E^T in the LM head.

    Returns (mean_weighted_loss, n_scored_tokens).
    Raises NumericalError on a non-finite loss so the trainer can SKIP the
    step (and log it) instead of corrupting optimizer state.
    """
    assert_shape(logits, (None, None, None), "logits")
    if logits.shape[:2] != labels.shape:
        raise NumericalError(
            f"logits/labels misaligned: {tuple(logits.shape[:2])} vs "
            f"{tuple(labels.shape)} — visual-prefix padding bug?")

    shift_logits = logits[:, :-1, :].contiguous().float()  # fp32 CE: stable on MPS
    shift_labels = labels[:, 1:].contiguous()

    per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100, reduction="none",
        label_smoothing=label_smoothing,
    ).view(shift_labels.shape)                              # [B, T-1]

    mask = (shift_labels != -100).float()
    if weights is not None:
        w = weights[:, 1:].contiguous() * mask
    else:
        w = mask

    denom = w.sum().clamp_min(1.0)            # clamp: empty batch can't divide by 0
    loss = (per_tok * w).sum() / denom

    if not torch.isfinite(loss):
        raise NumericalError("non-finite loss")
    return loss, mask.sum()
