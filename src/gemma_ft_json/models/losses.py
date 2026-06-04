"""Loss for faithful image->JSON extraction.

Core objective: standard autoregressive (next-token) cross-entropy, SHAPED to be
faithful/extractive (per the design discussion). There is no non-generative "OCR
loss" for a generative decoder; faithfulness comes from (a) masking so only the
JSON target is supervised, (b) up-weighting content/value tokens over JSON
scaffolding (digits matter more than braces), and optionally a grounding term.
Stronger faithfulness (constrained decoding, RL with exact-match reward) layers
on at decode / post-training time.

NUMERICS: F.cross_entropy uses a stable log-softmax internally; we assert
finiteness so any blow-up is caught at its source, not as silent NaN grads.
"""
from __future__ import annotations

from typing import Optional, Set

import torch
import torch.nn.functional as F

from ..utils.checks import assert_finite

IGNORE_INDEX = -100


def faithful_lm_loss(
    logits: torch.Tensor,            # [B, T, V] over the FULL sequence (vision+text)
    labels: torch.Tensor,            # [B, T] with -100 on non-supervised positions
    content_token_weight: float = 1.0,
    scaffold_ids: Optional[Set[int]] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Masked, shifted, optionally content-weighted cross-entropy.

    SHIFT: position t predicts token t+1, so align logits[:, :-1] with
    labels[:, 1:] (standard causal-LM teacher forcing).

    MATRIX-MULTIPLY NOTE: `logits` already came from hidden @ W_lm^T inside the
    decoder. Cross-entropy = -log_softmax(logits)[target]; the log_softmax is the
    vocab-axis normalization, and gathering the target index equals the dot
    product of a one-hot target with the log-probabilities.
    """
    shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()      # [B, T-1]
    B, Tm1, V = shift_logits.shape

    flat_logits = shift_logits.view(-1, V)         # [(B*(T-1)), V]
    flat_labels = shift_labels.view(-1)            # [(B*(T-1))]

    # Per-token NLL, no reduction (so we can weight it). ignore_index -> 0 there.
    nll = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_INDEX,
                          reduction="none", label_smoothing=label_smoothing)

    valid = flat_labels != IGNORE_INDEX
    if valid.sum() == 0:
        return flat_logits.sum() * 0.0  # degenerate batch -> finite zero w/ grad

    # Content up-weighting: value tokens penalized more than JSON scaffolding.
    weights = torch.ones_like(nll)
    if scaffold_ids and content_token_weight != 1.0:
        scaffold = torch.zeros_like(flat_labels, dtype=torch.bool)
        for sid in scaffold_ids:
            scaffold |= (flat_labels == sid)
        is_content = valid & (~scaffold)
        weights = torch.where(is_content, torch.full_like(nll, content_token_weight), weights)

    loss = (nll * weights).sum() / weights[valid].sum().clamp_min(1.0)
    assert_finite(loss, "lm_loss")
    return loss
