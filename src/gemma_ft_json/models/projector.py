"""Vision->LLM projector.

Maps vision patch embeddings (vision embed_dim) into the decoder's embedding
space (hidden_size), producing the "soft visual tokens" prepended to text token
embeddings. This small MLP is the main trainable bridge in the LLaVA-style design.

Explicit dimension checks here because a mismatch (encoder dim != projector input)
is the most common wiring bug and, unguarded, surfaces later as a confusing
device error or a NaN loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.checks import check_shape, assert_finite


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, inner_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        hidden = out_dim * inner_mult
        # Two matmuls:
        #   h = x[B,N,in_dim] @ W1[in_dim,hidden] -> [B,N,hidden]
        #   y = gelu(h)       @ W2[hidden,out_dim] -> [B,N,out_dim]
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_shape(x, (None, None, self.in_dim), "projector_input")
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        check_shape(x, (None, None, self.out_dim), "projector_output")
        assert_finite(x, "projector_output")
        return x


def build_projector(projector_cfg, in_dim: int, out_dim: int) -> nn.Module:
    """Factory. One type today ('mlp'); kept extensible (e.g. a future Q-Former/
    resampler can be added here without touching callers)."""
    if projector_cfg.type != "mlp":
        raise ValueError(f"Unknown projector type: {projector_cfg.type}")
    return MLPProjector(in_dim, out_dim, projector_cfg.inner_mult, projector_cfg.dropout)
