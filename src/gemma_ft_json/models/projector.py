"""RMS-Calibrated Projector: vision tokens -> Gemma's embedding space.

THE critical fix for "LoRA fine-tuning is not working" with a text-only LLM:

1. There must be a *trainable bridge* from pixels to the LM input. LoRA on
   Gemma's attention alone gives the model literally nothing visual to
   attend to — gradients w.r.t. the image do not exist. This module is that
   bridge (LLaVA-style MLP projector).

2. SCALE MISMATCH is the silent killer. Gemma multiplies token embeddings by
   sqrt(hidden_size) (~25.3 for hidden=640) before the first layer. A
   random-init MLP outputs vectors with RMS ~1, i.e. ~25x *smaller* than
   what the LM expects — the visual tokens are effectively whispers the
   attention layers ignore, the model collapses to language-prior guessing,
   and val loss plateaus exactly the way a "broken LoRA run" looks.
   Fix: LayerNorm the projector output, then multiply by a scalar
   *calibrated at startup to the empirical RMS of Gemma's real text
   embeddings* (robust to whether the transformers version scales inside or
   outside `embed_tokens`), times a learnable fine-adjustment gain.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.registry import Registry
from .guards import assert_shape, check_finite

PROJECTORS = Registry("projector")


@PROJECTORS.register("rms_mlp")
class RMSCalibratedProjector(nn.Module):
    def __init__(self, in_dim: int, lm_dim: int, hidden_mult: int = 2,
                 rms_calibrate: bool = True):
        super().__init__()
        h = lm_dim * hidden_mult
        # 2-layer MLP (LLaVA-1.5 finding: MLP > single linear for alignment).
        # Each Linear is a matmul: [B, T, in_dim] @ [in_dim, h] -> [B, T, h],
        # then [B, T, h] @ [h, lm_dim] -> [B, T, lm_dim].
        self.fc1 = nn.Linear(in_dim, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, lm_dim)
        self.out_norm = nn.LayerNorm(lm_dim)      # forces output RMS == 1 exactly
        self.rms_calibrate = rms_calibrate
        # target_rms is filled by `calibrate()`; gain is a learnable trim knob.
        self.register_buffer("target_rms", torch.tensor(1.0))
        self.gain = nn.Parameter(torch.tensor(1.0))
        nn.init.trunc_normal_(self.fc1.weight, std=0.02); nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02); nn.init.zeros_(self.fc2.bias)

    @torch.no_grad()
    def calibrate(self, embed_layer: nn.Module, sample_ids: torch.Tensor) -> float:
        """Measure the RMS of REAL text embeddings (post any internal Gemma
        scaling) and store it as the projector's output magnitude target.
        Called once by the VLM constructor — version-proof by construction."""
        e = embed_layer(sample_ids)                       # [1, N, lm_dim]
        rms = e.float().pow(2).mean().sqrt().item()
        self.target_rms.fill_(rms)
        return rms

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        assert_shape(vision_tokens, (None, None, self.fc1.in_features), "vision_tokens")
        x = self.fc2(self.act(self.fc1(vision_tokens)))   # two matmuls, see above
        x = self.out_norm(x)                              # RMS ~ 1
        if self.rms_calibrate:
            # Re-amplify to the LM's native embedding magnitude (~sqrt(hidden)).
            x = x * (self.target_rms * self.gain)
        x = check_finite(x, "projected_visual_embeds", sanitize=True)
        return x                                          # [B, T_vis, lm_dim]
