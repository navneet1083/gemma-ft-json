"""TableViT-Lite: a from-scratch vision encoder biased toward documents.

WHY from scratch?  The "no Hugging Face downloads" constraint removes
CLIP/SigLIP. Donut showed an encoder can learn to read documents when
pretrained on *synthetic* renders — and tables are a far narrower visual
domain than natural images (binary-ish colors, axis-aligned strokes, text
glyphs), so a small encoder (~11M params) trained on our synthetic corpus
is sufficient and fits comfortably in 48 GB unified memory next to Gemma.

Architecture (input 448x448):
    Conv stem  : 4 stride-2 convs  -> [B, D, 28, 28]   (stride 16 total)
                 (convs > naive patchify for thin gridlines / small glyphs:
                  overlapping receptive fields preserve stroke continuity)
    + learned 2-D positional embedding (tables are intrinsically 2-D; row/col
      position IS the structure signal)
    Transformer: `depth` pre-norm blocks over the 784 tokens
    PixelShuffle merge 2x2 -> 196 tokens of dim 4*D
        (downsamples the SEQUENCE, not the information: 4 neighbours are
         concatenated channel-wise, then the projector mixes them. 196 visual
         tokens keep Gemma's quadratic attention affordable on MPS.)
Output: [B, 196, 4*embed_dim]  -> consumed by the projector.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..utils.registry import Registry
from .guards import assert_shape, check_finite

VISION_ENCODERS = Registry("vision_encoder")


class _Block(nn.Module):
    """Standard pre-norm transformer block (LN -> MHSA -> LN -> MLP).

    Pre-norm (LayerNorm BEFORE the sublayer) is chosen over post-norm because
    it keeps gradient magnitudes bounded in from-scratch training — the
    single most effective architectural NaN preventative.
    """

    def __init__(self, dim: int, heads: int, drop: float):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * 4, dim), nn.Dropout(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MHSA internally computes softmax(Q @ K^T / sqrt(d)) @ V :
        #   Q,K,V: [B, T, dim] -> attention matrix [B, heads, T, T] -> ctx [B, T, dim]
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


@VISION_ENCODERS.register("table_vit_lite")
class TableViTLite(nn.Module):
    def __init__(self, image_size: int = 448, embed_dim: int = 256, depth: int = 6,
                 num_heads: int = 4, patch_stride: int = 16, merge_factor: int = 2,
                 drop_rate: float = 0.0):
        super().__init__()
        if image_size % patch_stride:
            raise ValueError("image_size must be divisible by patch_stride")
        self.grid = image_size // patch_stride            # 28
        if self.grid % merge_factor:
            raise ValueError("grid must be divisible by merge_factor")
        self.merge = merge_factor
        self.out_grid = self.grid // merge_factor         # 14
        self.num_tokens = self.out_grid ** 2              # 196
        self.out_dim = embed_dim * merge_factor ** 2      # 1024

        d = embed_dim
        # Overlapping conv stem: 3 -> d/8 -> d/4 -> d/2 -> d, total stride 16.
        self.stem = nn.Sequential(
            nn.Conv2d(3, d // 8, 5, 2, 2), nn.GELU(),
            nn.Conv2d(d // 8, d // 4, 3, 2, 1), nn.GELU(),
            nn.Conv2d(d // 4, d // 2, 3, 2, 1), nn.GELU(),
            nn.Conv2d(d // 2, d, 3, 2, 1),
        )
        # Learned 2-D positional embedding, one vector per stem cell.
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [_Block(d, num_heads, drop_rate) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        # Small-std init keeps early activations ~N(0, .02) -> no fp blowups.
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        assert_shape(pixel_values, (None, 3, None, None), "pixel_values")
        x = self.stem(pixel_values)                       # [B, D, 28, 28]
        B, D, H, W = x.shape
        if H != self.grid or W != self.grid:
            # Loud failure beats a silent positional-embedding misalignment.
            raise RuntimeError(f"stem grid {H}x{W} != expected {self.grid}")
        x = x.flatten(2).transpose(1, 2)                  # [B, 784, D]
        x = x + self.pos                                  # broadcast add
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = check_finite(x, "vision_tokens", sanitize=True)

        # ---- pixel-shuffle token merge: [B, 28*28, D] -> [B, 14*14, 4D] ----
        m, g = self.merge, self.grid
        x = x.view(B, g, g, D)
        x = x.view(B, g // m, m, g // m, m, D)            # split into m x m cells
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()      # group the 4 neighbours
        x = x.view(B, self.num_tokens, D * m * m)         # concat channel-wise
        assert_shape(x, (B, self.num_tokens, self.out_dim), "merged_vision_tokens")
        return x
