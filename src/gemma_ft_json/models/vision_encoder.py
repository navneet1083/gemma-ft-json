"""Vision encoders.

Two backends behind one factory:

* `FromScratchViT` — a compact, dependency-free ViT in plain PyTorch. Downloads
  NOTHING. Random-init runs the pipeline; `load_local_weights` loads a local .pt.
  (A ViT is the right *architecture*; ideally you want contrastive,
  aspect-ratio-preserving pretraining — but with no local checkpoint, a
  from-scratch ViT trained on your corpus is the zero-download fallback.)

* `LocalSiglipEncoder` — wraps a SigLIP/ViT loaded OFFLINE from a local dir.

Both return patch embeddings [B, num_patches, embed_dim].
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from ..exceptions import WeightsNotFoundError
from ..utils.checks import check_shape, assert_finite


class _MHSA(nn.Module):
    """Multi-head self-attention (bidirectional; encoders are not causal).

    MATRIX-MULTIPLY NOTES (requested explicitly):
      * qkv: x[B,N,D] @ Wqkv[D,3D] -> [B,N,3D]
      * scores: q[B,h,N,d] @ k^T[B,h,d,N] -> [B,h,N,N], scaled by 1/sqrt(d)
      * context: softmax(scores)[B,h,N,N] @ v[B,h,N,d] -> [B,h,N,d]
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(t):  # [B,N,D] -> [B, heads, N, head_dim]
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,h,N,N]
        attn = attn.softmax(dim=-1)
        ctx = attn @ v                                  # [B,h,N,d]
        ctx = ctx.transpose(1, 2).contiguous().view(B, N, D)
        return self.proj(ctx)


class _Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)); x + mlp(norm(x))."""

    def __init__(self, dim: int, num_heads: int, mlp_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MHSA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_mult), nn.GELU(),
                                 nn.Linear(dim * mlp_mult, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FromScratchViT(nn.Module):
    def __init__(self, image_size=384, patch_size=16, embed_dim=384, depth=6, num_heads=6):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        # Strided conv == non-overlapping linear patchify.
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([_Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        check_shape(pixel_values, (None, 3, None, None), "vit_input")
        x = self.patch_embed(pixel_values)        # [B, D, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)          # [B, num_patches, D]
        check_shape(x, (None, self.num_patches, self.embed_dim), "vit_patches")
        x = x + self.pos_embed                    # broadcast add positions
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        assert_finite(x, "vit_output")
        return x

    def load_local_weights(self, path: str | Path) -> None:
        """Load a LOCAL state_dict (.pt). No network access."""
        path = Path(path)
        if not path.is_file():
            raise WeightsNotFoundError(f"Local ViT weights not found: {path}")
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)


class LocalSiglipEncoder(nn.Module):
    """Offline wrapper around a local SigLIP/ViT (transformers) vision tower."""

    def __init__(self, local_dir: str):
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as e:  # noqa: BLE001
            raise WeightsNotFoundError(
                "transformers required for local_siglip backend (`pip install -e '.[hf]'`)."
            ) from e
        import os
        if not local_dir or not os.path.isdir(local_dir):
            raise WeightsNotFoundError(f"Local SigLIP dir not found: '{local_dir}'")
        self.model = AutoModel.from_pretrained(local_dir, local_files_only=True)
        cfg = self.model.config
        self.embed_dim = getattr(cfg, "hidden_size", None) or cfg.vision_config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.model(pixel_values=pixel_values).last_hidden_state  # [B, N(+1), D]
        assert_finite(feats, "siglip_output")
        return feats


def build_vision_encoder(vision_cfg) -> nn.Module:
    """Factory selecting the vision backend from config."""
    if vision_cfg.backend == "local_siglip":
        enc = LocalSiglipEncoder(vision_cfg.local_dir)
    else:
        enc = FromScratchViT(image_size=vision_cfg.image_size, patch_size=vision_cfg.patch_size,
                             embed_dim=vision_cfg.embed_dim, depth=vision_cfg.depth,
                             num_heads=vision_cfg.num_heads)
    if vision_cfg.freeze:
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
    return enc
