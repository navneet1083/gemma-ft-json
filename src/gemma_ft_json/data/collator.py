"""Batch collation: dynamic right-padding + weight-tag resolution.

Dynamic padding (pad to the longest sequence IN THE BATCH, not to a global
max) matters on MPS: attention cost is quadratic in sequence length, and
table targets vary 3-4x in length.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from ..exceptions import ShapeError


class VLMCollator:
    def __init__(self, pad_token_id: int,
                 structure_weight: float = 0.6, content_weight: float = 1.4):
        self.pad_id = pad_token_id
        self.w_struct = structure_weight
        self.w_content = content_weight

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ShapeError("Empty batch passed to collator")
        max_len = max(b["input_ids"].numel() for b in batch)
        B = len(batch)

        input_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)   # pad never scored
        attn = torch.zeros((B, max_len), dtype=torch.long)
        weights = torch.zeros((B, max_len), dtype=torch.float32)

        for i, b in enumerate(batch):
            n = b["input_ids"].numel()
            input_ids[i, :n] = b["input_ids"]
            labels[i, :n] = b["labels"]
            attn[i, :n] = 1
            # Resolve tags -> real weights (see dataset.py for tag meaning).
            tags = b["weight_tags"]
            w = torch.ones(n)
            w[tags == -1.0] = self.w_struct
            w[tags == -2.0] = self.w_content
            weights[i, :n] = w

        pixel = torch.stack([b["pixel_values"] for b in batch])      # [B,3,H,W]
        if pixel.dim() != 4 or pixel.shape[1] != 3:
            raise ShapeError(f"pixel_values must be [B,3,H,W], got {tuple(pixel.shape)}")

        return {"pixel_values": pixel, "input_ids": input_ids,
                "attention_mask": attn, "labels": labels, "loss_weights": weights}
