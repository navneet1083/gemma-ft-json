"""Decoder backends + LoRA.

The decoder is Gemma 3 270M acting as the autoregressive text generator. Two
interchangeable backends behind a common (duck-typed) interface:

  * `LocalGemmaDecoder` — loads Gemma FROM A LOCAL DIRECTORY, fully offline.
  * `TinyStubDecoder`   — a small pure-PyTorch causal LM, no external weights, so
    the entire pipeline can be smoke-tested with ZERO downloads.

Interface used by the VLM:
    .hidden_size : int
    .vocab_size  : int
    .embed_tokens(input_ids[B,T]) -> embeds[B,T,H]
    .forward(inputs_embeds=[B,T,H], attention_mask=[B,T]) -> logits[B,T,V]

`inject_lora` implements LoRA WITHOUT the `peft` dependency, identically on both
backends. LoRA is our primary defense against catastrophic forgetting: base
weights stay frozen; only the small low-rank updates train.
"""
from __future__ import annotations

import math
import os
from typing import List

import torch
import torch.nn as nn

from ..exceptions import WeightsNotFoundError
from ..utils.checks import assert_finite


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around a frozen nn.Linear.

    out = base(x) + scaling * (dropout(x) @ A^T @ B^T)

    MATRIX-MULTIPLY NOTE: base weight W[out,in] is frozen. We learn A[r,in] and
    B[out,r] (r << in,out):  x[..,in] @ A^T[in,r] -> [..,r]  then  @ B^T[r,out]
    -> [..,out]. Trainable params drop from out*in to r*(in+out).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)  # freeze original projection
        in_f, out_f = base.in_features, base.out_features
        self.scaling = alpha / rank
        self.drop = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.zeros(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # A small, B=0 -> starts as no-op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * ((self.drop(x) @ self.A.t()) @ self.B.t())


def inject_lora(module: nn.Module, target_names: List[str], rank: int,
                alpha: int, dropout: float) -> int:
    """Recursively replace nn.Linear children whose attribute name is in
    `target_names` with LoRALinear. Returns count adapted. Works for the stub
    (q_proj/v_proj) and HF Gemma alike."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            n += 1
        else:
            n += inject_lora(child, target_names, rank, alpha, dropout)
    return n


# --------------------------------------------------------------------------- #
# Stub decoder (pure PyTorch, no weights)
# --------------------------------------------------------------------------- #
class _CausalSelfAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.h, self.d = num_heads, dim // num_heads
        self.scale = 1.0 / math.sqrt(self.d)
        # Named q_proj/k_proj/v_proj/o_proj so LoRA targeting finds q_proj/v_proj.
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale  # [B,h,T,T]
        scores = scores + attn_mask                       # additive causal+pad bias
        probs = scores.softmax(dim=-1)
        ctx = (probs @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(ctx)


class _DecBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _CausalSelfAttn(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, attn_mask):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyStubDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 640, depth: int = 4,
                 num_heads: int = 8, max_seq_len: int = 2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._embed = nn.Embedding(vocab_size, hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_DecBlock(hidden_size, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._embed(input_ids)

    def _build_mask(self, attention_mask: torch.Tensor, T: int) -> torch.Tensor:
        """Combine causal + padding masks into an additive bias [B,1,T,T]: 0 where
        allowed, large-negative where forbidden. We use finfo.min (not -inf) to
        avoid NaNs from softmax over an all-masked row."""
        device = attention_mask.device
        causal = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
        pad = attention_mask.view(attention_mask.size(0), 1, 1, T)  # 1=keep,0=pad
        allowed = causal * pad
        return (1.0 - allowed) * torch.finfo(torch.float32).min

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = inputs_embeds.shape
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        x = inputs_embeds + self.pos[:, :T, :]
        mask = self._build_mask(attention_mask, T)
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.lm_head(self.norm(x))   # [B,T,V] (matmul hidden @ W^T)
        assert_finite(logits, "stub_logits")
        return logits


# --------------------------------------------------------------------------- #
# Local offline Gemma decoder
# --------------------------------------------------------------------------- #
class LocalGemmaDecoder(nn.Module):
    """Loads Gemma from a LOCAL directory, offline. Never touches the network."""

    def __init__(self, local_dir: str, new_vocab_size: int | None = None):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM
        except Exception as e:  # noqa: BLE001
            raise WeightsNotFoundError(
                "transformers required for local_gemma backend (`pip install -e '.[hf]'`)."
            ) from e
        if not local_dir or not os.path.isdir(local_dir):
            raise WeightsNotFoundError(
                f"Local Gemma dir not found: '{local_dir}'. No download attempted (offline)."
            )
        # float32 for MPS stability; local_files_only forbids network.
        self.model = AutoModelForCausalLM.from_pretrained(
            local_dir, local_files_only=True, torch_dtype=torch.float32)
        if new_vocab_size is not None and new_vocab_size != self.model.config.vocab_size:
            self.model.resize_token_embeddings(new_vocab_size)  # grow for added tokens
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                         use_cache=False)
        assert_finite(out.logits, "gemma_logits")
        return out.logits


def build_decoder(decoder_cfg, tokenizer) -> nn.Module:
    """Factory selecting the decoder backend. `tokenizer.vocab_size` reflects any
    added structure/loc tokens, keeping embeddings and the LM head correctly sized."""
    if decoder_cfg.backend == "local_gemma":
        return LocalGemmaDecoder(decoder_cfg.local_dir, new_vocab_size=tokenizer.vocab_size)
    return TinyStubDecoder(vocab_size=tokenizer.vocab_size,
                           hidden_size=decoder_cfg.hidden_size,
                           max_seq_len=decoder_cfg.max_seq_len)
