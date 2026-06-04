"""Optimizer and scheduler factories.

Only TRAINABLE parameters (projector + LoRA adapters; encoder and base decoder are
frozen) reach the optimizer, so optimizer state stays tiny — important on MPS.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(params: Iterable[torch.nn.Parameter], optim_cfg) -> Optimizer:
    params = [p for p in params if p.requires_grad]
    if not params:
        raise ValueError(
            "No trainable parameters. Check that LoRA injected layers or that the "
            "projector is unfrozen."
        )
    if optim_cfg.name.lower() != "adamw":
        raise ValueError(f"Unsupported optimizer: {optim_cfg.name}")
    return AdamW(params, lr=optim_cfg.lr, betas=tuple(optim_cfg.betas),
                 eps=optim_cfg.eps, weight_decay=optim_cfg.weight_decay)


def build_scheduler(optimizer: Optimizer, optim_cfg, total_steps: int) -> LambdaLR:
    """Linear warmup then cosine decay (or constant after warmup). The lambda
    returns a MULTIPLIER on the base LR: rises 0->1 during warmup, then half-cosine
    to 0 (cosine) or stays 1 (constant)."""
    warmup = max(0, int(optim_cfg.warmup_steps))
    total = max(1, total_steps)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return step / warmup
        if optim_cfg.scheduler == "constant":
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
