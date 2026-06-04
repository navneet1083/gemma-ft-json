"""Checkpointing.

Saves a single dict with model/optimizer/scheduler state + run bookkeeping
(epoch, global_step, best_val). Writes are ATOMIC (temp file then os.replace) so an
interrupted save can never corrupt an existing checkpoint.

By default we save only TRAINABLE tensors (`save_trainable_only=True`): for the
offline Gemma the multi-GB base is reloaded from its local dir, so we avoid
duplicating it; only the projector + LoRA adapters are checkpointed.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..exceptions import CheckpointError


def _trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    train_names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in train_names}


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Any, epoch: int, global_step: int, best_val: float,
                    save_trainable_only: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model_state = _trainable_state_dict(model) if save_trainable_only else model.state_dict()
        payload = {
            "model": model_state,
            "trainable_only": save_trainable_only,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch, "global_step": global_step, "best_val": best_val,
        }
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic on POSIX
    except OSError as e:
        raise CheckpointError(f"Failed to save checkpoint {path}: {e}") from e


def load_checkpoint(path: str | Path, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Any = None,
                    map_location: str = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location)
    # strict=False: we may have saved trainable params only.
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return {"epoch": payload.get("epoch", 0), "global_step": payload.get("global_step", 0),
            "best_val": payload.get("best_val", float("inf")),
            "missing": missing, "unexpected": unexpected}
