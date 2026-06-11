"""Checkpointing with best-only retention + full resume support.

Files written under `checkpoints_dir`:
    last.pt  — every epoch (enables resume after crash/interrupt)
    best.pt  — ONLY when val loss improves (project requirement: "storing
               model weights only if it is learning / better than previous")
Each file contains trainable weights + optimizer + scheduler + RNG + counters.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..exceptions import CheckpointError


class CheckpointManager:
    def __init__(self, ckpt_dir: str, save_best_only: bool = True):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_val = float("inf")

    def _payload(self, model, optimizer, scheduler, epoch: int,
                 global_step: int) -> Dict[str, Any]:
        return {
            "model": model.trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_val": self.best_val,
            "rng": {"py": random.getstate(), "np": np.random.get_state(),
                    "torch": torch.get_rng_state()},
        }

    def save_epoch(self, model, optimizer, scheduler, epoch: int,
                   global_step: int, val_loss: Optional[float]) -> Dict[str, bool]:
        """Always refresh last.pt; refresh best.pt iff val improved."""
        improved = val_loss is not None and val_loss < self.best_val
        if improved:
            self.best_val = float(val_loss)
        try:
            payload = self._payload(model, optimizer, scheduler, epoch, global_step)
            torch.save(payload, self.dir / "last.pt")
            if improved or not self.save_best_only:
                torch.save(payload, self.dir / "best.pt")
        except Exception as exc:
            raise CheckpointError(f"Saving checkpoint failed: {exc}") from exc
        return {"saved_best": improved}

    def resume(self, path: str, model, optimizer=None, scheduler=None) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise CheckpointError(f"Resume checkpoint not found: {p}")
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            model.load_trainable_state_dict(ck["model"])
            if optimizer is not None and ck.get("optimizer"):
                optimizer.load_state_dict(ck["optimizer"])
            if scheduler is not None and ck.get("scheduler"):
                scheduler.load_state_dict(ck["scheduler"])
            self.best_val = ck.get("best_val", float("inf"))
            rng = ck.get("rng")
            if rng:
                random.setstate(rng["py"]); np.random.set_state(rng["np"])
                torch.set_rng_state(rng["torch"].cpu())
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"Resume from {p} failed: {exc}") from exc
        return {"epoch": ck["epoch"], "global_step": ck["global_step"],
                "best_val": self.best_val}
