"""Training loop implementing every requested behavior:

  * resume training (auto-loads runs_dir/last.ckpt when cfg.training.resume),
  * descriptive train & val loss (console + JSONL metrics stream),
  * step/epoch accounting and optimizer/LR logging,
  * "logging skip" cadence (log_every_steps / eval_every_steps),
  * records which epoch produced each checkpoint,
  * SAVE-BEST-ONLY: weights written only when val loss improves,
  * tqdm progress bars,
  * gradient clipping + finiteness checks as NaN/explosion guards.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import Config
from ..utils.checks import assert_finite
from ..utils.logging_utils import JsonlMetricLogger
from .optim import build_optimizer, build_scheduler
from .checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger("gemma_ft_json")


class Trainer:
    def __init__(self, cfg: Config, model: torch.nn.Module, train_loader: DataLoader,
                 val_loader: Optional[DataLoader], device: torch.device, run_dir: str | Path):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        steps_per_epoch = math.ceil(len(train_loader) / cfg.training.grad_accum_steps)
        self.total_optimizer_steps = steps_per_epoch * cfg.training.epochs
        if cfg.training.max_steps > 0:
            self.total_optimizer_steps = min(self.total_optimizer_steps, cfg.training.max_steps)

        self.optimizer = build_optimizer(self.model.parameters(), cfg.optim)
        self.scheduler = build_scheduler(self.optimizer, cfg.optim, self.total_optimizer_steps)

        self.epoch = 0
        self.global_step = 0          # counts OPTIMIZER steps
        self.best_val = float("inf")

        self.metrics = JsonlMetricLogger(self.run_dir / cfg.logging.metrics_filename)
        self.last_ckpt = self.run_dir / "last.ckpt"
        self.best_ckpt = self.run_dir / "best.ckpt"

        if cfg.training.resume and self.last_ckpt.is_file():
            state = load_checkpoint(self.last_ckpt, self.model, self.optimizer,
                                    self.scheduler, map_location=str(device))
            self.epoch, self.global_step, self.best_val = (
                state["epoch"], state["global_step"], state["best_val"])
            logger.info("Resumed from %s at epoch=%d step=%d best_val=%.4f",
                        self.last_ckpt, self.epoch, self.global_step, self.best_val)

        n_train = self.model.num_trainable() if hasattr(self.model, "num_trainable") else \
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info("Trainable params: %s / %s (%.3f%%)",
                    f"{n_train:,}", f"{n_total:,}", 100.0 * n_train / max(1, n_total))

    def _move(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _forward_loss(self, batch) -> torch.Tensor:
        return self.model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
                          attention_mask=batch["attention_mask"], labels=batch["labels"])["loss"]

    @torch.no_grad()
    def evaluate(self) -> Optional[float]:
        """Mean validation loss; None if no val set."""
        if self.val_loader is None:
            return None
        self.model.eval()
        total, count = 0.0, 0
        for batch in tqdm(self.val_loader, desc="val", leave=False):
            loss = self._forward_loss(self._move(batch))
            total += float(loss.item()); count += 1
        self.model.train()
        return total / max(1, count)

    def _maybe_save_best(self, val_loss: Optional[float]) -> None:
        """Save best.ckpt ONLY when val loss strictly improves (spec)."""
        if val_loss is None:
            return
        improved = val_loss < self.best_val
        self.metrics.log({"event": "val", "epoch": self.epoch, "step": self.global_step,
                          "val_loss": val_loss, "best_val": min(self.best_val, val_loss),
                          "improved": bool(improved)})
        logger.info("[val] epoch=%d step=%d val_loss=%.4f (best=%.4f)%s",
                    self.epoch, self.global_step, val_loss, min(self.best_val, val_loss),
                    "  <-- improved, saving best" if improved else "")
        if improved or not self.cfg.training.save_best_only:
            self.best_val = min(self.best_val, val_loss)
            save_checkpoint(self.best_ckpt, self.model, self.optimizer, self.scheduler,
                            epoch=self.epoch, global_step=self.global_step, best_val=self.best_val)

    def _save_last(self) -> None:
        save_checkpoint(self.last_ckpt, self.model, self.optimizer, self.scheduler,
                        epoch=self.epoch, global_step=self.global_step, best_val=self.best_val)

    def fit(self) -> None:
        cfg = self.cfg
        accum = cfg.training.grad_accum_steps
        self.model.train()
        logger.info("Starting training: %d epoch(s), ~%d optimizer steps.",
                    cfg.training.epochs, self.total_optimizer_steps)

        for epoch in range(self.epoch, cfg.training.epochs):
            self.epoch = epoch
            running, seen = 0.0, 0
            self.optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=True)

            for it, batch in enumerate(pbar):
                batch = self._move(batch)
                loss = self._forward_loss(batch)
                assert_finite(loss, "train_loss")          # NaN guard before backward
                # Scale by accum so the effective batch is bs*accum with correct averaging.
                (loss / accum).backward()
                running += float(loss.item()); seen += 1

                is_step = ((it + 1) % accum == 0) or (it + 1 == len(self.train_loader))
                if is_step:
                    # Grad clip = explosion guard; the returned norm is logged so
                    # spikes are visible before they become NaNs.
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        cfg.optim.grad_clip_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    if self.global_step % cfg.training.log_every_steps == 0:
                        avg = running / max(1, seen)
                        lr = self.scheduler.get_last_lr()[0]
                        self.metrics.log({"event": "train_step", "epoch": epoch,
                                          "step": self.global_step, "loss": avg, "lr": lr,
                                          "grad_norm": float(grad_norm), "time": time.time()})
                        pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}",
                                         gnorm=f"{float(grad_norm):.2f}")
                        running, seen = 0.0, 0

                    if cfg.training.eval_every_steps and \
                       self.global_step % cfg.training.eval_every_steps == 0:
                        self._maybe_save_best(self.evaluate())

                    if cfg.training.max_steps > 0 and self.global_step >= cfg.training.max_steps:
                        logger.info("Reached max_steps=%d; stopping.", cfg.training.max_steps)
                        self._save_last()
                        self.metrics.close()
                        return

            if not cfg.training.eval_every_steps:
                self._maybe_save_best(self.evaluate())
            self._save_last()
            logger.info("Epoch %d complete. last.ckpt saved.", epoch)

        self.metrics.close()
        logger.info("Training finished. best_val=%.4f", self.best_val)
