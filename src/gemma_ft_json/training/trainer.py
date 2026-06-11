"""Trainer: epoch loop, grad accumulation, NaN skip-policy, resume, tqdm,
dual logging (human .log + machine metrics.jsonl), best-only checkpointing.

NaN/Inf policy (3 lines of defense, in order):
  1. Forward guards sanitize stray non-finite *activations* (guards.py).
  2. A non-finite *loss* raises NumericalError -> the step is SKIPPED, the
     skip is logged & counted, optimizer state stays clean.
  3. Gradients are clipped to `grad_clip_norm`; if the pre-clip norm itself
     is non-finite, the accumulated grads are zeroed and the step skipped.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config.loader import AppConfig
from ..exceptions import NumericalError
from ..utils.logging_utils import MetricsWriter, get_logger
from .checkpoint import CheckpointManager
from .losses import weighted_causal_lm_loss


def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """LR = linear warmup then cosine decay to 0. Warmup is non-negotiable
    here: the projector is random-init and its first gradients into a frozen
    LM are large; warmup is the cheapest NaN insurance there is."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return LambdaLR(optimizer, fn)


class Trainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader,
                 cfg: AppConfig, device: torch.device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        t = cfg.training

        self.logger = get_logger("trainer", cfg.paths.log_file)
        self.metrics = MetricsWriter(cfg.paths.metrics_file)

        self.optimizer = AdamW(
            model.param_groups(t.lr_projector, t.lr_lora),
            weight_decay=t.weight_decay, betas=(0.9, 0.98), eps=1e-6)

        steps_per_epoch = math.ceil(len(train_loader) / t.grad_accum_steps)
        total = steps_per_epoch * t.epochs
        self.scheduler = _cosine_with_warmup(
            self.optimizer, int(total * t.warmup_ratio), total)

        self.ckpt = CheckpointManager(cfg.paths.checkpoints_dir, t.save_best_only)
        self.start_epoch, self.global_step = 0, 0
        self.skipped_steps = 0

        if t.resume_from:
            state = self.ckpt.resume(t.resume_from, self.model,
                                     self.optimizer, self.scheduler)
            self.start_epoch = state["epoch"] + 1
            self.global_step = state["global_step"]
            self.logger.info(
                f"RESUMED from {t.resume_from}: next epoch {self.start_epoch}, "
                f"global_step {self.global_step}, best_val {state['best_val']:.4f}")

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"Trainable params: {n_train/1e6:.2f}M / {n_total/1e6:.1f}M total "
            f"({100*n_train/n_total:.2f}%) | device={device} | stage={t.stage}")
        self.metrics.write("run_config", stage=t.stage, epochs=t.epochs,
                           batch_size=t.batch_size, grad_accum=t.grad_accum_steps,
                           trainable_params=n_train)

    # ------------------------------------------------------------------ #
    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _step_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model(pixel_values=batch["pixel_values"],
                         input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"],
                         labels=batch["labels"])
        # loss_weights are aligned to the TEXT sequence; pad a neutral block
        # over the visual prefix exactly the way the model padded labels.
        w = batch["loss_weights"]
        B = w.shape[0]
        vis_pad = torch.ones(B, self.model.num_visual_tokens, device=w.device)
        w_fused = torch.cat([w[:, :1], vis_pad, w[:, 1:]], dim=1)
        loss, _ = weighted_causal_lm_loss(
            out["logits"], out["labels"], w_fused,
            self.cfg.training.label_smoothing)
        return loss

    # ------------------------------------------------------------------ #
    def train(self) -> Dict[str, float]:
        t = self.cfg.training
        for epoch in range(self.start_epoch, t.epochs):
            self.model.train()
            epoch_loss, epoch_tokens, n_loss = 0.0, 0, 0
            self.optimizer.zero_grad(set_to_none=True)

            bar = tqdm(self.train_loader,
                       desc=f"epoch {epoch+1}/{t.epochs} [{t.stage}]",
                       dynamic_ncols=True, leave=True)
            t0 = time.time()
            for i, batch in enumerate(bar):
                batch = self._to_device(batch)
                try:
                    loss = self._step_loss(batch) / t.grad_accum_steps
                    loss.backward()
                except NumericalError as exc:
                    # SKIP policy: log, count, drop grads, continue training.
                    self.skipped_steps += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    self.logger.warning(
                        f"SKIPPED micro-step e{epoch+1} i{i}: {exc} "
                        f"(total skips: {self.skipped_steps})")
                    self.metrics.write("skip", epoch=epoch + 1, micro_step=i,
                                       reason=str(exc), total_skips=self.skipped_steps)
                    continue

                if (i + 1) % t.grad_accum_steps == 0 or (i + 1) == len(self.train_loader):
                    # ---- gradient hygiene before the optimizer touches state ----
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        self.model.trainable_parameters(), t.grad_clip_norm)
                    if not torch.isfinite(gnorm):
                        self.skipped_steps += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        self.logger.warning(
                            f"SKIPPED optimizer step e{epoch+1}: non-finite grad norm "
                            f"(total skips: {self.skipped_steps})")
                        self.metrics.write("skip", epoch=epoch + 1, micro_step=i,
                                           reason="non-finite grad norm",
                                           total_skips=self.skipped_steps)
                        continue
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    step_loss = loss.item() * t.grad_accum_steps
                    epoch_loss += step_loss
                    n_loss += 1
                    lr = self.scheduler.get_last_lr()[0]
                    bar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{lr:.2e}",
                                    gnorm=f"{float(gnorm):.2f}",
                                    skips=self.skipped_steps)
                    if self.global_step % t.log_every_steps == 0:
                        self.logger.info(
                            f"e{epoch+1} step {self.global_step} | "
                            f"loss {step_loss:.4f} | lr {lr:.2e} | "
                            f"gnorm {float(gnorm):.2f}")
                        self.metrics.write("train_step", epoch=epoch + 1,
                                           step=self.global_step, loss=step_loss,
                                           lr=lr, grad_norm=float(gnorm))

            train_avg = epoch_loss / max(1, n_loss)
            val_loss = None
            if (epoch + 1) % t.val_every_epochs == 0:
                val_loss = self.evaluate(epoch)

            saved = self.ckpt.save_epoch(self.model, self.optimizer,
                                         self.scheduler, epoch,
                                         self.global_step, val_loss)
            dur = time.time() - t0
            self.logger.info(
                f"EPOCH {epoch+1} done in {dur/60:.1f} min | train {train_avg:.4f} | "
                f"val {val_loss if val_loss is None else f'{val_loss:.4f}'} | "
                f"best {self.ckpt.best_val:.4f} | "
                f"{'>> new best.pt <<' if saved['saved_best'] else 'no improvement, best.pt unchanged'} | "
                f"skipped {self.skipped_steps}")
            self.metrics.write("epoch", epoch=epoch + 1, train_loss=train_avg,
                               val_loss=val_loss, best_val=self.ckpt.best_val,
                               saved_best=saved["saved_best"],
                               duration_sec=dur, skipped=self.skipped_steps)
        return {"best_val": self.ckpt.best_val, "skipped": self.skipped_steps}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, epoch: int) -> float:
        self.model.eval()
        tot, n = 0.0, 0
        for batch in tqdm(self.val_loader, desc="validating",
                          dynamic_ncols=True, leave=False):
            batch = self._to_device(batch)
            try:
                tot += self._step_loss(batch).item()
                n += 1
            except NumericalError:
                continue                       # one bad val batch never aborts eval
        val = tot / max(1, n)
        self.logger.info(f"VAL e{epoch+1}: loss {val:.4f} over {n} batches")
        self.metrics.write("val", epoch=epoch + 1, val_loss=val, batches=n)
        self.model.train()
        return val
