#!/usr/bin/env python3
"""CLI twin of notebook 04: end-to-end training driven by configs/config.yaml.

Typical curriculum (edit training.stage in YAML between runs):
    1. stage=align + dataset stage "read"/"linearize"  (projector warm-up)
    2. stage=sft   (LoRA + projector, JSON targets)    <- the real task
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader

from gemma_ft_json.config import load_config
from gemma_ft_json.data import TableImageJsonDataset, VLMCollator
from gemma_ft_json.models import GemmaVisionForJSON, load_gemma_local
from gemma_ft_json.training import Trainer
from gemma_ft_json.utils import (get_logger, resolve_device, resolve_dtype,
                                 set_seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--resume", default=None,
                    help="checkpoint path; overrides training.resume_from")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg.training.resume_from = args.resume
    set_seed(cfg.training.seed)
    log = get_logger("train", cfg.paths.log_file)

    device = resolve_device(cfg.device.preferred, cfg.device.mps_fallback_env)
    dtype = resolve_dtype(cfg.device.dtype)
    log.info(f"device={device} dtype={dtype}")

    gemma, tok = load_gemma_local(cfg.paths.gemma_model_dir, dtype)
    model = GemmaVisionForJSON(gemma, tok, cfg.model,
                               image_size=cfg.dataset.image_size,
                               stage=cfg.training.stage)

    stage = cfg.training.stage if cfg.training.stage != "align" else "linearize"
    mk = lambda manifest, shuffle: DataLoader(  # noqa: E731
        TableImageJsonDataset(manifest, tok, cfg.dataset.image_size,
                              stage=stage, max_seq_len=cfg.model.max_seq_len),
        batch_size=cfg.training.batch_size, shuffle=shuffle,
        num_workers=cfg.training.num_workers,
        collate_fn=VLMCollator(tok.pad_token_id,
                               cfg.training.structure_loss_weight,
                               cfg.training.content_loss_weight),
        pin_memory=False,  # pin_memory is a no-op/warning source on MPS
        persistent_workers=cfg.training.num_workers > 0)

    trainer = Trainer(model, mk(cfg.paths.manifest_train, True),
                      mk(cfg.paths.manifest_val, False), cfg, device)
    result = trainer.train()
    log.info(f"DONE: best_val={result['best_val']:.4f}, "
             f"skipped_steps={result['skipped']}")


if __name__ == "__main__":
    main()
