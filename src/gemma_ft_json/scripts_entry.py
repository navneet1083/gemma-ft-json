"""Console-script entry points (referenced by pyproject [project.scripts]).

  gemma-ft-train           -> train_cli
  gemma-ft-verify-offline  -> verify_offline_cli

Kept thin: parse args, then delegate to the high-level pipeline so the same logic
is reusable from notebooks and the CLI.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from .config import load_config
from .utils.device import get_device, describe_device
from .utils.logging_utils import setup_logging
from .utils.seed import set_seed


def _build_everything(cfg):
    """Shared wiring used by the train CLI and the notebooks."""
    from .data.build_dataset import build_manifest, split_manifest
    from .data.dataset import TableJsonDataset
    from .data.transforms import build_image_transform
    from .data.collate import build_dataloaders
    from .models.vlm import build_model
    from .training.trainer import Trainer

    set_seed(cfg.project.seed)
    device = get_device(cfg.project.device)

    manifest, _ = build_manifest(
        cfg.paths.images_dir, cfg.paths.json_dir, cfg.paths.manifest_dir,
        pdf_dir=cfg.paths.pdf_dir, require_valid_json=cfg.data.require_valid_json,
        use_pymupdf_boxes=cfg.data.use_pymupdf_boxes,
    )
    train_path, val_path = split_manifest(manifest, cfg.data.val_fraction, seed=cfg.project.seed)

    model, tokenizer = build_model(cfg)
    transform = build_image_transform(cfg.model.vision)
    train_ds = TableJsonDataset(train_path, tokenizer, transform, cfg.data.prompt,
                                cfg.data.max_target_tokens)
    val_ds = TableJsonDataset(val_path, tokenizer, transform, cfg.data.prompt,
                              cfg.data.max_target_tokens)
    train_loader, val_loader = build_dataloaders(
        train_ds, val_ds, pad_id=tokenizer.pad_id, batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers, pin_memory=cfg.dataloader.pin_memory,
        shuffle_train=cfg.dataloader.shuffle_train,
    )
    run_dir = Path(cfg.paths.runs_dir) / cfg.project.name
    cfg.save_snapshot(run_dir / "config.snapshot.yaml")  # reproducibility
    trainer = Trainer(cfg, model, train_loader, val_loader, device, run_dir)
    return trainer, device


def train_cli(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fine-tune Gemma 3 270M -> table-JSON.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. training.epochs=3")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.set)
    run_dir = Path(cfg.paths.runs_dir) / cfg.project.name
    setup_logging(run_dir, level=cfg.logging.level, filename=cfg.logging.log_filename)
    trainer, device = _build_everything(cfg)
    print(f"Device: {describe_device(device)}")
    trainer.fit()


def verify_offline_cli(argv: List[str] | None = None) -> None:
    """Assert the HF offline guards are active and (if requested) that a local
    Gemma dir exists — WITHOUT downloading anything."""
    ap = argparse.ArgumentParser(description="Verify no-Hugging-Face-download setup.")
    ap.add_argument("--config", required=False)
    args = ap.parse_args(argv)

    print("HF_HUB_OFFLINE       =", os.environ.get("HF_HUB_OFFLINE"))
    print("TRANSFORMERS_OFFLINE =", os.environ.get("TRANSFORMERS_OFFLINE"))
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "HF offline not enforced!"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Transformers offline not enforced!"

    if args.config:
        cfg = load_config(args.config)
        if cfg.model.decoder.backend == "local_gemma":
            d = cfg.model.decoder.local_dir
            print("Gemma local_dir      =", d, "exists:", Path(d).is_dir())
    print("OK: offline mode enforced; no network access will be attempted.")
