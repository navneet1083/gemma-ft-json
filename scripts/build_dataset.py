#!/usr/bin/env python3
"""CLI twin of notebook 01: build the synthetic table corpus from config."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from gemma_ft_json.config import load_config
from gemma_ft_json.data import build_dataset
from gemma_ft_json.utils import get_logger, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.dataset.seed)
    log = get_logger("build_dataset", cfg.paths.log_file)

    for split, n, manifest, off in (
        ("train", cfg.dataset.num_train_samples, cfg.paths.manifest_train, 0),
        ("val", cfg.dataset.num_val_samples, cfg.paths.manifest_val, 10_000),
    ):
        bar = tqdm(total=n, desc=f"render {split}")
        wrote = build_dataset(cfg.dataset, cfg.paths.raw_images_dir, manifest,
                              n, split, seed_offset=off,
                              progress_cb=lambda i, t: bar.update(1))
        bar.close()
        log.info(f"{split}: wrote {wrote} samples -> {manifest}")


if __name__ == "__main__":
    main()
