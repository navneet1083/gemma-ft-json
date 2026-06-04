#!/usr/bin/env python
"""Train from a YAML config.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --set training.epochs=3 dataloader.batch_size=4

(Equivalent console script after `pip install -e .`:  gemma-ft-train --config ...)
"""
import sys
from pathlib import Path

# Allow running without installation: add src/ to the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemma_ft_json.scripts_entry import train_cli  # noqa: E402

if __name__ == "__main__":
    train_cli()
