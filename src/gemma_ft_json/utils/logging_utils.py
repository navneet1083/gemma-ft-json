"""Logging utilities.

Two complementary mechanisms:

1. `setup_logging` — standard python logging to BOTH console and a `train.log`
   file (human-readable progress / warnings / exceptions).

2. `JsonlMetricLogger` — appends one JSON object per logged step to
   `metrics.jsonl` and flushes immediately. This machine-readable stream is what
   notebook 05 reads to plot train/val curves *while training is still running*
   — append-only and safe to tail/re-read at any time.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict


def setup_logging(log_dir: str | Path, level: str = "INFO",
                  filename: str = "train.log") -> logging.Logger:
    """Configure logging to console + file. Idempotent per process (clears old
    handlers so notebook re-runs don't duplicate lines)."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gemma_ft_json")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_dir / filename)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


class JsonlMetricLogger:
    """Append-only JSONL metric writer (one flat dict per line)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)  # line-buffered append

    def log(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()  # explicit flush -> live plotting during training

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # closing must never crash a run
            pass
