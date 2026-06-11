"""Dual-channel logging.

1. `get_logger`     -> human-readable .log file + console (tqdm-safe).
2. `MetricsWriter`  -> append-only JSONL of every metric event. The plotting
   notebook *tails this file while training is still running*, which is what
   enables live loss curves without TensorBoard.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict


def get_logger(name: str, log_file: str | None = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:          # idempotent: safe to call from notebooks repeatedly
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class MetricsWriter:
    """Append one JSON object per line; flushed immediately so a second
    process (the plotting notebook) can read it mid-training."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        rec: Dict[str, Any] = {"event": event, "ts": time.time(), **fields}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
