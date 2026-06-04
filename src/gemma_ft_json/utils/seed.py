"""Reproducibility helpers."""
from __future__ import annotations
import os, random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch (incl. CUDA/MPS) for reproducible runs.

    Full determinism on MPS/GPU is not always achievable per-op, but this removes
    the dominant sources of run-to-run variance.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
