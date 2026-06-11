"""Device + offline-mode helpers (MPS-first, per project requirement)."""
from __future__ import annotations

import os

import torch


def enforce_offline() -> None:
    """HARD guarantee: no Hugging Face Hub traffic, ever.

    These env vars make `transformers` refuse any network call; loading is
    only possible from a *local* directory. Called at import time by the
    model loader, before transformers is imported.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def resolve_device(preferred: str = "auto", mps_fallback_env: bool = True) -> torch.device:
    """Pick the compute device. On a Mac M4 this resolves to MPS.

    `PYTORCH_ENABLE_MPS_FALLBACK=1` lets the few ops MPS does not implement
    silently fall back to CPU instead of crashing mid-epoch.
    """
    if mps_fallback_env:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    """Map config string -> torch dtype. fp16 is intentionally not offered:
    MPS has no GradScaler, so fp16 training NaNs almost immediately."""
    table = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if name not in table:
        raise ValueError(f"dtype must be one of {list(table)}, got {name!r}")
    return table[name]
