"""Device selection. Priority MPS (Apple Silicon) > CUDA > CPU, matching M4."""
from __future__ import annotations
import torch


def get_device(preference: str = "auto") -> torch.device:
    """Return a torch.device honoring an explicit preference or auto-selecting."""
    pref = (preference or "auto").lower()

    def _mps_ok() -> bool:
        # is_built() guards wheels compiled without MPS support.
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()

    if pref == "mps":
        return torch.device("mps") if _mps_ok() else torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if pref == "cpu":
        return torch.device("cpu")
    if _mps_ok():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "mps":
        return "Apple MPS (Metal) — unified memory"
    if device.type == "cuda":
        return f"CUDA: {torch.cuda.get_device_name(device.index or 0)}"
    return "CPU"
