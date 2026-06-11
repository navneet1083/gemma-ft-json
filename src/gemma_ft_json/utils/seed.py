"""Reproducibility helper."""
import random

import numpy as np
import torch


def set_seed(seed: int = 1337) -> None:
    """Seed python / numpy / torch (incl. MPS) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
