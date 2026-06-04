"""gemma_ft_json
================
Modular, offline, MPS-friendly fine-tuning of **Gemma 3 270M** (text-only) as the
decoder of a small Vision-Language Model that reads table/document images and
emits **structured JSON**.

Design goals
------------
* Plug-and-play modules (encoder / projector / decoder / tokenizer / loss) so the
  Gemma decoder fuses easily with other models.
* Config-driven (YAML); no dataset path or hyper-parameter is hard-coded.
* Runs end-to-end with **zero network access** via pure-PyTorch fallbacks, while
  still supporting real Gemma/SigLIP weights loaded *locally*.

Hugging Face restriction
------------------------
This package NEVER downloads weights from the HF Hub. We force offline mode HERE,
at first import, BEFORE `transformers`/`huggingface_hub` are imported anywhere —
so any accidental Hub fetch raises locally instead of hitting the network.
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported MPS ops -> CPU

__version__ = "0.1.0"
__all__ = ["__version__"]
