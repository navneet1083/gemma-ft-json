"""Strictly-offline loader for a LOCAL Gemma 3 270M directory.

`transformers` is used purely as a *file format reader* for weights already
on disk. enforce_offline() is called BEFORE importing transformers, so even
a mis-typed path can never trigger a Hub download — it fails loudly instead.
"""
from __future__ import annotations

from pathlib import Path

from ..exceptions import ModelLoadError
from ..utils.device import enforce_offline

enforce_offline()  # must precede the transformers import below

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def load_gemma_local(model_dir: str, dtype: torch.dtype = torch.float32):
    """Return (model, tokenizer) from a local folder, or raise ModelLoadError
    with an actionable message listing offline acquisition channels."""
    p = Path(model_dir).expanduser()
    if not p.is_dir() or not (p / "config.json").exists():
        raise ModelLoadError(
            f"Gemma model directory invalid: {p}\n"
            "Expected config.json + tokenizer files + model.safetensors.\n"
            "Offline acquisition options (NO Hugging Face Hub):\n"
            "  1) Kaggle Models: kagglehub.model_download('google/gemma-3/"
            "transformers/gemma-3-270m')\n"
            "  2) Direct download from ai.google.dev (Gemma terms page)\n"
            "  3) Copy the folder from another machine\n"
            "Then set paths.gemma_model_dir in configs/config.yaml.")
    try:
        tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(p), local_files_only=True, torch_dtype=dtype,
            attn_implementation="eager",   # SDPA on MPS is fine, but eager is
        )                                  # the most numerically conservative
    except Exception as exc:
        raise ModelLoadError(f"Failed loading local Gemma from {p}: {exc}") from exc
    if tok.pad_token_id is None:           # Gemma defines <pad>; belt & braces
        tok.pad_token = tok.eos_token
    return model, tok
