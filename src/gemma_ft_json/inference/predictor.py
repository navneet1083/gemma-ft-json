"""Inference / prediction.

Rebuilds the exact model from a saved config snapshot, loads the best checkpoint
(trainable params only — the frozen base reloads from its local dir), and turns a
single image into a JSON string. Used by notebook 06 and the FastAPI service.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from ..config import Config, load_config
from ..data.transforms import build_image_transform
from ..models.vlm import build_model
from ..training.checkpoint import load_checkpoint
from ..utils.device import get_device, describe_device


class Predictor:
    def __init__(self, config_snapshot: str | Path, checkpoint_path: str | Path,
                 device: Optional[str] = None, max_new_tokens: int = 1024):
        self.cfg: Config = load_config(config_snapshot)
        self.device = get_device(device or self.cfg.project.device)
        self.max_new_tokens = max_new_tokens

        # Rebuild model + tokenizer identically to training, then load weights.
        self.model, self.tokenizer = build_model(self.cfg)
        load_checkpoint(checkpoint_path, self.model, map_location=str(self.device))
        self.model.to(self.device).eval()

        self.transform = build_image_transform(self.cfg.model.vision)
        self._prompt_ids = self.tokenizer.encode(self.cfg.data.prompt, add_bos=True, add_eos=False)

    def describe(self) -> str:
        return f"Predictor on {describe_device(self.device)} | prompt={self.cfg.data.prompt!r}"

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> str:
        """Image path -> predicted JSON string (greedy decoding)."""
        with Image.open(image_path) as im:
            pixel_values = self.transform(im).unsqueeze(0).to(self.device)  # [1,3,S,S]
        prompt_ids = torch.tensor(self._prompt_ids, dtype=torch.long, device=self.device)
        out_ids = self.model.generate(pixel_values, prompt_ids,
                                      max_new_tokens=self.max_new_tokens,
                                      eos_id=self.tokenizer.eos_id)
        return self.tokenizer.decode(out_ids)
