"""Inference: image -> JSON, with a bracket-balance JSON guard.

Consistency engineering for small models: a 270M decoder occasionally
truncates or unbalances brackets. `repair_json` deterministically closes
open strings/brackets so downstream consumers always receive parseable
JSON, plus a `valid` flag telling you whether repair was needed.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

import torch
from PIL import Image

from ..config.loader import AppConfig
from ..data.dataset import PROMPT_BY_STAGE, image_to_tensor
from ..models.vlm import GemmaVisionForJSON


def repair_json(text: str) -> Tuple[Optional[dict], bool]:
    """Try strict parse; else trim trailing garbage and close open
    quotes/brackets in stack order. Returns (obj_or_None, was_strictly_valid)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass
    stack, in_str, esc = [], False, False
    for ch in text:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str and ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif not in_str and ch in "}]":
            if stack:
                stack.pop()
    fixed = text + ('"' if in_str else "") + "".join(reversed(stack))
    fixed = fixed.rstrip(",")
    try:
        return json.loads(fixed), False
    except json.JSONDecodeError:
        return None, False


class Predictor:
    """Loads best.pt and exposes `predict(image) -> dict`. Used by notebook 06."""

    def __init__(self, model: GemmaVisionForJSON, cfg: AppConfig,
                 device: torch.device):
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.device = device
        self.tok = model.tok
        # <end_of_turn> terminates generation in Gemma chat format.
        ids = self.tok.encode("<end_of_turn>", add_special_tokens=False)
        self.eot_id = ids[-1] if ids else None

    @torch.no_grad()
    def predict(self, image: Image.Image, max_new_tokens: Optional[int] = None,
                temperature: Optional[float] = None) -> Dict:
        ic = self.cfg.inference
        pixel = image_to_tensor(image, self.cfg.dataset.image_size)
        pixel = pixel.unsqueeze(0).to(self.device)            # [1,3,H,W]
        prompt_ids = torch.tensor(
            [self.tok.encode(PROMPT_BY_STAGE["json"], add_special_tokens=True)],
            device=self.device)

        out_ids = self.model.generate_from_image(
            pixel, prompt_ids,
            max_new_tokens=max_new_tokens or ic.max_new_tokens,
            temperature=ic.temperature if temperature is None else temperature,
            eot_id=self.eot_id)
        raw = self.tok.decode(out_ids.tolist(), skip_special_tokens=True)

        obj, valid = (repair_json(raw) if ic.json_guard else (None, False))
        if not ic.json_guard:
            try:
                obj, valid = json.loads(raw), True
            except json.JSONDecodeError:
                obj, valid = None, False
        return {"raw_text": raw, "json": obj, "strictly_valid": valid}
