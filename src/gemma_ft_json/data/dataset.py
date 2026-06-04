"""PyTorch Dataset for image -> JSON supervised fine-tuning.

Each item produces the TEXT side of one example:

    input_ids = [BOS] + prompt_ids + target_ids + [EOS]
    labels    = [-100  ...(prompt masked)...] + target_ids + [EOS]

WHY MASK THE PROMPT?  We only grade/learn the *target* JSON, not the fixed
instruction. Masked positions use ignore_index (-100) -> zero loss there.

The image is returned as a preprocessed pixel tensor; projecting it into the
decoder's embedding space and prepending the visual "soft tokens" happen in the
model (models/vlm.py), keeping the Dataset model-agnostic (plug-and-play).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..exceptions import DataError
from ..tokenization import BaseTokenizer
from .transforms import ImageTransform

IGNORE_INDEX = -100  # standard CrossEntropy ignore label


class TableJsonDataset(Dataset):
    def __init__(self, manifest_path: str | Path, tokenizer: BaseTokenizer,
                 transform: ImageTransform, prompt: str = "Extract the table as JSON.",
                 max_target_tokens: int = 1024):
        self.tokenizer = tokenizer
        self.transform = transform
        self.prompt = prompt
        self.max_target_tokens = max_target_tokens

        self.records: List[Dict] = []
        path = Path(manifest_path)
        if not path.is_file():
            raise DataError(f"Manifest not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self.records.append(json.loads(line))
        if not self.records:
            raise DataError(f"Manifest is empty: {path}")

        # Pre-tokenize the (constant) prompt once.
        self._prompt_ids = self.tokenizer.encode(self.prompt, add_bos=True, add_eos=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        try:
            with Image.open(rec["image_path"]) as im:
                pixel_values = self.transform(im)  # [3, S, S]
        except (OSError, KeyError) as e:
            raise DataError(f"Failed to load image for record {idx}: {e}") from e

        target_ids = self.tokenizer.encode(rec["target"], add_bos=False, add_eos=True)
        target_ids = target_ids[: self.max_target_tokens]

        input_ids = self._prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(self._prompt_ids) + list(target_ids)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "length": torch.tensor(len(input_ids), dtype=torch.long),
        }
