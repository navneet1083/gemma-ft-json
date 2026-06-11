"""PyTorch Dataset for (table image -> JSON) supervised fine-tuning.

Design notes
------------
* The dataset returns *token ids*, not strings: tokenization is the slow,
  deterministic part, so doing it here lets DataLoader workers parallelize
  it off the MPS device.
* `stage` selects the curriculum target:
      "read"      -> read_text     (alignment stage A: pseudo-OCR, Donut-style)
      "linearize" -> linearized    (alignment stage B: structure as markdown)
      "sft"/"json"-> json          (final task)
* Labels use -100 over prompt tokens so loss is computed ONLY on the target
  (visual-token positions are masked later by the collator/model since their
  count is a model property, not a data property).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..exceptions import DataError

# Mean/std of ~white document images; near-identity normalization keeps the
# conv stem's input distribution centered without ImageNet statistics
# (we have no ImageNet pretraining anyway — encoder is trained from scratch).
_MEAN, _STD = 0.9, 0.2

PROMPT_BY_STAGE = {
    "read": "<start_of_turn>user\nRead all text in the table image.<end_of_turn>\n<start_of_turn>model\n",
    "linearize": "<start_of_turn>user\nLinearize the table image row by row.<end_of_turn>\n<start_of_turn>model\n",
    "json": "<start_of_turn>user\nExtract the table in the image as a JSON object.<end_of_turn>\n<start_of_turn>model\n",
}
PROMPT_BY_STAGE["sft"] = PROMPT_BY_STAGE["json"]
TARGET_KEY_BY_STAGE = {"read": "read_text", "linearize": "linearized",
                       "json": "json", "sft": "json"}


def image_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    """PIL -> float tensor [3, size, size] in roughly N(0,1) range.

    Shape contract is asserted because a silent channel/size mismatch here
    would surface 4 modules later as an inscrutable matmul error.
    """
    if img.size != (size, size):
        canvas = Image.new("RGB", (size, size), "white")
        img.thumbnail((size, size), Image.LANCZOS)
        canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        img = canvas
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)            # HWC -> CHW
    t = (t - _MEAN) / _STD
    if t.shape != (3, size, size):
        raise DataError(f"image tensor shape {tuple(t.shape)} != (3,{size},{size})")
    return t


class TableImageJsonDataset(Dataset):
    """Manifest-driven dataset. Plug-and-play: any JSONL with the keys
    {image, json, linearized, read_text} works, so you can later swap in
    REAL annotated tables without touching the trainer."""

    def __init__(self, manifest_path: str, tokenizer, image_size: int,
                 stage: str = "sft", max_seq_len: int = 768):
        self.path = Path(manifest_path)
        if not self.path.exists():
            raise DataError(
                f"Manifest not found: {self.path}. Run notebook 01 / "
                "scripts/build_dataset.py first.")
        if stage not in PROMPT_BY_STAGE:
            raise DataError(f"Unknown stage {stage!r}; use {list(PROMPT_BY_STAGE)}")
        self.records: List[Dict] = []
        with open(self.path) as fh:
            for ln, line in enumerate(fh):
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DataError(f"Corrupt manifest line {ln} in {self.path}") from exc
        self.tok = tokenizer
        self.image_size = image_size
        self.stage = stage
        self.max_seq_len = max_seq_len
        # Token-id sets used for structure-vs-content loss weighting.
        # JSON scaffolding chars are easy to learn (always present), so they
        # are down-weighted; cell content carries the real signal.
        self.structure_ids = set()
        for ch in ['{', '}', '[', ']', ':', ',', '"', '",', '"}', '"]', '":']:
            for tid in self.tok.encode(ch, add_special_tokens=False):
                self.structure_ids.add(tid)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        try:
            img = Image.open(rec["image"])
        except Exception as exc:
            raise DataError(f"Cannot open image {rec['image']}") from exc
        pixel = image_to_tensor(img, self.image_size)

        prompt = PROMPT_BY_STAGE[self.stage]
        target = rec[TARGET_KEY_BY_STAGE[self.stage]] + "<end_of_turn>"

        p_ids = self.tok.encode(prompt, add_special_tokens=True)    # includes BOS
        t_ids = self.tok.encode(target, add_special_tokens=False)

        ids = (p_ids + t_ids)[: self.max_seq_len]
        n_prompt = min(len(p_ids), len(ids))
        labels = [-100] * n_prompt + ids[n_prompt:]                  # loss on target only

        # Per-token weight TAGS, resolved to real weights in the collator
        # (where training-config values are known):
        #   1.0  -> neutral (prompt positions; masked by -100 anyway)
        #  -1.0  -> JSON *structure* token ({ } [ ] : , ")  -> down-weighted
        #  -2.0  -> JSON *content* token (cell text/digits) -> up-weighted
        weights = [1.0] * n_prompt
        if self.stage in ("json", "sft"):
            for tid in ids[n_prompt:]:
                weights.append(-1.0 if tid in self.structure_ids else -2.0)
        else:
            weights += [1.0] * (len(ids) - n_prompt)

        return {
            "pixel_values": pixel,                                   # [3,H,W]
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weight_tags": torch.tensor(weights, dtype=torch.float32),
        }
