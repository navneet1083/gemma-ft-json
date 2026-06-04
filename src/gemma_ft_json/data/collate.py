"""Batching: a padding collator + dataloader factory.

Variable-length text is right-padded to the batch maximum. Padding positions get
pad_id in input_ids, 0 in attention_mask, and IGNORE_INDEX in labels (so padding
never contributes to the loss).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .dataset import IGNORE_INDEX, TableJsonDataset


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(int(b["length"]) for b in batch)
        bsz = len(batch)

        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)  # [B,3,S,S]

        for i, b in enumerate(batch):
            n = int(b["length"])
            input_ids[i, :n] = b["input_ids"]
            labels[i, :n] = b["labels"]
            attention_mask[i, :n] = 1

        return {"pixel_values": pixel_values, "input_ids": input_ids,
                "labels": labels, "attention_mask": attention_mask}


def build_dataloaders(train_ds: TableJsonDataset, val_ds: Optional[TableJsonDataset],
                      pad_id: int, batch_size: int, num_workers: int = 2,
                      pin_memory: bool = False, shuffle_train: bool = True
                      ) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train/val dataloaders sharing one collator."""
    collate = Collator(pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=collate, drop_last=False)
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory,
                                collate_fn=collate, drop_last=False)
    return train_loader, val_loader
