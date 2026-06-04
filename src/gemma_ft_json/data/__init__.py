"""Data subpackage: curation, transforms, dataset, and collation."""
from .transforms import build_image_transform, ImageTransform
from .dataset import TableJsonDataset, IGNORE_INDEX
from .collate import Collator, build_dataloaders
from .build_dataset import (
    build_manifest, split_manifest, iter_image_paths, pair_image_json,
    validate_record, target_to_string, extract_boxes_pymupdf,
)

__all__ = [
    "build_image_transform", "ImageTransform",
    "TableJsonDataset", "IGNORE_INDEX",
    "Collator", "build_dataloaders",
    "build_manifest", "split_manifest", "iter_image_paths", "pair_image_json",
    "validate_record", "target_to_string", "extract_boxes_pymupdf",
]
