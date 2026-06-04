"""Dataset curation / manifest building.

Turns raw inputs (images + per-image GT JSON, optionally source PDFs) into clean
train/val *manifests* — JSONL where each line is one record:

    {"image_path": "...", "target": "<json string>", "boxes": [...optional...]}

Small, individually-exposed utilities so notebook 01 can demo each step:
    iter_image_paths -> pair_image_json -> validate_record -> target_to_string
    -> (extract_boxes_pymupdf) -> build_manifest -> split_manifest

PHILOSOPHY (see README): for 90%-table data, structured JSON is the right target
(flattened text destroys the grid). We validate JSON, optionally attach per-word
boxes from a digital PDF text layer (free grounding labels), and drop unparseable
pages so we never *teach the model to hallucinate* from noisy labels.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from ..exceptions import DataError

logger = logging.getLogger("gemma_ft_json")

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def iter_image_paths(images_dir: str | Path) -> Iterator[Path]:
    """Yield every supported image under `images_dir` (sorted -> deterministic)."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise DataError(f"images_dir does not exist: {images_dir}")
    for p in sorted(images_dir.rglob("*")):
        if p.suffix.lower() in _IMAGE_EXTS:
            yield p


def pair_image_json(image_path: Path, json_dir: str | Path) -> Optional[Path]:
    """Find GT JSON for an image by matching stem. Returns None if absent."""
    candidate = Path(json_dir) / f"{image_path.stem}.json"
    return candidate if candidate.is_file() else None


def _load_json(path: Path) -> object:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise DataError(f"Unreadable/invalid JSON {path}: {e}") from e


def validate_record(obj: object) -> bool:
    """True if the GT looks like usable table JSON (non-empty dict/list).
    Tighten to enforce your concrete schema (e.g. require 'rows'/'columns')."""
    if isinstance(obj, (dict, list)):
        return len(obj) > 0
    return False


def target_to_string(obj: object) -> str:
    """Canonical compact JSON string: sorted keys, no spaces -> one stable target
    so the model learns a single serialization instead of chasing formatting."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_boxes_pymupdf(pdf_path: Path, page_index: int = 0) -> List[Dict]:
    """OPTIONAL: per-word bounding boxes from a *digital* PDF page (free grounding
    labels). Returns [{"text": str, "bbox": [x0,y0,x1,y1]}]. Import is guarded."""
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # noqa: BLE001
        raise DataError("PyMuPDF not installed. `pip install -e '.[curation]'`.") from e
    doc = fitz.open(pdf_path)
    if page_index >= len(doc):
        return []
    words = doc[page_index].get_text("words")  # (x0,y0,x1,y1, word, block, line, wordno)
    return [{"text": w[4], "bbox": [w[0], w[1], w[2], w[3]]} for w in words]


def build_manifest(
    images_dir: str | Path,
    json_dir: str | Path,
    out_dir: str | Path,
    pdf_dir: str | Path = "",
    require_valid_json: bool = True,
    use_pymupdf_boxes: bool = False,
    manifest_name: str = "manifest.jsonl",
) -> Tuple[Path, Dict[str, int]]:
    """Build a manifest.jsonl pairing each image with its JSON target.
    Returns (manifest_path, stats)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / manifest_name

    stats = {"total": 0, "kept": 0, "no_json": 0, "invalid_json": 0}
    pdf_dir = Path(pdf_dir) if pdf_dir else None

    with open(manifest_path, "w", encoding="utf-8") as out:
        for img in iter_image_paths(images_dir):
            stats["total"] += 1
            jpath = pair_image_json(img, json_dir)
            if jpath is None:
                stats["no_json"] += 1
                continue
            try:
                obj = _load_json(jpath)
            except DataError:
                stats["invalid_json"] += 1
                continue
            if require_valid_json and not validate_record(obj):
                stats["invalid_json"] += 1
                continue

            record: Dict[str, object] = {
                "image_path": str(img.resolve()),
                "target": target_to_string(obj),
            }
            if use_pymupdf_boxes and pdf_dir is not None:
                pdf_candidate = pdf_dir / f"{img.stem}.pdf"
                if pdf_candidate.is_file():
                    try:
                        record["boxes"] = extract_boxes_pymupdf(pdf_candidate)
                    except DataError as e:
                        logger.warning("Box extraction failed for %s: %s", img.name, e)

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    if stats["kept"] == 0:
        raise DataError(
            f"No usable (image, JSON) pairs found among {stats['total']} images. "
            f"no_json={stats['no_json']} invalid_json={stats['invalid_json']}"
        )
    logger.info("Manifest written: %s (%s)", manifest_path, stats)
    return manifest_path, stats


def split_manifest(manifest_path: str | Path, val_fraction: float,
                   seed: int = 42) -> Tuple[Path, Path]:
    """Shuffle and split a manifest into train.jsonl / val.jsonl.

    NOTE: random split. For a stricter generalization test, split BY DOCUMENT
    SOURCE (all pages of one source on one side) so eval uses unseen layouts.
    """
    manifest_path = Path(manifest_path)
    lines = [ln for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    random.Random(seed).shuffle(lines)
    n_val = int(round(len(lines) * val_fraction))
    val_lines, train_lines = lines[:n_val], lines[n_val:]

    train_path = manifest_path.with_name("train.jsonl")
    val_path = manifest_path.with_name("val.jsonl")
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    logger.info("Split: %d train / %d val", len(train_lines), len(val_lines))
    return train_path, val_path
