"""Synthetic complex-table corpus generator (Donut-style synthetic pretraining).

WHY synthetic? Two of the project's hard constraints meet here:
  * No Hugging Face downloads  -> no public table datasets pulled from the Hub.
  * Gemma 3 270M is text-only  -> the new vision pathway is random-init and
    needs *thousands* of perfectly-labelled (image, JSON) pairs to align.
A renderer gives us infinite, pixel-perfect ground truth for free, with a
difficulty curriculum (merged headers, row spans, borderless tables) that
mirrors real "complex table architectures".

Each sample carries THREE aligned targets used by the 3-stage curriculum:
  read_text  -> raw left-to-right, top-to-bottom cell text   (stage READ)
  linearized -> markdown-ish row serialization                (stage LINEARIZE)
  json       -> the final structured object                   (stage JSON/SFT)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..config.loader import DatasetCfg
from ..exceptions import DataError

# --------------------------------------------------------------------------- #
# Vocabulary for cell content (kept domain-generic; replace with your domain
# nouns to specialize). Numbers/dates/currency teach digit-level reading.
# --------------------------------------------------------------------------- #
_WORDS = ("alpha beta gamma delta omega sigma metro hydro solar lunar nova "
          "quartz cobalt amber onyx delta vertex apex prism orbit flux "
          "ledger invoice asset margin yield basis quota tariff freight "
          "north south east west retail bulk export import domestic").split()
_HEADERS = ("Item Name Region Category Status Owner Code SKU Type Group "
            "Q1 Q2 Q3 Q4 Total Price Qty Amount Rate Date Score Rank").split()


def _rand_cell(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.35:                                   # number / currency
        v = round(rng.uniform(0, 99999), rng.choice([0, 1, 2]))
        return rng.choice([f"{v}", f"${v}", f"{v}%"])
    if kind < 0.50:                                   # date
        return f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{rng.randint(2019,2026)}"
    n = rng.randint(1, 2)                             # words
    return " ".join(rng.choice(_WORDS).capitalize() for _ in range(n))


@dataclass
class TableSpec:
    """Logical description of one table BEFORE rendering — this is the
    ground truth; the image is merely its projection onto pixels."""
    headers: List[str]
    header_groups: Optional[List[Tuple[str, int]]]   # (group_label, span) or None
    rows: List[List[str]]
    row_spans: List[Tuple[int, int]]                 # (row_idx, col_idx) cells merged downward
    style: Dict


class SyntheticTableGenerator:
    """Exposed utility consumed by notebook 01. Deterministic under `seed`."""

    def __init__(self, cfg: DatasetCfg):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    # ------------------------- logical structure -------------------------- #
    def sample_spec(self) -> TableSpec:
        rng = self.rng
        n_rows = rng.randint(self.cfg.min_rows, self.cfg.max_rows)
        n_cols = rng.randint(self.cfg.min_cols, self.cfg.max_cols)
        headers = rng.sample(_HEADERS, n_cols)

        header_groups = None
        if n_cols >= 4 and rng.random() < self.cfg.p_merged_header:
            # Split columns into 2 contiguous merged super-headers.
            cut = rng.randint(2, n_cols - 2)
            header_groups = [(rng.choice(_WORDS).capitalize(), cut),
                             (rng.choice(_WORDS).capitalize(), n_cols - cut)]

        rows = [[_rand_cell(rng) for _ in range(n_cols)] for _ in range(n_rows)]

        row_spans: List[Tuple[int, int]] = []
        if n_rows >= 3 and rng.random() < self.cfg.p_row_span:
            r, c = rng.randint(0, n_rows - 2), rng.randint(0, n_cols - 1)
            rows[r + 1][c] = rows[r][c]              # value repeats logically
            row_spans.append((r, c))                 # but renders as ONE merged cell

        style = {
            "grid": rng.random() >= self.cfg.p_no_gridlines,
            "zebra": rng.random() < self.cfg.p_zebra,
            "font_size": rng.randint(13, 19),
            "header_bg": rng.choice(["#dde6f0", "#e8e8e8", "#f3e3d3", "#dcefdc"]),
            "pad": rng.randint(6, 12),
        }
        return TableSpec(headers, header_groups, rows, row_spans, style)

    # ----------------------------- targets -------------------------------- #
    @staticmethod
    def to_json_obj(spec: TableSpec) -> Dict:
        obj: Dict = {"columns": spec.headers, "rows": spec.rows}
        if spec.header_groups:
            obj["column_groups"] = [
                {"label": g, "span": s} for g, s in spec.header_groups]
        return obj

    @staticmethod
    def to_linearized(spec: TableSpec) -> str:
        lines = [" | ".join(spec.headers)]
        lines += [" | ".join(r) for r in spec.rows]
        return "\n".join(lines)

    @staticmethod
    def to_read_text(spec: TableSpec) -> str:
        toks: List[str] = []
        if spec.header_groups:
            toks += [g for g, _ in spec.header_groups]
        toks += spec.headers
        for r in spec.rows:
            toks += r
        return " ".join(toks)

    # ----------------------------- rendering ------------------------------ #
    def render(self, spec: TableSpec, size: int) -> Image.Image:
        """Rasterize the spec. Pixel-perfect by construction: the JSON target
        is derived from the SAME spec object, so labels can never drift."""
        st = spec.style
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", st["font_size"])
        except OSError:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", st["font_size"])
            except OSError:
                font = ImageFont.load_default()

        n_cols = len(spec.headers)
        pad = st["pad"]

        def tw(s: str) -> int:
            bb = font.getbbox(s)
            return bb[2] - bb[0]

        col_w = [max([tw(spec.headers[c])] + [tw(r[c]) for r in spec.rows]) + 2 * pad
                 for c in range(n_cols)]
        line_h = font.getbbox("Ag")[3] + 2 * pad
        extra_header = 1 if spec.header_groups else 0
        W = sum(col_w) + 2 * pad
        H = (len(spec.rows) + 1 + extra_header) * line_h + 2 * pad

        img = Image.new("RGB", (max(W, 64), max(H, 64)), "white")
        d = ImageDraw.Draw(img)
        x0, y = pad, pad

        if spec.header_groups:                       # merged super-header band
            x = x0
            ci = 0
            for label, span in spec.header_groups:
                w = sum(col_w[ci:ci + span]); ci += span
                d.rectangle([x, y, x + w, y + line_h], fill=st["header_bg"],
                            outline="black" if st["grid"] else None)
                d.text((x + (w - tw(label)) / 2, y + pad), label, fill="black", font=font)
                x += w
            y += line_h

        x = x0                                        # column headers
        for c, h in enumerate(spec.headers):
            d.rectangle([x, y, x + col_w[c], y + line_h], fill=st["header_bg"],
                        outline="black" if st["grid"] else None)
            d.text((x + pad, y + pad), h, fill="black", font=font)
            x += col_w[c]
        y += line_h

        merged = {(r + 1, c) for r, c in spec.row_spans}
        for ri, row in enumerate(spec.rows):          # body
            if st["zebra"] and ri % 2 == 1:
                d.rectangle([x0, y, x0 + sum(col_w), y + line_h], fill="#f5f5f5")
            x = x0
            for ci, cell in enumerate(row):
                if st["grid"]:
                    top = y if (ri, ci) not in {(r + 1, c) for r, c in spec.row_spans} else y
                    d.rectangle([x, y, x + col_w[ci], y + line_h], outline="black")
                if (ri, ci) not in merged:            # skip text in merged-into cell
                    d.text((x + pad, y + pad), cell, fill="black", font=font)
                x += col_w[ci]
            y += line_h

        # AR-preserving pad to a square canvas of `size` (matches model input).
        img.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        return canvas


def build_dataset(cfg: DatasetCfg, raw_dir: str, manifest_path: str,
                  n_samples: int, split: str, seed_offset: int = 0,
                  progress_cb=None) -> int:
    """Materialize `n_samples` (image, targets) pairs + a JSONL manifest.

    Exposed function utility for notebook 01. Returns #samples written.
    """
    out_dir = Path(raw_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = SyntheticTableGenerator(cfg)
    gen.rng.seed(cfg.seed + seed_offset)

    n_written = 0
    with open(manifest_path, "w") as mf:
        for i in range(n_samples):
            try:
                spec = gen.sample_spec()
                img = gen.render(spec, cfg.image_size)
                img_path = out_dir / f"{split}_{i:06d}.png"
                img.save(img_path)
                rec = {
                    "image": str(img_path),
                    "json": json.dumps(SyntheticTableGenerator.to_json_obj(spec),
                                       separators=(",", ":")),
                    "linearized": SyntheticTableGenerator.to_linearized(spec),
                    "read_text": SyntheticTableGenerator.to_read_text(spec),
                }
                mf.write(json.dumps(rec) + "\n")
                n_written += 1
            except Exception as exc:    # one bad sample must not kill the build
                raise DataError(f"Failed generating sample {i} ({split}): {exc}") from exc
            if progress_cb:
                progress_cb(i + 1, n_samples)
    return n_written
