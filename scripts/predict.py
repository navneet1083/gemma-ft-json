#!/usr/bin/env python
"""Predict JSON for a single image using a trained best.ckpt.

Usage:
    python scripts/predict.py \
        --snapshot runs/gemma-ft-json/config.snapshot.yaml \
        --checkpoint runs/gemma-ft-json/best.ckpt \
        --image /path/to/table.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemma_ft_json.inference import Predictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="config.snapshot.yaml saved by training")
    ap.add_argument("--checkpoint", required=True, help="path to best.ckpt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    predictor = Predictor(args.snapshot, args.checkpoint, max_new_tokens=args.max_new_tokens)
    print(predictor.describe())
    print(predictor.predict(args.image))


if __name__ == "__main__":
    main()
