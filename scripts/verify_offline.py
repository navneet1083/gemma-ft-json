#!/usr/bin/env python
"""Verify the no-Hugging-Face-download setup (does not touch the network).

Usage:
    python scripts/verify_offline.py
    python scripts/verify_offline.py --config configs/default.yaml
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemma_ft_json.scripts_entry import verify_offline_cli  # noqa: E402

if __name__ == "__main__":
    verify_offline_cli()
