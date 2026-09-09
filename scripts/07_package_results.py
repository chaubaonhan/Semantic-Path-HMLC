#!/usr/bin/env python3
"""Phase 07 -- Bundle every table/prediction/audit artifact (but not raw
model checkpoints) into one zip for archiving or download.

Example:
    python scripts/07_package_results.py --output-dir outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_path_hmlc.packaging import package_results  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--bundle-dir", default=None, help="Scratch directory for the uncompressed bundle")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    bundle_root = Path(args.bundle_dir) if args.bundle_dir else out_dir / "_results_bundle_scratch"
    zip_path = package_results(out_dir, bundle_root)
    print(f"Done: {zip_path}")


if __name__ == "__main__":
    main()
