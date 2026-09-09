#!/usr/bin/env python3
"""Phase 01 -- Tokenize once, then save the reusable token cache.

Run this once before training (``02_run_core_and_ablations.py`` and every
other training script validate the dataset fingerprint against this cache
and refuse to run on a stale/misaligned one).

Example:
    python scripts/01_tokenize_cache.py --run-mode paper
    python scripts/01_tokenize_cache.py --encoders vinai/phobert-base-v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_path_hmlc.config import Config  # noqa: E402
from semantic_path_hmlc.pipeline import prepare_dataset  # noqa: E402
from semantic_path_hmlc.tokenization import build_token_cache  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default=None, help="Override Config.data_path")
    p.add_argument("--output-dir", default=None, help="Override Config.output_dir")
    p.add_argument("--token-cache-dir", default=None, help="Override Config.token_cache_dir")
    p.add_argument("--run-mode", default=None, choices=["paper", "smoke"])
    p.add_argument(
        "--encoders", nargs="*", default=None,
        help="Encoders to tokenize (default: the full encoder-robustness grid in paper mode, "
             "or just the primary encoder in smoke mode).",
    )
    p.add_argument("--overwrite", action="store_true", help="Rebuild caches even if a current one exists")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.data_path:
        cfg.data_path = args.data_path
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.token_cache_dir:
        cfg.token_cache_dir = args.token_cache_dir
    if args.run_mode:
        cfg.run_mode = args.run_mode

    state = prepare_dataset(cfg)
    df, dataset_fingerprint = state["df"], state["dataset_fingerprint"]

    encoders = args.encoders or (
        cfg.encoder_robustness_variants if cfg.run_mode == "paper" else (cfg.encoder_name,)
    )
    print(f"Preparing {len(encoders)} encoder cache(s): {tuple(encoders)}")
    for encoder_name in encoders:
        build_token_cache(df, cfg, encoder_name, dataset_fingerprint, overwrite=args.overwrite)

    print("\nTOKENIZATION PHASE COMPLETE. Run 02_run_core_and_ablations.py next.")


if __name__ == "__main__":
    main()
