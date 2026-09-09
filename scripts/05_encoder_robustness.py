#!/usr/bin/env python3
"""Phase 05 -- Encoder-robustness grid: retrain ``flat`` and
``semantic_path_hmlc`` with five pretrained encoders (paper Section 6.6 /
Table 5).

Requires token caches for every encoder in
``Config.encoder_robustness_variants`` (build them with
``01_tokenize_cache.py``; non-PhoBERT encoders use max_length=512 -- see
``Config.encoder_max_length_overrides``).

Example:
    python scripts/05_encoder_robustness.py --run-mode paper
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_path_hmlc.config import Config  # noqa: E402
from semantic_path_hmlc.encoder_robustness import run_encoder_robustness_grid  # noqa: E402
from semantic_path_hmlc.pipeline import prepare_dataset  # noqa: E402
from semantic_path_hmlc.training import train_one  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--token-cache-dir", default=None)
    p.add_argument("--run-mode", default=None, choices=["paper", "smoke"])
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

    def train_one_fn(variant, seed, encoder_name=None):
        return train_one(
            variant, seed, cfg, state["tax"], state["df"], state["split_col"],
            state["out_dir"], state["local_out_dir"], state["device"], state["amp_dtype"],
            state["dataset_fingerprint"], state["cache_ram"], state["path_class_weights"],
            encoder_name=encoder_name,
        )

    encoder_robustness = run_encoder_robustness_grid(cfg, train_one_fn, state["out_dir"])
    print(encoder_robustness)
    print("\nENCODER-ROBUSTNESS GRID COMPLETE. Run 06_export_evidence_and_qualitative.py next.")


if __name__ == "__main__":
    main()
