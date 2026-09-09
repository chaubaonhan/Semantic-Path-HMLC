#!/usr/bin/env python3
"""Phase 02 -- Train the five core variants and the five clean ablations
(three seeds each), then write the main results table.

Requires token caches built by ``01_tokenize_cache.py`` for
``Config.encoder_name`` (the primary encoder, PhoBERT-base-v2 by default).

Re-running is safe and cheap: a run whose metrics/checkpoint/prediction
files already exist and match the current pipeline version is skipped
instead of retrained (see ``semantic_path_hmlc.training.train_one``).

Example:
    python scripts/02_run_core_and_ablations.py --run-mode paper
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_path_hmlc.config import Config  # noqa: E402
from semantic_path_hmlc.pipeline import prepare_dataset  # noqa: E402
from semantic_path_hmlc.tables import build_main_summary  # noqa: E402
from semantic_path_hmlc.training import default_run_specs, train_one  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--token-cache-dir", default=None)
    p.add_argument("--run-mode", default=None, choices=["paper", "smoke"])
    p.add_argument("--force-retrain", action="store_true")
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
    run_specs = default_run_specs(cfg, bool(state["competing_stop_paths"]))
    n_core = sum(v in cfg.core_variants for v, _ in run_specs)
    print(f"Single-path paper schedule: {len(run_specs)} runs = {n_core} core + {len(run_specs) - n_core} clean ablation runs.")

    all_results = []
    tables_dir = state["out_dir"] / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for variant, seed in run_specs:
        metrics = train_one(
            variant, seed, cfg, state["tax"], state["df"], state["split_col"],
            state["out_dir"], state["local_out_dir"], state["device"], state["amp_dtype"],
            state["dataset_fingerprint"], state["cache_ram"], state["path_class_weights"],
            force_retrain=args.force_retrain,
        )
        all_results.append(metrics)
        pd.DataFrame(all_results).to_csv(tables_dir / "all_seed_results.csv", index=False)

    results = pd.DataFrame(all_results)
    print(results)
    summary = build_main_summary(results, state["out_dir"])
    print(summary)
    print("\nCORE + ABLATION TRAINING COMPLETE. Run 03_build_tables.py next.")


if __name__ == "__main__":
    main()
