#!/usr/bin/env python3
"""Phase 04 -- Controlled known-taxonomy zero-example and few-shot terminal-
path evaluation (paper Section 6.5 / Table 4).

Requires the primary encoder's token cache (``01_tokenize_cache.py``).
Does not require phase 02 to have run first (it trains its own controlled-
protocol models), but sharing the same ``Config`` and dataset keeps the
taxonomy and splits identical.

Example:
    python scripts/04_zero_shot_few_shot.py --run-mode paper
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_path_hmlc.config import Config  # noqa: E402
from semantic_path_hmlc.pipeline import prepare_dataset  # noqa: E402
from semantic_path_hmlc.zero_shot import (  # noqa: E402
    build_controlled_protocol, build_controlled_summary, train_controlled_one,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--token-cache-dir", default=None)
    p.add_argument("--run-mode", default=None, choices=["paper", "smoke"])
    p.add_argument("--skip-few-shot", action="store_true")
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
    tax, df, out_dir = state["tax"], state["df"], state["out_dir"]

    protocol = build_controlled_protocol(df, tax, cfg, state["split_col"], out_dir)
    print(protocol["manifest"])
    print({
        "controlled_train_documents": len(protocol["train"]),
        "seen_test_documents": len(protocol["seen_test"]),
        "unseen_dev_paths": len(protocol["dev_path_ids"]),
        "unseen_test_paths": len(protocol["test_path_ids"]),
        "unseen_test_documents": len(protocol["unseen_test"]),
    })

    controlled_rows = []
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if cfg.run_controlled_zero_shot:
        for variant in cfg.controlled_variants:
            for seed in cfg.controlled_seeds:
                metrics = train_controlled_one(
                    variant, seed, protocol, cfg, tax, df, out_dir, state["local_out_dir"],
                    state["device"], state["amp_dtype"], state["dataset_fingerprint"], state["cache_ram"], k=0,
                )
                controlled_rows.append(metrics)
                pd.DataFrame(controlled_rows).to_csv(tables_dir / "controlled_zero_shot_results.csv", index=False)

    if cfg.run_few_shot and not args.skip_few_shot:
        for k in cfg.few_shot_values:
            for seed in cfg.controlled_seeds:
                metrics = train_controlled_one(
                    "semantic_path_hmlc", seed, protocol, cfg, tax, df, out_dir, state["local_out_dir"],
                    state["device"], state["amp_dtype"], state["dataset_fingerprint"], state["cache_ram"], k=int(k),
                )
                controlled_rows.append(metrics)
                pd.DataFrame(controlled_rows).to_csv(tables_dir / "controlled_zero_and_few_shot_results.csv", index=False)

    controlled_results = pd.DataFrame(controlled_rows)
    print(controlled_results)
    if not controlled_results.empty:
        summary = build_controlled_summary(controlled_results, out_dir)
        print(summary)

    print("\nZERO-SHOT / FEW-SHOT EVALUATION COMPLETE. Run 05_encoder_robustness.py next.")


if __name__ == "__main__":
    main()
