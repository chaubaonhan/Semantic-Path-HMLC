#!/usr/bin/env python3
"""Phase 03 -- Depth/long-tail subgroup tables, seen/unseen diagnostics, the
same-checkpoint calibration ablation, and paired bootstrap significance
tests. Reads the predictions written by ``02_run_core_and_ablations.py``.

Example:
    python scripts/03_build_tables.py --run-mode paper
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_path_hmlc.config import Config  # noqa: E402
from semantic_path_hmlc.pipeline import prepare_dataset  # noqa: E402
from semantic_path_hmlc.tables import (  # noqa: E402
    build_calibration_ablation, build_seen_unseen_table, run_paired_bootstrap_pair, subgroup_metrics,
)
from semantic_path_hmlc.training import default_run_specs, run_identity  # noqa: E402


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
    tax, out_dir = state["tax"], state["out_dir"]
    run_specs = default_run_specs(cfg, bool(state["competing_stop_paths"]))

    def rid(variant, seed, encoder_name=None):
        run_id, _, _ = run_identity(cfg, out_dir, variant, seed, encoder_name or cfg.encoder_name)
        return run_id

    tables_dir = out_dir / "tables"
    for variant, seed in run_specs:
        npz_path = out_dir / "predictions" / f"{rid(variant, seed)}.npz"
        if not npz_path.exists():
            print(f"SKIP missing predictions: {npz_path}")
            continue
        subgroup = subgroup_metrics(npz_path, tax, state["path_counts_train"])
        subgroup.insert(0, "seed", seed)
        subgroup.insert(0, "variant", variant)
        subgroup.to_csv(tables_dir / f"subgroups__{rid(variant, seed)}.csv", index=False)

    train_supported_mask = state["path_counts_train"].cpu().numpy() > 0
    seen_unseen = build_seen_unseen_table(run_specs, rid, out_dir, tax, train_supported_mask)
    print(seen_unseen)

    calibration = build_calibration_ablation(cfg.seeds, rid, out_dir, tax)
    print(calibration)

    seed0 = cfg.seeds[0]
    variants_present = {v for v, _ in run_specs}
    if {"flat", "path_hmlc"}.issubset(variants_present):
        run_paired_bootstrap_pair("flat", "path_hmlc", seed0, rid, out_dir, tax, "paired_bootstrap.json")
    if {"path_hmlc", "semantic_path_hmlc"}.issubset(variants_present):
        run_paired_bootstrap_pair(
            "path_hmlc", "semantic_path_hmlc", seed0, rid, out_dir, tax, "paired_bootstrap__path_vs_semantic.json"
        )

    print("\nTABLE-BUILDING COMPLETE. Run 04_zero_shot_few_shot.py next.")


if __name__ == "__main__":
    main()
