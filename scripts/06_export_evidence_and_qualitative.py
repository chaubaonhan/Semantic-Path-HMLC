#!/usr/bin/env python3
"""Phase 06 -- Export the machine-readable paper-evidence package and the
qualitative example / error-analysis tables. Reads whatever phase 02-05
tables are present in ``output_dir/tables`` (missing ones degrade
gracefully to empty tables rather than failing).

Example:
    python scripts/06_export_evidence_and_qualitative.py --run-mode paper
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_path_hmlc.config import Config  # noqa: E402
from semantic_path_hmlc.evidence import export_evidence  # noqa: E402
from semantic_path_hmlc.pipeline import prepare_dataset  # noqa: E402
from semantic_path_hmlc.qualitative import controlled_unseen_examples, qualitative_examples  # noqa: E402
from semantic_path_hmlc.training import run_identity  # noqa: E402
from semantic_path_hmlc.zero_shot import controlled_run_id  # noqa: E402


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


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
    tax, df, out_dir = state["tax"], state["df"], state["out_dir"]
    tables_dir = out_dir / "tables"

    results = _read_csv_or_empty(tables_dir / "all_seed_results.csv")
    controlled_results = _read_csv_or_empty(tables_dir / "controlled_zero_and_few_shot_results.csv")
    if controlled_results.empty:
        controlled_results = _read_csv_or_empty(tables_dir / "controlled_zero_shot_results.csv")
    encoder_robustness = _read_csv_or_empty(tables_dir / "encoder_robustness.csv")
    controlled_manifest = _read_csv_or_empty(out_dir / "splits" / "controlled_zero_shot_manifest.csv")
    split_counts = df[state["split_col"]].value_counts().to_dict()

    export_evidence(
        cfg, out_dir, state["task_diagnosis"], state["node_stats"], split_counts,
        len(state["train_paths"]), len(state["test_paths"]), len(state["unseen_test_paths"]),
        len(state["competing_stop_paths"]), len(tax.paths), results,
        controlled_manifest, controlled_results, encoder_robustness,
    )

    def rid(variant, seed, encoder_name=None):
        run_id, _, _ = run_identity(cfg, out_dir, variant, seed, encoder_name or cfg.encoder_name)
        return run_id

    def controlled_rid(variant, seed, k):
        return controlled_run_id(cfg, variant, seed, k)

    if (out_dir / "predictions" / f"{rid('semantic_path_hmlc', cfg.seeds[0])}.npz").exists():
        qualitative_examples(df, tax, cfg, out_dir, rid, variant="semantic_path_hmlc", n_examples=12, only_mismatches=False)
        qualitative_examples(df, tax, cfg, out_dir, rid, variant="semantic_path_hmlc", n_examples=6, only_mismatches=True, random_state=7)
    else:
        print("Skipping qualitative_examples: no semantic_path_hmlc predictions found (run 02 first).")

    controlled_unseen_examples(df, tax, out_dir, controlled_rid, seed=cfg.seeds[0])

    print("\nEVIDENCE + QUALITATIVE EXPORT COMPLETE. Run 07_package_results.py next.")


if __name__ == "__main__":
    main()
