"""Export a machine-readable evidence package summarizing every audited
statistic, main result, ablation, controlled zero/few-shot result, and
encoder-robustness result (paper-writing support). Unchanged from the
original notebook.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .config import Config


def export_evidence(
    cfg: Config, out_dir: Path, task_diagnosis: Dict[str, Any], node_stats: pd.DataFrame,
    split_counts: Dict[str, int], train_paths: int, test_paths: int, unseen_test_paths: int,
    competing_stop_paths: int, total_terminal_paths: int, results: pd.DataFrame,
    controlled_manifest: pd.DataFrame, controlled_results: pd.DataFrame,
    encoder_robustness: pd.DataFrame,
) -> Path:
    evidence = {
        "config": asdict(cfg),
        "task_diagnosis": task_diagnosis,
        "node_statistics": node_stats.to_dict(orient="records"),
        "split_counts": split_counts,
        "train_valid_paths": train_paths,
        "test_valid_paths": test_paths,
        "unseen_test_paths": unseen_test_paths,
        "stop_audit": {
            "terminal_paths_with_child_vs_stop_choice": competing_stop_paths,
            "terminal_leaf_paths_with_forced_stop": total_terminal_paths - competing_stop_paths,
            "adaptive_stop_claim_allowed": bool(competing_stop_paths),
        },
        "metric_definitions": {
            "path_macro_f1_supported": "macro-F1 over labels present in the evaluated split/subset",
            "path_macro_f1_all_taxonomy": "macro-F1 over all fixed taxonomy terminal paths",
            "hierarchical_f1": "micro node-overlap F1 excluding ROOT",
            "hierarchical_f1_example_macro": "mean per-document node-overlap F1 excluding ROOT",
        },
        "semantic_method": {
            "label_description": "full taxonomy prefix text",
            "label_encoder": "frozen initial pretrained encoder representations",
            "taxonomy_fusion": "semantic node features plus graph message passing",
            "child_scorer": "shared document-parent-child compatibility without action-specific bias",
            "stop_scorer": "shared document-node-depth compatibility",
            "controlled_training_mask": "held-out paths are excluded from the training softmax denominator",
        },
        "main_results": results.to_dict(orient="records"),
        "controlled_zero_shot_manifest": controlled_manifest.to_dict(orient="records"),
        "controlled_zero_few_shot_results": controlled_results.to_dict(orient="records"),
        "claim_guards": {
            "natural_unseen_is_primary_evidence": False,
            "zero_example_claim_scope": "held-out terminal paths in a taxonomy known in advance",
            "controlled_zero_shot_executed": not controlled_results.empty,
            "few_shot_claim_allowed": bool(not controlled_results.empty and (controlled_results["k_shot"] > 0).any()),
            "adaptive_stop_claim_allowed": bool(competing_stop_paths),
        },
        "encoder_robustness": encoder_robustness.to_dict(orient="records"),
        "smoke_mode_warning": cfg.run_mode != "paper",
    }
    out_path = Path(out_dir) / "paper_evidence.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    print(f"Saved evidence package: {out_path}")
    print("DO NOT write numerical claims from smoke mode. Re-run with cfg.run_mode='paper'.")
    return out_path
