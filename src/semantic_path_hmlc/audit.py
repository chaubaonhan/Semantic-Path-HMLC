"""Taxonomy audit: nodes per level, documents per stopping level, duplicate
names, and the single-path task-diagnosis assertion.

Unchanged from the original notebook logic (Section 2 / Section "1. Conservative
schema resolution" of the paper's Problem Formulation).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .common import all_prefixes, stable_text_hash
from .config import Config


def run_taxonomy_audit(df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """Compute the full-dataset taxonomy audit and enforce the single-path
    task assertion (raises if any document has more than one gold path).

    Returns a dict with the augmented ``df`` plus every audit artifact.
    """
    df = df.copy()
    df["n_gold_paths"] = df["paths"].map(len)
    df["terminal_depths"] = df["paths"].map(lambda ps: tuple(len(p) for p in ps))
    df["text_hash"] = df["text"].map(stable_text_hash)

    all_paths = sorted({p for paths in df["paths"] for p in paths})
    all_nodes = sorted({node for path in all_paths for node in all_prefixes(path)}, key=lambda x: (len(x), x))
    surface_to_nodes: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    for node in all_nodes:
        surface_to_nodes[node[-1]].append(node)

    max_depth = max(map(len, all_nodes))
    node_rows = []
    for depth in range(1, max_depth + 1):
        nodes_d = [n for n in all_nodes if len(n) == depth]
        surfaces_d = {n[-1] for n in nodes_d}
        terminating_paths_d = [p for p in all_paths if len(p) == depth]
        docs_ending_d = int(sum(sum(len(p) == depth for p in ps) for ps in df["paths"]))
        node_rows.append({
            "depth": depth,
            "canonical_nodes": len(nodes_d),
            "unique_surface_names": len(surfaces_d),
            "valid_terminal_paths": len(terminating_paths_d),
            "gold_path_assignments_ending_here": docs_ending_d,
        })
    node_stats = pd.DataFrame(node_rows)

    duplicates = []
    for surface, nodes in surface_to_nodes.items():
        if len(nodes) > 1:
            duplicates.append({
                "surface_name": surface,
                "canonical_node_count": len(nodes),
                "canonical_nodes": [list(x) for x in nodes],
            })
    duplicate_names = (
        pd.DataFrame(duplicates).sort_values("canonical_node_count", ascending=False)
        if duplicates else pd.DataFrame(columns=["surface_name", "canonical_node_count", "canonical_nodes"])
    )

    task_diagnosis = {
        "documents": int(len(df)),
        "documents_with_multiple_gold_paths": int((df["n_gold_paths"] > 1).sum()),
        "max_gold_paths_per_document": int(df["n_gold_paths"].max()),
        "mean_gold_paths_per_document": float(df["n_gold_paths"].mean()),
        "mean_path_depth_per_assignment": float(np.mean([len(p) for ps in df["paths"] for p in ps])),
        "canonical_nodes_total": len(all_nodes),
        "unique_surface_names_total": len(surface_to_nodes),
        "valid_terminal_paths_total": len(all_paths),
        "duplicate_surface_names": len(duplicate_names),
        "exact_duplicate_document_rows": int(df.duplicated("text_hash", keep=False).sum()),
        "task_is_genuinely_multilabel": bool((df["n_gold_paths"] > 1).any()),
    }

    audit_dir = Path(out_dir) / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    node_stats.to_csv(audit_dir / "nodes_by_depth.csv", index=False)
    duplicate_names.to_csv(audit_dir / "duplicate_surface_names.csv", index=False)
    with open(audit_dir / "task_diagnosis.json", "w", encoding="utf-8") as f:
        import json
        json.dump(task_diagnosis, f, ensure_ascii=False, indent=2)

    print(node_stats)
    print(task_diagnosis)
    print(duplicate_names.head(20))

    if task_diagnosis["task_is_genuinely_multilabel"]:
        raise ValueError(
            "This pipeline is intentionally single-path, but at least one document has multiple "
            "gold paths. Use a separately designed multi-label pipeline instead of silently "
            "changing the objective."
        )
    if not (df["n_gold_paths"] == 1).all():
        raise AssertionError("Every retained document must have exactly one complete gold path.")
    df["gold_path"] = df["paths"].str[0]
    print("TASK CONFIRMED: single-path hierarchical classification; one terminal path per article.")

    return {
        "df": df,
        "all_paths": all_paths,
        "all_nodes": all_nodes,
        "node_stats": node_stats,
        "duplicate_names": duplicate_names,
        "task_diagnosis": task_diagnosis,
        "max_depth": max_depth,
    }


def apply_smoke_sampling(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Sample rows for a fast pipeline smoke-test. Never used for reported
    numbers: the audit above always runs on the complete usable dataset
    first, so smoke sampling cannot overwrite paper statistics."""
    if cfg.run_mode == "smoke" and len(df) > cfg.smoke_rows:
        df = df.sample(cfg.smoke_rows, random_state=cfg.split_seed).reset_index(drop=True)
        print(f"SMOKE MODE: sampled {len(df):,} rows for training only. Set run_mode='paper' for publishable results.")
    return df
