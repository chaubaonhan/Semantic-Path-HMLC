"""Qualitative examples: gold path vs. categorical prediction (paper Section
7.1 / Table with hand-selected error cases). Unchanged from the original
notebook.

Note: running this against your own copy of the dataset will write article
text snippets into ``out_dir/tables/qualitative_examples__*``. Those files
are runtime outputs, not part of this repository, and should not be
published unless you have the right to redistribute the underlying text.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .taxonomy import Taxonomy


def qualitative_examples(
    df: pd.DataFrame, tax: Taxonomy, cfg: Config, out_dir: Path, run_identity_fn,
    variant: str = "semantic_path_hmlc", seed: Optional[int] = None,
    n_examples: int = 12, top_k_pred: int = 5, only_mismatches: bool = False,
    random_state: int = 2026,
) -> pd.DataFrame:
    """Create reproducible single-path qualitative and error-analysis tables."""
    seed = seed if seed is not None else cfg.seeds[0]
    run_id = run_identity_fn(variant, seed)
    npz_path = Path(out_dir) / "predictions" / f"{run_id}.npz"
    assert npz_path.exists(), f"No predictions found at {npz_path}. Train '{variant}' (seed {seed}) first."

    z = np.load(npz_path)
    y_true = z["y_true_path_id"].astype(np.int64)
    probs = z["probs"].astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True).clip(min=1e-12)
    row_ids = z["row_ids"].astype(np.int64)
    predicted = probs.argmax(axis=1)

    text_by_row = df["text"]
    rng = np.random.default_rng(random_state)
    order = np.arange(len(row_ids))
    if only_mismatches:
        order = order[y_true != predicted]
    rng.shuffle(order)
    order = order[:n_examples]

    rows = []
    for position in order:
        row_id = int(row_ids[position])
        gold_id = int(y_true[position])
        pred_id = int(predicted[position])
        top_idx = np.argsort(-probs[position])[:top_k_pred]
        rows.append({
            "position_in_npz": int(position),
            "row_id": row_id,
            "text_snippet": (
                str(text_by_row.loc[row_id])[:280] + "..."
                if len(str(text_by_row.loc[row_id])) > 280 else str(text_by_row.loc[row_id])
            ),
            "gold_path": " > ".join(tax.paths[gold_id]),
            "predicted_path": f'{" > ".join(tax.paths[pred_id])} (p={probs[position, pred_id]:.4f})',
            f"top_{top_k_pred}": " | ".join(
                f'#{rank}: {" > ".join(tax.paths[pid])} (p={probs[position, pid]:.4f})'
                for rank, pid in enumerate(top_idx, start=1)
            ),
            "exact_match": gold_id == pred_id,
            "probability_sum": float(probs[position].sum()),
        })

    table = pd.DataFrame(rows)
    out_stub = Path(out_dir) / "tables" / f"qualitative_examples__{run_id}"
    out_stub.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(f"{out_stub}.csv", index=False)
    table.to_json(f"{out_stub}.json", orient="records", force_ascii=False, indent=2)
    print(f"Saved {len(table)} qualitative examples -> {out_stub}.csv / .json")
    return table


def controlled_unseen_examples(
    df: pd.DataFrame, tax: Taxonomy, out_dir: Path, controlled_run_id_fn,
    seed: int = 13, n_examples: int = 12,
) -> pd.DataFrame:
    from .metrics import per_example_hierarchical_f1

    run_id = controlled_run_id_fn("semantic_path_hmlc", seed, 0)
    npz_path = Path(out_dir) / "predictions" / f"{run_id}.npz"
    if not npz_path.exists():
        print(f"Controlled prediction not found: {npz_path}")
        return pd.DataFrame()
    with np.load(npz_path) as z:
        y = z["unseen_y_true_path_id"].astype(np.int64)
        probs = z["unseen_probs"].astype(np.float32)
        row_ids = z["unseen_row_ids"].astype(np.int64)
    probs /= probs.sum(axis=1, keepdims=True).clip(min=1e-12)
    pred = probs.argmax(1)
    example_hf = per_example_hierarchical_f1(y, pred, tax)
    order = np.lexsort((row_ids, -example_hf, -(pred == y).astype(np.int8)))[:n_examples]
    rows = []
    for position in order:
        top_ids = np.argsort(-probs[position])[:5]
        row_id = int(row_ids[position])
        rows.append({
            "row_id": row_id,
            "text_snippet": str(df.loc[row_id, "text"])[:300],
            "gold_path": " > ".join(tax.paths[int(y[position])]),
            "predicted_path": " > ".join(tax.paths[int(pred[position])]),
            "exact_match": bool(pred[position] == y[position]),
            "hierarchical_f1": float(example_hf[position]),
            "top_5": " | ".join(
                f"{rank}: {' > '.join(tax.paths[int(path_id)])} ({probs[position, path_id]:.4f})"
                for rank, path_id in enumerate(top_ids, start=1)
            ),
        })
    table = pd.DataFrame(rows)
    tables_dir = Path(out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / f"controlled_unseen_examples__seed{seed}.csv", index=False)
    return table
