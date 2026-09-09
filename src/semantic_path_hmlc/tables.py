"""Paper tables: mean +/- SD, depth/long-tail subgroups, seen/unseen
diagnostics, the same-checkpoint calibration ablation, and paired bootstrap
significance tests (paper Sections 6.1-6.4). Unchanged from the original
notebook.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .metrics import evaluate_arrays, per_example_hierarchical_f1, scores_to_probabilities
from .taxonomy import Taxonomy

METRIC_COLS = [
    "exact_path_accuracy", "path_macro_f1_supported",
    "path_macro_f1_all_taxonomy", "path_weighted_f1",
    "top_3_accuracy", "top_5_accuracy", "multiclass_nll",
    "multiclass_brier", "top_label_ece", "hierarchical_precision",
    "hierarchical_recall", "hierarchical_f1",
    "hierarchical_f1_example_macro",
]


def build_main_summary(results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    summary_rows = []
    for variant, group in results.groupby("variant"):
        row = {"variant": variant, "n_seeds": int(group["seed"].nunique())}
        for metric in METRIC_COLS:
            mean = group[metric].mean()
            sd = group[metric].std(ddof=1)
            row[metric] = f"{mean:.4f} ± {sd:.4f}" if len(group) > 1 else f"{mean:.4f}"
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    tables_dir = Path(out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "main_results_mean_sd.csv", index=False)
    with open(tables_dir / "main_results_mean_sd.tex", "w", encoding="utf-8") as handle:
        handle.write(summary.to_latex(index=False, escape=True))
    return summary


def subgroup_metrics(npz_path: Path, taxonomy: Taxonomy, path_counts_train) -> pd.DataFrame:
    z = np.load(npz_path)
    y = z["y_true_path_id"].astype(np.int64)
    probs = z["probs"].astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True).clip(min=1e-12)
    path_freq = path_counts_train.cpu().numpy()
    depths = taxonomy.path_depths.cpu().numpy()[y]
    out = []

    for depth in sorted(np.unique(depths)):
        mask = depths == depth
        metrics_at_depth = evaluate_arrays(y[mask], probs[mask], taxonomy)
        out.append({"group_type": "gold_depth", "group": str(int(depth)), "n_docs": int(mask.sum()), **metrics_at_depth})

    bins = [(0, 0), (1, 9), (10, 99), (100, 999), (1000, np.inf)]
    gold_frequency = path_freq[y]
    for lo, hi in bins:
        mask = (gold_frequency >= lo) & (gold_frequency <= hi)
        if mask.any():
            metrics_in_bin = evaluate_arrays(y[mask], probs[mask], taxonomy)
            out.append({
                "group_type": "train_path_frequency",
                "group": f"{lo}-{int(hi) if np.isfinite(hi) else 'inf'}",
                "n_docs": int(mask.sum()), "n_paths": int(np.unique(y[mask]).size),
                **metrics_in_bin,
            })
    return pd.DataFrame(out)


def build_seen_unseen_table(run_specs, run_identity_fn, out_dir: Path, taxonomy: Taxonomy, train_supported_mask: np.ndarray) -> pd.DataFrame:
    """``run_identity_fn(variant, seed)`` should return the same run_id used
    when training (see :func:`semantic_path_hmlc.training.run_identity`)."""
    seen_unseen_rows = []
    for variant, seed in run_specs:
        run_id = run_identity_fn(variant, seed)
        npz_path = Path(out_dir) / "predictions" / f"{run_id}.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path) as z:
            y = z["y_true_path_id"].astype(np.int64)
            probs = z["probs"].astype(np.float32)
        probs /= probs.sum(axis=1, keepdims=True).clip(min=1e-12)
        group_masks = {
            "all_test": np.ones(len(y), dtype=bool),
            "seen_train_path": train_supported_mask[y],
            "unseen_train_path": ~train_supported_mask[y],
        }
        for group_name, mask in group_masks.items():
            if not mask.any():
                continue
            group_metrics = evaluate_arrays(y[mask], probs[mask], taxonomy)
            seen_unseen_rows.append({
                "variant": variant, "seed": seed, "group": group_name,
                "n_docs": int(mask.sum()), "n_paths": int(np.unique(y[mask]).size),
                **group_metrics,
            })
    table = pd.DataFrame(seen_unseen_rows)
    tables_dir = Path(out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / "seen_unseen_metrics.csv", index=False)
    return table


def build_calibration_ablation(seeds, run_identity_fn, out_dir: Path, taxonomy: Taxonomy) -> pd.DataFrame:
    """Correct calibration ablation: evaluate T=1 and fitted T on the SAME
    checkpoint/logits. Temperature scaling cannot change argmax, exact
    accuracy, or any F1 score -- this is asserted below."""
    calibration_rows = []
    for variant in ("path_hmlc", "semantic_path_hmlc"):
        for seed in seeds:
            run_id = run_identity_fn(variant, seed)
            npz_path = Path(out_dir) / "predictions" / f"{run_id}.npz"
            if not npz_path.exists():
                continue
            with np.load(npz_path) as z:
                y = z["y_true_path_id"].astype(np.int64)
                calibrated_probs = z["probs"].astype(np.float32)
                temperature = float(np.asarray(z["temperature"]).reshape(-1)[0])
                raw_scores = z["raw_path_scores"].astype(np.float32) if "raw_path_scores" in z.files else None
            calibrated_probs /= calibrated_probs.sum(axis=1, keepdims=True).clip(min=1e-12)
            if raw_scores is None:
                raw_scores = temperature * np.log(calibrated_probs.clip(min=1e-12))
            uncalibrated_probs = scores_to_probabilities(raw_scores, temperature=1.0)
            recalibrated_probs = scores_to_probabilities(raw_scores, temperature=temperature)
            uncal = evaluate_arrays(y, uncalibrated_probs, taxonomy)
            cal = evaluate_arrays(y, recalibrated_probs, taxonomy)
            assert np.array_equal(uncalibrated_probs.argmax(1), recalibrated_probs.argmax(1)), (
                "Temperature scaling unexpectedly changed argmax"
            )
            for state_name, values in (("uncalibrated_T1", uncal), ("temperature_scaled", cal)):
                calibration_rows.append({
                    "variant": variant, "seed": seed, "state": state_name, "temperature": temperature,
                    "exact_path_accuracy": values["exact_path_accuracy"],
                    "path_macro_f1_supported": values["path_macro_f1_supported"],
                    "multiclass_nll": values["multiclass_nll"],
                    "multiclass_brier": values["multiclass_brier"],
                    "top_label_ece": values["top_label_ece"],
                })
    table = pd.DataFrame(calibration_rows)
    tables_dir = Path(out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / "calibration_same_checkpoint.csv", index=False)
    return table


def paired_bootstrap_delta(y: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray,
                            taxonomy: Taxonomy, n_boot: int = 2000, seed: int = 2026) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    pred_a = probs_a.argmax(axis=1)
    pred_b = probs_b.argmax(axis=1)
    exact_delta = (pred_b == y).astype(np.float64) - (pred_a == y).astype(np.float64)
    hier_delta = (
        per_example_hierarchical_f1(y, pred_b, taxonomy) - per_example_hierarchical_f1(y, pred_a, taxonomy)
    )
    sampled_exact, sampled_hier = [], []
    for _ in tqdm(range(n_boot), desc="paired bootstrap"):
        idx = rng.integers(0, len(y), len(y))
        sampled_exact.append(exact_delta[idx].mean())
        sampled_hier.append(hier_delta[idx].mean())
    return {
        "delta_exact_path_accuracy": float(exact_delta.mean()),
        "exact_ci95_low": float(np.quantile(sampled_exact, 0.025)),
        "exact_ci95_high": float(np.quantile(sampled_exact, 0.975)),
        "delta_hierarchical_f1": float(hier_delta.mean()),
        "hierarchical_ci95_low": float(np.quantile(sampled_hier, 0.025)),
        "hierarchical_ci95_high": float(np.quantile(sampled_hier, 0.975)),
    }


def run_paired_bootstrap_pair(name_a: str, name_b: str, seed: int, run_identity_fn, out_dir: Path, taxonomy: Taxonomy, out_name: str):
    run_a = run_identity_fn(name_a, seed)
    run_b = run_identity_fn(name_b, seed)
    a = np.load(Path(out_dir) / "predictions" / f"{run_a}.npz")
    b = np.load(Path(out_dir) / "predictions" / f"{run_b}.npz")
    assert np.array_equal(a["row_ids"], b["row_ids"]), "Prediction rows are not aligned"
    assert np.array_equal(a["y_true_path_id"], b["y_true_path_id"]), "Gold path ids differ"
    result = paired_bootstrap_delta(
        a["y_true_path_id"].astype(np.int64), a["probs"].astype(np.float32), b["probs"].astype(np.float32), taxonomy,
    )
    import json
    tables_dir = Path(out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / out_name, "w") as handle:
        json.dump(result, handle, indent=2)
    print(result)
    return result
