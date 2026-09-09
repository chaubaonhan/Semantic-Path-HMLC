"""Encoder-robustness grid: retrain ``flat`` and ``semantic_path_hmlc`` with
several pretrained encoders on the same split/protocol (paper Section 6.6).
Unchanged from the original notebook.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd

from .config import Config


def run_encoder_robustness_grid(cfg: Config, train_one_fn: Callable[..., Dict[str, Any]], out_dir: Path) -> pd.DataFrame:
    """``train_one_fn(variant, seed, encoder_name=...)`` should be a partial
    application of :func:`semantic_path_hmlc.training.train_one` with every
    other argument (cfg, tax, df, ...) already bound."""
    encoder_list = (cfg.encoder_name,) if cfg.run_mode == "smoke" else cfg.encoder_robustness_variants
    n_enc_runs = len(encoder_list) * len(cfg.encoder_robustness_seeds) * 2
    print(
        f"Encoder robustness grid: {len(encoder_list)} encoders x {len(cfg.encoder_robustness_seeds)} "
        f"seed(s) x 2 variants = {n_enc_runs} runs (cached runs are skipped automatically)"
    )

    rows = []
    tables_dir = Path(out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for enc_name in encoder_list:
        for seed in cfg.encoder_robustness_seeds:
            for variant in ("flat", "semantic_path_hmlc"):
                rows.append(train_one_fn(variant, seed, encoder_name=enc_name))
                pd.DataFrame(rows).to_csv(tables_dir / "encoder_robustness.csv", index=False)

    encoder_robustness = pd.DataFrame(rows)
    summary_rows = []
    for (encoder, variant), group in encoder_robustness.groupby(["encoder", "variant"]):
        row = {"encoder": encoder, "variant": variant, "n_seeds": group["seed"].nunique()}
        for metric in (
            "exact_path_accuracy", "path_macro_f1_supported",
            "path_macro_f1_all_taxonomy", "hierarchical_f1", "hierarchical_f1_example_macro",
        ):
            mean, sd = group[metric].mean(), group[metric].std(ddof=1)
            row[metric] = f"{mean:.4f} ± {sd:.4f}" if len(group) > 1 else f"{mean:.4f}"
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables_dir / "encoder_robustness_mean_sd.csv", index=False)
    with open(tables_dir / "encoder_robustness_mean_sd.tex", "w", encoding="utf-8") as handle:
        handle.write(summary.to_latex(index=False, escape=True))
    return encoder_robustness
