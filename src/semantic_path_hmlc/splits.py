"""Leakage-safe train/val/test splitting.

Exact-duplicate texts are assigned as groups so no duplicated document can
appear in more than one split. The primary split stratifies on each group's
rarest gold path (pooling rare strata) before a two-stage
``train_test_split``. Unchanged from the original notebook.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config


def rarest_path_key(paths: Sequence[Tuple[str, ...]], counts: Mapping[Tuple[str, ...], int]) -> str:
    path = min(paths, key=lambda p: (counts[p], p))
    return " / ".join(path)


def make_grouped_stratified_split(frame: pd.DataFrame, seed: int) -> pd.Series:
    path_counts = Counter(p for paths in frame["paths"] for p in paths)
    groups = []
    for h, g in frame.groupby("text_hash", sort=False):
        union_paths = sorted({p for ps in g["paths"] for p in ps})
        groups.append({"text_hash": h, "stratum": rarest_path_key(union_paths, path_counts), "n": len(g)})
    gdf = pd.DataFrame(groups)
    vc = gdf["stratum"].value_counts()
    gdf["safe_stratum"] = gdf["stratum"].where(gdf["stratum"].map(vc) >= 3, "__RARE__")
    stratify_1 = gdf["safe_stratum"] if gdf["safe_stratum"].value_counts().min() >= 2 else None
    train_g, temp_g = train_test_split(gdf, test_size=0.20, random_state=seed, stratify=stratify_1)
    vc2 = temp_g["safe_stratum"].value_counts()
    stratify_2 = temp_g["safe_stratum"] if len(vc2) > 1 and vc2.min() >= 2 else None
    val_g, test_g = train_test_split(temp_g, test_size=0.50, random_state=seed, stratify=stratify_2)
    mapping = {
        **{x: "train" for x in train_g.text_hash},
        **{x: "val" for x in val_g.text_hash},
        **{x: "test" for x in test_g.text_hash},
    }
    return frame["text_hash"].map(mapping)


def assign_splits(df: pd.DataFrame, cfg: Config, out_dir: Path) -> Tuple[pd.DataFrame, str]:
    """Adds ``split_stratified`` (and ``split_temporal`` when a usable ``year``
    column is present) to ``df``. Returns ``(df, split_col)`` where
    ``split_col`` is the column selected by ``cfg.primary_split``.
    """
    df = df.copy()
    df["split_stratified"] = make_grouped_stratified_split(df, cfg.split_seed)
    if df["year"].notna().all() and {2025, 2026}.issubset(set(df["year"].astype(int).unique())):
        df["split_temporal"] = np.select(
            [df["year"].astype(int).between(2019, 2024), df["year"].astype(int).eq(2025), df["year"].astype(int).eq(2026)],
            ["train", "val", "test"],
            default="outside",
        )
    else:
        df["split_temporal"] = "unavailable"

    split_col = f"split_{cfg.primary_split}"
    if split_col not in df or not {"train", "val", "test"}.issubset(set(df[split_col])):
        raise ValueError(
            f"Requested split {cfg.primary_split!r} is unavailable. "
            f"Counts: {df.get(split_col, pd.Series()).value_counts().to_dict()}"
        )

    splits_dir = Path(out_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    df[["text_hash", "year", "split_stratified", "split_temporal"]].to_parquet(
        splits_dir / "split_manifest.parquet", index=False
    )
    print(df[split_col].value_counts())

    train_paths = {p for ps in df.loc[df[split_col] == "train", "paths"] for p in ps}
    test_paths = {p for ps in df.loc[df[split_col] == "test", "paths"] for p in ps}
    unseen_test_paths = test_paths - train_paths
    print({"train_paths": len(train_paths), "test_paths": len(test_paths), "unseen_test_paths": len(unseen_test_paths)})

    return df, split_col
