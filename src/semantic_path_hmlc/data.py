"""Conservative schema resolution and taxonomy-path parsing.

Logic is unchanged from the original notebooks. The only generalization is
that ``load_raw_dataset`` no longer tries to mount Google Drive; it simply
reads the parquet file at ``cfg.data_path`` (local path, mounted network
drive, or any other path the caller resolves beforehand).

The dataset itself (article text and gold taxonomy paths) is not included in
this repository -- see the README's Data Availability section.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import Config

TEXT_CANDIDATES = ("content", "text", "body", "article", "full_text", "description", "title_content")
YEAR_CANDIDATES = ("year", "publication_year", "published_year", "date_year")
PATH_CANDIDATES = ("paths", "label_paths", "gold_paths", "category_paths", "path", "label_path", "category_path")
LEVEL_PATTERNS = (
    re.compile(r"^(?:level|lvl|label|category|cat|path)[_\- ]?(\d+)$", re.I),
    re.compile(r"^l(\d+)$", re.I),
)


def first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {str(c).lower(): str(c) for c in columns}
    return next((lowered[x.lower()] for x in candidates if x.lower() in lowered), None)


def infer_level_cols(columns: Sequence[str]) -> Tuple[str, ...]:
    found = []
    for col in columns:
        for pattern in LEVEL_PATTERNS:
            match = pattern.match(str(col).strip())
            if match:
                found.append((int(match.group(1)), str(col)))
                break
    return tuple(c for _, c in sorted(found))


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and not x.strip():
        return True
    return False


def clean_label(x: Any) -> str:
    text = re.sub(r"\s+", " ", str(x)).strip()
    if not text:
        raise ValueError("Empty label after normalization")
    return text


def parse_literal_maybe(value: str) -> Any:
    value = value.strip()
    if not value:
        return value
    if value[0] not in "[({":
        return value
    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            return value


def normalize_paths_value(value: Any, delimiter: Optional[str]) -> List[Tuple[str, ...]]:
    """Return a list of gold paths, where every path is a tuple of labels."""
    if is_missing(value):
        return []
    if isinstance(value, str):
        value = parse_literal_maybe(value)
    if isinstance(value, str):
        if not delimiter:
            raise ValueError(
                "The path column contains plain strings. Set CFG.path_delimiter explicitly; "
                "automatic splitting is disabled because '/' and '>' may be part of a label."
            )
        return [tuple(clean_label(x) for x in value.split(delimiter) if str(x).strip())]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError(f"Unsupported path value type: {type(value)}")
    if len(value) == 0:
        return []

    # ["L1", "L2"] is one path; [["L1", "L2"], ["A", "B"]] is multi-path.
    if all(not isinstance(x, (list, tuple, np.ndarray, dict)) for x in value):
        return [tuple(clean_label(x) for x in value if not is_missing(x))]

    paths = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("path", item.get("labels", item.get("nodes")))
        if isinstance(item, str):
            item = parse_literal_maybe(item)
        if isinstance(item, str):
            if not delimiter:
                raise ValueError("A nested path is a string; set CFG.path_delimiter.")
            item = item.split(delimiter)
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if not isinstance(item, (list, tuple)):
            raise TypeError(f"Unsupported nested path type: {type(item)}")
        path = tuple(clean_label(x) for x in item if not is_missing(x))
        if path:
            paths.append(path)
    return sorted(set(paths))


def load_raw_dataset(cfg: Config) -> pd.DataFrame:
    """Read the parquet dataset at ``cfg.data_path``.

    The dataset is not distributed with this repository; provide your own
    parquet file with the schema documented in the README (or point
    ``cfg.data_path`` / ``SPHMLC_DATA_PATH`` at your copy).
    """
    if not Path(cfg.data_path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {cfg.data_path}. This repository does not "
            "redistribute the underlying news corpus -- see the README's "
            "Data Availability section."
        )
    df_raw = pd.read_parquet(cfg.data_path)
    print("shape:", df_raw.shape)
    print("columns:", df_raw.columns.tolist())
    return df_raw


def resolve_schema_and_build_frame(df_raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Resolve text/year/path columns and build the working frame.

    Returns a DataFrame with columns ``text``, ``year``, ``paths`` (a list of
    gold taxonomy paths per row -- exactly one for every row in the released
    dataset; see :mod:`semantic_path_hmlc.audit`).
    """
    text_col = cfg.text_col or first_existing(df_raw.columns, TEXT_CANDIDATES)
    year_col = cfg.year_col or first_existing(df_raw.columns, YEAR_CANDIDATES)
    paths_col = cfg.paths_col or first_existing(df_raw.columns, PATH_CANDIDATES)
    level_cols = cfg.level_cols or infer_level_cols(df_raw.columns)

    if text_col is None:
        raise ValueError(f"Cannot infer text column. Set cfg.text_col. Columns: {df_raw.columns.tolist()}")
    if paths_col is None and not level_cols:
        raise ValueError(
            "Cannot infer hierarchical labels. Set cfg.paths_col for a path/list-of-paths column "
            "or cfg.level_cols for one label column per depth."
        )
    if paths_col is not None and level_cols:
        print(f"Both paths_col={paths_col!r} and level_cols={level_cols} found; using paths_col.")

    df = pd.DataFrame({"text": df_raw[text_col].fillna("").astype(str)})
    if year_col:
        df["year"] = pd.to_numeric(df_raw[year_col], errors="coerce").astype("Int64")
    else:
        df["year"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    if paths_col:
        df["paths"] = [
            normalize_paths_value(x, cfg.path_delimiter)
            for x in tqdm(df_raw[paths_col], desc="parse paths")
        ]
    else:
        def row_to_one_path(row: pd.Series) -> List[Tuple[str, ...]]:
            path = tuple(clean_label(row[c]) for c in level_cols if not is_missing(row[c]))
            return [path] if path else []

        df["paths"] = [
            row_to_one_path(row)
            for _, row in tqdm(df_raw[list(level_cols)].iterrows(), total=len(df_raw), desc="parse levels")
        ]

    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[(df["text"].str.len() > 0) & (df["paths"].map(len) > 0)].reset_index(drop=True)

    print({
        "text_col": text_col, "year_col": year_col, "paths_col": paths_col,
        "level_cols": level_cols, "usable_rows": len(df),
    })
    return df
