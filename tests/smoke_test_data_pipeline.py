"""Offline functional smoke test for the data/audit/splits/tokenization
plumbing, using a tiny synthetic dataset shaped like the real schema
(``sublabel`` column holding a list-of-labels path per row). No network
access, no GPU, no real dataset needed.

Run: python tests/smoke_test_data_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from semantic_path_hmlc.audit import apply_smoke_sampling, run_taxonomy_audit
from semantic_path_hmlc.config import Config
from semantic_path_hmlc.data import resolve_schema_and_build_frame
from semantic_path_hmlc.splits import assign_splits
from semantic_path_hmlc.taxonomy import Taxonomy
from semantic_path_hmlc.tokenization import (
    CachedTokenDataset, LengthBucketBatchSampler, cached_collate, dataset_token_fingerprint, sequence_buckets,
)


def make_synthetic_raw(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    taxonomy_paths = [
        ["giao duc"], ["giao duc", "tuyen sinh"], ["giao duc", "tuyen sinh", "tu van"],
        ["the gioi"], ["the gioi", "quan su"],
        ["kinh doanh", "vi mo"], ["kinh doanh", "doanh nghiep"],
        ["giai tri", "am nhac"],
    ]
    rows = []
    for i in range(n):
        path = taxonomy_paths[rng.integers(0, len(taxonomy_paths))]
        rows.append({
            "content": f"bai bao so {i} noi dung ngau nhien " * rng.integers(3, 8),
            "sublabel": path,
            "year": int(rng.integers(2019, 2027)),
        })
    return pd.DataFrame(rows)


def main():
    cfg = Config(run_mode="paper")
    df_raw = make_synthetic_raw()
    df = resolve_schema_and_build_frame(df_raw, cfg)
    assert set(df.columns) >= {"text", "year", "paths"}
    print(f"resolve_schema_and_build_frame OK: {len(df)} rows")

    audit = run_taxonomy_audit(df, Path("/tmp/smoke_out"))
    df = audit["df"]
    assert audit["task_diagnosis"]["task_is_genuinely_multilabel"] is False
    print(f"run_taxonomy_audit OK: {len(audit['all_paths'])} valid paths, {len(audit['all_nodes'])} nodes")

    df = apply_smoke_sampling(df, cfg)  # no-op in paper mode
    df, split_col = assign_splits(df, cfg, Path("/tmp/smoke_out"))
    assert {"train", "val", "test"}.issubset(set(df[split_col]))
    print(f"assign_splits OK: split_col={split_col}, counts={df[split_col].value_counts().to_dict()}")

    tax = Taxonomy(audit["all_paths"])
    fp = dataset_token_fingerprint(df)
    assert isinstance(fp, str) and len(fp) == 64
    print(f"Taxonomy + dataset_token_fingerprint OK: fp={fp[:12]}...")

    # Fake a token cache directly in memory (skips real tokenizer/network).
    max_length = cfg.max_length
    n = len(df)
    lengths = np.random.default_rng(1).integers(4, max_length, size=n).astype(np.int32)
    token_cache = {
        "input_ids": torch.randint(1, 100, (n, max_length), dtype=torch.int32),
        "attention_mask": torch.zeros(n, max_length, dtype=torch.uint8),
        "lengths": torch.from_numpy(lengths).to(torch.int16),
    }
    for i, L in enumerate(lengths):
        token_cache["attention_mask"][i, :L] = 1

    subset = df[df[split_col] == "train"]
    ds = CachedTokenDataset(subset, tax, token_cache)
    boundaries = sequence_buckets(max_length)
    sampler = LengthBucketBatchSampler(ds.lengths, batch_size=8, boundaries=boundaries, seed=13, shuffle=True)
    batches = list(sampler)
    assert sum(len(b) for b in batches) == len(ds)
    batch_indices = batches[0]
    items = [ds[i] for i in batch_indices]
    collated = cached_collate(items, n_paths=len(tax.paths), boundaries=boundaries)
    assert collated["input_ids"].shape[0] == len(items)
    assert collated["path_ids"].shape[0] == len(items)
    print(f"CachedTokenDataset + LengthBucketBatchSampler + cached_collate OK: {len(batches)} batches, first batch shape={tuple(collated['input_ids'].shape)}")

    print("\nALL DATA-PIPELINE SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
