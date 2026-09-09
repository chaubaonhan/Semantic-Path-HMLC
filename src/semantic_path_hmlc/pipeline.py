"""Shared setup used by every pipeline script: load the dataset, run the
taxonomy audit, assign splits, build the taxonomy/action graph, and compute
class weights. Deterministic given the same ``Config`` (same dataset file,
same seeds), so it is safe to call independently from each stage script --
later stages do not need the earlier stage's in-memory state, only its
cached outputs on disk (token caches, checkpoints, metrics/predictions).
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from .audit import apply_smoke_sampling, run_taxonomy_audit
from .common import ensure_dirs, get_device_and_amp_dtype
from .config import Config, autotune_for_device
from .data import load_raw_dataset, resolve_schema_and_build_frame
from .splits import assign_splits
from .taxonomy import Taxonomy, compute_path_class_weights
from .tokenization import dataset_token_fingerprint


def prepare_dataset(cfg: Config) -> Dict[str, Any]:
    autotune_for_device(cfg)
    device, amp_dtype = get_device_and_amp_dtype()

    out_dir = Path(cfg.output_dir)
    local_out_dir = Path(cfg.local_output_dir)
    token_cache_dir = Path(cfg.token_cache_dir)
    ensure_dirs(
        out_dir, token_cache_dir, local_out_dir,
        out_dir / "audit", out_dir / "splits", out_dir / "checkpoints",
        out_dir / "predictions", out_dir / "tables",
        local_out_dir / "checkpoints", local_out_dir / "predictions", local_out_dir / "tables",
    )
    print(asdict(cfg))

    df_raw = load_raw_dataset(cfg)
    df = resolve_schema_and_build_frame(df_raw, cfg)
    del df_raw

    audit = run_taxonomy_audit(df, out_dir)
    df = audit["df"]
    df = apply_smoke_sampling(df, cfg)

    df, split_col = assign_splits(df, cfg, out_dir)
    train_paths = {p for ps in df.loc[df[split_col] == "train", "paths"] for p in ps}
    test_paths = {p for ps in df.loc[df[split_col] == "test", "paths"] for p in ps}
    unseen_test_paths = test_paths - train_paths

    tax = Taxonomy(audit["all_paths"])
    competing_stop_paths = [path for path in tax.paths if len(tax.children.get(path, [])) > 0]
    print({
        "nodes_including_root": len(tax.nodes), "actions": len(tax.actions),
        "valid_paths": len(tax.paths), "max_depth": audit["max_depth"],
        "terminal_paths_with_child_vs_stop_choice": len(competing_stop_paths),
        "terminal_leaf_paths_with_forced_stop": len(tax.paths) - len(competing_stop_paths),
    })
    if not competing_stop_paths:
        print("WARNING: every STOP is forced at a leaf; STOP cannot be claimed as an adaptive decision.")

    path_counts_train, path_class_weights = compute_path_class_weights(
        df[df[split_col] == "train"], tax, cfg
    )

    dataset_fingerprint = dataset_token_fingerprint(df)
    print(f"Dataset token fingerprint: {dataset_fingerprint}")

    return {
        "cfg": cfg,
        "device": device,
        "amp_dtype": amp_dtype,
        "out_dir": out_dir,
        "local_out_dir": local_out_dir,
        "token_cache_dir": token_cache_dir,
        "df": df,
        "split_col": split_col,
        "tax": tax,
        "task_diagnosis": audit["task_diagnosis"],
        "node_stats": audit["node_stats"],
        "duplicate_names": audit["duplicate_names"],
        "train_paths": train_paths,
        "test_paths": test_paths,
        "unseen_test_paths": unseen_test_paths,
        "competing_stop_paths": competing_stop_paths,
        "path_counts_train": path_counts_train,
        "path_class_weights": path_class_weights,
        "dataset_fingerprint": dataset_fingerprint,
        "cache_ram": {},
    }
