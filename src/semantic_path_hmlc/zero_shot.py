"""Controlled known-taxonomy zero-example and few-shot evaluation
(paper Section 5.6 / Algorithm 3 analogue). Unchanged from the original
notebook, with global state replaced by explicit arguments.

Held-out terminal paths retain their names and taxonomy positions, but all of
their documents are removed from training and masked from the training
softmax denominator. Generalized zero-shot (GZSL) evaluation uses the full
taxonomy output space; restricted (ZSL) evaluation normalizes only over the
held-out test paths.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import Config
from .metrics import evaluate_arrays, fit_temperature, per_example_hierarchical_f1, predict_scores, scores_to_probabilities
from .models import LabelSemanticClassifier, SemanticPathHMLC
from .taxonomy import Taxonomy
from .tokenization import (
    CachedTokenDataset, LengthBucketBatchSampler, cached_collate, load_token_cache, sequence_buckets,
)
from .training import build_model, build_optimizer


def one_path_id(frame: pd.DataFrame, taxonomy: Taxonomy) -> np.ndarray:
    return np.asarray([taxonomy.path_to_id[paths[0]] for paths in frame["paths"]], dtype=np.int64)


def round_robin_stratified_paths(candidate_ids: np.ndarray, taxonomy: Taxonomy, n_select: int, seed: int) -> List[int]:
    """Select paths across top-level categories and depths without document leakage."""
    rng = np.random.default_rng(seed)
    groups: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for path_id in candidate_ids:
        path = taxonomy.paths[int(path_id)]
        groups[(path[0], len(path))].append(int(path_id))
    for ids in groups.values():
        rng.shuffle(ids)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected: List[int] = []
    while len(selected) < n_select and any(groups[key] for key in keys):
        for key in keys:
            if groups[key] and len(selected) < n_select:
                selected.append(groups[key].pop())
    return selected


def build_controlled_protocol(frame: pd.DataFrame, taxonomy: Taxonomy, cfg: Config, split_col: str, out_dir: Path) -> Dict[str, Any]:
    path_ids = one_path_id(frame, taxonomy)
    total_counts = np.bincount(path_ids, minlength=len(taxonomy.paths))
    original_train = frame[split_col].eq("train").to_numpy()
    original_test = frame[split_col].eq("test").to_numpy()
    train_counts = np.bincount(path_ids[original_train], minlength=len(taxonomy.paths))

    # A held-out path must have real training support before removal and a sibling
    # terminal path that can remain supervised under the same immediate parent.
    parent_to_paths: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for path_id, path in enumerate(taxonomy.paths):
        parent_to_paths[path[:-1]].append(path_id)
    eligible = []
    for path_id, path in enumerate(taxonomy.paths):
        if len(path) < cfg.controlled_min_depth:
            continue
        if total_counts[path_id] < cfg.controlled_min_documents:
            continue
        if train_counts[path_id] == 0:
            continue
        siblings_with_train = sum(
            train_counts[sibling_id] > 0 for sibling_id in parent_to_paths[path[:-1]] if sibling_id != path_id
        )
        if siblings_with_train < 1:
            continue
        eligible.append(path_id)
    eligible = np.asarray(eligible, dtype=np.int64)
    if eligible.size < 4:
        raise RuntimeError(
            "Too few eligible controlled hold-out paths. Inspect the path-frequency "
            "table before lowering cfg.controlled_min_documents."
        )

    n_holdout = max(2, int(round(eligible.size * cfg.controlled_holdout_fraction)))
    n_holdout = min(n_holdout, eligible.size - 1)
    selected = round_robin_stratified_paths(eligible, taxonomy, n_holdout, cfg.controlled_split_seed)
    n_dev = max(1, int(round(len(selected) * cfg.controlled_dev_fraction)))
    n_dev = min(n_dev, len(selected) - 1)
    dev_path_ids = np.asarray(selected[:n_dev], dtype=np.int64)
    test_path_ids = np.asarray(selected[n_dev:], dtype=np.int64)
    all_holdout_ids = np.concatenate([dev_path_ids, test_path_ids])

    is_dev_path = np.isin(path_ids, dev_path_ids)
    is_test_path = np.isin(path_ids, test_path_ids)
    is_any_holdout = np.isin(path_ids, all_holdout_ids)
    controlled_train = frame.loc[original_train & ~is_any_holdout].copy()
    controlled_train_ids = one_path_id(controlled_train, taxonomy)
    controlled_train_counts = np.bincount(controlled_train_ids, minlength=len(taxonomy.paths))
    train_supported = controlled_train_counts > 0

    seen_val_mask = frame[split_col].eq("val").to_numpy() & ~is_any_holdout
    seen_val_mask &= train_supported[path_ids]
    seen_test_mask = original_test & ~is_any_holdout
    seen_test_mask &= train_supported[path_ids]

    protocol = {
        "train": controlled_train,
        "seen_val": frame.loc[seen_val_mask].copy(),
        "seen_test": frame.loc[seen_test_mask].copy(),
        "unseen_dev": frame.loc[is_dev_path].copy(),
        "unseen_test": frame.loc[is_test_path].copy(),
        "dev_path_ids": dev_path_ids,
        "test_path_ids": test_path_ids,
        "train_counts": controlled_train_counts,
    }

    manifest_rows = []
    for role, ids in (("unseen_dev", dev_path_ids), ("unseen_test", test_path_ids)):
        for path_id in ids:
            path_id = int(path_id)
            manifest_rows.append({
                "role": role, "path_id": path_id, "path": " > ".join(taxonomy.paths[path_id]),
                "depth": len(taxonomy.paths[path_id]),
                "documents_before_holdout": int(total_counts[path_id]),
                "original_train_documents_removed": int(train_counts[path_id]),
                "original_test_documents": int(np.logical_and(original_test, path_ids == path_id).sum()),
            })
    manifest = pd.DataFrame(manifest_rows)
    splits_dir = Path(out_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(splits_dir / "controlled_zero_shot_manifest.csv", index=False)
    protocol["manifest"] = manifest
    protocol_fingerprint = hashlib.sha256(
        json.dumps(
            {"split_seed": cfg.controlled_split_seed, "dev_path_ids": dev_path_ids.tolist(), "test_path_ids": test_path_ids.tolist()},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    protocol["fingerprint"] = protocol_fingerprint
    return protocol


def protocol_class_weights(train_frame: pd.DataFrame, taxonomy: Taxonomy, cfg: Config) -> torch.Tensor:
    counts = np.bincount(one_path_id(train_frame, taxonomy), minlength=len(taxonomy.paths))
    counts_t = torch.from_numpy(counts).float()
    weights = torch.zeros_like(counts_t)
    supported = counts_t > 0
    if supported.any():
        mean_count = counts_t[supported].mean()
        weights[supported] = (mean_count / counts_t[supported]).pow(cfg.class_balance_power).clamp(max=cfg.max_class_weight)
        weights[supported] /= weights[supported].mean()
    return weights


def make_frame_loader(
    frame: pd.DataFrame, tax: Taxonomy, cfg: Config, encoder_name: str, seed: int, shuffle: bool,
    dataset_fingerprint: str, n_rows: int, cache_ram: Dict[str, Any],
) -> DataLoader:
    max_length = cfg.encoder_max_length_overrides.get(encoder_name, cfg.max_length)
    token_cache = load_token_cache(encoder_name, cfg, dataset_fingerprint, n_rows, cache_ram)
    length_factor = max(1, math.ceil(max_length / cfg.max_length))
    batch_size = max(1, (cfg.batch_size if shuffle else cfg.eval_batch_size) // length_factor)
    boundaries = sequence_buckets(max_length)
    dataset = CachedTokenDataset(frame, tax, token_cache)
    common = dict(
        dataset=dataset, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        collate_fn=partial(cached_collate, n_paths=len(tax.paths), boundaries=boundaries),
        batch_sampler=LengthBucketBatchSampler(dataset.lengths, batch_size, boundaries, seed, shuffle=shuffle),
    )
    if cfg.num_workers > 0:
        common["prefetch_factor"] = 4
    return DataLoader(**common)


def add_few_shot_examples(protocol: Dict[str, Any], df: pd.DataFrame, tax: Taxonomy, k: int, seed: int):
    if k == 0:
        return protocol["train"].copy(), protocol["unseen_test"].copy(), []
    rng = np.random.default_rng(seed)
    unseen = protocol["unseen_test"]
    unseen_ids = one_path_id(unseen, tax)
    shot_indices: List[int] = []
    for path_id in protocol["test_path_ids"]:
        candidates = unseen.index[unseen_ids == int(path_id)].to_numpy()
        if len(candidates) <= k:
            raise RuntimeError(
                f"Path {int(path_id)} has {len(candidates)} documents, not enough for k={k} "
                "plus a non-empty evaluation remainder."
            )
        shot_indices.extend(rng.choice(candidates, size=k, replace=False).tolist())
    adapted_train = pd.concat([protocol["train"], df.loc[sorted(shot_indices)]], axis=0).sort_index()
    adapted_unseen_test = unseen.drop(index=shot_indices)
    return adapted_train, adapted_unseen_test, sorted(shot_indices)


def restricted_probabilities(probs: np.ndarray, allowed_ids: np.ndarray) -> np.ndarray:
    restricted = np.zeros_like(probs, dtype=np.float32)
    restricted[:, allowed_ids] = probs[:, allowed_ids]
    restricted /= restricted.sum(axis=1, keepdims=True).clip(min=1e-12)
    return restricted


def harmonic_mean(a: float, b: float) -> float:
    return float(2 * a * b / max(a + b, 1e-12))


def controlled_run_id(cfg: Config, variant: str, seed: int, k: int) -> str:
    mode_tag = "__smoke" if cfg.run_mode == "smoke" else ""
    return f"controlled_zs__{variant}__k{k}__split{cfg.controlled_split_seed}__seed{seed}{mode_tag}"


def train_controlled_one(
    variant: str, seed: int, protocol: Dict[str, Any], cfg: Config, tax: Taxonomy, df: pd.DataFrame,
    out_dir: Path, local_out_dir: Path, device: torch.device, amp_dtype: torch.dtype,
    dataset_fingerprint: str, cache_ram: Dict[str, Any], k: int = 0, force_retrain: bool = False,
) -> Dict[str, Any]:
    if variant not in cfg.controlled_variants:
        raise ValueError(f"Unsupported controlled variant: {variant}")
    out_dir, local_out_dir = Path(out_dir), Path(local_out_dir)
    run_id = controlled_run_id(cfg, variant, seed, k)
    drive_ckpt = out_dir / "checkpoints" / f"{run_id}.pt"
    metrics_path = out_dir / "tables" / f"metrics__{run_id}.json"
    drive_pred = out_dir / "predictions" / f"{run_id}.npz"
    local_ckpt = local_out_dir / "checkpoints" / drive_ckpt.name
    local_pred = local_out_dir / "predictions" / drive_pred.name
    for d in (drive_ckpt.parent, metrics_path.parent, drive_pred.parent, local_ckpt.parent, local_pred.parent):
        d.mkdir(parents=True, exist_ok=True)

    adapted_train, unseen_eval, shot_rows = add_few_shot_examples(protocol, df, tax, k, seed)
    run_fingerprint = hashlib.sha256((protocol["fingerprint"] + json.dumps(shot_rows)).encode()).hexdigest()
    if not force_retrain and metrics_path.exists() and drive_ckpt.exists() and drive_pred.exists():
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        from .training import METRIC_SCHEMA_VERSION, pipeline_version_for_variant
        if (
            cached.get("protocol_fingerprint") == run_fingerprint
            and cached.get("training_pipeline_version") == pipeline_version_for_variant(variant)
            and cached.get("metric_schema_version") == METRIC_SCHEMA_VERSION
        ):
            print(f"SKIP controlled cached result: {run_id}")
            return cached

    n_rows = len(df)
    loaders = {
        "train": make_frame_loader(adapted_train, tax, cfg, cfg.encoder_name, seed, True, dataset_fingerprint, n_rows, cache_ram),
        "val": make_frame_loader(protocol["seen_val"], tax, cfg, cfg.encoder_name, seed, False, dataset_fingerprint, n_rows, cache_ram),
        "seen_test": make_frame_loader(protocol["seen_test"], tax, cfg, cfg.encoder_name, seed, False, dataset_fingerprint, n_rows, cache_ram),
        "unseen_test": make_frame_loader(unseen_eval, tax, cfg, cfg.encoder_name, seed, False, dataset_fingerprint, n_rows, cache_ram),
    }
    class_weights = protocol_class_weights(adapted_train, tax, cfg)
    model = build_model(variant, seed, tax, cfg, default_path_class_weights=None, class_weights=class_weights).to(device)
    if isinstance(model, (SemanticPathHMLC, LabelSemanticClassifier)):
        model.initialize_label_features()
    optimizer, _ = build_optimizer(model, cfg, device)
    steps_per_epoch = math.ceil(len(loaders["train"]) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg.warmup_ratio), total_steps)
    scaler = GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)

    best_hf1, patience_left = -1.0, cfg.patience
    train_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        iterator = tqdm(loaders["train"], desc=f"{run_id} e{epoch}", dynamic_ncols=True)
        for step, batch in enumerate(iterator, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["path_ids"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                outputs = model(input_ids, attention_mask)
                loss, _ = model.compute_loss(outputs, targets)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            if step % cfg.grad_accum_steps == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            iterator.set_postfix(loss=f"{float(loss.detach()) * cfg.grad_accum_steps:.4f}")

        y_val, val_scores, _ = predict_scores(model, loaders["val"], device, amp_dtype)
        val_metrics = evaluate_arrays(y_val, scores_to_probabilities(val_scores), tax)
        if val_metrics["hierarchical_f1"] > best_hf1:
            best_hf1 = val_metrics["hierarchical_f1"]
            patience_left = cfg.patience
            torch.save({"model": model.state_dict(), "epoch": epoch}, local_ckpt)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    train_seconds = time.perf_counter() - train_start

    try:
        state = torch.load(local_ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(local_ckpt, map_location=device)
    model.load_state_dict(state["model"])
    y_val, val_scores, _ = predict_scores(model, loaders["val"], device, amp_dtype)
    temperature = fit_temperature(val_scores, y_val, device, cfg.temperature_max_iter)
    state["temperature"] = temperature
    torch.save(state, local_ckpt)

    y_seen, seen_scores, seen_rows = predict_scores(model, loaders["seen_test"], device, amp_dtype)
    y_unseen, unseen_scores, unseen_rows = predict_scores(model, loaders["unseen_test"], device, amp_dtype)
    p_seen = scores_to_probabilities(seen_scores, temperature)
    p_unseen = scores_to_probabilities(unseen_scores, temperature)
    p_unseen_restricted = restricted_probabilities(p_unseen, protocol["test_path_ids"])
    seen_metrics = evaluate_arrays(y_seen, p_seen, tax)
    unseen_gzsl = evaluate_arrays(y_unseen, p_unseen, tax, supported_labels=protocol["test_path_ids"])
    unseen_zsl = evaluate_arrays(y_unseen, p_unseen_restricted, tax, supported_labels=protocol["test_path_ids"])
    from .training import METRIC_SCHEMA_VERSION, pipeline_version_for_variant
    metrics: Dict[str, Any] = {
        "training_pipeline_version": pipeline_version_for_variant(variant),
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "protocol": "controlled_known_taxonomy_zero_example",
        "protocol_fingerprint": run_fingerprint,
        "variant": variant, "seed": seed, "k_shot": k,
        "best_epoch": int(state["epoch"]), "temperature": float(temperature),
        "train_seconds": float(train_seconds),
        "n_train_documents": int(len(adapted_train)),
        "n_seen_test_documents": int(len(y_seen)),
        "n_unseen_test_documents": int(len(y_unseen)),
        "n_unseen_test_paths": int(len(protocol["test_path_ids"])),
        **{f"seen_{name}": value for name, value in seen_metrics.items()},
        **{f"unseen_gzsl_{name}": value for name, value in unseen_gzsl.items()},
        **{f"unseen_zsl_{name}": value for name, value in unseen_zsl.items()},
    }
    metrics["gzsl_harmonic_exact_accuracy"] = harmonic_mean(seen_metrics["exact_path_accuracy"], unseen_gzsl["exact_path_accuracy"])
    metrics["gzsl_harmonic_macro_f1"] = harmonic_mean(seen_metrics["path_macro_f1_supported"], unseen_gzsl["path_macro_f1_supported"])
    metrics["gzsl_harmonic_hierarchical_f1"] = harmonic_mean(seen_metrics["hierarchical_f1"], unseen_gzsl["hierarchical_f1"])

    unseen_pred = p_unseen.argmax(1)
    unseen_example_hf = per_example_hierarchical_f1(y_unseen, unseen_pred, tax)
    per_path_rows = []
    for path_id in protocol["test_path_ids"]:
        mask = y_unseen == int(path_id)
        per_path_rows.append({
            "run_id": run_id, "variant": variant, "seed": seed, "k_shot": k,
            "path_id": int(path_id), "path": " > ".join(tax.paths[int(path_id)]),
            "n_documents": int(mask.sum()),
            "gzsl_exact_accuracy": float((unseen_pred[mask] == y_unseen[mask]).mean()),
            "gzsl_hierarchical_f1_example_macro": float(unseen_example_hf[mask].mean()),
        })
    pd.DataFrame(per_path_rows).to_csv(out_dir / "tables" / f"controlled_per_path__{run_id}.csv", index=False)

    np.savez_compressed(
        local_pred,
        seen_y_true_path_id=y_seen.astype(np.int64), seen_probs=p_seen.astype(np.float16), seen_row_ids=seen_rows.astype(np.int64),
        unseen_y_true_path_id=y_unseen.astype(np.int64), unseen_probs=p_unseen.astype(np.float16),
        unseen_restricted_probs=p_unseen_restricted.astype(np.float16), unseen_row_ids=unseen_rows.astype(np.int64),
        heldout_path_ids=protocol["test_path_ids"].astype(np.int64), few_shot_row_ids=np.asarray(shot_rows, dtype=np.int64),
        temperature=np.asarray(temperature, dtype=np.float32),
    )
    shutil.copy2(local_ckpt, drive_ckpt)
    shutil.copy2(local_pred, drive_pred)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    del model, optimizer, scheduler, scaler, loaders, state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def build_controlled_summary(controlled_results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    controlled_metric_cols = [
        "seen_exact_path_accuracy", "unseen_gzsl_exact_path_accuracy", "unseen_zsl_exact_path_accuracy",
        "gzsl_harmonic_exact_accuracy", "unseen_gzsl_path_macro_f1_supported",
        "unseen_gzsl_hierarchical_f1", "gzsl_harmonic_hierarchical_f1",
    ]
    rows = []
    for (variant, k_shot), group in controlled_results.groupby(["variant", "k_shot"]):
        row = {"variant": variant, "k_shot": int(k_shot), "n_seeds": int(group["seed"].nunique())}
        for metric in controlled_metric_cols:
            mean, sd = group[metric].mean(), group[metric].std(ddof=1)
            row[metric] = f"{mean:.4f} ± {sd:.4f}" if len(group) > 1 else f"{mean:.4f}"
        rows.append(row)
    summary = pd.DataFrame(rows)
    tables_dir = Path(out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "controlled_zero_few_shot_mean_sd.csv", index=False)
    with open(tables_dir / "controlled_zero_few_shot_mean_sd.tex", "w", encoding="utf-8") as handle:
        handle.write(summary.to_latex(index=False, escape=True))
    return summary
