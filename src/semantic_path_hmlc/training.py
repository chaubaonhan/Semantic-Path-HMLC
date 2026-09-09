"""Model construction, data loaders, and the single-run training loop
(paper Section 5, Algorithm 2). Unchanged from the original notebook, with
global state (``CFG``, ``OUT``, ``df``, ``tax`` ...) replaced by explicit
function arguments so the pipeline can be driven from a script or another
process.
"""
from __future__ import annotations

import gc
import json
import math
import shutil
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .common import encoder_tag, seed_everything
from .config import Config
from .metrics import evaluate_arrays, fit_temperature, predict_scores, scores_to_probabilities
from .models import FlatPathClassifier, LabelSemanticClassifier, PathHMLC, SemanticPathHMLC
from .taxonomy import Taxonomy
from .tokenization import (
    CachedTokenDataset, LengthBucketBatchSampler, cached_collate, load_token_cache, sequence_buckets,
)

TRAIN_PIPELINE_VERSION = "single_path_conditional_softmax_v3"
SEMANTIC_PIPELINE_VERSION = "semantic_shared_child_stop_v1"
METRIC_SCHEMA_VERSION = "supported_macro_f1_v2"

SEMANTIC_VARIANTS = {
    "semantic_path_hmlc", "semantic_path_hmlc_wo_graph",
    "semantic_path_hmlc_wo_label_semantics",
    "semantic_path_hmlc_wo_shared_stop",
    "semantic_path_hmlc_wo_balance",
    "semantic_path_hmlc_wo_semantic_loss",
}


def pipeline_version_for_variant(variant: str) -> str:
    uses_label_semantics = variant == "label_semantic" or variant.startswith("semantic_")
    return SEMANTIC_PIPELINE_VERSION if uses_label_semantics else TRAIN_PIPELINE_VERSION


def build_model(
    variant: str, seed: int, tax: Taxonomy, cfg: Config,
    default_path_class_weights: Optional[torch.Tensor] = None,
    encoder_name: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
):
    seed_everything(seed, cfg.strict_determinism)
    encoder_name = encoder_name or cfg.encoder_name
    if variant == "flat":
        return FlatPathClassifier(encoder_name, len(tax.paths), cfg, class_weights=class_weights)
    if variant == "label_semantic":
        return LabelSemanticClassifier(
            encoder_name, tax, cfg, class_weights=class_weights,
            default_path_class_weights=default_path_class_weights,
        )
    if variant == "hierarchical_softmax":
        return PathHMLC(
            encoder_name, tax, cfg, use_graph=False, use_balance=False, use_calibration=True,
            class_weights=class_weights, default_path_class_weights=default_path_class_weights,
        )
    if variant == "path_hmlc":
        return PathHMLC(
            encoder_name, tax, cfg, use_graph=True, use_balance=True, use_calibration=True,
            class_weights=class_weights, default_path_class_weights=default_path_class_weights,
        )
    if variant in SEMANTIC_VARIANTS:
        return SemanticPathHMLC(
            encoder_name, tax, cfg,
            use_graph=variant != "semantic_path_hmlc_wo_graph",
            use_label_semantics=variant != "semantic_path_hmlc_wo_label_semantics",
            use_shared_stop=variant != "semantic_path_hmlc_wo_shared_stop",
            use_balance=variant != "semantic_path_hmlc_wo_balance",
            use_semantic_loss=variant != "semantic_path_hmlc_wo_semantic_loss",
            class_weights=class_weights, default_path_class_weights=default_path_class_weights,
        )
    raise ValueError(f"Unsupported single-path variant: {variant}")


def make_loaders(
    cfg: Config, tax: Taxonomy, df: pd.DataFrame, split_col: str,
    dataset_fingerprint: str, cache_ram: Dict[str, Any],
    encoder_name: Optional[str] = None, seed: int = 13,
    local_cache_root: Path = None,
) -> Dict[str, DataLoader]:
    encoder_name = encoder_name or cfg.encoder_name
    max_length = cfg.encoder_max_length_overrides.get(encoder_name, cfg.max_length)
    token_cache = load_token_cache(encoder_name, cfg, dataset_fingerprint, len(df), cache_ram, local_cache_root)
    length_factor = max(1, math.ceil(max_length / cfg.max_length))
    train_bs = max(1, cfg.batch_size // length_factor)
    eval_bs = max(1, cfg.eval_batch_size // length_factor)
    boundaries = sequence_buckets(max_length)
    collate = partial(cached_collate, n_paths=len(tax.paths), boundaries=boundaries)
    loaders = {}

    for split in ("train", "val", "test"):
        subset_frame = df[df[split_col] == split]
        ds = CachedTokenDataset(subset_frame, tax, token_cache)
        common = dict(
            dataset=ds, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=cfg.num_workers > 0, collate_fn=collate,
        )
        if cfg.num_workers > 0:
            common["prefetch_factor"] = 4
        split_batch_size = train_bs if split == "train" else eval_bs
        common["batch_sampler"] = LengthBucketBatchSampler(
            ds.lengths, split_batch_size, boundaries, seed, shuffle=split == "train"
        )
        loaders[split] = DataLoader(**common)
    print(
        f"Loaders {encoder_name}: train_batch={train_bs}, eval_batch={eval_bs}, "
        f"max_length={max_length}, buckets={boundaries}"
    )
    return loaders


def run_identity(cfg: Config, out_dir: Path, variant: str, seed: int, encoder_name: str):
    is_primary = encoder_name == cfg.encoder_name
    tag = "" if is_primary else f"__{encoder_tag(encoder_name)}"
    mode_tag = "__smoke" if cfg.run_mode == "smoke" else ""
    run_id = f"{variant}{tag}__seed{seed}{mode_tag}"
    ckpt = Path(out_dir) / "checkpoints" / f"{run_id}.pt"
    metrics_path = Path(out_dir) / "tables" / f"metrics__{run_id}.json"
    return run_id, ckpt, metrics_path


def build_optimizer(model, cfg: Config, device: torch.device):
    encoder_params, head_params = [], []
    for name, parameter in model.named_parameters():
        (encoder_params if name.startswith("encoder.") else head_params).append(parameter)
    groups = [
        {"params": encoder_params, "lr": cfg.lr_encoder},
        {"params": head_params, "lr": cfg.lr_head},
    ]
    kwargs = {"weight_decay": cfg.weight_decay}
    if device.type == "cuda":
        kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(groups, **kwargs)
        fused = bool(kwargs.get("fused", False))
    except (TypeError, RuntimeError) as exc:
        print(f"Fused AdamW unavailable; using standard AdamW: {exc}")
        kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(groups, **kwargs)
        fused = False
    return optimizer, fused


def train_one(
    variant: str, seed: int, cfg: Config, tax: Taxonomy, df: pd.DataFrame, split_col: str,
    out_dir: Path, local_out_dir: Path, device: torch.device, amp_dtype: torch.dtype,
    dataset_fingerprint: str, cache_ram: Dict[str, Any],
    default_path_class_weights: Optional[torch.Tensor],
    encoder_name: Optional[str] = None, force_retrain: bool = False,
    local_cache_root: Path = None,
) -> Dict[str, Any]:
    """Train (or reuse a cached result for) one (variant, seed[, encoder])
    run. Behavior, hyperparameters, checkpoint/metric caching, and the
    validation-temperature calibration step are unchanged from the original
    notebook."""
    out_dir, local_out_dir = Path(out_dir), Path(local_out_dir)
    encoder_name = encoder_name or cfg.encoder_name
    expected_pipeline_version = pipeline_version_for_variant(variant)
    run_id, drive_ckpt, metrics_path = run_identity(cfg, out_dir, variant, seed, encoder_name)
    local_ckpt = local_out_dir / "checkpoints" / drive_ckpt.name
    local_history = local_out_dir / "tables" / f"history__{run_id}.csv"
    drive_history = out_dir / "tables" / local_history.name
    local_pred = local_out_dir / "predictions" / f"{run_id}.npz"
    drive_pred = out_dir / "predictions" / local_pred.name
    for d in (drive_ckpt.parent, metrics_path.parent, local_ckpt.parent, local_history.parent,
              drive_history.parent, local_pred.parent, drive_pred.parent):
        d.mkdir(parents=True, exist_ok=True)

    if not force_retrain and metrics_path.exists() and drive_ckpt.exists():
        with open(metrics_path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("training_pipeline_version") == expected_pipeline_version and drive_pred.exists():
            if cached.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
                with np.load(drive_pred) as z:
                    cached_y = z["y_true_path_id"].astype(np.int64)
                    cached_probs = z["probs"].astype(np.float32)
                cached_probs /= cached_probs.sum(axis=1, keepdims=True).clip(min=1e-12)
                cached.update(evaluate_arrays(cached_y, cached_probs, tax))
                cached["metric_schema_version"] = METRIC_SCHEMA_VERSION
                with open(metrics_path, "w", encoding="utf-8") as handle:
                    json.dump(cached, handle, ensure_ascii=False, indent=2)
                print(f"REFRESHED metrics without retraining: {run_id}")
            print(f"SKIP (cached result found): {run_id} -> {metrics_path.name}")
            return cached
        print(f"RETRAIN stale/incompatible result: {run_id}")

    loaders = make_loaders(cfg, tax, df, split_col, dataset_fingerprint, cache_ram, encoder_name, seed, local_cache_root)
    model = build_model(variant, seed, tax, cfg, default_path_class_weights, encoder_name).to(device)
    if isinstance(model, (SemanticPathHMLC, LabelSemanticClassifier)):
        model.initialize_label_features()
    optimizer, fused_optimizer = build_optimizer(model, cfg, device)
    steps_per_epoch = math.ceil(len(loaders["train"]) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg.warmup_ratio), total_steps)
    scaler = GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_hf1, patience_left = -1.0, cfg.patience
    history = []
    train_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        from collections import Counter
        running = Counter()
        iterator = tqdm(loaders["train"], desc=f"{variant} s{seed} e{epoch}", dynamic_ncols=True)
        for step, batch in enumerate(iterator, start=1):
            input_ids = batch["input_ids"].to(device, dtype=torch.long, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long, non_blocking=True)
            gold_path_ids = batch["path_ids"].to(device, dtype=torch.long, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                outputs = model(input_ids, attention_mask)
                loss, pieces = model.compute_loss(outputs, gold_path_ids)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            if step % cfg.grad_accum_steps == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running["loss"] += float(loss.detach()) * cfg.grad_accum_steps
            for key, value in pieces.items():
                running[key] += float(value)
            iterator.set_postfix(loss=f"{running['loss'] / step:.4f}", tokens=batch["sequence_length"])

        y_val, val_scores, _ = predict_scores(model, loaders["val"], device, amp_dtype)
        p_val = scores_to_probabilities(val_scores, temperature=1.0)
        val_metrics = evaluate_arrays(y_val, p_val, tax)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value / len(loaders["train"]) for key, value in running.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        pd.DataFrame(history).to_csv(local_history, index=False)

        if val_metrics["hierarchical_f1"] > best_hf1:
            best_hf1 = val_metrics["hierarchical_f1"]
            patience_left = cfg.patience
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": asdict(cfg)}, local_ckpt)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    train_seconds = time.perf_counter() - train_start
    train_peak_mb = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    try:
        state = torch.load(local_ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(local_ckpt, map_location=device)
    model.load_state_dict(state["model"])

    # Fit a single validation temperature after model selection. This affects
    # confidence/calibration only; categorical argmax predictions are unchanged.
    calibration_start = time.perf_counter()
    y_val, val_scores, _ = predict_scores(model, loaders["val"], device, amp_dtype)
    use_calibration = bool(getattr(model, "use_calibration", True))
    temperature = fit_temperature(val_scores, y_val, device, cfg.temperature_max_iter) if use_calibration else 1.0
    calibration_seconds = time.perf_counter() - calibration_start
    state["temperature"] = temperature
    torch.save(state, local_ckpt)
    print(f"Validation temperature for {run_id}: {temperature:.4f}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    infer_start = time.perf_counter()
    y_test, test_scores, test_rows = predict_scores(model, loaders["test"], device, amp_dtype)
    p_test = scores_to_probabilities(test_scores, temperature)
    infer_seconds = time.perf_counter() - infer_start
    infer_peak_mb = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    metrics = evaluate_arrays(y_test, p_test, tax)
    metrics.update({
        "training_pipeline_version": expected_pipeline_version,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "problem_type": "single_path_hierarchical_classification",
        "run_mode": cfg.run_mode,
        "decoder": "categorical_argmax",
        "variant": variant,
        "seed": seed,
        "encoder": encoder_name,
        "best_epoch": int(state["epoch"]),
        "temperature": float(temperature),
        "train_seconds": train_seconds,
        "calibration_seconds": calibration_seconds,
        "inference_seconds": infer_seconds,
        "articles_per_second": len(y_test) / max(infer_seconds, 1e-9),
        "peak_gpu_mb": max(train_peak_mb, infer_peak_mb),
        "fused_optimizer": fused_optimizer,
        "attention_backend_requested": cfg.attention_backend,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    })

    predicted_path_id = p_test.argmax(axis=1).astype(np.int64)
    np.savez_compressed(
        local_pred,
        y_true_path_id=y_test.astype(np.int64),
        raw_path_scores=test_scores.astype(np.float16),
        probs=p_test.astype(np.float16),
        row_ids=test_rows.astype(np.int64),
        predicted_path_id=predicted_path_id,
        temperature=np.asarray(temperature, dtype=np.float32),
    )
    shutil.copy2(local_ckpt, drive_ckpt)
    shutil.copy2(local_pred, drive_pred)
    if local_history.exists():
        shutil.copy2(local_history, drive_history)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    del model, optimizer, scheduler, scaler, loaders, state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def default_run_specs(cfg: Config, has_competing_stop_paths: bool):
    """The (variant, seed) schedule used for the main paper run: core
    variants + clean ablations, each on their configured seeds."""
    if cfg.run_mode == "smoke":
        run_specs = [("flat", cfg.seeds[0]), ("path_hmlc", cfg.seeds[0])]
    else:
        run_specs = [
            *((variant, seed) for variant in cfg.core_variants for seed in cfg.seeds),
            *((variant, seed) for variant in cfg.ablation_variants for seed in cfg.ablation_seeds),
        ]
    run_specs = list(dict.fromkeys(run_specs))
    if not has_competing_stop_paths:
        run_specs = [spec for spec in run_specs if spec[0] != "semantic_path_hmlc_wo_shared_stop"]
    return run_specs
