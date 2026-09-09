"""Single-path and hierarchical evaluation (paper Section 5.5).

Every prediction is one categorical distribution over the valid terminal
paths, decoded by argmax -- there are no per-label thresholds or fallback
decoding. Macro-F1 is reported both over labels supported by the evaluated
split (primary) and over the entire fixed taxonomy (secondary/diagnostic).
Calibration is measured with multiclass NLL, multiclass Brier score, and
top-label ECE. Unchanged from the original notebook.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from .taxonomy import Taxonomy


def scores_to_probabilities(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temperature = float(np.clip(temperature, 0.05, 20.0))
    scaled = scores.astype(np.float32, copy=True) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    np.exp(scaled, out=scaled)
    scaled /= scaled.sum(axis=1, keepdims=True).clip(min=1e-12)
    return scaled


def top_k_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    k = min(int(k), probs.shape[1])
    top = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float((top == y_true[:, None]).any(axis=1).mean())


def top_label_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    pred = probs.argmax(axis=1)
    confidence = probs[np.arange(len(y_true)), pred]
    correct = pred == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def hierarchical_scores(y_true: np.ndarray, y_pred: np.ndarray, taxonomy: Taxonomy):
    node_matrix = taxonomy.path_node_matrix.cpu().numpy().astype(bool)
    gold_nodes = node_matrix[y_true]
    pred_nodes = node_matrix[y_pred]
    intersection = np.logical_and(gold_nodes, pred_nodes).sum(axis=1)
    hp = float(intersection.sum() / np.maximum(pred_nodes.sum(), 1))
    hr = float(intersection.sum() / np.maximum(gold_nodes.sum(), 1))
    hf = float(2 * hp * hr / max(hp + hr, 1e-12))
    return hp, hr, hf


def per_example_hierarchical_f1(y_true: np.ndarray, y_pred: np.ndarray, taxonomy: Taxonomy) -> np.ndarray:
    node_matrix = taxonomy.path_node_matrix.cpu().numpy().astype(bool)
    gold_nodes = node_matrix[y_true]
    pred_nodes = node_matrix[y_pred]
    intersection = np.logical_and(gold_nodes, pred_nodes).sum(axis=1).astype(np.float64)
    precision = intersection / np.maximum(pred_nodes.sum(axis=1), 1)
    recall = intersection / np.maximum(gold_nodes.sum(axis=1), 1)
    return 2 * precision * recall / np.maximum(precision + recall, 1e-12)


def evaluate_arrays(y_true: np.ndarray, probs: np.ndarray, taxonomy: Taxonomy,
                     supported_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    if y_true.ndim != 1:
        raise ValueError(f"Expected one path id per document, got shape {y_true.shape}")
    if probs.shape != (len(y_true), len(taxonomy.paths)):
        raise ValueError(f"Probability shape mismatch: {probs.shape}")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=2e-4):
        raise ValueError(f"Softmax rows do not sum to one; max error={np.abs(row_sums - 1).max():.3g}")

    pred = probs.argmax(axis=1)
    all_taxonomy_labels = np.arange(len(taxonomy.paths), dtype=np.int64)
    if supported_labels is None:
        supported_labels = np.unique(y_true)
    supported_labels = np.asarray(supported_labels, dtype=np.int64)
    if supported_labels.ndim != 1 or supported_labels.size == 0:
        raise ValueError("supported_labels must contain at least one path id")
    hp, hr, hf = hierarchical_scores(y_true, pred, taxonomy)
    example_hf = per_example_hierarchical_f1(y_true, pred, taxonomy)
    true_probability = probs[np.arange(len(y_true)), y_true].clip(1e-12, 1.0)
    brier = (
        np.square(probs, dtype=np.float64).sum(axis=1)
        - 2.0 * true_probability + 1.0
    ).mean()
    return {
        "exact_path_accuracy": float((pred == y_true).mean()),
        # Primary macro-F1: average only over labels supported by this evaluation set.
        "path_macro_f1_supported": float(f1_score(
            y_true, pred, labels=supported_labels, average="macro", zero_division=0
        )),
        # Secondary diagnostic: average over every path in the fixed taxonomy.
        "path_macro_f1_all_taxonomy": float(f1_score(
            y_true, pred, labels=all_taxonomy_labels, average="macro", zero_division=0
        )),
        "path_weighted_f1": float(f1_score(
            y_true, pred, labels=all_taxonomy_labels, average="weighted", zero_division=0
        )),
        "top_3_accuracy": top_k_accuracy(y_true, probs, 3),
        "top_5_accuracy": top_k_accuracy(y_true, probs, 5),
        "multiclass_nll": float(-np.log(true_probability).mean()),
        "multiclass_brier": float(brier),
        "top_label_ece": top_label_ece(y_true, probs),
        "hierarchical_precision": hp,
        "hierarchical_recall": hr,
        "hierarchical_f1": hf,
        "hierarchical_f1_example_macro": float(example_hf.mean()),
        "n_supported_paths": int(supported_labels.size),
    }


@torch.inference_mode()
def predict_scores(model, loader, device: torch.device, amp_dtype: torch.dtype):
    model.eval()
    scores_all, true_all, rows_all = [], [], []
    for batch in tqdm(loader, desc="predict", leave=False):
        input_ids = batch["input_ids"].to(device, dtype=torch.long, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.long, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
        ):
            scores = model(input_ids, attention_mask)["path_logits"]
        scores_all.append(scores.float().cpu().numpy())
        true_all.append(batch["path_ids"].numpy().astype(np.int64, copy=False))
        rows_all.append(batch["row_ids"].numpy())
    return np.concatenate(true_all), np.concatenate(scores_all), np.concatenate(rows_all)


def fit_temperature(scores: np.ndarray, y_true: np.ndarray, device: torch.device, max_iter: int = 30) -> float:
    """Fit one positive temperature on validation NLL; argmax predictions never change."""
    logits = torch.from_numpy(scores).to(device, dtype=torch.float32)
    labels = torch.from_numpy(np.asarray(y_true, dtype=np.int64)).to(device)
    log_temperature = nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0).cpu())
    del logits, labels, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return temperature
