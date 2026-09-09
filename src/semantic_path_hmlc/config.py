"""Experiment configuration.

This mirrors the ``Config`` dataclass used in the original research notebooks
exactly (same field names, defaults, hyperparameters, seeds, and variant
lists), with one deliberate change: filesystem paths no longer default to a
personal Google Drive mount (e.g. ``/content/drive/MyDrive/<project>/...``).
Instead they default to plain relative folders and can be overridden with
environment variables or CLI flags, so the same code runs unmodified on any
machine.

Environment variable overrides (all optional):
    SPHMLC_DATA_PATH         -> Config.data_path
    SPHMLC_OUTPUT_DIR        -> Config.output_dir
    SPHMLC_TOKEN_CACHE_DIR   -> Config.token_cache_dir
    SPHMLC_LOCAL_OUTPUT_DIR  -> Config.local_output_dir
    SPHMLC_RUN_MODE          -> Config.run_mode ("smoke" or "paper")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    # ---- Paths (generalized: no hardcoded personal Drive paths) -----------
    data_path: str = field(default_factory=lambda: _env("SPHMLC_DATA_PATH", "data/dataset.parquet"))
    output_dir: str = field(default_factory=lambda: _env("SPHMLC_OUTPUT_DIR", "outputs"))
    token_cache_dir: str = field(default_factory=lambda: _env("SPHMLC_TOKEN_CACHE_DIR", "token_cache"))
    local_output_dir: str = field(default_factory=lambda: _env("SPHMLC_LOCAL_OUTPUT_DIR", "local_runs"))

    # Schema: the released dataset schema is
    # ['index', 'src', 'sublabel', 'tag', 'day', 'month', 'year', 'source', 'content', 'author'].
    # 'sublabel' stores each article's full taxonomy path as a list, e.g. ["xe"] or
    # ["xe", "thi truong"] (root -> leaf). It is intentionally hardcoded (not
    # auto-detected) because its name is not a generic candidate below.
    text_col: Optional[str] = "content"
    year_col: Optional[str] = "year"
    paths_col: Optional[str] = "sublabel"     # each cell: path OR list of paths
    level_cols: Tuple[str, ...] = ()          # one path, e.g. ("level_1", ..., "level_5")
    path_delimiter: Optional[str] = None      # e.g. " > "; JSON/list paths need no delimiter

    encoder_name: str = "vinai/phobert-base-v2"
    max_length: int = 256       # PhoBERT does not support 512
    batch_size: int = 32        # fallback; auto-tuned for CUDA GPUs below
    eval_batch_size: int = 64
    grad_accum_steps: int = 1
    epochs: int = 5
    patience: int = 2
    lr_encoder: float = 2e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    hidden_dim: int = 256
    graph_layers: int = 2
    dropout: float = 0.15
    num_workers: int = 2
    attention_backend: str = "sdpa"  # falls back to eager when unsupported
    strict_determinism: bool = False    # seeded, but allows faster CUDA kernels

    # Single-path objective and post-hoc calibration.
    class_balance_power: float = 0.5  # inverse-sqrt class frequency for Path-HMLC
    max_class_weight: float = 10.0
    temperature_max_iter: int = 30
    label_max_length: int = 48
    label_encode_batch_size: int = 256
    semantic_loss_weight: float = 0.10
    semantic_temperature: float = 0.07

    # Reproducibility and execution.
    seeds: Tuple[int, ...] = (13, 21, 42)
    ablation_seeds: Tuple[int, ...] = (13, 21, 42)
    split_seed: int = 2026
    run_mode: str = field(default_factory=lambda: _env("SPHMLC_RUN_MODE", "paper"))
    smoke_rows: int = 4000
    primary_split: str = "stratified"  # or "temporal"

    # All variants below are valid for one mutually exclusive terminal path.
    # `hierarchical_softmax` is the graph-free conditional-softmax baseline.
    core_variants: Tuple[str, ...] = (
        "flat", "hierarchical_softmax", "label_semantic",
        "path_hmlc", "semantic_path_hmlc",
    )
    ablation_variants: Tuple[str, ...] = (
        "semantic_path_hmlc_wo_graph",
        "semantic_path_hmlc_wo_label_semantics",
        "semantic_path_hmlc_wo_shared_stop",
        "semantic_path_hmlc_wo_balance",
        "semantic_path_hmlc_wo_semantic_loss",
    )
    model_variants: Tuple[str, ...] = (
        "flat", "hierarchical_softmax", "label_semantic",
        "path_hmlc", "semantic_path_hmlc",
        "semantic_path_hmlc_wo_graph",
        "semantic_path_hmlc_wo_label_semantics",
        "semantic_path_hmlc_wo_shared_stop",
        "semantic_path_hmlc_wo_balance",
        "semantic_path_hmlc_wo_semantic_loss",
    )

    # Controlled zero-example protocol. Taxonomy and label text remain known,
    # while every document of held-out terminal paths is removed from training.
    run_controlled_zero_shot: bool = True
    controlled_split_seed: int = 3107
    controlled_holdout_fraction: float = 0.10
    controlled_dev_fraction: float = 0.25
    controlled_min_documents: int = 20
    controlled_min_depth: int = 2
    controlled_variants: Tuple[str, ...] = (
        "flat", "label_semantic", "path_hmlc", "semantic_path_hmlc"
    )
    controlled_seeds: Tuple[int, ...] = (13, 21, 42)
    run_few_shot: bool = True
    few_shot_values: Tuple[int, ...] = (1, 5, 10)

    # Encoder robustness: retrain flat and full Semantic Path-HMLC with several
    # pretrained encoders on the same split/protocol. PhoBERT-family encoders are
    # automatically word-segmented (pyvi) before tokenization; other encoders
    # receive raw text, matching each one's expected input format.
    encoder_robustness_variants: Tuple[str, ...] = (
        "vinai/phobert-base-v2",          # primary encoder, monolingual Vietnamese
        "Qualcomm-AI-Research/BamiBERT",  # monolingual Vietnamese, raw text, 2048-ctx
        "uitnlp/CafeBERT",                # XLM-R continued-pretrained on Vietnamese
        "xlm-roberta-base",               # multilingual
        "bert-base-multilingual-cased",   # multilingual (mBERT)
    )
    encoder_robustness_seeds: Tuple[int, ...] = (13, 21, 42)
    encoder_max_length_overrides: Dict[str, int] = field(default_factory=lambda: {
        "Qualcomm-AI-Research/BamiBERT": 512,
        "uitnlp/CafeBERT": 512,
        "xlm-roberta-base": 512,
        "bert-base-multilingual-cased": 512,
    })  # PhoBERT keeps CFG.max_length (256); other robustness-grid encoders use 512.


def autotune_for_device(cfg: Config) -> None:
    """GPU-aware batch-size auto-tuning, unchanged from the original notebooks.

    Only execution-speed knobs are touched here -- epochs, learning rates, and
    seeds are untouched, so this does not change what "paper mode" trains or
    how results are scored.
    """
    import os as _os

    import torch

    if not torch.cuda.is_available():
        return
    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    cpu_count = _os.cpu_count() or 2
    if total_mem_gb >= 70:        # A100-80GB, primary PhoBERT at length 256
        cfg.batch_size, cfg.eval_batch_size, cfg.grad_accum_steps = 512, 512, 1
    elif total_mem_gb >= 35:      # A100-40GB, primary PhoBERT at length 256
        cfg.batch_size, cfg.eval_batch_size, cfg.grad_accum_steps = 256, 256, 1
    elif total_mem_gb >= 20:      # L4 / V100
        cfg.batch_size, cfg.eval_batch_size, cfg.grad_accum_steps = 48, 96, 1
    # Smaller GPUs keep the conservative fallback declared above.
    cfg.num_workers = min(8, max(2, cpu_count - 1))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(
        f"GPU auto-tune: {gpu_name} ({total_mem_gb:.1f} GB, {cpu_count} CPUs) -> "
        f"batch_size={cfg.batch_size}, eval_batch_size={cfg.eval_batch_size}, "
        f"grad_accum_steps={cfg.grad_accum_steps} (effective train batch = "
        f"{cfg.batch_size * cfg.grad_accum_steps}), num_workers={cfg.num_workers}."
    )
