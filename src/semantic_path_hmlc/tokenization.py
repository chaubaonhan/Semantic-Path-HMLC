"""Tokenization and token-cache management.

Phase 01 (build, ``build_token_cache``) tokenizes the whole dataset once per
encoder -- with CPU word segmentation for PhoBERT-family encoders -- and
writes one tensor cache per encoder to ``cfg.token_cache_dir``, plus a
dataset-fingerprint sidecar file.

Phase 02 (load, ``load_token_cache`` + ``CachedTokenDataset``) is
deliberately read-only with respect to token caches: a changed dataset, row
order, encoder, or maximum length raises an error instead of silently
training on misaligned tokens.

Unchanged from the original notebooks, except that the Drive-to-local-disk
copy step is now a generic "copy to a local cache directory" step (it still
matters off-Colab whenever the cache directory is a slower network mount and
a faster local scratch disk is available; set ``local_cache_root`` to the
same path as ``token_cache_dir`` to skip the copy entirely).
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .common import encoder_tag, needs_word_segmentation
from .config import Config
from .taxonomy import Taxonomy


def dataset_token_fingerprint(frame: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(f"rows={len(frame)}\n".encode())
    for text_hash in frame["text_hash"].astype(str):
        h.update(text_hash.encode())
        h.update(b"\n")
    return h.hexdigest()


def token_cache_paths(token_cache_dir: Path, encoder_name: str, max_length: int) -> Tuple[Path, Path]:
    stem = f"{encoder_tag(encoder_name)}__L{max_length}"
    token_cache_dir = Path(token_cache_dir)
    return token_cache_dir / f"{stem}.pt", token_cache_dir / f"{stem}.meta.json"


# --------------------------------------------------------------------------
# Phase 01: build token caches (run once, offline, before training)
# --------------------------------------------------------------------------

_WORKER_TOKENIZER = None
_WORKER_WORD_SEGMENT = False
_WORKER_MAX_LENGTH = None


def _iter_text_chunks(texts: List[str], chunk_size: int):
    for start in range(0, len(texts), chunk_size):
        yield start, texts[start:start + chunk_size]


def _copy_with_progress(src: Path, dst: Path, description: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".copying")
    total = src.stat().st_size
    with src.open("rb") as fin, tmp.open("wb") as fout, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=description
    ) as bar:
        while True:
            block = fin.read(16 * 1024 * 1024)
            if not block:
                break
            fout.write(block)
            bar.update(len(block))
    os.replace(tmp, dst)


def _existing_cache_is_current(cache_path: Path, meta_path: Path, expected: Dict[str, Any]) -> bool:
    if not cache_path.exists() or not meta_path.exists():
        return False
    try:
        current = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    keys = ("dataset_fingerprint", "encoder_name", "max_length", "rows")
    return all(current.get(k) == expected.get(k) for k in keys)


def _init_token_worker(encoder_name: str, max_length: int, word_segment: bool) -> None:
    global _WORKER_TOKENIZER, _WORKER_WORD_SEGMENT, _WORKER_MAX_LENGTH
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        encoder_name, use_fast=True, local_files_only=True
    )
    _WORKER_WORD_SEGMENT = word_segment
    _WORKER_MAX_LENGTH = max_length


def _encode_chunk_in_worker(task):
    start, texts = task
    if _WORKER_WORD_SEGMENT:
        from pyvi import ViTokenizer
        texts = [ViTokenizer.tokenize(t) for t in texts]
    enc = _WORKER_TOKENIZER(
        texts, padding="max_length", truncation=True, max_length=_WORKER_MAX_LENGTH,
        return_attention_mask=True, return_tensors="np",
    )
    ids = np.asarray(enc["input_ids"], dtype=np.int32)
    mask = np.asarray(enc["attention_mask"], dtype=np.uint8)
    return start, ids, mask


def _write_encoded_result(result, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> int:
    start, ids_np, mask_np = result
    stop = start + ids_np.shape[0]
    input_ids[start:stop].copy_(torch.from_numpy(ids_np))
    attention_mask[start:stop].copy_(torch.from_numpy(mask_np))
    return ids_np.shape[0]


def _encode_slow_parallel(texts, encoder_name, max_length, input_ids, attention_mask, progress, n_workers, max_in_flight, batch_rows):
    tasks = iter(_iter_text_chunks(texts, batch_rows))
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_token_worker,
        initargs=(encoder_name, max_length, needs_word_segmentation(encoder_name)),
    ) as executor:
        pending = set()
        for _ in range(max_in_flight):
            try:
                pending.add(executor.submit(_encode_chunk_in_worker, next(tasks)))
            except StopIteration:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                progress.update(_write_encoded_result(future.result(), input_ids, attention_mask))
                try:
                    pending.add(executor.submit(_encode_chunk_in_worker, next(tasks)))
                except StopIteration:
                    pass


def _encode_fast_rust(texts, tokenizer, max_length, input_ids, attention_mask, progress, n_workers, batch_rows):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(n_workers)
    for start, batch_texts in _iter_text_chunks(texts, batch_rows):
        enc = tokenizer(
            batch_texts, padding="max_length", truncation=True, max_length=max_length,
            return_attention_mask=True, return_tensors="np",
        )
        ids_np = np.asarray(enc["input_ids"], dtype=np.int32)
        mask_np = np.asarray(enc["attention_mask"], dtype=np.uint8)
        progress.update(_write_encoded_result((start, ids_np, mask_np), input_ids, attention_mask))


def build_token_cache(
    df: pd.DataFrame, cfg: Config, encoder_name: str,
    dataset_fingerprint: str, overwrite: bool = False,
    n_workers: int = None, slow_batch_rows: int = 2048, fast_batch_rows: int = 8192,
    local_build_dir: Path = None,
) -> Path:
    """Tokenize the whole dataset once for ``encoder_name`` and write a
    reusable tensor cache. Safe to re-run: skips a cache that already
    matches the current dataset fingerprint / encoder / max length."""
    max_length = cfg.encoder_max_length_overrides.get(encoder_name, cfg.max_length)
    cache_path, meta_path = token_cache_paths(cfg.token_cache_dir, encoder_name, max_length)
    expected = {
        "format_version": 3,
        "dataset_fingerprint": dataset_fingerprint,
        "encoder_name": encoder_name,
        "max_length": int(max_length),
        "rows": int(len(df)),
        "word_segmented": bool(needs_word_segmentation(encoder_name)),
    }
    if not overwrite and _existing_cache_is_current(cache_path, meta_path, expected):
        print(f"SKIP current cache: {cache_path} ({cache_path.stat().st_size / 2**30:.2f} GiB)")
        return cache_path

    n_workers = n_workers or max(1, min(32, (os.cpu_count() or 2) - 1))
    max_in_flight = max(2, n_workers * 2)

    tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    use_processes = needs_word_segmentation(encoder_name) or not tokenizer.is_fast
    texts = df["text"].tolist()
    n_rows = len(texts)
    input_ids = torch.empty((n_rows, max_length), dtype=torch.int32)
    attention_mask = torch.empty((n_rows, max_length), dtype=torch.uint8)
    started = time.perf_counter()
    mode = f"{n_workers} CPU processes" if use_processes else f"Rust/Rayon {n_workers} threads"
    print(f"\nEncoder: {encoder_name} | rows={n_rows:,} | max_length={max_length} | mode={mode}")

    with tqdm(total=n_rows, desc=f"tokenize {encoder_tag(encoder_name)}", unit="rows", dynamic_ncols=True) as progress:
        if use_processes:
            del tokenizer
            gc.collect()
            _encode_slow_parallel(
                texts, encoder_name, max_length, input_ids, attention_mask, progress,
                n_workers, max_in_flight, slow_batch_rows,
            )
            tokenizer_class = "worker-loaded slow/PhoBERT tokenizer"
        else:
            tokenizer_class = tokenizer.__class__.__name__
            _encode_fast_rust(texts, tokenizer, max_length, input_ids, attention_mask, progress, n_workers, fast_batch_rows)
            del tokenizer

    elapsed = time.perf_counter() - started
    meta = {
        **expected,
        "tokenizer_class": tokenizer_class,
        "parallel_mode": mode,
        "tokenize_workers": n_workers,
        "tokenization_seconds": elapsed,
        "rows_per_second": n_rows / max(elapsed, 1e-9),
        "input_ids_dtype": str(input_ids.dtype),
        "attention_mask_dtype": str(attention_mask.dtype),
    }
    local_dir = Path(local_build_dir) if local_build_dir else Path(cfg.token_cache_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / cache_path.name
    print(f"Saving tensor package: {local_path}")
    torch.save({"input_ids": input_ids, "attention_mask": attention_mask, "meta": meta}, local_path)
    if local_path != cache_path:
        _copy_with_progress(local_path, cache_path, f"copy cache {encoder_tag(encoder_name)}")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(meta_tmp, meta_path)
    if local_path != cache_path:
        local_path.unlink(missing_ok=True)
    print(
        f"DONE: {cache_path} ({cache_path.stat().st_size / 2**30:.2f} GiB) | "
        f"{meta['rows_per_second']:.1f} rows/s | {elapsed / 60:.1f} min"
    )
    del input_ids, attention_mask, texts
    gc.collect()
    return cache_path


# --------------------------------------------------------------------------
# Phase 02: load pre-tokenized tensors into RAM (training-time, read-only)
# --------------------------------------------------------------------------

def load_token_cache(
    encoder_name: str, cfg: Config, dataset_fingerprint: str, n_rows: int,
    cache_ram: Dict[str, Any], local_cache_root: Path = None,
) -> Dict[str, Any]:
    """Load a token cache built by :func:`build_token_cache`, verifying it
    still matches the current dataset/encoder/max-length. Raises instead of
    silently training on a stale or misaligned cache. ``cache_ram`` is a
    dict used to memoize already-loaded caches within one process."""
    if encoder_name in cache_ram:
        return cache_ram[encoder_name]

    max_length = cfg.encoder_max_length_overrides.get(encoder_name, cfg.max_length)
    cache_path, meta_path = token_cache_paths(cfg.token_cache_dir, encoder_name, max_length)
    if not cache_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing token cache for {encoder_name}: {cache_path}. "
            "Run scripts/01_tokenize_cache.py first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "dataset_fingerprint": dataset_fingerprint,
        "encoder_name": encoder_name,
        "max_length": int(max_length),
        "rows": int(n_rows),
    }
    mismatches = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
    if mismatches:
        raise ValueError(
            f"Stale or misaligned token cache for {encoder_name}: {mismatches}. "
            "Rebuild it with scripts/01_tokenize_cache.py; do not train on this cache."
        )

    local_root = Path(local_cache_root) if local_cache_root else Path(cfg.token_cache_dir)
    local_path = local_root / cache_path.name
    if local_path != cache_path and (not local_path.exists() or local_path.stat().st_size != cache_path.stat().st_size):
        _copy_with_progress(cache_path, local_path, f"cache -> local {encoder_tag(encoder_name)}")
    elif local_path == cache_path:
        local_path = cache_path
    print(f"Loading {local_path.name} into CPU RAM ...")
    try:
        payload = torch.load(local_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(local_path, map_location="cpu")
    if tuple(payload["input_ids"].shape) != (n_rows, max_length):
        raise ValueError(f"Unexpected input_ids shape: {tuple(payload['input_ids'].shape)}")
    if tuple(payload["attention_mask"].shape) != (n_rows, max_length):
        raise ValueError(f"Unexpected attention_mask shape: {tuple(payload['attention_mask'].shape)}")
    payload["lengths"] = payload["attention_mask"].sum(1).to(torch.int16)
    cache_ram[encoder_name] = payload
    size_gib = sum(x.numel() * x.element_size() for x in (payload["input_ids"], payload["attention_mask"])) / 2**30
    print(f"Token cache ready in RAM: {encoder_name} | {size_gib:.2f} GiB | {n_rows:,} rows x {max_length}")
    return payload


class CachedTokenDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, taxonomy: Taxonomy, token_cache: Dict[str, Any]):
        self.row_ids = frame.index.to_numpy(dtype=np.int64)
        self.path_ids = [taxonomy.path_to_id[paths[0]] for paths in frame["paths"]]
        self.input_ids = token_cache["input_ids"]
        self.attention_mask = token_cache["attention_mask"]
        row_tensor = torch.from_numpy(self.row_ids)
        self.lengths = token_cache["lengths"].index_select(0, row_tensor).numpy().astype(np.int32)

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, idx):
        row = int(self.row_ids[idx])
        return self.input_ids[row], self.attention_mask[row], self.path_ids[idx], row, int(self.lengths[idx])


def sequence_buckets(max_length: int) -> Tuple[int, ...]:
    return tuple(sorted({max(8, int(max_length * ratio)) for ratio in (0.25, 0.50, 0.75, 1.0)}))


class LengthBucketBatchSampler:
    def __init__(self, lengths, batch_size: int, boundaries: Sequence[int], seed: int, shuffle: bool):
        self.lengths = np.asarray(lengths)
        self.batch_size = int(batch_size)
        self.boundaries = np.asarray(boundaries)
        self.seed = int(seed)
        self.epoch = 0
        self.shuffle = bool(shuffle)
        self.bucket_ids = np.searchsorted(self.boundaries, self.lengths, side="left")
        if self.bucket_ids.max(initial=0) >= len(self.boundaries):
            raise ValueError("A cached sequence is longer than the final bucket boundary")

    def __len__(self):
        return sum(
            math.ceil(int((self.bucket_ids == bucket_id).sum()) / self.batch_size)
            for bucket_id in range(len(self.boundaries))
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for bucket_id in range(len(self.boundaries)):
            indices = np.flatnonzero(self.bucket_ids == bucket_id)
            if self.shuffle:
                rng.shuffle(indices)
            batches.extend(
                indices[i:i + self.batch_size].tolist()
                for i in range(0, len(indices), self.batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def cached_collate(batch, n_paths: int, boundaries: Sequence[int]):
    max_valid = max(x[4] for x in batch)
    crop_length = next(boundary for boundary in boundaries if max_valid <= boundary)
    path_ids = torch.tensor([x[2] for x in batch], dtype=torch.long)
    return {
        "input_ids": torch.stack([x[0][:crop_length] for x in batch], dim=0),
        "attention_mask": torch.stack([x[1][:crop_length] for x in batch], dim=0),
        "path_ids": path_ids,
        "row_ids": torch.tensor([x[3] for x in batch], dtype=torch.long),
        "sequence_length": crop_length,
    }
