"""Small utilities shared across the pipeline (unchanged from the notebooks)."""
from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def seed_everything(seed: int, strict_determinism: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = strict_determinism
    torch.backends.cudnn.benchmark = not strict_determinism


def get_device_and_amp_dtype() -> Tuple[torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    return device, amp_dtype


def needs_word_segmentation(encoder_name: str) -> bool:
    return "phobert" in encoder_name.lower()


def encoder_tag(encoder_name: str) -> str:
    return encoder_name.rsplit("/", 1)[-1].lower().replace(".", "-")


def stable_text_hash(text: str) -> str:
    normalized = re.sub(r"[^\w]+", " ", text.lower(), flags=re.UNICODE).strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def all_prefixes(path: Tuple[str, ...]) -> List[Tuple[str, ...]]:
    return [path[:i] for i in range(1, len(path) + 1)]


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
