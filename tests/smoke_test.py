"""Offline functional smoke test.

Exercises the taxonomy/action factorization, all four model forward/loss
paths, and the metrics module on tiny synthetic data -- without any network
access or GPU. Pretrained-encoder loading is monkeypatched with a small
random Transformer-like stub so this runs anywhere.

This is a development sanity check for this refactor, not a unit-test suite
for the paper's numerical claims (those require the real dataset and a GPU).

Run: python tests/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import semantic_path_hmlc.models as models_mod
from semantic_path_hmlc.config import Config
from semantic_path_hmlc.metrics import evaluate_arrays, fit_temperature, scores_to_probabilities
from semantic_path_hmlc.taxonomy import Taxonomy, compute_action_counts, compute_path_class_weights


# ---------------------------------------------------------------------------
# Stubs so this test needs no network access / real pretrained weights.
# ---------------------------------------------------------------------------
class _FakeEncoderConfig:
    hidden_size = 16


class _FakeEncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeEncoder(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=16):
        super().__init__()
        self.config = _FakeEncoderConfig()
        self.config.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids=None, attention_mask=None):
        return _FakeEncoderOutput(self.embed(input_ids))


class _FakeTokenizerOutput(dict):
    def __getattr__(self, item):
        return self[item]


class _FakeTokenizer:
    def __call__(self, texts, padding=True, truncation=True, max_length=48, return_tensors="pt"):
        n = len(texts)
        ids = torch.randint(1, 64, (n, 8))
        mask = torch.ones(n, 8, dtype=torch.long)
        return _FakeTokenizerOutput(input_ids=ids, attention_mask=mask)


def _fake_load_pretrained_encoder(cfg, encoder_name):
    return _FakeEncoder()


def _fake_tokenizer_from_pretrained(*args, **kwargs):
    return _FakeTokenizer()


def main():
    models_mod.load_pretrained_encoder = _fake_load_pretrained_encoder
    models_mod.AutoTokenizer.from_pretrained = staticmethod(_fake_tokenizer_from_pretrained)

    # ---- Taxonomy -----------------------------------------------------
    paths = [
        ("giao duc",), ("giao duc", "tuyen sinh"), ("giao duc", "tuyen sinh", "tu van"),
        ("the gioi",), ("the gioi", "quan su"),
        ("kinh doanh", "vi mo"), ("kinh doanh", "doanh nghiep"),
    ]
    tax = Taxonomy(paths)
    assert len(tax.paths) == len(paths)
    assert tax.stop_action_id[("giao duc",)] in tax.parent_action_ids[("giao duc",)]
    print(f"Taxonomy OK: {len(tax.nodes)} nodes, {len(tax.actions)} actions, {len(tax.paths)} paths")

    frame = pd.DataFrame({"paths": [[p] for p in paths] + [[paths[0]], [paths[2]]]})
    counts, parent_counts = compute_action_counts(frame, tax)
    assert counts.sum() > 0
    path_counts_train, weights = compute_path_class_weights(frame, tax, Config())
    assert weights.shape[0] == len(tax.paths)
    print("compute_action_counts / compute_path_class_weights OK")

    # ---- Models: forward + loss for all four architectures -------------
    cfg = Config(hidden_dim=16, dropout=0.1, graph_layers=2)
    batch = 4
    input_ids = torch.randint(1, 64, (batch, 10))
    attention_mask = torch.ones(batch, 10, dtype=torch.long)
    gold = torch.randint(0, len(tax.paths), (batch,))

    from semantic_path_hmlc.models import (
        FlatPathClassifier, LabelSemanticClassifier, PathHMLC, SemanticPathHMLC,
    )

    flat = FlatPathClassifier("dummy", len(tax.paths), cfg)
    out = flat(input_ids, attention_mask)
    loss, pieces = flat.compute_loss(out, gold)
    assert torch.isfinite(loss)
    print(f"FlatPathClassifier OK: loss={float(loss):.4f}")

    hsm = PathHMLC("dummy", tax, cfg, use_graph=False, use_balance=False,
                   default_path_class_weights=weights)
    out = hsm(input_ids, attention_mask)
    loss, _ = hsm.compute_loss(out, gold)
    assert torch.isfinite(loss)
    assert torch.allclose(out["path_log_probs"].exp().sum(dim=1), torch.ones(batch), atol=1e-3)
    print(f"PathHMLC (hierarchical_softmax config) OK: loss={float(loss):.4f}")

    ph = PathHMLC("dummy", tax, cfg, use_graph=True, use_balance=True,
                  default_path_class_weights=weights)
    out = ph(input_ids, attention_mask)
    loss, _ = ph.compute_loss(out, gold)
    assert torch.isfinite(loss)
    print(f"PathHMLC (full graph+balance) OK: loss={float(loss):.4f}")

    lsc = LabelSemanticClassifier("dummy", tax, cfg, default_path_class_weights=weights)
    out = lsc(input_ids, attention_mask)
    loss, _ = lsc.compute_loss(out, gold)
    assert torch.isfinite(loss)
    print(f"LabelSemanticClassifier OK: loss={float(loss):.4f}")

    sph = SemanticPathHMLC("dummy", tax, cfg, default_path_class_weights=weights)
    out = sph(input_ids, attention_mask)
    loss, pieces = sph.compute_loss(out, gold)
    assert torch.isfinite(loss)
    assert "semantic_ce" in pieces
    assert torch.allclose(out["path_log_probs"].exp().sum(dim=1), torch.ones(batch), atol=1e-3)
    print(f"SemanticPathHMLC (full) OK: loss={float(loss):.4f}, pieces={list(pieces)}")

    # Ablation configurations should also run without shape errors.
    for kwargs in [
        dict(use_graph=False), dict(use_label_semantics=False), dict(use_shared_stop=False),
        dict(use_balance=False), dict(use_semantic_loss=False),
    ]:
        m = SemanticPathHMLC("dummy", tax, cfg, default_path_class_weights=weights, **kwargs)
        out = m(input_ids, attention_mask)
        loss, _ = m.compute_loss(out, gold)
        assert torch.isfinite(loss)
    print("SemanticPathHMLC ablation configs OK")

    # ---- Metrics --------------------------------------------------------
    n_paths = len(tax.paths)
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(50, n_paths)).astype(np.float32)
    probs = scores_to_probabilities(scores)
    y_true = rng.integers(0, n_paths, size=50)
    metrics = evaluate_arrays(y_true, probs, tax)
    assert 0.0 <= metrics["exact_path_accuracy"] <= 1.0
    assert 0.0 <= metrics["hierarchical_f1"] <= 1.0
    temperature = fit_temperature(scores, y_true, torch.device("cpu"), max_iter=5)
    assert temperature > 0
    print(f"metrics OK: exact_acc={metrics['exact_path_accuracy']:.3f}, hf1={metrics['hierarchical_f1']:.3f}, T={temperature:.3f}")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
