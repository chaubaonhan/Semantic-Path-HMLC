"""Canonical taxonomy and action factorization (paper Section 3.2-3.3 /
Algorithm 1). Unchanged from the original notebook.

For every active parent node, actions are its children plus ``STOP`` when
that parent is itself a valid terminal path. A local conditional softmax
selects exactly one valid action; a complete path's log-probability is the
sum of its edge actions' and terminal STOP action's local log-probabilities.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from .common import all_prefixes
from .config import Config

ROOT: Tuple[str, ...] = tuple()


class Taxonomy:
    def __init__(self, paths: Sequence[Tuple[str, ...]]):
        self.paths = sorted(set(paths), key=lambda x: (len(x), x))
        self.nodes = [ROOT] + sorted({n for p in self.paths for n in all_prefixes(p)}, key=lambda x: (len(x), x))
        self.node_to_id = {n: i for i, n in enumerate(self.nodes)}
        self.children: Dict[Tuple[str, ...], List[Tuple[str, ...]]] = defaultdict(list)
        for node in self.nodes[1:]:
            self.children[node[:-1]].append(node)
        for parent in self.children:
            self.children[parent] = sorted(set(self.children[parent]))

        self.actions: List[Tuple[str, Tuple[str, ...], Optional[Tuple[str, ...]]]] = []
        self.parent_action_ids: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
        self.edge_action_id: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], int] = {}
        self.stop_action_id: Dict[Tuple[str, ...], int] = {}
        terminal_set = set(self.paths)
        for parent in self.nodes:
            for child in self.children.get(parent, []):
                aid = len(self.actions)
                self.actions.append(("EDGE", parent, child))
                self.parent_action_ids[parent].append(aid)
                self.edge_action_id[(parent, child)] = aid
            if parent != ROOT and parent in terminal_set:
                aid = len(self.actions)
                self.actions.append(("STOP", parent, None))
                self.parent_action_ids[parent].append(aid)
                self.stop_action_id[parent] = aid

        self.path_to_id = {p: i for i, p in enumerate(self.paths)}
        path_action_ids = []
        for p in self.paths:
            ids = []
            parent = ROOT
            for child in all_prefixes(p):
                ids.append(self.edge_action_id[(parent, child)])
                parent = child
            ids.append(self.stop_action_id[p])
            path_action_ids.append(ids)
        self.path_action_ids = path_action_ids
        self.max_actions_per_path = max(map(len, path_action_ids))
        matrix = torch.zeros(len(self.paths), len(self.actions), dtype=torch.float32)
        for pid, ids in enumerate(path_action_ids):
            matrix[pid, ids] = 1.0
        self.path_action_matrix = matrix
        node_matrix = torch.zeros(len(self.paths), len(self.nodes) - 1, dtype=torch.float32)
        for pid, path in enumerate(self.paths):
            for node in all_prefixes(path):
                node_matrix[pid, self.node_to_id[node] - 1] = 1.0
        self.path_node_matrix = node_matrix
        self.path_depths = torch.tensor([len(p) for p in self.paths], dtype=torch.long)

        edge_src, edge_dst = [], []
        for node in self.nodes[1:]:
            a, b = self.node_to_id[node[:-1]], self.node_to_id[node]
            edge_src += [a, b]
            edge_dst += [b, a]
        self.graph_src = torch.tensor(edge_src, dtype=torch.long)
        self.graph_dst = torch.tensor(edge_dst, dtype=torch.long)

    def active_parent_targets(self, paths: Sequence[Tuple[str, ...]]) -> Dict[Tuple[str, ...], List[int]]:
        positives: Dict[Tuple[str, ...], set] = defaultdict(set)
        for p in paths:
            parent = ROOT
            for child in all_prefixes(p):
                positives[parent].add(self.edge_action_id[(parent, child)])
                parent = child
            positives[parent].add(self.stop_action_id[p])
        return {parent: sorted(ids) for parent, ids in positives.items()}


def compute_action_counts(frame: pd.DataFrame, taxonomy: Taxonomy) -> Tuple[torch.Tensor, torch.Tensor]:
    counts = torch.zeros(len(taxonomy.actions), dtype=torch.float32)
    parent_counts: Dict[Tuple[str, ...], int] = Counter()
    for paths in frame["paths"]:
        targets = taxonomy.active_parent_targets(paths)
        for parent, positive_ids in targets.items():
            parent_counts[parent] += 1
            counts[positive_ids] += 1
    parent_count_per_action = torch.tensor(
        [parent_counts[parent] for _, parent, _ in taxonomy.actions], dtype=torch.float32
    )
    return counts, parent_count_per_action


def compute_path_class_weights(
    train_frame: pd.DataFrame, taxonomy: Taxonomy, cfg: Config
) -> torch.Tensor:
    """Capped inverse-square-root frequency weights at the complete-path
    level, renormalized to mean one over supported paths (Section 4,
    "Class balancing and training support")."""
    path_counts_train = torch.zeros(len(taxonomy.paths), dtype=torch.float32)
    for paths in train_frame["paths"]:
        for p in paths:
            path_counts_train[taxonomy.path_to_id[p]] += 1

    seen_path_mask = path_counts_train > 0
    path_class_weights = torch.zeros_like(path_counts_train)
    if seen_path_mask.any():
        mean_count = path_counts_train[seen_path_mask].mean()
        path_class_weights[seen_path_mask] = (
            mean_count / path_counts_train[seen_path_mask]
        ).pow(cfg.class_balance_power).clamp(max=cfg.max_class_weight)
        path_class_weights[seen_path_mask] /= path_class_weights[seen_path_mask].mean()
    print({
        "seen_train_paths": int(seen_path_mask.sum()),
        "unseen_train_paths": int((~seen_path_mask).sum()),
        "class_weight_min": float(path_class_weights[seen_path_mask].min()) if seen_path_mask.any() else None,
        "class_weight_max": float(path_class_weights[seen_path_mask].max()) if seen_path_mask.any() else None,
    })
    return path_counts_train, path_class_weights
