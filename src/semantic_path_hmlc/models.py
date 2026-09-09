"""Model architectures (paper Section 4, Algorithm 1).

``FlatPathClassifier``   -- non-hierarchical categorical baseline.
``PathHMLC``             -- taxonomy-graph conditional-softmax classifier
                             (also used, with ``use_graph=False,
                             use_balance=False``, as the graph-free
                             "Hierarchical Softmax" baseline).
``LabelSemanticClassifier`` -- dual-encoder label-description baseline.
``SemanticPathHMLC``     -- the proposed model: frozen full-prefix label
                             features, taxonomy-graph fusion, shared
                             EDGE/STOP scoring, semantic-logit fusion, and
                             an auxiliary semantic objective.

All logic (forward passes, loss functions, hyperparameter defaults) is
unchanged from the original research notebook.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .config import Config
from .common import needs_word_segmentation
from .taxonomy import Taxonomy


def load_pretrained_encoder(cfg: Config, encoder_name: str):
    if cfg.attention_backend == "sdpa":
        try:
            return AutoModel.from_pretrained(encoder_name, attn_implementation="sdpa")
        except (TypeError, ValueError, NotImplementedError) as exc:
            print(f"SDPA unavailable for {encoder_name}; falling back to eager attention: {exc}")
    return AutoModel.from_pretrained(encoder_name)


def taxonomy_node_texts(taxonomy: Taxonomy, encoder_name: str) -> List[str]:
    """Full-prefix label descriptions disambiguate repeated node names."""
    texts = ["phân loại tin tức"] + [" > ".join(node) for node in taxonomy.nodes[1:]]
    if needs_word_segmentation(encoder_name):
        from pyvi import ViTokenizer
        texts = [ViTokenizer.tokenize(text) for text in texts]
    return texts


class TreeGraphEncoder(nn.Module):
    def __init__(self, n_nodes: int, dim: int, layers: int, dropout: float,
                 src: torch.Tensor, dst: torch.Tensor):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.self_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.neighbor_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)

    def forward(self, use_graph: bool = True,
                initial_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.embedding.weight if initial_features is None else initial_features
        if not use_graph:
            return h
        for self_fc, nbr_fc, norm in zip(
            self.self_layers, self.neighbor_layers, self.norms
        ):
            agg = torch.zeros_like(h)
            deg = torch.zeros(h.size(0), 1, device=h.device, dtype=h.dtype)
            agg.index_add_(0, self.dst, h[self.src])
            deg.index_add_(
                0, self.dst,
                torch.ones(self.dst.numel(), 1, device=h.device, dtype=h.dtype),
            )
            nbr = agg / deg.clamp_min(1.0)
            h = norm(h + self.dropout(F.gelu(self_fc(h) + nbr_fc(nbr))))
        return h


class PathHMLC(nn.Module):
    """Graph-enhanced conditional-softmax classifier over valid terminal paths.

    At every active taxonomy parent exactly one action is selected: descend to one
    child, or STOP when the current node is a valid terminal. Local action
    probabilities are normalized among siblings, and complete-path log-probability is
    the sum of its local log-probabilities. The final renormalization is a numerical
    safeguard and gives one categorical distribution over all valid paths.

    Used with ``use_graph=False, use_balance=False`` as the "Hierarchical
    Softmax" baseline in Table 1 of the paper.
    """

    def __init__(self, encoder_name: str, taxonomy: Taxonomy, cfg: Config,
                 use_graph: bool = True, use_balance: bool = True,
                 use_calibration: bool = True,
                 class_weights: Optional[torch.Tensor] = None,
                 default_path_class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.tax = taxonomy
        self.use_graph = use_graph
        self.use_balance = use_balance
        self.use_calibration = use_calibration
        self.restrict_training_to_supported = class_weights is not None
        self.encoder = load_pretrained_encoder(cfg, encoder_name)
        enc_dim = self.encoder.config.hidden_size
        self.doc_proj = nn.Sequential(
            nn.Linear(enc_dim, cfg.hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.tree_encoder = TreeGraphEncoder(
            len(taxonomy.nodes), cfg.hidden_dim, cfg.graph_layers, cfg.dropout,
            taxonomy.graph_src, taxonomy.graph_dst,
        )
        self.parent_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.child_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.edge_bias = nn.Parameter(torch.zeros(len(taxonomy.actions)))
        self.stop_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2 + 3, cfg.hidden_dim), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, 1),
        )
        self.register_buffer("path_action_matrix", taxonomy.path_action_matrix)
        effective_weights = (
            default_path_class_weights if class_weights is None else class_weights
        )
        if effective_weights is None:
            effective_weights = torch.ones(len(taxonomy.paths))
        self.register_buffer("path_class_weights", effective_weights.detach().clone())
        self.register_buffer("train_supported_path_mask", effective_weights > 0)

        parent_keys = list(taxonomy.parent_action_ids)
        parent_to_group = {parent: i for i, parent in enumerate(parent_keys)}
        action_parent_group = []
        action_parent_ids, action_child_ids, action_is_stop, action_depth = [], [], [], []
        for kind, parent, child in taxonomy.actions:
            action_parent_group.append(parent_to_group[parent])
            action_parent_ids.append(taxonomy.node_to_id[parent])
            action_child_ids.append(taxonomy.node_to_id[child] if child is not None else 0)
            action_is_stop.append(kind == "STOP")
            action_depth.append(len(parent))
        self.n_parent_groups = len(parent_keys)
        self.register_buffer("action_parent_group", torch.tensor(action_parent_group, dtype=torch.long))
        self.register_buffer("action_parent_ids", torch.tensor(action_parent_ids, dtype=torch.long))
        self.register_buffer("action_child_ids", torch.tensor(action_child_ids, dtype=torch.long))
        self.register_buffer("action_is_stop", torch.tensor(action_is_stop, dtype=torch.bool))
        self.register_buffer("action_depth", torch.tensor(action_depth, dtype=torch.float32))
        self.register_buffer(
            "stop_action_indices",
            torch.tensor([i for i, value in enumerate(action_is_stop) if value], dtype=torch.long),
        )
        self.max_path_depth = max(1, int(taxonomy.path_depths.max().item()))

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.doc_proj(pooled)

    def action_logits(self, doc: torch.Tensor) -> torch.Tensor:
        node_h = self.tree_encoder(self.use_graph)
        parent_h = node_h[self.action_parent_ids]
        child_h = node_h[self.action_child_ids]
        edge_context = self.parent_proj(parent_h) + doc[:, None, :]
        logits = (
            edge_context * self.child_proj(child_h)[None, :, :]
        ).sum(-1) / math.sqrt(doc.size(-1))
        logits = logits + self.edge_bias

        stop_idx = self.stop_action_indices
        ph = parent_h[stop_idx].unsqueeze(0).expand(doc.size(0), -1, -1)
        query = doc.unsqueeze(1).expand(-1, stop_idx.numel(), -1)
        depth = (self.action_depth[stop_idx] / self.max_path_depth).view(1, -1, 1)
        depth = depth.expand(doc.size(0), -1, -1)
        compatibility = F.cosine_similarity(query, ph, dim=-1).unsqueeze(-1)
        query_norm = query.norm(dim=-1, keepdim=True) / math.sqrt(query.size(-1))
        stop_features = torch.cat([query, ph, depth, compatibility, query_norm], dim=-1)
        stop_values = self.stop_mlp(stop_features).squeeze(-1) + self.edge_bias[stop_idx]
        # AMP may produce bfloat16 heads while the preallocated logits are float32.
        logits[:, stop_idx] = stop_values.to(dtype=logits.dtype)
        return logits

    def conditional_log_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Segment log-softmax over the actions belonging to each taxonomy parent."""
        groups = self.action_parent_group.unsqueeze(0).expand(logits.size(0), -1)
        maxima = logits.new_full((logits.size(0), self.n_parent_groups), -torch.inf)
        maxima.scatter_reduce_(1, groups, logits, reduce="amax", include_self=True)
        shifted = logits - maxima.gather(1, groups)
        denominators = logits.new_zeros((logits.size(0), self.n_parent_groups))
        denominators.scatter_add_(1, groups, shifted.exp())
        return shifted - denominators.gather(1, groups).clamp_min(1e-12).log()

    def forward(self, input_ids, attention_mask):
        doc = self.encode(input_ids, attention_mask)
        action_logits = self.action_logits(doc)
        action_log_probs = self.conditional_log_softmax(action_logits)
        path_log_probs = action_log_probs @ self.path_action_matrix.t()
        path_log_probs = path_log_probs - torch.logsumexp(path_log_probs, dim=1, keepdim=True)
        return {
            "action_logits": action_logits,
            "action_log_probs": action_log_probs,
            "path_log_probs": path_log_probs,
            "path_logits": path_log_probs,
        }

    def compute_loss(self, outputs, gold_path_ids):
        target = gold_path_ids.to(device=outputs["path_log_probs"].device, dtype=torch.long)
        path_log_probs = outputs["path_log_probs"]
        if self.restrict_training_to_supported:
            path_log_probs = path_log_probs.masked_fill(
                ~self.train_supported_path_mask.unsqueeze(0), -torch.inf
            )
            path_log_probs = path_log_probs - torch.logsumexp(path_log_probs, dim=1, keepdim=True)
        weights = self.path_class_weights.to(path_log_probs.dtype) if self.use_balance else None
        loss = F.nll_loss(path_log_probs, target, weight=weights)
        return loss, {"categorical_nll": loss.detach()}


class LabelSemanticClassifier(nn.Module):
    """Dual-encoder label-description baseline for controlled zero-shot tests."""

    def __init__(self, encoder_name: str, taxonomy: Taxonomy, cfg: Config,
                 class_weights: Optional[torch.Tensor] = None,
                 default_path_class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.tax = taxonomy
        self.use_calibration = True
        self.restrict_training_to_supported = class_weights is not None
        self.encoder = load_pretrained_encoder(cfg, encoder_name)
        enc_dim = self.encoder.config.hidden_size
        self.doc_proj = nn.Sequential(
            nn.Linear(enc_dim, cfg.hidden_dim), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.LayerNorm(cfg.hidden_dim),
        )
        self.label_proj = nn.Sequential(
            nn.Linear(enc_dim, cfg.hidden_dim), nn.GELU(), nn.LayerNorm(cfg.hidden_dim)
        )
        self.logit_scale_raw = nn.Parameter(torch.tensor(2.0))
        try:
            tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
        except (TypeError, ValueError):
            tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=False)
        encoded = tokenizer(
            taxonomy_node_texts(taxonomy, encoder_name), padding=True, truncation=True,
            max_length=cfg.label_max_length, return_tensors="pt",
        )
        self.register_buffer("label_input_ids", encoded["input_ids"], persistent=False)
        self.register_buffer("label_attention_mask", encoded["attention_mask"], persistent=False)
        self.register_buffer(
            "raw_label_features", torch.zeros(len(taxonomy.nodes), enc_dim), persistent=True
        )
        self.register_buffer("label_features_ready", torch.tensor(False), persistent=True)
        self.register_buffer(
            "terminal_node_ids",
            torch.tensor([taxonomy.node_to_id[path] for path in taxonomy.paths], dtype=torch.long),
        )
        effective_weights = (
            default_path_class_weights if class_weights is None else class_weights
        )
        if effective_weights is None:
            effective_weights = torch.ones(len(taxonomy.paths))
        self.register_buffer("path_class_weights", effective_weights.detach().clone())
        self.register_buffer("train_supported_path_mask", effective_weights > 0)

    @torch.no_grad()
    def initialize_label_features(self):
        if bool(self.label_features_ready.item()):
            return
        was_training = self.encoder.training
        self.encoder.eval()
        rows = []
        for start in range(0, len(self.label_input_ids), self.cfg.label_encode_batch_size):
            ids = self.label_input_ids[start:start + self.cfg.label_encode_batch_size]
            mask = self.label_attention_mask[start:start + self.cfg.label_encode_batch_size]
            output = self.encoder(input_ids=ids, attention_mask=mask)
            float_mask = mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
            pooled = (output.last_hidden_state * float_mask).sum(1) / float_mask.sum(1).clamp_min(1)
            rows.append(pooled.float())
        self.raw_label_features.copy_(torch.cat(rows, dim=0))
        self.label_features_ready.fill_(True)
        self.encoder.train(was_training)

    def encode(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
        pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.doc_proj(pooled)

    def forward(self, input_ids, attention_mask):
        self.initialize_label_features()
        doc = F.normalize(self.encode(input_ids, attention_mask), dim=-1)
        label_h = self.label_proj(self.raw_label_features.to(doc.dtype))
        path_h = F.normalize(label_h[self.terminal_node_ids], dim=-1)
        scale = self.logit_scale_raw.exp().clamp(max=100.0)
        return {"path_logits": scale * (doc @ path_h.t())}

    def compute_loss(self, outputs, gold_path_ids):
        target = gold_path_ids.to(outputs["path_logits"].device, dtype=torch.long)
        logits = outputs["path_logits"]
        if self.restrict_training_to_supported:
            logits = logits.masked_fill(
                ~self.train_supported_path_mask.unsqueeze(0), -torch.inf
            )
        weights = self.path_class_weights.to(logits.dtype)
        loss = F.cross_entropy(logits, target, weight=weights)
        return loss, {"categorical_ce": loss.detach()}


class SemanticPathHMLC(PathHMLC):
    """Known-taxonomy zero-example model with semantic, shared action scoring.

    Label descriptions are encoded once with the initial pretrained document encoder.
    Their raw representations are frozen buffers; trainable projections, graph message
    passing, and shared child/STOP scorers learn task-specific compatibility. No test
    document or held-out-label example is used to construct these representations.
    """

    def __init__(self, encoder_name: str, taxonomy: Taxonomy, cfg: Config,
                 use_graph: bool = True, use_label_semantics: bool = True,
                 use_shared_stop: bool = True, use_balance: bool = True,
                 use_semantic_loss: bool = True,
                 class_weights: Optional[torch.Tensor] = None,
                 default_path_class_weights: Optional[torch.Tensor] = None):
        super().__init__(
            encoder_name, taxonomy, cfg, use_graph=use_graph,
            use_balance=use_balance, use_calibration=True,
            class_weights=class_weights,
            default_path_class_weights=default_path_class_weights,
        )
        self.encoder_name = encoder_name
        self.use_label_semantics = use_label_semantics
        self.use_shared_stop = use_shared_stop
        self.use_semantic_loss = use_semantic_loss and use_label_semantics
        enc_dim = self.encoder.config.hidden_size

        try:
            tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
        except (TypeError, ValueError):
            tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=False)
        encoded = tokenizer(
            taxonomy_node_texts(taxonomy, encoder_name), padding=True, truncation=True,
            max_length=cfg.label_max_length, return_tensors="pt",
        )
        self.register_buffer("label_input_ids", encoded["input_ids"], persistent=False)
        self.register_buffer("label_attention_mask", encoded["attention_mask"], persistent=False)
        self.register_buffer(
            "raw_label_features", torch.zeros(len(taxonomy.nodes), enc_dim), persistent=True
        )
        self.register_buffer("label_features_ready", torch.tensor(False), persistent=True)
        self.label_proj = nn.Sequential(
            nn.Linear(enc_dim, cfg.hidden_dim), nn.GELU(), nn.LayerNorm(cfg.hidden_dim)
        )
        self.label_fusion = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.LayerNorm(cfg.hidden_dim),
        )
        self.semantic_doc_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.semantic_scale_raw = nn.Parameter(torch.zeros(()))
        self.per_stop_head = nn.Linear(cfg.hidden_dim, len(self.stop_action_indices))
        self.register_buffer(
            "terminal_node_ids",
            torch.tensor([taxonomy.node_to_id[path] for path in taxonomy.paths], dtype=torch.long),
        )

    @torch.no_grad()
    def initialize_label_features(self):
        if bool(self.label_features_ready.item()) or not self.use_label_semantics:
            return
        was_training = self.encoder.training
        self.encoder.eval()
        rows = []
        for start in range(0, len(self.label_input_ids), self.cfg.label_encode_batch_size):
            ids = self.label_input_ids[start:start + self.cfg.label_encode_batch_size]
            mask = self.label_attention_mask[start:start + self.cfg.label_encode_batch_size]
            output = self.encoder(input_ids=ids, attention_mask=mask)
            float_mask = mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
            pooled = (output.last_hidden_state * float_mask).sum(1) / float_mask.sum(1).clamp_min(1)
            rows.append(pooled.float())
        self.raw_label_features.copy_(torch.cat(rows, dim=0))
        self.label_features_ready.fill_(True)
        self.encoder.train(was_training)

    def semantic_node_features(self) -> torch.Tensor:
        id_features = self.tree_encoder.embedding.weight
        if not self.use_label_semantics:
            return id_features
        self.initialize_label_features()
        semantic_features = self.label_proj(self.raw_label_features.to(id_features.dtype))
        return self.label_fusion(torch.cat([id_features, semantic_features], dim=-1))

    def action_logits_with_nodes(self, doc: torch.Tensor, node_h: torch.Tensor) -> torch.Tensor:
        parent_h = node_h[self.action_parent_ids]
        child_h = node_h[self.action_child_ids]
        edge_context = self.parent_proj(parent_h) + doc[:, None, :]
        # No action-specific bias: the compatibility function is shared across labels.
        logits = (
            edge_context * self.child_proj(child_h)[None, :, :]
        ).sum(-1) / math.sqrt(doc.size(-1))

        stop_idx = self.stop_action_indices
        if self.use_shared_stop:
            ph = parent_h[stop_idx].unsqueeze(0).expand(doc.size(0), -1, -1)
            query = doc.unsqueeze(1).expand(-1, stop_idx.numel(), -1)
            depth = (self.action_depth[stop_idx] / self.max_path_depth).view(1, -1, 1)
            depth = depth.expand(doc.size(0), -1, -1)
            compatibility = F.cosine_similarity(query, ph, dim=-1).unsqueeze(-1)
            query_norm = query.norm(dim=-1, keepdim=True) / math.sqrt(query.size(-1))
            features = torch.cat([query, ph, depth, compatibility, query_norm], dim=-1)
            stop_values = self.stop_mlp(features).squeeze(-1)
            # Keep indexed assignment dtype-safe under float16/bfloat16 autocast.
            logits[:, stop_idx] = stop_values.to(dtype=logits.dtype)
        else:
            # Per-terminal classifiers cannot transfer a learned STOP rule to held-out paths.
            stop_values = self.per_stop_head(doc)
            logits[:, stop_idx] = stop_values.to(dtype=logits.dtype)
        return logits

    def forward(self, input_ids, attention_mask):
        doc = self.encode(input_ids, attention_mask)
        initial_nodes = self.semantic_node_features()
        node_h = self.tree_encoder(self.use_graph, initial_features=initial_nodes)
        action_logits = self.action_logits_with_nodes(doc, node_h)
        action_log_probs = self.conditional_log_softmax(action_logits)
        structural_path_scores = action_log_probs @ self.path_action_matrix.t()
        structural_path_scores = structural_path_scores - torch.logsumexp(
            structural_path_scores, dim=1, keepdim=True
        )

        if self.use_label_semantics:
            doc_sem = F.normalize(self.semantic_doc_proj(doc), dim=-1)
            path_sem = F.normalize(node_h[self.terminal_node_ids], dim=-1)
            semantic_logits = doc_sem @ path_sem.t()
            combined_scores = structural_path_scores + F.softplus(self.semantic_scale_raw) * semantic_logits
        else:
            semantic_logits = None
            combined_scores = structural_path_scores
        path_log_probs = combined_scores - torch.logsumexp(combined_scores, dim=1, keepdim=True)
        return {
            "action_logits": action_logits, "action_log_probs": action_log_probs,
            "structural_path_scores": structural_path_scores,
            "semantic_logits": semantic_logits,
            "path_log_probs": path_log_probs, "path_logits": path_log_probs,
        }

    def compute_loss(self, outputs, gold_path_ids):
        target = gold_path_ids.to(outputs["path_log_probs"].device, dtype=torch.long)
        path_log_probs = outputs["path_log_probs"]
        if self.restrict_training_to_supported:
            path_log_probs = path_log_probs.masked_fill(
                ~self.train_supported_path_mask.unsqueeze(0), -torch.inf
            )
            path_log_probs = path_log_probs - torch.logsumexp(path_log_probs, dim=1, keepdim=True)
        weights = self.path_class_weights.to(path_log_probs.dtype) if self.use_balance else None
        path_nll = F.nll_loss(path_log_probs, target, weight=weights)
        loss = path_nll
        pieces = {"categorical_nll": path_nll.detach()}
        if self.use_semantic_loss:
            semantic_train_logits = outputs["semantic_logits"] / self.cfg.semantic_temperature
            if self.restrict_training_to_supported:
                semantic_train_logits = semantic_train_logits.masked_fill(
                    ~self.train_supported_path_mask.unsqueeze(0), -torch.inf
                )
            semantic_ce = F.cross_entropy(semantic_train_logits, target, weight=weights)
            loss = loss + self.cfg.semantic_loss_weight * semantic_ce
            pieces["semantic_ce"] = semantic_ce.detach()
        return loss, pieces


class FlatPathClassifier(nn.Module):
    """Non-hierarchical categorical baseline over the valid path vocabulary."""

    def __init__(self, encoder_name: str, n_paths: int, cfg: Config,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.use_calibration = True
        self.restrict_training_to_supported = class_weights is not None
        self.encoder = load_pretrained_encoder(cfg, encoder_name)
        dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(dim, n_paths)
        effective_weights = torch.ones(n_paths) if class_weights is None else class_weights
        self.register_buffer("path_class_weights", effective_weights.detach().clone())
        self.register_buffer("train_supported_path_mask", effective_weights > 0)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
        return {"path_logits": self.head(self.dropout(pooled))}

    def compute_loss(self, outputs, gold_path_ids):
        target = gold_path_ids.to(device=outputs["path_logits"].device, dtype=torch.long)
        logits = outputs["path_logits"]
        if self.restrict_training_to_supported:
            logits = logits.masked_fill(
                ~self.train_supported_path_mask.unsqueeze(0), -torch.inf
            )
        weights = self.path_class_weights.to(logits.dtype) if self.restrict_training_to_supported else None
        loss = F.cross_entropy(logits, target, weight=weights)
        return loss, {"categorical_ce": loss.detach()}
