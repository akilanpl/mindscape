"""Original R5.6 trainable graph transition model; no procedural answer paths."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    width: int = 48
    message_steps: int = 2
    node_types: int = 16
    relation_types: int = 8
    concepts: int = 32
    max_nodes: int = 256
    max_edges: int = 2048
    max_concepts: int = 16
    candidate_classes: int = 16


@dataclass
class GraphInput:
    node_features: torch.Tensor
    node_types: torch.Tensor
    edges: torch.Tensor
    relation_types: torch.Tensor
    action: torch.Tensor
    concept_ids: torch.Tensor
    domain: int

    def to(self, device):
        return GraphInput(*(getattr(self, name).to(device) for name in
                            ('node_features', 'node_types', 'edges', 'relation_types', 'action', 'concept_ids')),
                          domain=self.domain)

    def validate(self, config):
        n = self.node_features.shape[0]
        if self.node_features.ndim != 2 or self.node_features.shape[1] != 8 or not 1 <= n <= config.max_nodes:
            raise ValueError('node_features must have shape [1..max_nodes,8]')
        if self.node_types.shape != (n,) or self.node_types.dtype != torch.long:
            raise ValueError('node_types must be int64 [N]')
        if self.edges.ndim != 2 or self.edges.shape[0] != 2 or self.edges.dtype != torch.long:
            raise ValueError('edges must be int64 [2,E]')
        e = self.edges.shape[1]
        if e > config.max_edges or self.relation_types.shape != (e,) or self.relation_types.dtype != torch.long:
            raise ValueError('relation_types must be int64 [E] within edge budget')
        if self.action.shape != (4,) or self.concept_ids.ndim != 1 or self.concept_ids.dtype != torch.long:
            raise ValueError('action must be [4]; concept_ids int64 [K]')
        if len(self.concept_ids) > config.max_concepts or self.domain not in (0, 1, 2):
            raise ValueError('Invalid concept count or domain')
        for values, bound in ((self.node_types, config.node_types), (self.edges, n),
                              (self.relation_types, config.relation_types), (self.concept_ids, config.concepts)):
            if values.numel() and (values.min().item() < 0 or values.max().item() >= bound):
                raise ValueError('Graph index out of bounds')
        if not torch.isfinite(self.node_features).all() or not torch.isfinite(self.action).all():
            raise ValueError('Nonfinite observed features')
        tensors = [self.node_features, self.node_types, self.edges, self.relation_types, self.action, self.concept_ids]
        if len({v.device for v in tensors}) != 1:
            raise ValueError('All graph tensors must share a device')


class R56Model(nn.Module):
    """Shared typed relation core with learned domain heads and optional GRU state.

    All outputs are predictions. Verifiers and deterministic scheduling are outside
    this module. There is no initialized/pretrained success claim.
    """
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if not 8 <= c.width <= 512 or not 1 <= c.message_steps <= 8:
            raise ValueError('Require width 8..512 and message_steps 1..8')
        w = c.width
        self.features = nn.Linear(8, w)
        self.types = nn.Embedding(c.node_types, w)
        self.relations = nn.Embedding(c.relation_types, w)
        self.concepts = nn.Embedding(c.concepts, w)
        self.domains = nn.Embedding(3, w)
        self.action = nn.Linear(4, w)
        self.message = nn.Sequential(nn.Linear(w * 4, w), nn.Tanh(), nn.Linear(w, w))
        self.update = nn.Sequential(nn.Linear(w * 3, w), nn.Tanh(), nn.Linear(w, w))
        self.norm = nn.LayerNorm(w)
        self.working_memory = nn.GRUCell(w, w)
        self.state_fusion = nn.Sequential(nn.Linear(2 * w, w), nn.Tanh())
        self.math_digit = nn.Linear(w, 10)
        self.math_carry = nn.Linear(w, 2)
        self.coding_choice = nn.Linear(w, c.candidate_classes)
        self.physics_effect = nn.Linear(w, 2)
        self.action_score = nn.Linear(w, 4)

    def forward(self, graph: GraphInput, memory=None, *, stateless=False):
        c = self.config
        graph.validate(c)
        if graph.node_features.device != self.features.weight.device:
            raise ValueError('Model and graph devices differ; no silent device fallback')
        features = graph.node_features.to(dtype=self.features.weight.dtype)
        action = torch.tanh(self.action(graph.action.to(dtype=features.dtype)))
        concept = (self.concepts(graph.concept_ids).mean(0) if graph.concept_ids.numel()
                   else torch.zeros(c.width, device=features.device, dtype=features.dtype))
        context = action + concept + self.domains.weight[graph.domain]
        h = torch.tanh(self.features(features) + self.types(graph.node_types) + context)
        source, target = graph.edges
        for _ in range(c.message_steps):
            aggregate = torch.zeros_like(h)
            degree = torch.zeros((len(h), 1), device=h.device, dtype=h.dtype)
            if source.numel():
                message = self.message(torch.cat((h[source], h[target], self.relations(graph.relation_types),
                                                   context.expand(len(source), -1)), dim=-1))
                aggregate.index_add_(0, target, message)
                degree.index_add_(0, target, torch.ones((len(target), 1), device=h.device, dtype=h.dtype))
            aggregate = aggregate / degree.clamp(min=1)
            h = self.norm(h + self.update(torch.cat((h, aggregate, context.expand(len(h), -1)), dim=-1)))
        pooled = h.mean(0)
        if memory is not None and (memory.shape != (c.width,) or memory.device != pooled.device or not torch.isfinite(memory).all()):
            raise ValueError('Memory must be a finite model-device width-vector')
        prior = torch.zeros_like(pooled) if memory is None or stateless else memory
        new_memory = torch.tanh(self.working_memory(pooled.unsqueeze(0), prior.unsqueeze(0)).squeeze(0))
        state = pooled if stateless else self.state_fusion(torch.cat((pooled, new_memory), dim=-1))
        return {'math_digit_logits': self.math_digit(state), 'math_carry_logits': self.math_carry(state),
                'coding_logits': self.coding_choice(state), 'physics_delta': self.physics_effect(state),
                'action_scores': self.action_score(state), 'memory': new_memory,
                'pooled': pooled, 'node_states': h}

    def predict(self, problem: dict, *, stateless=False):
        from .native_domains import infer_problem
        return infer_problem(self, problem, stateless=stateless)
