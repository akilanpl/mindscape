"""Neural prediction and empirical concept retrieval. No arithmetic teacher calls."""

import itertools
import math

import numpy as np
import torch
from torch import nn

from .environment import MAX_DIGITS, OPS, RAW_OFFSET, Claim

INPUT = 320
GROUPS = (3, 120, 10, 10, 10, 10)
LOCAL_SIZE = sum(GROUPS)
ANSWER_SIZE = MAX_DIGITS * 10 + 2


def local_features(state):
    x = np.zeros(INPUT, dtype=np.float32)
    x[OPS.index(state.operation)] = 1
    for offset, value in zip((3, 13, 23), (state.left, state.right, state.incoming)):
        x[offset + value] = 1
    # Relational predicates expose no arithmetic target; equal-info baseline sees all of them.
    x[33:40] = [
        state.left == state.right,
        state.left < state.right,
        state.left == 0,
        state.right == 0,
        state.incoming > 0,
        state.left == 9,
        state.right == 9,
    ]
    x[40] = 1  # observation type: local transition
    return x


def task_features(visible):
    x = np.zeros(INPUT, dtype=np.float32)
    operands, operations = visible["operands"], visible["operations"]
    for j, number in enumerate(operands):
        text = str(abs(number))[::-1]
        if len(text) > 17:
            raise ValueError("Input digit bound exceeded")
        # Training tasks have <=8-digit original operands; rollout signs may see longer intermediate.
        for i, d in enumerate(text[:8]):
            x[48 + j * 80 + i * 10 + int(d)] = 1
        x[288 + j] = number < 0
    for j, op in enumerate(operations):
        x[291 + j * 3 + OPS.index(op)] = 1
    if len(operands) >= 2:
        x[297:300] = [
            abs(operands[0]) < abs(operands[1]),
            abs(operands[0]) == abs(operands[1]),
            operands[0] == 0,
        ]
    x[41] = 1
    return x


def claim_features(claim):
    return np.asarray(
        [
            claim.operation / 2,
            (claim.raw + 20) / 120,
            claim.digit / 9,
            claim.carry / 9,
            claim.next_digit / 9,
            claim.next_carry / 9,
        ],
        dtype=np.float32,
    )


def decode_claim(labels):
    return Claim(int(labels[0]), int(labels[1]) - RAW_OFFSET, *(int(x) for x in labels[2:]))


def decode_answer(labels):
    values = list(map(int, labels[:MAX_DIGITS]))
    text = "".join(str(d) for d in reversed(values)).lstrip("0") or "0"
    return int(("-" if int(labels[-1]) and text != "0" else "") + text)


class TransitionNet(nn.Module):
    def __init__(self, seed=0, pretrained_size=0):
        super().__init__()
        torch.manual_seed(seed)
        self.seed, self.pretrained_size = seed, pretrained_size
        self.encoder = nn.Sequential(
            nn.Linear(INPUT, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh()
        )
        if pretrained_size:
            # Trainable residual bottleneck adapter over frozen contextual representations.
            self.adapter_down = nn.Linear(pretrained_size, 8, bias=False)
            self.adapter_up = nn.Linear(8, 128, bias=False)
        self.local_head = nn.Linear(128, LOCAL_SIZE)
        self.answer_head = nn.Linear(128, ANSWER_SIZE)
        self.sign_head = nn.Linear(128, 2)
        self.consistency_head = nn.Sequential(nn.Linear(134, 64), nn.Tanh(), nn.Linear(64, 2))

    def hidden(self, x, pretrained=None):
        h = self.encoder(x)
        if self.pretrained_size:
            if pretrained is None:
                raise ValueError("Frozen pretrained features required")
            h = h + self.adapter_up(torch.tanh(self.adapter_down(pretrained)))
        return h

    def forward(self, x, pretrained=None):
        h = self.hidden(x, pretrained)
        return self.local_head(h), self.answer_head(h), self.sign_head(h)

    def consistency(self, x, claims, pretrained=None):
        return self.consistency_head(torch.cat((self.hidden(x, pretrained), claims), dim=1))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def split_logits(logits):
    return logits.split(GROUPS, dim=-1)


class Predictor:
    identifier = "v3.math.neural_transition"
    policy_forward_calls = 1
    transition_forward_calls = 1

    def __init__(self, model, memory=None, planning=True, encoder=None):
        self.model = model.eval()
        self.memory = memory or {}
        self.planning = planning
        self.encoder = encoder
        self.cache = {}
        self.forward_calls = 0
        self.memory_hits = 0
        self.plan_candidates = 0

    @property
    def parameter_count(self):
        return self.model.parameter_count

    def _pretrained(self, xs):
        return self.encoder.encode(xs) if self.encoder else None

    @torch.inference_mode()
    def predict_local(self, state):
        key = state.key
        if key in self.cache:
            claim, diag = self.cache[key]
            return claim, dict(diag, cache_hit=True)
        x = torch.from_numpy(local_features(state)[None, :])
        pre = self._pretrained(x.numpy())
        logits = self.model(x, pre)[0]
        self.forward_calls += 1
        heads = [h[0].clone() for h in split_logits(logits)]
        retrieved = self.memory.get(key)
        if retrieved is not None:
            self.memory_hits += 1
            # A finite demonstrated concept instance, not a calculator. Same training labels as baseline.
            for head, target in zip(heads, retrieved.labels()):
                head[target] += 1.0
        labels = [int(h.argmax()) for h in heads]
        alternatives = [decode_claim(labels)]
        if self.planning:
            # Bounded neural-only search; never scores against arithmetic or target output.
            for digit, carry in itertools.product(
                heads[4].topk(2).indices.tolist(), heads[5].topk(2).indices.tolist()
            ):
                candidate = labels.copy()
                candidate[4], candidate[5] = digit, carry
                claim = decode_claim(candidate)
                if claim not in alternatives:
                    alternatives.append(claim)
        cx = torch.from_numpy(np.stack([claim_features(c) for c in alternatives]))
        repeated = x.repeat(len(alternatives), 1)
        repeated_pre = pre.repeat(len(alternatives), 1) if pre is not None else None
        consistent = self.model.consistency(repeated, cx, repeated_pre).softmax(-1)[:, 1]
        self.forward_calls += 1
        self.plan_candidates += len(alternatives)
        logprobs = [h.log_softmax(-1) for h in heads]
        scores = [
            sum(float(h[v]) for h, v in zip(logprobs, c.labels()))
            + (2 * math.log(max(float(consistent[i]), 1e-8)) if self.planning else 0)
            for i, c in enumerate(alternatives)
        ]
        selected = max(range(len(scores)), key=lambda i: scores[i])
        claim = alternatives[selected]
        diag = {
            "logits": [h.tolist() for h in heads],
            "consistency": float(consistent[selected]),
            "candidate_claims": [c.labels() for c in alternatives],
            "candidate_scores": scores,
            "retrieved_training_instance": retrieved is not None,
            "cache_hit": False,
            "syntactic_candidate_count": int(np.prod(GROUPS)),
            "semantic_mask": False,
        }
        self.cache[key] = claim, diag
        return claim, diag

    @torch.inference_mode()
    def predict_sign(self, visible):
        from .training import sign_features

        x = torch.from_numpy(sign_features(visible)[None, :])
        logits = self.model(x, self._pretrained(x.numpy()))[2][0]
        self.forward_calls += 1
        return int(logits.argmax()), {"logits": logits.tolist(), "kind": "neural_prediction"}

    @torch.inference_mode()
    def answer(self, visible):
        x = torch.from_numpy(task_features(visible)[None, :])
        scores = self.model(x, self._pretrained(x.numpy()))[1]
        self.forward_calls += 1
        labels = [int(h.argmax()) for h in scores[0].split([10] * MAX_DIGITS + [2])]
        return decode_answer(labels), scores[0].tolist()

    # Common V3 backend interface. Factorized mathematical actions require the richer
    # math rollout, while this view scores written-digit actions in the generic kernel.
    def logits(self, observation, candidates):
        from .environment import LocalState

        if observation.domain != "mathematics" or len(observation.values) != 4:
            raise ValueError("Math observation requires operation,left,right,incoming")
        op, a, b, c = map(int, observation.values)
        _claim, diag = self.predict_local(LocalState(OPS[op], a, b, c))
        return [
            diag["logits"][4][action.value] if 0 <= action.value <= 9 else -1e9
            for action in candidates
        ]

    def predict_transition(self, state, action, confidence):
        from mindscape.v3.schema import Prediction, State

        from .environment import LocalState

        op, a, b, c = map(int, state.observation.values)
        claim, _ = self.predict_local(LocalState(OPS[op], a, b, c))
        return Prediction(
            action,
            State(state.observation, (float(claim.next_digit), float(claim.next_carry)), True),
            confidence,
        )

    def checkpoint(self):
        import hashlib

        return {
            "identifier": self.identifier,
            "weights_sha256": hashlib.sha256(
                b"".join(
                    t.detach().cpu().numpy().tobytes() for t in self.model.state_dict().values()
                )
            ).hexdigest(),
            "memory_size": len(self.memory),
            "planning": self.planning,
        }
