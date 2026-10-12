"""OBSERVE -> REPRESENT -> PROPOSE -> PREDICT -> ACT -> VERIFY -> UPDATE.

Verifier access occurs after the action. No retries use held-out judgments. Policies
receive observations and syntax candidates only; memory retrieval is partitioned.
"""

import math
import time
from dataclasses import asdict

from .memory import Memory
from .schema import Experience, Provenance, State, digest


class BoundedPolicy:
    def __init__(self, max_candidates=128):
        if not 2 <= max_candidates <= 4096:
            raise ValueError("Candidate budget out of bounds")
        self.max_candidates = max_candidates

    def choose(self, logits, candidates):
        if not 2 <= len(candidates) <= self.max_candidates:
            raise ValueError("Singleton/empty/over-budget action sets are forbidden")
        if len(set(candidates)) != len(candidates) or len(logits) != len(candidates):
            raise ValueError("Malformed candidate scores")
        if any(not math.isfinite(x) for x in logits):
            raise ValueError("Nonfinite model scores")
        selected = max(range(len(logits)), key=lambda i: logits[i])
        norm = sum(math.exp(x - max(logits)) for x in logits)
        confidence = math.exp(logits[selected] - max(logits)) / norm
        return candidates[selected], confidence


class Kernel:
    def __init__(self, backend, memory=None, policy=None):
        self.backend = backend
        self.memory = memory if memory is not None else Memory()
        self.policy = policy if policy is not None else BoundedPolicy()

    def run(self, task, domain, seed, source_hash):
        started, cpu_started = time.perf_counter(), time.process_time()
        events = []

        def record(stage, value):
            events.append({"stage": stage, "value": value})

        observation = domain.observe(task)
        record("OBSERVE", asdict(observation))
        state = State(observation)
        retrieved = self.memory.retrieve(observation.domain, "train", exclude_task=task.identity)
        record(
            "REPRESENT",
            {
                "state": asdict(state),
                "retrieved_training_evidence": [digest(asdict(e)) for e in retrieved],
                "retrieval_used_by_model": False,
            },
        )
        candidates = domain.candidates(state)
        logits = self.backend.logits(observation, candidates)
        action, confidence = self.policy.choose(logits, candidates)
        record(
            "PROPOSE",
            {
                "candidates": [asdict(a) for a in candidates],
                "logits": logits,
                "action": asdict(action),
                "confidence_uncalibrated": confidence,
            },
        )
        prediction = self.backend.predict_transition(state, action, confidence)
        record("PREDICT", asdict(prediction))
        transition = domain.execute(state, action)
        record("ACT", asdict(transition))
        predicted_values, actual_values = prediction.state.written, transition.after.written
        transition_mse = (
            sum((a - b) ** 2 for a, b in zip(predicted_values, actual_values)) / len(actual_values)
            if len(predicted_values) == len(actual_values) and actual_values
            else None
        )
        evaluation = domain.evaluate(task, transition)
        record("VERIFY", asdict(evaluation))
        provenance = Provenance(
            task.identity, task.split, digest(self.backend.checkpoint()), source_hash, seed
        )
        experience = Experience(transition, evaluation, provenance)
        # Eval traces retained separately; they never update train concept memory or weights.
        key = self.memory.add(experience) if task.split == "train" else None
        record("UPDATE", {"training_memory_key": key, "weights_updated": False})
        return {
            "task": asdict(task),
            "provenance": asdict(provenance),
            "events": events,
            "success": evaluation.success,
            "valid": evaluation.valid,
            "latency_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "model_calls": (
                self.backend.policy_forward_calls + self.backend.transition_forward_calls
            ),
            "policy_forward_calls": self.backend.policy_forward_calls,
            "transition_forward_calls": self.backend.transition_forward_calls,
            "candidate_count": len(candidates),
            "parameter_count": self.backend.parameter_count,
            "transition_prediction_mse": transition_mse,
        }


def replay(trace, domain):
    """Replay actual transitions without inference and independently recompute judgments."""
    from .domains import Task
    from .schema import ActualTransition, _construct

    task_data = trace["task"]
    task = Task(
        task_data["domain"], tuple(task_data["inputs"]), task_data["split"], task_data["family"]
    )
    stages = [e["stage"] for e in trace["events"]]
    if stages != ["OBSERVE", "REPRESENT", "PROPOSE", "PREDICT", "ACT", "VERIFY", "UPDATE"]:
        raise ValueError("Invalid cycle order")
    actual = _construct(ActualTransition, trace["events"][4]["value"])
    proposal = trace["events"][2]["value"]
    if asdict(actual.action) != proposal["action"] or actual.before != State(domain.observe(task)):
        raise ValueError("Proposal or observation binding changed")
    if [asdict(a) for a in domain.candidates(actual.before)] != proposal["candidates"]:
        raise ValueError("Candidate set changed")
    chosen, _ = BoundedPolicy().choose(proposal["logits"], domain.candidates(actual.before))
    if chosen != actual.action or domain.execute(actual.before, actual.action) != actual:
        raise ValueError("Execution mismatch")
    judgment = domain.evaluate(task, actual)
    if asdict(judgment) != trace["events"][5]["value"] or judgment.success != trace["success"]:
        raise ValueError("Evaluation mismatch")
    return judgment
