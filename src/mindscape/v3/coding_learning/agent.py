"""Bounded hypothesis -> prediction -> real visible execution -> working update loop."""

import json
import time
from dataclasses import asdict

from mindscape.v3.schema import digest

from .backend import parse, prompt
from .concepts import extract
from .planning import choose

CONDITIONS = (
    "plain",
    "structured",
    "typed",
    "mindscape",
    "no_memory",
    "no_planning",
    "no_concepts",
    "word_memory",
)


def feedback(observed):
    return {
        k: v
        for k, v in observed.items()
        if k in ("all_passed", "passed", "total", "observed", "error")
    }


def solve(task, backend, executor, memory, evaluator, condition, max_rounds=2, max_tokens=256):
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    started = time.perf_counter()
    repository = dict(task.repository)
    initial = executor.run(repository, task.visible_tests)
    events = []
    attempts = []
    working = []
    snapshots = []
    current = initial
    memory_hash = digest(memory.records)
    for turn in range(max_rounds):
        structural = condition in ("mindscape", "no_memory", "no_planning", "word_memory")
        use_memory = condition in ("mindscape", "no_planning", "no_concepts", "word_memory")
        retrieved = (
            memory.retrieve(
                repository,
                task.contract,
                strategy="words" if condition in ("word_memory", "no_concepts") else "structured",
            )
            if use_memory
            else []
        )
        # The three simple baselines receive the same training-example budget: two fixed exemplars.
        examples = (
            retrieved
            if use_memory
            else (memory.records[:2] if condition in ("plain", "structured", "typed") else [])
        )
        extracted = extract(repository, task.contract)
        model_concepts = (
            {
                "features": extracted["features"],
                "concepts": extracted["concepts"],
                "functions": extracted["functions"],
            }
            if structural
            else None
        )
        payload = prompt(
            task, repository, feedback(current), model_concepts, examples, condition, working
        )
        events.append(
            {
                "stage": "READ_REPRESENT",
                "repository_hash": digest(repository),
                "concepts": extracted if structural else None,
                "training_memory_ids": [r["task_id"] for r in examples],
            }
        )
        response, timing = backend.generate(payload, max_tokens)
        record = {
            "turn": turn,
            "payload": payload,
            "response": response,
            "timing": timing,
            "parse_error": None,
            "selected": None,
            "predictions": [],
            "candidates": [],
        }
        try:
            proposals = parse(response, repository)
            index, predictions = choose(
                repository,
                proposals,
                task.contract,
                retrieved,
                evaluator,
                condition in ("mindscape", "no_memory", "no_concepts", "word_memory"),
                structural,
            )
            record.update(
                selected=index, predictions=predictions, candidates=[asdict(p) for p in proposals]
            )
            proposed = proposals[index].apply(repository)
            events.append(
                {
                    "stage": "HYPOTHESIZE_PROPOSE_PREDICT",
                    "hypothesis": proposals[index].hypothesis,
                    "predictions": predictions,
                    "selected": index,
                }
            )
            current = executor.run(proposed, task.visible_tests)
            record["actual_visible_result"] = current
            repository = proposed
            snapshots.append(dict(repository))
            events.append(
                {"stage": "ACT_OBSERVE", "repository": dict(repository), "actual": current}
            )
            working.append(
                {"hypothesis": proposals[index].hypothesis, "visible_result": feedback(current)}
            )
            events.append(
                {
                    "stage": "UPDATE",
                    "persistent_memory_added": False,
                    "working_evidence": working[-1],
                }
            )
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            record["parse_error"] = str(error)
            working.append({"parse_error": str(error)})
        attempts.append(record)
        if record.get("actual_visible_result", {}).get("all_passed"):
            break
    if digest(memory.records) != memory_hash:
        raise ValueError("Evaluation contaminated persistent memory")
    return {
        "task_id": task.identity,
        "condition": condition,
        "family": task.family,
        "split": task.split,
        "repository": repository,
        "snapshots": snapshots,
        "initial_visible": initial,
        "attempts": attempts,
        "events": events,
        "model_calls": len(attempts),
        "wall_seconds": time.perf_counter() - started,
        "persistent_memory_unchanged": True,
        "visible_vm_calls": int(initial["actual_vm_call"])
        + sum(
            int(a.get("actual_visible_result", {}).get("actual_vm_call", False)) for a in attempts
        ),
        "success": None,
        "private_assessment": "deferred until learner terminates",
    }
