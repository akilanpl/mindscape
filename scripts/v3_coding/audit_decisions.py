"""Independently bind raw responses to parsed actions and recompute neural ranking."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mindscape.v3.coding_learning.backend import parse
from mindscape.v3.coding_learning.data import load_public
from mindscape.v3.coding_learning.memory import VerifiedMemory
from mindscape.v3.coding_learning.planning import ProposalEvaluator, choose
from mindscape.v3.schema import digest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="configs/v3_coding/benchmark_v1")
    p.add_argument("--run", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    tasks, _ = load_public(a.manifest)
    tasks = {t.identity: t for t in tasks}
    records = json.loads((Path(a.training) / "memory.json").read_text())
    memory = VerifiedMemory([t for t in tasks.values() if t.split == "train"], records)
    model = ProposalEvaluator(a.seed)
    model.load_state_dict(
        torch.load(Path(a.training) / f"evaluator-s{a.seed}.pt", weights_only=True)["state"]
    )
    counts = {
        "episodes": 0,
        "attempts": 0,
        "parse_failures": 0,
        "ranker_forward_calls": 0,
        "qwen_calls": 0,
        "candidate_sets": 0,
    }
    for path in Path(a.run).glob("*.json"):
        saved = json.loads(path.read_text())
        if "trace" not in saved:
            continue
        trace = saved["trace"]
        task = tasks[trace["task_id"]]
        repository = dict(task.repository)
        if saved["trace_hash"] != digest(trace):
            raise ValueError("Trace integrity mismatch")
        counts["episodes"] += 1
        for attempt in trace["attempts"]:
            counts["attempts"] += 1
            try:
                proposals = parse(attempt["response"], repository)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                if attempt["parse_error"] != str(error) or attempt["selected"] is not None:
                    raise ValueError("Parse-failure binding mismatch") from error
                counts["parse_failures"] += 1
                continue
            if [asdict(x) for x in proposals] != attempt["candidates"]:
                raise ValueError("Raw response/action mismatch")
            condition = trace["condition"]
            structural = condition in ("mindscape", "no_memory", "no_planning", "word_memory")
            retrieved = (
                memory.retrieve(
                    repository,
                    task.contract,
                    strategy="words"
                    if condition in ("word_memory", "no_concepts")
                    else "structured",
                )
                if condition in ("mindscape", "no_planning", "no_concepts", "word_memory")
                else []
            )
            index, predictions = choose(
                repository,
                proposals,
                task.contract,
                retrieved,
                model,
                condition in ("mindscape", "no_memory", "no_concepts", "word_memory"),
                structural,
            )
            if index != attempt["selected"]:
                raise ValueError("Selected neural rank changed")
            for prediction, recorded in zip(predictions, attempt["predictions"], strict=True):
                if prediction["private_success"] is not None:
                    raise ValueError("Private oracle channel")
                if not np.allclose(
                    [prediction["visible_usefulness"], prediction["seconds"]],
                    [recorded["visible_usefulness"], recorded["seconds"]],
                    atol=1e-5,
                    rtol=1e-5,
                ):
                    raise ValueError("Prediction changed")
            repository = proposals[index].apply(repository)
            counts["candidate_sets"] += 1
            counts["ranker_forward_calls"] += len(proposals)
        if repository != trace["repository"]:
            raise ValueError("Final source differs from executed edit sequence")
    Path(a.output).write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts))
