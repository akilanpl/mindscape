"""Only approved, actually executed training repairs enter persistent memory."""

import json
from dataclasses import asdict
from pathlib import Path

from mindscape.v3.schema import digest

from .concepts import extract, similarity


class VerifiedMemory:
    def __init__(self, training_tasks, records=()):
        self.allowed = {t.identity: t for t in training_tasks if t.split == "train"}
        if len(self.allowed) != len(training_tasks):
            raise ValueError("Memory approval must be train-only")
        self.records = list(records)
        if any(r.get("task_id") not in self.allowed for r in self.records):
            raise ValueError("Unapproved persisted memory")

    def admit(self, task, repository, executor, training_private):
        if task.split != "train" or task.identity not in self.allowed:
            raise ValueError("Evaluation experience cannot enter verified memory")
        approved = self.allowed[task.identity]
        if asdict(task) != asdict(approved):
            raise ValueError("Approved task changed")
        visible = executor.run(repository, task.visible_tests)
        private = executor.run(repository, training_private)
        if not visible["all_passed"] or not private["all_passed"]:
            raise ValueError("Successful real training execution required")
        record = {
            "task_id": task.identity,
            "contract": asdict(task.contract),
            "before": task.repository,
            "after": repository,
            "concepts": extract(task.repository, task.contract),
            "after_concepts": extract(repository, task.contract),
            "outcome": {"visible": visible, "approved_training_validation": private},
            "provenance": {
                "split": "train",
                "actual_execution": True,
                "before_hash": digest(task.repository),
                "after_hash": digest(repository),
            },
        }
        record["hash"] = digest(record)
        if not any(r["task_id"] == task.identity for r in self.records):
            self.records.append(record)
        return record["hash"]

    def retrieve(self, repository, contract, k=2, strategy="structured"):
        query = extract(repository, contract)

        def score(record):
            if strategy == "words":
                a = set(contract.description.lower().split())
                b = set(record["contract"]["description"].lower().split())
                return len(a & b) / max(1, len(a | b))
            return similarity(query, record["concepts"])

        return sorted(self.records, key=lambda r: (-score(r), r["task_id"]))[:k]

    def save(self, path):
        p = Path(path)
        if p.exists():
            raise ValueError("Persistent memory snapshot is immutable")
        p.write_text(json.dumps(self.records, indent=2))

    @classmethod
    def load(cls, path, tasks, executor, private):
        records = json.loads(Path(path).read_text())
        result = cls(tasks)
        for r in records:
            copy = dict(r)
            stored = copy.pop("hash")
            if digest(copy) != stored:
                raise ValueError("Memory hash changed")
            if r["task_id"] not in result.allowed:
                raise ValueError("Memory contamination")
            task = result.allowed[r["task_id"]]
            if r["before"] != task.repository:
                raise ValueError("Memory task binding changed")
            # Trust no Boolean from disk; independently re-execute training approvals.
            result.admit(task, r["after"], executor, private[r["task_id"]]["private_tests"])
        return result
