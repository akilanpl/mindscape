"""Partitioned structured memory: predictions cannot enter actual episodic evidence."""

from .schema import ActualTransition, Concept, Experience, Rule, digest, encode


class Memory:
    def __init__(self):
        self.episodes = {}
        self.concepts = {}
        self.rules = {}

    def add(self, experience: Experience):
        if (
            not isinstance(experience, Experience)
            or not isinstance(experience.transition, ActualTransition)
            or not experience.evaluation.valid
        ):
            raise ValueError("Only verified actual transitions may enter episodic memory")
        key = digest(encode(experience))
        self.episodes[key] = experience
        return key

    def retrieve(self, domain: str, split: str, limit: int = 8, exclude_task: str = ""):
        if split != "train":
            # Evaluation memory is never used as inference-time training/retrieval evidence.
            raise ValueError("Retrieval is restricted to training evidence")
        if not 0 <= limit <= 64:
            raise ValueError("Retrieval budget exceeded")
        return tuple(
            e
            for _, e in sorted(self.episodes.items())
            if e.transition.before.observation.domain == domain
            and e.provenance.split == "train"
            and e.provenance.task_id != exclude_task
        )[:limit]

    def store_concept(self, concept: Concept):
        self.concepts[concept.id] = concept

    def hypothesize(self, rule: Rule):
        if rule.concept_id not in self.concepts or any(
            key not in self.episodes for key in rule.evidence_ids
        ):
            raise ValueError("Unknown concept or evidence")
        if any(self.episodes[key].provenance.split != "train" for key in rule.evidence_ids):
            raise ValueError("Rules may only use training evidence")
        self.rules[digest(encode(rule))] = rule

    def save(self, path):
        from pathlib import Path

        from .schema import canonical

        payload = {
            "episodes": [encode(e) for _, e in sorted(self.episodes.items())],
            "concepts": [encode(c) for _, c in sorted(self.concepts.items())],
            "rules": [encode(r) for _, r in sorted(self.rules.items())],
        }
        Path(path).write_text(canonical({"payload": payload, "sha256": digest(payload)}))

    @classmethod
    def load(cls, path):
        import json
        from pathlib import Path

        from .schema import decode

        data = json.loads(Path(path).read_text())
        if digest(data["payload"]) != data["sha256"]:
            raise ValueError("Memory integrity mismatch")
        memory = cls()
        for episode in data["payload"]["episodes"]:
            memory.add(decode(episode))
        for concept in data["payload"]["concepts"]:
            memory.store_concept(decode(concept))
        for rule in data["payload"]["rules"]:
            memory.hypothesize(decode(rule))
        return memory
