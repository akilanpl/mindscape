"""Bounded coding memories with provenance and explicit hypothetical records."""

import copy
import re
from dataclasses import dataclass, field


def words(text):
    return set(re.findall(r"[a-zA-Z_]+", text.lower()))


@dataclass
class CodingMemory:
    capacity: int = 5000
    episodic: list = field(default_factory=list)
    concepts: dict = field(default_factory=dict)
    working: list = field(default_factory=list)

    def __post_init__(self):
        for record in self.episodic:
            if record.get("verified") and not record.get("hypothetical"):
                signature = " ".join(sorted(words(record.get("problem", ""))))
                self.concepts[signature] = {
                    "specification_words": signature,
                    "repair_method": "source edit validated by actual visible tests",
                }

    def reset_working(self):
        self.working = []

    def remember(self, record, *, verified=False, hypothetical=False):
        value = copy.deepcopy(record)
        value["verified"] = verified
        value["hypothetical"] = hypothetical
        self.working.append(value)
        if verified and not hypothetical:
            self.episodic.append(value)
            self.episodic = self.episodic[-self.capacity :]
            signature = " ".join(sorted(words(value.get("problem", ""))))
            self.concepts[signature] = {
                "specification_words": signature,
                "repair_method": "source edit validated by actual visible tests",
            }

    def retrieve(self, problem, k=1):
        query = words(problem)
        candidates = [x for x in self.episodic if x.get("verified") and not x.get("hypothetical")]

        def score(x):
            tokens = words(x.get("problem", ""))
            return len(tokens & query) / max(1, len(tokens | query))

        return sorted(candidates, key=score, reverse=True)[:k]

    def dream(self, state, candidates, limit=2):
        # These are proposals, never executed outcomes or successful experiences.
        return [
            {
                "hypothetical": True,
                "verified": False,
                "repository": copy.deepcopy(state.files),
                "proposal": copy.deepcopy(x),
            }
            for x in candidates[:limit]
        ]
