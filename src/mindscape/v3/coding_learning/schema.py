"""Public learner records are separate from private scorer/teacher records."""

from dataclasses import asdict, dataclass

from mindscape.v3.schema import Concept, Relationship, digest


@dataclass(frozen=True)
class Contract:
    entry: str
    description: str
    inputs: tuple[str, ...]
    outputs: str
    invariants: tuple[str, ...]


@dataclass(frozen=True)
class RepairTask:
    identity: str
    split: str
    family: str
    repository: dict[str, str]
    contract: Contract
    visible_tests: dict

    def visible(self):
        return {
            "task_id": self.identity,
            "repository": dict(self.repository),
            "contract": asdict(self.contract),
            "visible_tests": self.visible_tests,
        }


@dataclass(frozen=True)
class RepairHypothesis:
    mechanism: str
    evidence: tuple[str, ...]
    predicted_usefulness: float
    predicted_seconds: float
    kind: str = "hypothetical_unverified"


@dataclass(frozen=True)
class RepairProposal:
    replacements: dict[str, str]
    hypothesis: str = ""

    def apply(self, repository):
        if not self.replacements or len(self.replacements) > 4:
            raise ValueError("One to four existing files required")
        for path, content in self.replacements.items():
            if path not in repository or not isinstance(content, str) or len(content) > 16000:
                raise ValueError("Unsafe edit")
        return dict(repository) | self.replacements


def task_from_dict(row):
    c = row["contract"]
    c = Contract(
        c["entry"], c["description"], tuple(c["inputs"]), c["outputs"], tuple(c["invariants"])
    )
    return RepairTask(
        row["identity"], row["split"], row["family"], row["repository"], c, row["visible_tests"]
    )


def concept_record(mechanism, evidence):
    return Concept(
        digest([mechanism, evidence]),
        "python_repair",
        mechanism,
        tuple(Relationship("program", "static_evidence", e) for e in evidence),
    )
