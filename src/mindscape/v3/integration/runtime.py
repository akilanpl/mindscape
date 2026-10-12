"""Typed domain bridge; actual execution and private scoring follow prediction."""

import json
from dataclasses import asdict, dataclass

from mindscape.v3.kernel import BoundedPolicy, Kernel
from mindscape.v3.schema import (
    Action,
    ActualTransition,
    Entity,
    Evaluation,
    Goal,
    Observation,
    Prediction,
    State,
    digest,
)


@dataclass(frozen=True)
class Request:
    domain: str
    inputs: dict

    def __post_init__(self):
        allowed = {
            "mathematics": {"operands", "operations"},
            "coding": {"repository", "contract", "visible_tests"},
            "physics": {"family", "seed", "index", "horizon", "target"},
        }
        if self.domain not in allowed or set(self.inputs) != allowed[self.domain]:
            raise ValueError("Unknown domain or unexpected/private input fields")
        digest(self.inputs)  # Reject NaN, non-JSON objects.
        object.__setattr__(self, "inputs", json.loads(json.dumps(self.inputs)))
        if self.domain == "mathematics":
            from mindscape.v3.math_learning.environment import Problem

            Problem(
                tuple(self.inputs["operands"]),
                tuple(self.inputs["operations"]),
                "unit",
                "integration",
            )
        elif self.domain == "physics":
            x = self.inputs
            if x["family"] not in ("motion", "interaction") or type(x["seed"]) is not int:
                raise ValueError("Physics family/seed")
            if (
                type(x["index"]) is not int
                or x["index"] < 0
                or type(x["horizon"]) is not int
                or not 1 <= x["horizon"] <= 32
            ):
                raise ValueError("Bounded physics horizon/index")
            if len(x["target"]) != 2 or any(type(v) not in (int, float) for v in x["target"]):
                raise ValueError("Physics target")
        else:
            from mindscape.v3.coding_learning.execution import safe_source
            from mindscape.v3.coding_learning.schema import Contract

            safe_source(self.inputs["repository"])
            c = self.inputs["contract"]
            if set(c) != {"entry", "description", "inputs", "outputs", "invariants"}:
                raise ValueError("Contract fields")
            Contract(**c)

    @property
    def identity(self):
        return digest(asdict(self))

    @property
    def split(self):
        return "iid"  # Integration requests cannot update training memory.


@dataclass(frozen=True)
class Response:
    request_id: str
    domain: str
    prediction: dict
    actual: dict
    evaluation: dict
    trace: dict

    def __post_init__(self):
        if self.prediction.get("kind") != "hypothetical" or self.actual.get("kind") != "actual":
            raise ValueError("Prediction/actual separation")
        if not {"valid", "success", "evaluator", "reason"} <= set(self.evaluation):
            raise ValueError("Independent evaluation required")
        if (
            type(self.evaluation["valid"]) is not bool
            or type(self.evaluation["success"]) is not bool
        ):
            raise ValueError("Boolean evaluation required")
        digest(asdict(self))


class BridgeBackend:
    policy_forward_calls = 0
    transition_forward_calls = 0

    def __init__(self, adapter):
        self.adapter = adapter
        self.identifier = "v3.integration." + adapter.request.domain
        self.policy_forward_calls = 0
        self.transition_forward_calls = 0

    @property
    def parameter_count(self):
        counts = self.adapter.provenance["parameter_counts"]
        if not counts or any(type(n) is not int or n <= 0 for n in counts.values()):
            raise ValueError("Measured specialist parameter counts required")
        return sum(counts.values())

    def logits(self, observation, candidates):
        return self.adapter.scores

    def predict_transition(self, state, action, confidence):
        proposal = self.adapter.proposals[action.value]
        return Prediction(
            action,
            State(state.observation, tuple(self.adapter.summary(proposal)), True),
            confidence,
        )

    def checkpoint(self):
        return self.adapter.provenance


class Adapter:
    """Domain semantics stay in specialists; this only implements Kernel hooks."""

    def __init__(self, request, provenance):
        self.request = request
        self.provenance = provenance
        self.backend = BridgeBackend(self)

    def observe(self, task):
        self.public = self.observation()
        return Observation(
            task.domain,
            (),
            Goal("Submit specialist proposal for independent verification"),
            (Entity("public", (("json", json.dumps(self.public, sort_keys=True)),)),),
        )

    def candidates(self, state):
        self.proposals, self.scores = self.propose(self.public)
        if not 2 <= len(self.proposals) <= 128 or len(self.scores) != len(self.proposals):
            raise ValueError("Bounded proposal contract")
        return tuple(Action("submit_proposal", i) for i in range(len(self.proposals)))

    def summary(self, proposal):
        return proposal.get("summary", [])

    def execute(self, state, action):
        self.actual = self.act(self.proposals[action.value])
        self.actual["kind"] = "actual"
        return ActualTransition(
            state,
            action,
            State(state.observation, tuple(self.actual["summary"]), True),
            json.dumps(self.actual, sort_keys=True),
        )

    def evaluate(self, task, transition):
        result = self.verify(self.actual)
        return Evaluation(**result)

    def direct(self):
        """Independent specialist path without invoking the shared Kernel."""
        observation = self.observe(self.request)
        state = State(observation)
        candidates = self.candidates(state)
        action, confidence = BoundedPolicy().choose(self.scores, candidates)
        prediction = self.backend.predict_transition(state, action, confidence)
        actual = self.execute(state, action)
        evaluation = self.evaluate(self.request, actual)
        return self.pack(prediction, evaluation, {"path": "independent_specialist"})

    def pack(self, prediction, evaluation, trace):
        selected = prediction.action.value
        return Response(
            self.request.identity,
            self.request.domain,
            {**self.proposals[selected], "kind": "hypothetical"},
            self.actual,
            asdict(evaluation),
            trace,
        )


class Runtime:
    def __init__(self, adapters, source_hash):
        self.adapters = adapters
        self.source_hash = source_hash

    def run(self, request: Request) -> Response:
        request = Request(request.domain, request.inputs)
        adapter = self.adapters[request.domain](request)
        result = Kernel(adapter.backend).run(request, adapter, 11, self.source_hash)
        p = result["events"][3]["value"]
        action = Action(**p["action"])
        prediction = Prediction(
            action,
            State(
                Observation(request.domain, (), Goal("summary")), tuple(p["state"]["written"]), True
            ),
            p["confidence"],
        )
        evaluation = Evaluation(**result["events"][5]["value"])
        result["routing_is_learned"] = request.domain == "coding"
        result["persistent_memory_updated"] = False
        return adapter.pack(prediction, evaluation, result)
