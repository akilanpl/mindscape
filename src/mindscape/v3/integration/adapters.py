"""Model-backed adapters; arithmetic/VM/simulator verification is outside proposal code."""

from dataclasses import asdict

import numpy as np
import torch

from .runtime import Adapter


class Math(Adapter):
    def __init__(self, request, model, provenance):
        super().__init__(request, provenance)
        self.model = model

    def observation(self):
        return self.request.inputs

    def local(self, state):
        from mindscape.v3.math_learning.learner import (
            claim_features,
            decode_claim,
            local_features,
            split_logits,
        )

        x = torch.as_tensor(local_features(state)[None], device="mps")
        with torch.inference_mode():
            heads = [h[0] for h in split_logits(self.model(x)[0])]
            labels = [int(h.argmax().item()) for h in heads]
            candidates = []
            for digit in heads[4].topk(2).indices.tolist():
                for carry in heads[5].topk(2).indices.tolist():
                    row = labels.copy()
                    row[4], row[5] = digit, carry
                    candidates.append(decode_claim(row))
            cx = torch.as_tensor(np.stack([claim_features(c) for c in candidates]), device="mps")
            consistency = self.model.consistency(x.repeat(len(candidates), 1), cx).softmax(-1)[:, 1]
            logp = [h.log_softmax(-1) for h in heads]
            scores = [
                sum(float(h[v].item()) for h, v in zip(logp, c.labels()))
                + 2 * float(consistency[i].clamp(min=1e-8).log().item())
                for i, c in enumerate(candidates)
            ]
        self.calls += 2
        return candidates[int(np.argmax(scores))]

    def propose(self, public):
        from mindscape.v3.math_learning.environment import (
            LocalState,
            digits,
            materialize,
            normalize,
        )
        from mindscape.v3.math_learning.training import sign_features

        current = public["operands"][0]
        steps = []
        self.calls = 0

        def local(op, a, b, carry):
            state = LocalState(op, a, b, carry)
            claim = self.local(state)
            steps.append({"state": asdict(state), "claim": asdict(claim)})
            if len(steps) > 512:
                raise ValueError("Math action budget")
            return claim.next_digit, claim.next_carry

        def add(left, right, op):
            result, carry = [], 0
            for i in range(max(len(left), len(right)) + 1):
                digit, carry = local(
                    op, left[i] if i < len(left) else 0, right[i] if i < len(right) else 0, carry
                )
                result.append(digit)
            return result

        for op, right in zip(public["operations"], public["operands"][1:]):
            visible = {"operands": (current, right), "operations": (op,)}
            with torch.inference_mode():
                x = torch.as_tensor(sign_features(visible)[None], device="mps")
                sign = int(self.model(x)[2][0].argmax().item())
            self.calls += 1
            a, b, primitive = normalize(current, right, op)
            if primitive in ("add", "sub"):
                output = add(digits(a), digits(b), primitive)
            else:
                output = [0]
                for j, bdigit in enumerate(digits(b)):
                    row, carry = [0] * j, 0
                    for adigit in digits(a) + [0]:
                        digit, carry = local("mul", adigit, bdigit, carry)
                        row.append(digit)
                    output = add(output, row, "add")
            current = materialize(output, sign)
        self.backend.transition_forward_calls = self.calls
        return [
            {
                "answer": str(current),
                "steps": steps,
                "summary": [float(current)],
                "scaffold": "Preserved signed-magnitude scheduling; no teacher during prediction",
            },
            {"answer": None, "steps": [], "summary": []},
        ], [1.0, 0.0]  # Routing, not a learned answer selector.

    def act(self, proposal):
        from mindscape.v3.math_learning.environment import Claim, LocalState, execute_local

        written = [
            execute_local(LocalState(**s["state"]), Claim(**s["claim"])) for s in proposal["steps"]
        ]
        return {
            "answer": proposal["answer"],
            "written_transitions": written,
            "summary": proposal["summary"],
        }

    def verify(self, actual):
        from mindscape.v3.math_learning.environment import (
            Claim,
            LocalState,
            Problem,
            expected,
            teacher_local,
        )

        proposal = next(p for p in self.proposals if p["answer"] == actual["answer"])
        p = Problem(
            tuple(self.request.inputs["operands"]),
            tuple(self.request.inputs["operations"]),
            "unit",
            "integration",
        )
        strict = all(
            Claim(**s["claim"]) == teacher_local(LocalState(**s["state"]))
            for s in proposal["steps"]
        )
        success = actual["answer"] == str(expected(p))
        return {
            "valid": actual["answer"] is not None,
            "success": success,
            "evaluator": "independent arithmetic after write-only execution",
            "reason": f"exact_answer={success}; strict_local_transitions={strict}",
        }


class Coding(Adapter):
    def __init__(self, request, coder, evaluator, memory, executor, private, provenance):
        super().__init__(request, provenance)
        self.coder, self.evaluator, self.memory, self.executor = coder, evaluator, memory, executor
        self.private = private

    def observation(self):
        from mindscape.v3.coding_learning.agent import feedback

        self.initial = self.executor.run(
            self.request.inputs["repository"], self.request.inputs["visible_tests"]
        )
        return {**self.request.inputs, "actual_visible_result": feedback(self.initial)}

    def propose(self, public):
        from mindscape.v3.coding_learning.backend import parse, prompt
        from mindscape.v3.coding_learning.concepts import extract
        from mindscape.v3.coding_learning.planning import features
        from mindscape.v3.coding_learning.schema import RepairProposal, task_from_dict

        task = task_from_dict(
            {
                "identity": self.request.identity,
                "split": "iid",
                "family": "integration",
                **self.request.inputs,
            }
        )
        before = self.memory.records.copy()
        retrieved = self.memory.retrieve(task.repository, task.contract)
        payload = prompt(
            task,
            task.repository,
            public["actual_visible_result"],
            extract(task.repository, task.contract),
            retrieved,
            "mindscape",
            [],
        )
        response, timing = self.coder.generate(payload, 384)
        proposals = parse(response, task.repository)
        proposals.append(RepairProposal(dict(task.repository), "No change control"))
        predictions = [
            self.evaluator.predict(features(task.repository, p, task.contract, retrieved))
            for p in proposals
        ]
        assert before == self.memory.records
        self.backend.policy_forward_calls = 1 + len(predictions)
        results = [
            {
                "repository": p.apply(task.repository),
                "hypothesis": p.hypothesis,
                "prediction": prediction,
                "raw_model_response": response,
                "timing": timing,
                "approved_memory_ids": [r["task_id"] for r in retrieved],
                "summary": [],
            }
            for p, prediction in zip(proposals, predictions)
        ]
        return results, [p["visible_usefulness"] - 0.01 * p["seconds"] for p in predictions]

    def act(self, proposal):
        result = self.executor.run(proposal["repository"], self.request.inputs["visible_tests"])
        return {"repository": proposal["repository"], "visible": result, "summary": []}

    def verify(self, actual):
        result = self.executor.run(actual["repository"], self.private)
        actual["private_after_termination"] = result
        return {
            "valid": bool(actual["visible"].get("actual_vm_call")),
            "success": bool(result["all_passed"]),
            "evaluator": "independent private WASI suite",
            "reason": "Private scorer unavailable to generation/retrieval/evaluator",
        }


class Physics(Adapter):
    def __init__(self, request, model, provenance):
        super().__init__(request, provenance)
        self.model = model

    def observation(self):
        from mindscape.v3.physics_world.data import generate_scene
        from mindscape.v3.physics_world.environment import Environment
        from mindscape.v3.physics_world.schema import Action, observe

        x = self.request.inputs
        self.scene = generate_scene(x["family"], "iid", x["index"], x["seed"])
        self.env = Environment(self.scene)
        observations = [observe(self.env.state(), self.scene, 0)]
        for step in range(3):
            actual = self.env.step(Action.from_vector(np.zeros((3, 4))))
            observations.append(observe(actual, self.scene, step + 1))
        self.row = {
            "observations": np.stack([o["values"] for o in observations]),
            "masks": np.stack([o["mask"] for o in observations]),
            "contexts": np.stack([o["context"] for o in observations]),
        }
        return {k: v.tolist() for k, v in self.row.items()}

    def propose(self, public):
        from mindscape.v3_1.physics.evaluation import propose

        x = self.request.inputs
        proposal = propose(self.model, self.row, np.asarray(x["target"]), x["horizon"])
        proposal["summary"] = np.asarray(proposal["prediction"][-1]).flatten().tolist()
        self.backend.transition_forward_calls = x["horizon"]
        return [proposal, {"action": np.zeros((3, 4)).tolist(), "prediction": [], "summary": []}], [
            1.0,
            0.0,
        ]

    def act(self, proposal):
        from mindscape.v3.physics_world.schema import Action

        states = []
        for t in range(self.request.inputs["horizon"]):
            states.append(
                self.env.step(
                    Action.from_vector(proposal["action"] if t == 0 else np.zeros((3, 4)))
                ).copy()
            )
        self.states = np.asarray(states)
        return {
            "action": proposal["action"],
            "terminal": states[-1].tolist(),
            "summary": states[-1][:, :4].flatten().tolist(),
        }

    def verify(self, actual):
        from mindscape.v3.physics_world.environment import Environment
        from mindscape.v3.physics_world.schema import Action

        env = Environment(self.scene)
        for _ in range(3):
            env.step(Action.from_vector(np.zeros((3, 4))))
        replay = []
        for t in range(self.request.inputs["horizon"]):
            replay.append(
                env.step(
                    Action.from_vector(actual["action"] if t == 0 else np.zeros((3, 4)))
                ).copy()
            )
        np.testing.assert_array_equal(replay, self.states)
        predicted = np.asarray(self.proposals[0]["prediction"])
        active = self.states[0, :, 9] > 0
        actual["prediction_mse"] = float(
            ((predicted[:, active] - self.states[:, active, :4]) ** 2).mean()
        )
        distance = float(np.linalg.norm(self.states[-1, 0, :2] - self.request.inputs["target"]))
        actual["goal_distance"] = distance
        return {
            "valid": True,
            "success": distance <= 0.75,
            "evaluator": "independent Pymunk selected-action replay",
            "reason": "Simulated goal attainment; prediction MSE reported separately",
        }
