"""The same model/Prediction interfaces, with private SQL assessment external."""

import json
from dataclasses import asdict

from mindscape.models.base import Prediction
from mindscape.sql.environment import SqlAction, SqlEnvironment, SqlTask


class SqlBenchmarkModel:
    def __init__(self, coder, structured=True):
        self.coder = coder
        self.structured = structured
        self.identifier = "sql:typed_tools" if structured else "sql:plain_query"
        self.parameter_count = coder.parameter_count

    def predict(self, view):
        visible = view["observation"]
        learner = SqlTask(
            view["example_id"],
            view["problem"],
            visible["query"],
            visible["schema"],
            tuple(visible["visible_rows"]),
            (),
            "",
        )
        env = SqlEnvironment()
        env.reset(learner)
        initial = asdict(env.get_state())
        calls = self.coder.calls
        errors = []
        for step in range(5 if self.structured else 1):
            payload = learner.visible() | {"query": env.query}
            if self.structured:
                state = asdict(env.get_state())
                for observation in state["observations"]:
                    observation.pop("seconds", None)
                payload.update(
                    state=state,
                    tools=list(env.tools),
                    remaining_steps=5 - step,
                    controller_errors=errors[-2:],
                )
                system = "Repair the SQLite SELECT query. Choose one typed action as JSON with name and optional query. Available tools: inspect_schema, inspect_table, edit_query, execute_query, finish. Use actual visible execution. No explanation."
            else:
                system = "Repair the SQLite SELECT query for the specification. Return JSON with query containing the complete SELECT statement. No explanation."
            response = self.coder.generate(system, json.dumps(payload), 192)
            try:
                text = response.strip()
                if text.startswith("```"):
                    text = "\n".join(text.splitlines()[1:-1])
                value = json.loads(text)
                action = (
                    SqlAction(**value)
                    if self.structured
                    else SqlAction("edit_query", value["query"])
                )
                transition = env.step(action)
                if not transition.valid:
                    errors.append(transition.result["error"])
                if action.name == "finish" and transition.valid:
                    break
            except (ValueError, TypeError, KeyError) as error:
                errors.append(str(error))
        return Prediction(
            answer=env.query,
            trajectory={
                "initial_state": initial,
                "transitions": [asdict(t) for t in env.transitions],
            },
            model_calls=self.coder.calls - calls,
            diagnostics={"assessment": "deferred", "controller_errors": errors},
        )
