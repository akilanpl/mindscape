"""Training adapters use only train/validation examples supplied by orchestration."""
from dataclasses import asdict
import numpy as np
from mindscape.models.encoding import ACTION_NAMES, answer_labels, features, serialize_state
from mindscape.core.schema import Action, Goal, State
from mindscape.core.serialization import decode


def arrays(examples, kind):
    xs, ys = [], []
    for example in examples:
        operands = example.observation["operands"]
        if kind == "baseline":
            # Explicit answer-only projection: no trajectory/state access.
            view = example.view("answer_only", supervision=True)
            xs.append(features(view["observation"]["operands"]))
            ys.append(answer_labels(view["answer"]))
        elif kind == "mindscape":
            previous_action = None
            for transition in example.target_trajectory:
                state = decode(State, transition["state_before"])
                action = decode(Action, transition["action"])
                representation = serialize_state(example.problem, state, decode(Goal, example.goal), [action], previous_action)
                xs.append(features(operands, representation["state"], representation["previous_action"]))
                ys.append([ACTION_NAMES.index(action.name)])
                previous_action = action.name
        else:
            raise ValueError("Unknown learning regime")
    if not xs:
        raise ValueError("Cannot train on empty examples")
    return np.asarray(xs), np.asarray(ys, dtype=np.int64)
