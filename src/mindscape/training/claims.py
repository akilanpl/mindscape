"""A/B supervision and C positive-feedback replay from self-generated attempts."""
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np

from mindscape.core.schema import Observation, State
from mindscape.core.serialization import decode
from mindscape.data.backends import stable_hash
from mindscape.data.generation import load_dataset
from mindscape.environments.multiplication.claims import ClaimAction, ClaimsEnvironment, expected_value
from mindscape.memory.episodic import EpisodicMemory
from mindscape.models.encoding import answer_labels
from mindscape.models.numpy_backend import NumpyMLP
from mindscape.models.study import StudyModel
from mindscape.models.study_encoding import CONCEPT_SEED, VERSION, answer_features, candidate_scores, claim_labels, policy_features
from mindscape.training.optimizer import fit_fixed


CONDITIONS = ("answer_only", "structured", "trajectory", "experiential")


def supervised_arrays(examples, condition, no_state=False, no_relation=False, no_goal=False):
    xs, ys = [], []
    for example in examples:
        if condition in ("answer_only", "structured"):
            xs.append(answer_features(example.view("answer_only"), condition == "structured"))
            ys.append(answer_labels(example.target_answer))
        else:
            for t in example.target_trajectory:
                xs.append(policy_features(decode(State, t["state_before"]), no_state, no_relation, no_goal))
                ys.append(claim_labels(t["action"]["value"]))
    return np.asarray(xs), np.asarray(ys, dtype=np.int64)


def collect_experience(examples, model, memory, seed, attempts_per_problem=64, online=False):
    """No example target/trace is accessed. Accepted labels are the learner's own actions."""
    rng = np.random.default_rng(seed)
    xs, ys, keys, attempts, successes = [], [], [], 0, 0
    for example in examples:
        env = ClaimsEnvironment()
        env.reset(decode(Observation, example.view("experiential")["observation"]))
        records, tried = [], set()
        for _ in range(attempts_per_problem):
            if env.state.phase == "done":
                successes += 1
                break
            state = env.state
            x = policy_features(state, model.no_state, model.no_relation, model.no_goal)
            scores = candidate_scores(model.backend, x)[:100].copy()
            untried = [i for i in range(100) if i not in tried]
            if not untried: break
            if rng.random() < .75:
                value = int(rng.choice(untried))
            else:
                scores[list(tried)] = -np.inf
                value = int(np.argmax(scores))
            tried.add(value)
            action = ClaimAction(value, state.position, state.row_position)
            # Training-only scalar feedback; private numerical oracle never leaves this boundary.
            accepted = value == expected_value(state)
            attempts += 1
            if accepted:
                transition = env.step(action)
                xs.append(x); ys.append(claim_labels(value))
                if online:
                    _, gradients = model.backend.loss_and_gradients(x[None, :], np.asarray([claim_labels(value)]))
                    for key, gradient in gradients.items():
                        model.backend.weights[key] -= .01 * gradient
                tried = set()
                record = {"state": asdict(state), "action": asdict(action),
                          "event": asdict(transition.event), "result": asdict(transition.result),
                          "next_state": asdict(env.state), "reward": 1, "success": True}
            else:
                record = {"state": asdict(state), "action": asdict(action),
                          "event": {"name": "rejected_claim"}, "result": {"accepted": False},
                          "next_state": asdict(state), "reward": 0, "success": False}
            records.append(record)
        key = memory.store_episode(example.problem, records, env.state.answer, env.is_goal_reached(),
            {"example_id": example.example_id, "seed": seed, "condition": "experiential",
             "complete_target_received": False}, partition="train")
        keys.append(key)
    return np.asarray(xs), np.asarray(ys, dtype=np.int64), {"attempts": attempts,
        "accepted_samples": len(xs), "completed_episodes": successes, "episode_ids": keys,
        "full_target_trajectories_received": 0}


def train_claims(splits, manifest, config, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    if config["condition"] not in CONDITIONS:
        raise ValueError("Unknown study condition")
    ids = manifest["nested_subsets"].get(str(config["budget"]))
    if ids is None: raise ValueError("Unavailable training budget")
    lookup = {e.example_id: e for e in splits["train"]}
    examples = [lookup[key] for key in ids]
    backend = NumpyMLP([10] * 8 + [2], config["hidden"], config["seed"])
    model = StudyModel(backend, config["condition"], dream=config.get("dream", True),
        no_state=config.get("no_state", False), no_relation=config.get("no_relation", False),
        no_goal=config.get("no_goal", False))
    began = time.perf_counter()
    experience, logs, phases = None, [], []
    if config["condition"] == "experiential":
        memory = EpisodicMemory(output / "episodes.sqlite")
        x, y, experience = collect_experience(examples, model, memory, config["seed"], config["attempts_per_problem"], config.get("no_memory", False))
        # Replay retrieves only training experiences with positive scalar feedback.
        if not config.get("no_memory", False):
            replay_x, replay_y = [], []
            for _, episode in memory.retrieve("train", limit=len(examples)):
                for record in episode["trajectory"]:
                    if record["reward"] == 1:
                        replay_x.append(policy_features(decode(State, record["state"]), model.no_state, model.no_relation, model.no_goal))
                        replay_y.append(claim_labels(record["action"]["value"]))
            x, y = np.asarray(replay_x), np.asarray(replay_y, dtype=np.int64)
        if config.get("no_memory", False):
            metrics = {"learning_status": "online_positive_feedback", "updates": len(x)}
        elif len(x):
            steps = config["steps"] if not config.get("no_memory", False) else max(1, len(x) // config["batch_size"])
            logs, metrics = fit_fixed(backend, x, y, x, y, steps, config["batch_size"], config["learning_rate"], config["seed"], config["validation_every"])
        else:
            metrics = {"learning_status": "no_positive_feedback"}
    else:
        x, y = supervised_arrays(examples, config["condition"], model.no_state, model.no_relation, model.no_goal)
        vx, vy = supervised_arrays(splits["validation"], config["condition"], model.no_state, model.no_relation, model.no_goal)
        logs, metrics = fit_fixed(backend, x, y, vx, vy, config["steps"], config["batch_size"], config["learning_rate"], config["seed"], config["validation_every"])
    elapsed = time.perf_counter() - began
    metadata = {"config": config, "training_time": elapsed, "dataset_hash": stable_hash(manifest),
        "dataset_version": manifest["dataset_version"], "training_ids": ids,
        "training_rows": len(x), "regime": "answer_only" if config["condition"] == "answer_only" else
            "trajectory_supervised" if config["condition"] == "trajectory" else "structured" if config["condition"] == "structured" else "experiential",
        "checkpoint": str((output / "checkpoint").resolve()), "parameter_count": backend.parameter_count,
        "validation_metrics": metrics, "experience": experience, "concept_seed": CONCEPT_SEED,
        "feature_version": VERSION, "selection": "fixed_optimizer_steps_no_test_or_validation_selection",
        "software": {"python": platform.python_version(), "numpy": np.__version__,
            "git_revision": subprocess.run(["git", "rev-parse", "HEAD"],capture_output=True,text=True).stdout.strip()},
        "hardware": {"machine": platform.machine()}, "seed": config["seed"]}
    model.training_metadata = metadata
    model.save(output / "checkpoint")
    (output / "config.yaml").write_text(json.dumps({"training": config, "dataset_manifest": manifest}, indent=2))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in logs))
    return model, metadata
