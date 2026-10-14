import pytest

from mindscape.alpha_ood.benchmark import (
    Task,
    check_identities,
    digest,
    make_public_suite,
    memorization,
    reference,
    transition_lookup,
)
from mindscape.alpha_ood.scoring import paired_interval, replay, wilson


def fixture():
    tasks = (Task(2, (("add", 3), ("mul", 4)), "ood", "operation_compositions"),
             Task(8, (("sub", 2),), "ood", "borrow_chains"))
    labels = {t.identity: list(reference(t)) for t in tasks}  # Synthetic only.
    rows = [{"identity": t.identity, "condition": c, "seed": s,
             "states": labels[t.identity] if c == "recurrent" else [0] * len(t.steps)}
            for c in ("recurrent", "feedforward") for s in (11, 23, 47) for t in tasks]
    seal = {"prediction_hash": digest(rows), "label_commitment": digest(labels),
            "custodian_attested_precedence": True}
    return tasks, labels, rows, seal


def test_public_generation_is_deterministic_disjoint_and_unlabelled():
    tasks = make_public_suite(train_n=40, validation_n=10, family_n=12)
    assert tasks == make_public_suite(train_n=40, validation_n=10, family_n=12)
    assert check_identities(tasks, set())["count"] == 122
    assert all(set(t.public()) == {"identity", "start", "steps", "split", "family"} for t in tasks)
    with pytest.raises(ValueError, match="historical"):
        check_identities(tasks, {tasks[0].identity})
    with pytest.raises(ValueError, match="duplicate"):
        check_identities(tasks + (tasks[0],), set())


def test_identity_cannot_hide_split_leakage():
    a = Task(1, (("add", 2),), "train", "familiar")
    b = Task(1, (("add", 2),), "ood", "operand_combinations")
    assert a.identity == b.identity


def test_reference_and_controls_use_predicted_feedback():
    task = Task(1, (("add", 2), ("mul", 4)), "train", "familiar")
    assert reference(task) == (3, 12)
    assert memorization(task, {}) == (0, 0)
    assert transition_lookup(task, {(1, "add", 2): 7, (3, "mul", 4): 12}) == (7, 0)


def test_replay_independent_denominator_and_failures():
    tasks, labels, rows, seal = fixture()
    report = replay(tasks, rows, labels, seal, ("recurrent", "feedforward"), (11, 23, 47))
    assert report["unique_tasks"] == 2
    assert report["metrics"]["recurrent"]["11"]["overall"]["exact"] == 1
    assert report["comparisons"]["feedforward:overall"]["independent_tasks"] == 2
    assert len(report["failure_traces"]) == 6
    assert report["scientific_status"] == "UNVERIFIED"


@pytest.mark.parametrize("kind", ["tamper", "missing", "duplicate", "no_attestation", "boolean", "label"])
def test_replay_rejects_invalid_evidence(kind):
    tasks, labels, rows, seal = fixture()
    if kind == "tamper":
        rows[0]["states"] = [9, 9]
    elif kind == "missing":
        rows.pop()
        seal["prediction_hash"] = digest(rows)
    elif kind == "duplicate":
        rows.append(rows[0])
        seal["prediction_hash"] = digest(rows)
    elif kind == "no_attestation":
        seal["custodian_attested_precedence"] = False
    elif kind == "boolean":
        rows[0]["states"] = [True, 20]
        seal["prediction_hash"] = digest(rows)
    elif kind == "label":
        labels[tasks[0].identity] = [5]
        seal["label_commitment"] = digest(labels)
    with pytest.raises(ValueError):
        replay(tasks, rows, labels, seal, ("recurrent", "feedforward"), (11, 23, 47))


def test_uncertainty_and_invalid_denominators():
    assert paired_interval([1., 1.], draws=100) == (1., 1.)
    low, high = wilson(0, 36)
    assert low == 0 and 0 < high < 0.1
    with pytest.raises(ValueError):
        wilson(0, 0)


def test_opaque_historical_digest_compatibility():
    task = Task(2, (("add", 3),), "train", "familiar")
    historical = digest({"operands": [2, 3], "operations": ["add"], "namespace": "R5.2-math"})
    with pytest.raises(ValueError, match="historical"):
        check_identities((task,), {historical})


def test_empty_tasks_and_nonfinite_uncertainty_rejected():
    _, labels, rows, seal = fixture()
    with pytest.raises(ValueError, match="empty"):
        replay((), rows, labels, seal, ("recurrent",), (11,))
    with pytest.raises(ValueError):
        paired_interval([float("nan")])
