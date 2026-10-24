"""Pure data/oracle checks; no model fitting or optimizer imports."""
import copy
import pytest
from research.r6.datasets.mathematics import (
    canonical_key, carry_borrow_category, generate, generate_with_metadata,
    reference_answer, structured_answer, verify_record,
)


def test_determinism_and_independent_verification():
    rows, metadata = generate_with_metadata("train", 171, 512)
    assert rows == generate("train", 171, 512)
    assert metadata["accepted"] == 512
    assert metadata["attempts"] == metadata["accepted"] + sum(metadata["rejected"].values())
    assert all(verify_record(row) for row in rows)
    assert metadata["max_input_bytes"] < 192
    assert len(metadata["template_counts"]) == 3
    assert all(not row["features"] for row in rows)


def test_global_disjointness_different_or_same_seeds():
    sets = []
    templates = []
    for split, seed, count in (("train", 171, 512), ("dev", 172, 64),
                               ("final_iid", 173, 96), ("final_ood", 174, 96)):
        rows = generate(split, seed, count)
        keys = {row["canonical_key"] for row in rows}
        assert len(keys) == count
        assert all(not (keys & previous) for previous in sets)
        sets.append(keys)
        templates.append({row["template_id"] for row in rows})
    assert templates[0] == templates[1] == templates[2]
    assert not templates[0] & templates[3]
    same_seed_sets = [{r["canonical_key"] for r in generate(split, 1, 100)}
                      for split in ("train", "dev", "final_iid")]
    assert not same_seed_sets[0] & same_seed_sets[1]
    assert not same_seed_sets[1] & same_seed_sets[2]


@pytest.mark.parametrize("left,right", [
    ("12 + 34", "34 + 12"), ("12 - 34", "(-34) + 12"),
    ("12 * (-34)", "(-34) * 12"), ("12 + (34 + 56)", "(56 + 12) + 34"),
    ("-(12 * (-34))", "12 * 34"),
])
def test_canonical_commutative_and_sign_equivalence(left, right):
    assert canonical_key(left) == canonical_key(right)


@pytest.mark.parametrize("expression,answer", [
    ("(-73) * (-29)", "2117"), ("99 - (-99)", "198"),
    ("(12 + 34) * (-56)", "-2576"), ("99 - 99", "0"),
])
def test_exact_ast_oracle(expression, answer):
    assert reference_answer(expression) == answer


@pytest.mark.parametrize("expression", ["__import__('os')", "2 ** 3", "1 / 0", "True + 2", "10000 + 1", "a + 1"])
def test_reject_unsupported_or_unsafe_expression(expression):
    with pytest.raises(ValueError):
        reference_answer(expression)


def test_reject_bad_structured_inputs_and_target_tampering():
    row = generate("final_ood", 5, 1)[0]
    mutated = copy.deepcopy(row)
    mutated["target"]["answer"] = "wrong"
    with pytest.raises(ValueError):
        verify_record(mutated)
    mutated = copy.deepcopy(row)
    mutated["structured_input"]["operands"][0] += 1
    with pytest.raises(ValueError):
        verify_record(mutated)
    with pytest.raises(ValueError):
        structured_answer({"operands": [1, 2], "operators": ["/"]})


def test_ood_has_both_unseen_arity_and_range():
    rows = generate("final_ood", 99, 96)
    assert sum(len(row["structured_input"]["operands"]) == 3 for row in rows) == 48
    assert sum(all(abs(n) >= 100 for n in row["structured_input"]["operands"]) for row in rows) == 48
    assert all(verify_record(row) for row in rows)


def test_carry_and_borrow_strata():
    def row(a, op, b):
        return {"structured_input": {"operands": [a, b], "operators": [op]}}
    assert carry_borrow_category(row(99, "+", 11)) == "carry"
    assert carry_borrow_category(row(11, "+", 22)) == "no-carry"
    assert carry_borrow_category(row(20, "-", 11)) == "borrow"
    assert carry_borrow_category(row(33, "-", 11)) == "no-borrow"
    assert carry_borrow_category(row(-33, "-", 11)) == "signed"


@pytest.mark.parametrize("split,seed,count", [("invalid", 1, 2), ("train", 1, -1), ("train", True, 1), ("train", 1, 1.5)])
def test_invalid_generation_request(split, seed, count):
    with pytest.raises(ValueError):
        generate(split, seed, count)


def test_empty_generation():
    rows, metadata = generate_with_metadata("train", 3, 0)
    assert rows == []
    assert metadata["attempts"] == 0
