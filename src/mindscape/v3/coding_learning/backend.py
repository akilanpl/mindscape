"""Offline Qwen proposal generation, bounded parsers and hash-bound requests."""

import json
import re

from .schema import RepairProposal


def parse(text, repository):
    blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    raw = blocks[0] if blocks else text
    start = raw.find("{")
    if start < 0:
        raise ValueError("No JSON proposal")
    value, end = json.JSONDecoder().raw_decode(raw[start:])
    if raw[start + end :].strip():
        raise ValueError("Trailing response text")
    if not isinstance(value, dict) or set(value) - {"candidates", "hypothesis"}:
        raise ValueError("Invalid proposal fields")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 2:
        raise ValueError("One or two candidate repairs required")
    result = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"repository"}:
            raise ValueError("Invalid candidate envelope")
        replacement = candidate["repository"]
        if not isinstance(replacement, dict):
            raise TypeError("Repository mapping required")
        p = RepairProposal(replacement, str(value.get("hypothesis", ""))[:2000])
        p.apply(repository)
        result.append(p)
    return result


def prompt(task, repository, observed, concepts, memory, condition, working):
    payload = {
        "repository": repository,
        "contract": task.visible()["contract"],
        "visible_tests": task.visible_tests,
        "actual_visible_result": observed,
        "previous_attempts": working[-1:],
    }
    if condition != "plain":
        payload["structured_contract"] = task.visible()["contract"]
    if concepts is not None:
        payload["static_concepts"] = concepts
    if memory:
        payload["approved_training_experiences"] = [
            {
                "contract": r["contract"],
                "before": r["before"],
                "after": r["after"],
                "verified_training_only": True,
            }
            for r in memory
        ]
    payload["instruction"] = (
        'Propose two distinct repairs if useful. Return only JSON: {"hypothesis":"brief failure mechanism", "candidates":[{"repository":{"existing_file.py":"complete replacement source"}}]}. Only existing files may change; multi-file replacements allowed. Do not write tests. Private evaluation is inaccessible.'
    )
    if condition == "typed":
        payload["tool_schema"] = {
            "name": "replace_files",
            "arguments": "existing path -> complete source",
            "next_tool": "run_visible_tests",
        }
    return payload
