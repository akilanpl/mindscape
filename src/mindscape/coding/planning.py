"""Bounded static hypotheses; syntax checks are never actual test evidence."""

import ast
import copy
from dataclasses import asdict, replace

from mindscape.coding.environment import ACTIONS
from mindscape.coding.schema import CodeAction


class BoundedPlanner:
    def plan(self, state, proposal):
        alternative = (
            CodeAction("run_tests")
            if proposal.name in ("edit", "patch")
            else CodeAction("inspect_repo")
        )
        candidates = [proposal, alternative]
        records = []
        for index, action in enumerate(candidates):
            files = dict(state.files)
            legal = action.name in ACTIONS
            reason = "Tool preconditions checked; runtime result unknown"
            if action.path is not None and action.path not in files:
                legal = False
                reason = "Outside repository"
            if legal and action.name == "edit":
                legal = (
                    action.path in files
                    and isinstance(action.content, str)
                    and len(action.content) <= 64000
                )
                if legal:
                    files[action.path] = action.content
            elif legal and action.name == "patch":
                legal = bool(
                    action.path in files
                    and action.old
                    and isinstance(action.new, str)
                    and files[action.path].count(action.old) == 1
                )
                if legal:
                    files[action.path] = files[action.path].replace(action.old, action.new, 1)
            if action.name == "inspect_file" and action.path not in files:
                legal = False
            if action.name == "inspect_symbol" and not action.symbol:
                legal = False
            if action.name == "search" and not action.query:
                legal = False
            syntax_valid = None
            if legal and action.name in ("edit", "patch"):
                try:
                    ast.parse(files[action.path])
                    syntax_valid = True
                except (SyntaxError, TypeError):
                    syntax_valid = False
                    reason = "Hypothetical source contains syntax error"
            score = (0.7 if index == 0 else 0.5) if legal else 0.0
            if syntax_valid is False:
                score = 0.1
            predicted = replace(
                state,
                files=copy.deepcopy(files),
                hypothetical=True,
                progress="hypothetical_unverified",
            )
            records.append(
                {
                    "action": asdict(action),
                    "predicted_state": asdict(predicted),
                    "hypothetical_next_action": asdict(CodeAction("run_tests")),
                    "score": score,
                    "reason": reason,
                    "evidence_kind": "hypothetical_result",
                    "actual_tests_passed": None,
                    "legal_prediction": legal,
                    "syntax_prediction": syntax_valid,
                }
            )
        selected = max(range(len(records)), key=lambda i: records[i]["score"])
        return candidates[selected], records
