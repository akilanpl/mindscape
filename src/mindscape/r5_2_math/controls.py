"""Finite demonstration controls; never described as induced rules."""

from .verification import context_key


class Memory:
    def __init__(self, memory, sign_memory, fallback=None):
        self.memory = memory
        self.sign_memory = sign_memory
        self.fallback = fallback
        self.hits = 0

    def reset(self):
        if self.fallback:
            self.fallback.reset()

    def local(self, state):
        if self.fallback:
            d, c, diag = self.fallback.local(state)
        else:
            d, c, diag = 0, 0, {"kind": "zero_default_unverified"}
        if context_key(state) in self.memory:
            d, c = self.memory[context_key(state)]
            diag = {"kind": "exact_training_demonstration", "not_rule_induction": True}
            self.hits += 1
        return int(d), int(c), diag

    def sign(self, a, b, op):
        if self.fallback:
            s, diag = self.fallback.sign(a, b, op)
        else:
            s, diag = 0, {"kind": "zero_sign_default"}
        key = f"{a}|{b}|{op}"
        if key in self.sign_memory:
            s = self.sign_memory[key]
            diag = {"kind": "exact_training_sign_demo"}
        return int(s), diag
