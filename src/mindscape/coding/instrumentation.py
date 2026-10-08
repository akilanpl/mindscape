"""Process-local stage timers; record evidence without changing policy inputs."""

import time
from contextlib import AbstractContextManager
from functools import wraps


class StageMeter(AbstractContextManager):
    def __init__(self):
        self.samples = {}
        self.originals = []

    def __enter__(self):
        from mindscape.coding.environment import CodeRepairEnvironment
        from mindscape.coding.memory import CodingMemory
        from mindscape.coding.model import LocalCoder
        from mindscape.coding.planning import BoundedPlanner
        from mindscape.coding.sandbox import WasiSandbox

        for owner, name, stage in [
            (CodeRepairEnvironment, "get_state", "state_construction"),
            (CodingMemory, "retrieve", "memory_retrieval"),
            (CodingMemory, "dream", "planning"),
            (BoundedPlanner, "plan", "planning"),
            (WasiSandbox, "execute", "test_execution"),
            (CodeRepairEnvironment, "final_evaluate", "verification"),
            (CodeRepairEnvironment, "step", "environment_execution"),
            (LocalCoder, "generate", "model_generation"),
        ]:
            original = getattr(owner, name)
            self.originals.append((owner, name, original))
            setattr(owner, name, self._wrapper(original, stage))
        return self

    def _wrapper(self, original, stage):
        @wraps(original)
        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.samples.setdefault(stage, []).append(time.perf_counter() - start)

        return measured

    def __exit__(self, *args):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        return False

    def summary(self):
        return {
            stage: {"calls": len(values), "seconds": sum(values), "samples_seconds": values}
            for stage, values in self.samples.items()
        }
