import unittest
from pathlib import Path

from mindscape.v3.coding_learning.execution import Executor
from mindscape.v3.coding_safety import GuardedExecutor, guarded_source, same_types

RUNTIME = Path("work/coding/runtime")


class GuardedExecutionTests(unittest.TestCase):
    def test_indirect_builtins_static_rejected(self):
        with self.assertRaisesRegex(ValueError, "reflective"):
            guarded_source({"solution.py": "def f(x):\n return __builtins__['eval']('40+2')\n"})
        with self.assertRaises(ValueError):
            guarded_source({"solution.py": "def __probe__():\n return 0\n"})

    def test_recursive_types(self):
        self.assertFalse(same_types([True], [1]))
        self.assertFalse(same_types({"x": [1]}, {"x": [1.0]}))
        self.assertTrue(same_types({"x": [1, False]}, {"x": [2, True]}))

    @unittest.skipUnless((RUNTIME / "python.cwasm").exists(), "Pinned WASI copy required")
    def test_guest_only_reflection_probe_and_guard(self):
        source = {"solution.py": "def f(x):\n return __builtins__['eval']('40+2')\n"}
        suite = {
            "entry": "solution.f",
            "cases": [{"args": [0], "expected": 42, "expected_type": "int"}],
        }
        # This trusted regression fixture executes only in the original WASI guest.
        self.assertTrue(Executor(RUNTIME).run(source, suite)["all_passed"])
        blocked = GuardedExecutor(RUNTIME).run(source, suite)
        self.assertFalse(blocked["all_passed"])
        self.assertFalse(blocked["actual_vm_call"])

    @unittest.skipUnless((RUNTIME / "python.cwasm").exists(), "Pinned WASI copy required")
    def test_nested_output_boolean_cannot_impersonate_integer(self):
        source = {"solution.py": "def f(x):\n return [True]\n"}
        suite = {
            "entry": "solution.f",
            "cases": [{"args": [0], "expected": [1], "expected_type": "list"}],
        }
        self.assertTrue(Executor(RUNTIME).run(source, suite)["all_passed"])
        self.assertFalse(GuardedExecutor(RUNTIME).run(source, suite)["all_passed"])

    @unittest.skipUnless((RUNTIME / "python.cwasm").exists(), "Pinned WASI copy required")
    def test_nested_mutation_boolean_cannot_impersonate_integer(self):
        source = {"solution.py": "def f(values):\n values[0] = True\n return 0\n"}
        suite = {
            "entry": "solution.f",
            "cases": [{"args": [[1]], "expected": 0, "expected_type": "int", "unchanged": True}],
        }
        self.assertTrue(Executor(RUNTIME).run(source, suite)["all_passed"])
        self.assertFalse(GuardedExecutor(RUNTIME).run(source, suite)["all_passed"])
