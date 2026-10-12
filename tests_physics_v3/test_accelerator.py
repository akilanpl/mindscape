import os
import unittest
from unittest.mock import patch

from mindscape.v3.physics_world.accelerator import require_mps


class AcceleratorTests(unittest.TestCase):
    def test_unavailable_fails_before_allocation(self):
        with (
            patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}),
            patch("torch.backends.mps.is_built", return_value=True),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.ones") as allocation,
        ):
            with self.assertRaisesRegex(RuntimeError, "Mandatory MPS unavailable"):
                require_mps()
            allocation.assert_not_called()

    def test_cpu_fallback_rejected_before_device_probe(self):
        for value in ("1", "true", "yes"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": value}),
                patch("torch.backends.mps.is_available") as available,
            ):
                with self.assertRaisesRegex(RuntimeError, "fallback forbidden"):
                    require_mps()
                available.assert_not_called()

    def test_unbuilt_fails_closed(self):
        with (
            patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}),
            patch("torch.backends.mps.is_built", return_value=False),
            patch("torch.ones") as allocation,
        ):
            with self.assertRaises(RuntimeError):
                require_mps()
            allocation.assert_not_called()
