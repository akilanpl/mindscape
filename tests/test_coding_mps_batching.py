"""Blocking scheduler/device regressions; test doubles are not GPU measurements."""

import os
import time
from unittest.mock import patch

import pytest

from mindscape.coding.batching import evaluate_batch
from mindscape.coding.device import select_device


def test_accelerated_device_rejects_cpu_fallback():
    with pytest.raises(RuntimeError, match="CPU fallback forbidden"):
        select_device("cpu", require_mps=True)
    with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK":"1"}), pytest.raises(RuntimeError, match="fallback is forbidden"):
            select_device("mps", require_mps=True)


def test_unavailable_mps_fails_before_model_loading():
    with patch("torch.backends.mps.is_available", return_value=False), pytest.raises(RuntimeError, match="Main neural evaluation stopped"):
            select_device("mps", require_mps=True)


class SchedulerDouble:
    revision = "test-double-only"
    def generate_batch(self, requests):
        return [(r[1], {"test_double":True,"actual_batch_size":len(requests)}) for r in requests]


def test_independent_requests_preserve_episode_causality_and_accounting():
    recorded=[]
    def solve(job, client):
        first=client.generate("test",str(job),256,11)
        second=client.generate("test",first+":next",256,11)
        return {"id":job,"result":second,"calls":client.calls,"timings":client.timings}
    assert evaluate_batch(SchedulerDouble(),range(9),solve,recorded.append,4)==9
    assert {r["id"] for r in recorded}==set(range(9))
    assert all(r["calls"]==2 and r["result"]==str(r["id"])+":next" for r in recorded)
    assert all(len(r["timings"])==2 for r in recorded)


def test_deadline_drains_pending_episode_requests_without_fake_completions():
    recorded=[]
    def solve(job,client):
        return client.generate("test",str(job),256,11)
    assert evaluate_batch(SchedulerDouble(),range(8),solve,recorded.append,4,time.time()-1)==0
    assert not recorded
