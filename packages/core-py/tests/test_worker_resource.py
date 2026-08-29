"""Tests for domains.infrastructure.resource_manager — ResourceAllocation; domains.infrastructure.model_worker — WorkerHealth, WorkerStreamStalledError."""

from domains.infrastructure.resource_manager import ResourceAllocation
from domains.infrastructure.model_worker import WorkerHealth, WorkerStreamStalledError


class TestResourceAllocation:
    def test_defaults(self):
        ra = ResourceAllocation()
        assert ra.workload_mode == "balanced"
        assert ra.compute_threads == 0
        assert ra.inference_pool_size == 0

    def test_summary(self):
        ra = ResourceAllocation()
        s = ra.summary()
        assert isinstance(s, str)
        assert "balanced" in s

    def test_custom(self):
        ra = ResourceAllocation(compute_threads=4, io_threads=2, inference_pool_size=8)
        assert ra.compute_threads == 4
        assert ra.io_threads == 2
        assert ra.inference_pool_size == 8


class TestWorkerHealth:
    def test_defaults(self):
        wh = WorkerHealth()
        assert wh.alive is False
        assert wh.requests_served == 0
        assert wh.errors == 0
        assert wh.crashed is False

    def test_custom(self):
        wh = WorkerHealth(pid=1234, alive=True, requests_served=100, errors=2)
        assert wh.pid == 1234
        assert wh.alive is True
        assert wh.requests_served == 100


class TestWorkerStreamStalledError:
    def test_is_runtime_error(self):
        assert issubclass(WorkerStreamStalledError, RuntimeError)

    def test_message(self):
        err = WorkerStreamStalledError("stream stalled")
        assert str(err) == "stream stalled"
