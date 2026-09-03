"""Tests for training runtime protocol — registration and stub."""
from __future__ import annotations

from domains.training.runtime_protocol import (
    _NoOpRuntime,
    get_training_runtime,
    set_training_runtime,
)


class TestNoOpRuntime:
    def test_register_does_nothing(self):
        stub = _NoOpRuntime()
        stub.register("j1", {"task": "train"})  # should not raise

    def test_get_returns_none(self):
        stub = _NoOpRuntime()
        assert stub.get("j1") is None

    def test_sync_does_nothing(self):
        stub = _NoOpRuntime()
        stub.sync("j1")  # should not raise


class TestSetGetRuntime:
    def test_set_and_get(self):
        class FakeRuntime:
            def register(self, job_id, job, cancel_event=None, config=None):
                pass
            def get(self, job_id):
                return {"id": job_id}
            def sync(self, job_id):
                pass

        set_training_runtime(FakeRuntime())
        rt = get_training_runtime()
        assert rt.get("j1") == {"id": "j1"}

    def test_get_returns_stub_when_none(self):
        set_training_runtime(None)  # type: ignore
        rt = get_training_runtime()
        assert isinstance(rt, _NoOpRuntime)
