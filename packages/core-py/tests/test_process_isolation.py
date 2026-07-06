"""
Tests for process-level isolation: ModelWorkerProcess, ProcessGuard, and ModelServer integration.

These tests use a fake model class loaded in a subprocess to verify crash
isolation, auto-restart, and Queue-based RPC.
"""
import time
import os
import signal
import pytest
pytestmark = pytest.mark.slow
from typing import Any, Optional
from unittest.mock import MagicMock

# Path so spawned subprocess can import tests.helpers.fake_model
import sys as _sys
_test_helpers_dir = os.path.join(os.path.dirname(__file__), "helpers")
_extra_paths = [_test_helpers_dir]


# ── Tests: ModelWorkerProcess ──────────────────────────────────────────


class TestModelWorkerProcess:
    WORKER_KWARGS = {
        "model_cls_path": "fake_model.FakeTestModel",
        "model_kwargs": {"reply": "worker says hi"},
        "worker_id": "test-worker",
        "generate_timeout": 5.0,
        "extra_sys_paths": _extra_paths,
    }

    @pytest.fixture
    def worker(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        w = ModelWorkerProcess(**self.WORKER_KWARGS)
        w.start()
        yield w
        w.stop()

    def test_start_and_stop(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker.start()
        assert worker.alive
        worker.stop()
        assert not worker.alive

    def test_generate(self, worker):
        result = worker.generate("hello")
        assert isinstance(result, dict)
        assert "text" in result
        assert result["text"] == "worker says hi"
        assert result["tokens_generated"] >= 0

    def test_health_check_after_generate(self, worker):
        worker.generate("hello")
        health = worker.health_check()
        assert health.alive
        assert health.requests_served >= 1
        assert health.pid is not None

    def test_generate_timeout(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        kwargs = dict(self.WORKER_KWARGS)
        kwargs["model_kwargs"] = {"reply": "slow", "delay": 3.0}
        kwargs["generate_timeout"] = 1.0

        worker = ModelWorkerProcess(**kwargs)
        worker.start()
        with pytest.raises(TimeoutError):
            worker.generate("hello")
        worker.stop()

    def test_worker_crash_detected(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker.start()
        pid = worker._process.pid
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        assert not worker.alive
        health = worker.health_check()
        assert not health.alive
        worker.stop()

    def test_generate_on_dead_worker_raises(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker.start()
        worker.stop()
        with pytest.raises(RuntimeError, match="not alive"):
            worker.generate("hello")

    def test_double_start_safe(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker.start()
        worker.start()
        assert worker.alive
        worker.stop()


# ── Tests: ProcessGuard ────────────────────────────────────────────────


class TestProcessGuard:
    GUARD_KWARGS = {
        "model_cls_path": "fake_model.FakeTestModel",
        "model_kwargs": {"reply": "guarded hello"},
        "worker_id": "test-guard",
        "generate_timeout": 5.0,
        "max_restarts": 2,
        "restart_delay": 0.5,
        "health_check_interval": 0.5,
        "extra_sys_paths": _extra_paths,
    }

    def test_start_and_stop(self):
        from domains.infrastructure.process_guard import ProcessGuard

        guard = ProcessGuard(**self.GUARD_KWARGS)
        guard.start()
        assert guard.alive
        guard.stop()
        assert not guard.alive

    def test_generate(self):
        from domains.infrastructure.process_guard import ProcessGuard

        guard = ProcessGuard(**self.GUARD_KWARGS)
        guard.start()
        result = guard.generate("hello")
        assert result["text"] == "guarded hello"
        guard.stop()

    def test_health_report(self):
        from domains.infrastructure.process_guard import ProcessGuard

        guard = ProcessGuard(**self.GUARD_KWARGS)
        guard.start()
        guard.generate("hello")
        h = guard.health()
        assert h["alive"]
        assert h["worker_id"] == "test-guard"
        assert h["requests_served"] >= 1
        assert h["restart_count"] == 0
        guard.stop()

    def test_health_report_not_started(self):
        from domains.infrastructure.process_guard import ProcessGuard

        guard = ProcessGuard(**self.GUARD_KWARGS)
        h = guard.health()
        assert not h["alive"]

    def test_crash_and_restart(self):
        from domains.infrastructure.process_guard import ProcessGuard

        cb_calls = []
        guard = ProcessGuard(**self.GUARD_KWARGS)
        guard.on_crash(lambda wid: cb_calls.append(("crash", wid)))
        guard.on_restart(lambda wid: cb_calls.append(("restart", wid)))
        guard.start()

        old_pid = guard._worker._process.pid
        os.kill(old_pid, signal.SIGKILL)

        time.sleep(5.0)

        assert guard.alive
        new_pid = guard._worker._process.pid
        assert new_pid != old_pid
        assert ("crash", "test-guard") in cb_calls
        assert ("restart", "test-guard") in cb_calls

        result = guard.generate("hello")
        assert result["text"] == "guarded hello"

        h = guard.health()
        assert h["restart_count"] >= 1

        guard.stop()

    def test_max_restarts_exhausted(self):
        from domains.infrastructure.process_guard import ProcessGuard

        guard = ProcessGuard(**self.GUARD_KWARGS)
        guard.start()

        # Kill workers until max_restarts exhausted
        for _ in range(5):
            if guard._worker is not None and guard._worker._process is not None:
                pid = guard._worker._process.pid
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            # Wait for guard to detect and (if eligible) restart
            time.sleep(2.5)

        h = guard.health()
        assert h["exhausted"], f"restarts={h['restart_count']} max={h['max_restarts']}"
        guard.stop()


# ── Tests: ModelServer with ProcessGuard ───────────────────────────────


class TestModelServerWithGuard:
    @pytest.fixture
    def model_server_with_guard(self):
        pytest.importorskip("torch")
        from domains.infrastructure.model_server import ModelServer
        from domains.infrastructure.process_guard import ProcessGuard

        import torch
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.zeros(1, 10, dtype=torch.long)
        mock_model.device = "cpu"

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 0
        inputs_mock = MagicMock()
        inputs_mock.__getitem__.return_value = torch.zeros(1, 5, dtype=torch.long)
        inputs_mock.get.return_value = None
        mock_tokenizer.return_value = inputs_mock

        guard = ProcessGuard(
            model_cls_path="fake_model.FakeTestModel",
            model_kwargs={"reply": "process isolated"},
            worker_id="test-isolated",
            generate_timeout=5.0,
            max_restarts=1,
            restart_delay=0.5,
            health_check_interval=0.5,
            extra_sys_paths=_extra_paths,
        )
        guard.start()

        server = ModelServer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            model_id="test-with-guard",
            process_guard=guard,
            enable_warmup=False,
        )
        yield server, guard
        guard.stop()

    @pytest.mark.asyncio
    async def test_generate_delegates_to_guard(self, model_server_with_guard):
        server, guard = model_server_with_guard
        result = await server.generate("hello")
        assert result["text"] == "process isolated"
        assert result["tokens_generated"] >= 0

    @pytest.mark.asyncio
    async def test_stream_fallback_to_direct_model(self, model_server_with_guard):
        server, guard = model_server_with_guard
        tokens = []
        async for text in server.generate_stream("hello", max_new_tokens=5):
            tokens.append(text)
        assert isinstance(tokens, list)

    @pytest.mark.asyncio
    async def test_metrics_with_guard(self, model_server_with_guard):
        server, guard = model_server_with_guard
        await server.generate("hello")
        snap = server.get_metrics_snapshot()
        assert snap["requests_total"] >= 1
        assert snap["requests_completed"] >= 1
