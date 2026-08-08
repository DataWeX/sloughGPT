"""
Tests for process-level isolation: ModelWorkerProcess, ProcessGuard, and ModelServer integration.

These tests use a fake model class loaded in a subprocess to verify crash
isolation, auto-restart, and Queue-based RPC.
"""
import multiprocessing as mp
import time
import os
import signal
import queue
import sys
import threading
import types
import pytest
pytestmark = pytest.mark.slow
from typing import Any, Optional
from unittest.mock import MagicMock

# Path so spawned subprocess can import tests.helpers.fake_model
import sys as _sys
_test_helpers_dir = os.path.join(os.path.dirname(__file__), "helpers")
_extra_paths = [_test_helpers_dir]


# ── Fake SloNet worker (no heavy imports) ───────────────────────────────
# This module-level function is pickleable and can be spawned in a subprocess.
# It follows the same Queue protocol as _worker_loop but with a trivial model.

def _fake_slo_worker_main(
    req_q: mp.Queue,
    resp_q: mp.Queue,
    hb_q: mp.Queue,
    slnc_path: str,
    model_id: str,
    worker_id: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Fake SloNet worker for testing — no real model loaded."""
    import queue as _queue
    import gc as _gc
    import traceback as _tb

    hb_q.put_nowait(("ready", os.getpid()))
    requests_served = 0

    while True:
        try:
            cmd, payload = req_q.get(timeout=0.5)
        except _queue.Empty:
            hb_q.put_nowait(("alive", os.getpid()))
            continue

        if cmd == "stop":
            break

        if cmd == "generate":
            try:
                session_id, prompt, gen_kwargs = payload
                resp_q.put_nowait(("result", session_id, {
                    "text": f"slo({model_id}): {prompt}",
                    "tokens_generated": len(prompt.split()),
                    "elapsed_ms": 1.0,
                }))
                requests_served += 1
            except Exception as e:
                try:
                    resp_q.put_nowait(("error", session_id, f"{type(e).__name__}: {e}"))
                except Exception:
                    pass

        if cmd == "generate_stream":
            try:
                session_id, prompt, gen_kwargs = payload
                text = f"slo({model_id}): {prompt}"
                for word in text.split():
                    resp_q.put_nowait(("token", session_id, word + " "))
                resp_q.put_nowait(("result", session_id, {
                    "text": "",
                    "tokens_generated": len(text.split()),
                    "elapsed_ms": 1.0,
                }))
                requests_served += 1
            except Exception as e:
                try:
                    resp_q.put_nowait(("error", session_id, f"{type(e).__name__}: {e}"))
                except Exception:
                    pass

    _gc.collect()
    hb_q.put_nowait(("dead", os.getpid()))


# ── Spawned worker mains for start() failure paths ───────────────────────
# Pickled by reference into the child process (same mechanism as the fake
# SloNet worker above), so they never need a real model or torch.

def _dead_worker_main(
    req_q: mp.Queue,
    resp_q: mp.Queue,
    hb_q: mp.Queue,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Worker that reports ``dead`` before ever reporting ``ready``."""
    hb_q.put_nowait(("dead", os.getpid()))


def _silent_worker_main(
    req_q: mp.Queue,
    resp_q: mp.Queue,
    hb_q: mp.Queue,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Worker that exits without sending any heartbeat."""
    return


def _silent_alive_worker_main(
    req_q: mp.Queue,
    resp_q: mp.Queue,
    hb_q: mp.Queue,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Worker that stays alive for a while without sending any heartbeat."""
    time.sleep(2.0)


# ── Fake process / queue objects for parent-side error paths ─────────────

class _FakeProc:
    """Fake multiprocessing.Process: ``is_alive()`` returns a scripted sequence."""

    def __init__(self, alive_results):
        self._alive_results = list(alive_results)
        self.pid = 424242
        self.killed = False

    def is_alive(self):
        return self._alive_results.pop(0) if self._alive_results else False

    def join(self, timeout=None):
        pass

    def kill(self):
        self.killed = True


class _AlwaysAliveProc:
    """Fake multiprocessing.Process that stays alive indefinitely."""

    def __init__(self):
        self.pid = 424242

    def is_alive(self):
        return True

    def join(self, timeout=None):
        pass

    def kill(self):
        pass


class _OkQueue:
    """Queue stub that accepts puts and exposes close/join_thread."""

    def put_nowait(self, item):
        pass

    def close(self):
        pass

    def join_thread(self):
        pass


class _RaisingQueue(_OkQueue):
    """Queue stub whose put_nowait always raises."""

    def __init__(self, exc=None):
        self._exc = exc or RuntimeError("injected put failure")

    def put_nowait(self, item):
        raise self._exc


class _ScriptedQueue(_OkQueue):
    """Queue stub that yields scripted items, then raises ``end``."""

    def __init__(self, items, end=queue.Empty):
        self._items = list(items)
        self._end = end

    def get(self, timeout=None):
        if self._items:
            return self._items.pop(0)
        raise self._end()

    def get_nowait(self):
        return self.get(timeout=0)


class _PoisonQueue:
    """Queue stub whose close/join_thread raise — for _cleanup_queues errors."""

    def close(self):
        raise OSError("close failed")

    def join_thread(self):
        raise RuntimeError("join failed")


class _FakeWorker:
    """Minimal worker stub for ProcessGuard monitor-loop tests."""

    def __init__(self, alive=True):
        self._alive = alive

    @property
    def alive(self):
        return self._alive


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

    # ── Parent-side error/edge paths ──────────────────────────────────

    def test_stop_when_never_started(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker.stop()  # no-op: _process is None

    def test_stop_send_failure_swallowed(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._req_q = _RaisingQueue()
        worker._resp_q = _OkQueue()
        worker._hb_q = _OkQueue()
        worker._process = _FakeProc([False])
        worker.stop()

    def test_stop_kills_unresponsive_process(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._req_q = _OkQueue()
        worker._resp_q = _OkQueue()
        worker._hb_q = _OkQueue()
        proc = _FakeProc([True])
        worker._process = proc
        worker.stop()
        assert proc.killed

    def test_cleanup_queues_swallows_errors(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._req_q = _PoisonQueue()
        worker._resp_q = _PoisonQueue()
        worker._hb_q = _PoisonQueue()
        worker._cleanup_queues()  # close()/join_thread() failures are swallowed

    def test_start_fails_when_worker_reports_dead(self, monkeypatch):
        import domains.infrastructure.model_worker as mw_mod
        from domains.infrastructure.model_worker import ModelWorkerProcess

        monkeypatch.setattr(mw_mod, "_hf_worker_main", _dead_worker_main)
        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        try:
            with pytest.raises(RuntimeError, match="failed to start"):
                worker.start()
        finally:
            worker.stop()

    def test_start_fails_when_worker_dies_silently(self, monkeypatch):
        import domains.infrastructure.model_worker as mw_mod
        from domains.infrastructure.model_worker import ModelWorkerProcess

        monkeypatch.setattr(mw_mod, "_hf_worker_main", _silent_worker_main)
        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        try:
            with pytest.raises(RuntimeError, match="failed to start"):
                worker.start()
        finally:
            worker.stop()

    def test_start_retries_while_worker_alive_but_silent(self, monkeypatch):
        import domains.infrastructure.model_worker as mw_mod
        from domains.infrastructure.model_worker import ModelWorkerProcess

        monkeypatch.setattr(mw_mod, "_hf_worker_main", _silent_alive_worker_main)
        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        try:
            with pytest.raises(RuntimeError, match="failed to start"):
                worker.start()
        finally:
            worker.stop()

    def test_health_check_drains_heartbeats(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._hb_q = _ScriptedQueue([("alive", 1), ("dead", 2), ("ready", 3)])
        worker._process = None
        health = worker.health_check()
        assert health.alive is False
        assert health.crashed is True

    def test_health_check_handles_queue_error(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._hb_q = _ScriptedQueue([], end=OSError)
        worker._process = None
        health = worker.health_check()
        assert health.alive is False

    def test_generate_put_failure_raises(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._process = _FakeProc([True])
        worker._req_q = _RaisingQueue()
        worker._resp_q = _OkQueue()
        with pytest.raises(RuntimeError, match="Failed to send request"):
            worker.generate("hello")

    def test_generate_raises_when_worker_dies_during_generation(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._process = _FakeProc([True, False])
        worker._req_q = _OkQueue()
        worker._resp_q = _ScriptedQueue([])
        with pytest.raises(RuntimeError, match="crashed during generation"):
            worker.generate("hello")

    def test_generate_worker_error_raises(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._process = _FakeProc([True, True])
        worker._req_q = _OkQueue()
        worker._resp_q = _ScriptedQueue([("error", "boom")])
        with pytest.raises(RuntimeError, match="Worker generate error: boom"):
            worker.generate("hello")

    def test_generate_unknown_response_skipped(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._process = _FakeProc([True, True, True])
        worker._req_q = _OkQueue()
        worker._resp_q = _ScriptedQueue([
            ("weird", "x"),
            ("result", {"text": "ok", "tokens_generated": 1, "elapsed_ms": 1.0}),
        ])
        result = worker.generate("hello")
        assert result["text"] == "ok"

    def test_generate_ignores_stale_session_message(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._process = _FakeProc([True, True, True])
        worker._req_q = _OkQueue()
        worker._resp_q = _ScriptedQueue([
            ("result", "req-stale-0", {"text": "old", "tokens_generated": 1, "elapsed_ms": 1.0}),
            ("result", {"text": "ok", "tokens_generated": 1, "elapsed_ms": 1.0}),
        ])
        result = worker.generate("hello")
        assert result["text"] == "ok"

    def test_generate_stall_raises(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess
        from domains.infrastructure.model_worker import WorkerStreamStalledError

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._process = _AlwaysAliveProc()
        worker._req_q = _OkQueue()
        worker._resp_q = _ScriptedQueue([])
        worker._stall_timeout = 0.2
        with pytest.raises(WorkerStreamStalledError, match="stalled"):
            worker.generate("hello")

    def test_generate_stream_ignores_stale_session_tokens(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._process = _FakeProc([True, True, True, True])
        worker._req_q = _OkQueue()
        worker._resp_q = _ScriptedQueue([
            ("token", "req-stale-0", "junk "),
            ("token", "req-stale-0", "junk "),
            ("result", {"text": "ok", "tokens_generated": 0, "elapsed_ms": 1.0}),
        ])
        gen = worker.generate_stream("hello")
        with pytest.raises(StopIteration) as excinfo:
            next(gen)
        assert excinfo.value.value["tokens_generated"] == 0

    def test_generate_stream_stall_raises(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess
        from domains.infrastructure.model_worker import WorkerStreamStalledError

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._process = _AlwaysAliveProc()
        worker._req_q = _OkQueue()
        worker._resp_q = _ScriptedQueue([])
        worker._stall_timeout = 0.2
        with pytest.raises(WorkerStreamStalledError, match="stalled"):
            next(worker.generate_stream("hello"))

    def test_generate_stream_not_alive_raises(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._process = _FakeProc([False])
        with pytest.raises(RuntimeError, match="not alive"):
            next(worker.generate_stream("hello"))

    def test_generate_stream_put_failure_raises(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._process = _FakeProc([True])
        worker._req_q = _RaisingQueue()
        worker._resp_q = _OkQueue()
        with pytest.raises(RuntimeError, match="Failed to send stream request"):
            next(worker.generate_stream("hello"))

    def test_generate_stream_raises_when_worker_dies(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._process = _FakeProc([True, True, False])
        worker._req_q = _OkQueue()
        worker._resp_q = _ScriptedQueue([])
        with pytest.raises(RuntimeError, match="crashed during streaming"):
            next(worker.generate_stream("hello"))

    def test_generate_stream_worker_error_raises(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._process = _FakeProc([True, True])
        worker._req_q = _OkQueue()
        worker._resp_q = _ScriptedQueue([("error", "boom")])
        with pytest.raises(RuntimeError, match="generate_stream error: boom"):
            next(worker.generate_stream("hello"))

    def test_generate_stream_unknown_response_skipped(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._process = _FakeProc([True, True, True])
        worker._req_q = _OkQueue()
        worker._resp_q = _ScriptedQueue([
            ("weird", "x"),
            ("result", {"text": "ok", "tokens_generated": 0, "elapsed_ms": 1.0}),
        ])
        gen = worker.generate_stream("hello")
        with pytest.raises(StopIteration) as excinfo:
            next(gen)
        assert excinfo.value.value["tokens_generated"] == 0

    def test_generate_stream_timeout_returns_empty_result(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        worker = ModelWorkerProcess(**self.WORKER_KWARGS)
        worker._generate_timeout = 0.01
        worker._process = _AlwaysAliveProc()
        worker._req_q = _OkQueue()
        worker._resp_q = _ScriptedQueue([])
        gen = worker.generate_stream("hello")
        with pytest.raises(StopIteration) as excinfo:
            next(gen)
        assert excinfo.value.value == {}

    def test_context_manager_starts_and_stops(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        with ModelWorkerProcess(**self.WORKER_KWARGS) as worker:
            assert worker.alive
            result = worker.generate("hello")
            assert result["text"] == "worker says hi"
        assert not worker.alive


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

    def test_resolve_memory_limit_uses_explicit(self, tmp_path):
        from domains.infrastructure.process_guard import resolve_memory_limit_mb
        assert resolve_memory_limit_mb(str(tmp_path / "model.slnc"), 12345.0) == 12345.0

    def test_resolve_memory_limit_auto_sizes_from_file(self, tmp_path):
        from domains.infrastructure.process_guard import resolve_memory_limit_mb
        slnc = tmp_path / "model.slnc"
        slnc.write_bytes(b"\x00" * (1024 * 1024 * 100))  # 100 MB
        assert resolve_memory_limit_mb(str(slnc), 0) == 8192.0  # floor

    def test_resolve_memory_limit_auto_large_file(self, tmp_path):
        from domains.infrastructure.process_guard import resolve_memory_limit_mb
        slnc = tmp_path / "model.slnc"
        slnc.write_bytes(b"\x00" * (1024 * 1024 * 4096))  # 4 GB
        assert resolve_memory_limit_mb(str(slnc), None) == 4096 * 8

    def test_resolve_memory_limit_missing_file(self, tmp_path):
        from domains.infrastructure.process_guard import resolve_memory_limit_mb
        assert resolve_memory_limit_mb(str(tmp_path / "nope.slnc"), None) is None
        assert resolve_memory_limit_mb(None, 0) is None

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

    # ── Parent-side error/edge paths ──────────────────────────────────

    def test_generate_not_started_raises(self):
        from domains.infrastructure.process_guard import ProcessGuard

        guard = ProcessGuard(**self.GUARD_KWARGS)
        with pytest.raises(RuntimeError, match="not alive"):
            guard.generate("hello")

    def test_generate_stream_not_started_raises(self):
        from domains.infrastructure.process_guard import ProcessGuard

        guard = ProcessGuard(**self.GUARD_KWARGS)
        with pytest.raises(RuntimeError, match="not alive"):
            next(guard.generate_stream("hello"))

    def test_stall_recovers_and_restarts(self):
        from domains.infrastructure.process_guard import ProcessGuard
        from domains.infrastructure.model_worker import WorkerStreamStalledError

        guard = ProcessGuard(**self.GUARD_KWARGS)
        guard.start()
        try:
            guard._worker.generate = lambda *a, **k: (
                (_ for _ in ()).throw(WorkerStreamStalledError("simulated wedge"))
            )
            with pytest.raises(WorkerStreamStalledError, match="simulated wedge"):
                guard.generate("hello")
            assert guard._restart_count == 1
            assert guard.alive
            result = guard.generate("hello")
            assert result["text"] == "guarded hello"
        finally:
            guard.stop()

    def test_monitor_loop_continues_without_worker(self):
        from domains.infrastructure.process_guard import ProcessGuard

        guard = ProcessGuard(**self.GUARD_KWARGS)
        guard.health_check_interval = 0.01
        guard.restart_delay = 0.01
        guard._stop_monitor.clear()
        t = threading.Thread(target=guard._monitor_loop, daemon=True)
        t.start()
        time.sleep(0.05)
        guard._stop_monitor.set()
        t.join(timeout=2)
        assert not t.is_alive()

    def test_monitor_loop_handles_callback_errors(self, monkeypatch):
        from domains.infrastructure.process_guard import ProcessGuard

        guard = ProcessGuard(**self.GUARD_KWARGS)
        guard.health_check_interval = 0.01
        guard.restart_delay = 0.01
        guard._worker = _FakeWorker(alive=False)
        guard.on_crash(lambda wid: 1 / 0)
        guard.on_restart(lambda wid: 1 / 0)

        def _relaunch():
            guard._worker = _FakeWorker(alive=True)

        monkeypatch.setattr(guard, "_launch_worker", _relaunch)
        guard._stop_monitor.clear()
        t = threading.Thread(target=guard._monitor_loop, daemon=True)
        t.start()
        deadline = time.time() + 5.0
        while guard._restart_count == 0 and time.time() < deadline:
            time.sleep(0.01)
        assert guard._restart_count == 1
        guard._stop_monitor.set()
        t.join(timeout=2)
        assert not t.is_alive()


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


# ── Tests: SloNet Worker (ModelWorkerProcess, slnc_path mode) ──────────


class TestSloNetWorkerProcess:
    """Tests for ModelWorkerProcess in SloNet (pure-NumPy) mode.

    We patch ``_slo_worker_main`` with a trivial fake so no real .slnc
    file or SloNetChatProvider is needed.
    """

    WORKER_KWARGS = {
        "slnc_path": "/fake/test.slnc",
        "model_id": "test-slo",
        "worker_id": "slo-worker",
        "generate_timeout": 5.0,
        "extra_sys_paths": _extra_paths,
    }

    @pytest.fixture(autouse=True)
    def _patch_worker(self, monkeypatch):
        """Replace the real _slo_worker_main with our fake in the module namespace."""
        import domains.infrastructure.model_worker as mw_mod
        monkeypatch.setattr(mw_mod, "_slo_worker_main", _fake_slo_worker_main)

    def test_backend_property(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        w = ModelWorkerProcess(**self.WORKER_KWARGS)
        assert w.backend == "slo"

    def test_backend_hf_mode(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        w = ModelWorkerProcess(
            model_cls_path="fake_model.FakeTestModel",
            model_kwargs={"reply": "hi"},
            worker_id="hf-worker",
        )
        assert w.backend == "hf"

    def test_requires_either_slnc_or_hf(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        with pytest.raises(ValueError, match="requires either"):
            ModelWorkerProcess(worker_id="bad-worker")

    def test_start_and_stop(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        w = ModelWorkerProcess(**self.WORKER_KWARGS)
        w.start()
        assert w.alive
        w.stop()
        assert not w.alive

    def test_generate(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        w = ModelWorkerProcess(**self.WORKER_KWARGS)
        w.start()
        try:
            result = w.generate("hello world")
            assert isinstance(result, dict)
            assert "text" in result
            assert "slo(test-slo):" in result["text"]
            assert result["tokens_generated"] >= 0
        finally:
            w.stop()

    def test_generate_stream(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        w = ModelWorkerProcess(**self.WORKER_KWARGS)
        w.start()
        try:
            gen = w.generate_stream("hello world")
            tokens = []
            result = None
            try:
                while True:
                    tokens.append(next(gen))
            except StopIteration as e:
                result = e.value
            assert len(tokens) > 0
            assert all(isinstance(t, str) for t in tokens)
            assert isinstance(result, dict)
            assert "tokens_generated" in result
        finally:
            w.stop()

    def test_health_after_generate(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        w = ModelWorkerProcess(**self.WORKER_KWARGS)
        w.start()
        w.generate("test")
        h = w.health_check()
        assert h.alive
        assert h.requests_served >= 1
        assert h.pid is not None
        w.stop()

    def test_double_start_safe(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        w = ModelWorkerProcess(**self.WORKER_KWARGS)
        w.start()
        w.start()  # should be a no-op
        assert w.alive
        w.stop()

    def test_generate_on_dead_worker_raises(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        w = ModelWorkerProcess(**self.WORKER_KWARGS)
        w.start()
        w.stop()
        with pytest.raises(RuntimeError, match="not alive"):
            w.generate("hello")

    def test_worker_crash_detected(self):
        from domains.infrastructure.model_worker import ModelWorkerProcess

        w = ModelWorkerProcess(**self.WORKER_KWARGS)
        w.start()
        pid = w._process.pid
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        assert not w.alive
        h = w.health_check()
        assert not h.alive
        w.stop()


# ── Tests: ProcessGuard with SloNet params ──────────────────────────────


class TestProcessGuardSlo:
    """Tests for ProcessGuard in SloNet mode (slnc_path)."""

    GUARD_KWARGS = {
        "slnc_path": "/fake/test.slnc",
        "model_id": "test-slo",
        "worker_id": "slo-guard",
        "generate_timeout": 5.0,
        "max_restarts": 2,
        "restart_delay": 0.5,
        "health_check_interval": 0.5,
        "extra_sys_paths": _extra_paths,
    }

    @pytest.fixture(autouse=True)
    def _patch_worker(self, monkeypatch):
        import domains.infrastructure.model_worker as mw_mod
        monkeypatch.setattr(mw_mod, "_slo_worker_main", _fake_slo_worker_main)

    def test_start_and_stop(self):
        from domains.infrastructure.process_guard import ProcessGuard

        g = ProcessGuard(**self.GUARD_KWARGS)
        g.start()
        assert g.alive
        g.stop()
        assert not g.alive

    def test_generate(self):
        from domains.infrastructure.process_guard import ProcessGuard

        g = ProcessGuard(**self.GUARD_KWARGS)
        g.start()
        try:
            result = g.generate("hello")
            assert result["text"] == "slo(test-slo): hello"
        finally:
            g.stop()

    def test_health_report(self):
        from domains.infrastructure.process_guard import ProcessGuard

        g = ProcessGuard(**self.GUARD_KWARGS)
        g.start()
        try:
            g.generate("hello")
            h = g.health()
            assert h["alive"]
            assert h["worker_id"] == "slo-guard"
            assert h["requests_served"] >= 1
            assert h["restart_count"] == 0
        finally:
            g.stop()

    def test_health_report_not_started(self):
        from domains.infrastructure.process_guard import ProcessGuard

        g = ProcessGuard(**self.GUARD_KWARGS)
        h = g.health()
        assert not h["alive"]

    def test_crash_and_restart(self):
        from domains.infrastructure.process_guard import ProcessGuard

        cb_calls = []
        g = ProcessGuard(**self.GUARD_KWARGS)
        g.on_crash(lambda wid: cb_calls.append(("crash", wid)))
        g.on_restart(lambda wid: cb_calls.append(("restart", wid)))
        g.start()

        old_pid = g._worker._process.pid
        os.kill(old_pid, signal.SIGKILL)
        time.sleep(5.0)

        assert g.alive
        new_pid = g._worker._process.pid
        assert new_pid != old_pid
        assert ("crash", "slo-guard") in cb_calls
        assert ("restart", "slo-guard") in cb_calls

        result = g.generate("hello")
        assert result["text"] == "slo(test-slo): hello"

        h = g.health()
        assert h["restart_count"] >= 1
        g.stop()

    def test_max_restarts_exhausted(self):
        from domains.infrastructure.process_guard import ProcessGuard

        g = ProcessGuard(**self.GUARD_KWARGS)
        g.start()

        for _ in range(5):
            if g._worker is not None and g._worker._process is not None:
                pid = g._worker._process.pid
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(2.5)

        h = g.health()
        assert h["exhausted"], f"restarts={h['restart_count']} max={h['max_restarts']}"
        g.stop()

    def test_generate_stream(self):
        from domains.infrastructure.process_guard import ProcessGuard

        g = ProcessGuard(**self.GUARD_KWARGS)
        g.start()
        try:
            gen = g.generate_stream("hello world")
            tokens = []
            result = None
            try:
                while True:
                    tokens.append(next(gen))
            except StopIteration as e:
                result = e.value
            assert len(tokens) > 0
            assert all(isinstance(t, str) for t in tokens)
            assert result["tokens_generated"] >= 0
        finally:
            g.stop()

    def test_memory_mb_uses_psutil(self, monkeypatch):
        from domains.infrastructure.process_guard import ProcessGuard

        fake_psutil = types.ModuleType("psutil")

        class _NoSuchProcess(Exception):
            pass

        class _AccessDenied(Exception):
            pass

        class _MemInfo:
            rss = 100 * 1024 * 1024

        class _Proc:
            def __init__(self, pid):
                self.pid = pid

            def memory_info(self):
                return _MemInfo()

        fake_psutil.Process = _Proc
        fake_psutil.NoSuchProcess = _NoSuchProcess
        fake_psutil.AccessDenied = _AccessDenied
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        g = ProcessGuard(**self.GUARD_KWARGS)
        g.start()
        try:
            mem = g._memory_mb()
            assert mem == 100.0
        finally:
            g.stop()

    def test_memory_mb_swallows_psutil_error(self, monkeypatch):
        from domains.infrastructure.process_guard import ProcessGuard

        fake_psutil = types.ModuleType("psutil")

        class _NoSuchProcess(Exception):
            pass

        class _AccessDenied(Exception):
            pass

        def _raise(pid):
            raise _NoSuchProcess(pid)

        fake_psutil.Process = _raise
        fake_psutil.NoSuchProcess = _NoSuchProcess
        fake_psutil.AccessDenied = _AccessDenied
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        g = ProcessGuard(**self.GUARD_KWARGS)
        g.start()
        try:
            assert g._memory_mb() is None
        finally:
            g.stop()


# ── Tests: create_slo_guard factory ─────────────────────────────────────


class TestCreateSloGuard:
    def test_creates_guard_with_correct_params(self):
        from domains.infrastructure.process_guard import ProcessGuard

        g = ProcessGuard(
            slnc_path="/tmp/test.slnc",
            model_id="gpt2",
            max_restarts=3,
            restart_delay=1.0,
            worker_id="slo-guard-gpt2",
        )
        assert g._slnc_path == "/tmp/test.slnc"
        assert g._model_id == "gpt2"
        assert g.max_restarts == 3
        assert g.restart_delay == 1.0
        assert g.worker_id == "slo-guard-gpt2"

    def test_default_worker_id_convention(self):
        from domains.infrastructure.process_guard import ProcessGuard

        g = ProcessGuard(
            slnc_path="/tmp/mymodel.slnc",
            model_id="my-org/mymodel",
            worker_id="slo-guard-mymodel",
        )
        assert g.worker_id == "slo-guard-mymodel"

    def test_create_slo_guard_factory_signature(self):
        from domains.infrastructure.process_guard import create_slo_guard
        import inspect

        sig = inspect.signature(create_slo_guard)
        params = list(sig.parameters.keys())
        assert "slnc_path" in params
        assert "model_id" in params
        assert "quantize" in params
        assert "quant_bits" in params

    def test_create_slo_guard_started(self, monkeypatch):
        import domains.infrastructure.process_guard as pg_mod

        monkeypatch.setattr(pg_mod.ProcessGuard, "start", lambda self: None)
        guard = pg_mod.create_slo_guard(
            "/tmp/m.slnc",
            model_id="gpt2",
            max_restarts=1,
            quantize=True,
            quant_bits=4,
        )
        assert guard._slnc_path == "/tmp/m.slnc"
        assert guard._model_id == "gpt2"
        assert guard.worker_id == "slo-guard-gpt2"
        assert guard.max_restarts == 1
        assert guard._quantize is True
        assert guard._quant_bits == 4


class TestCreateModelGuard:
    def test_create_model_guard_configured(self, monkeypatch):
        import domains.infrastructure.process_guard as pg_mod

        monkeypatch.setattr(pg_mod.ProcessGuard, "start", lambda self: None)
        guard = pg_mod.create_model_guard(
            "my-org/model",
            max_restarts=2,
            restart_delay=0.5,
            memory_limit_mb=100.0,
            generate_timeout=3.0,
            max_concurrent=2,
        )
        assert guard.model_cls_path == "domains.infrastructure.hf_model_worker.hf_model_loader"
        assert guard.model_kwargs == {"model_id": "my-org/model", "device": "cpu"}
        assert guard.worker_id == "guard-model"
        assert guard.max_restarts == 2
        assert guard.restart_delay == 0.5
        assert guard.memory_limit_mb == 100.0
        assert guard.generate_timeout == 3.0
        assert guard._semaphore._value == 2
