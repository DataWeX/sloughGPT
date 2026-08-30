"""
Tests for domains.ops.wandb_server: inference counters + background flush loop.
"""
import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest

import domains.ops.wandb_server as ws
from domains.infrastructure.config import get_config


async def _wait_until_logged(log_payload, timeout=2.0):
    """Await the background loop's first payload, tolerant of scheduler load."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not log_payload.called:
        if loop.time() > deadline:
            break
        await asyncio.sleep(0.01)


@pytest.fixture(autouse=True)
def _reset_counters():
    ws._inference_total = 0
    ws._inference_latency_sum = 0.0
    ws._inference_tokens_sum = 0.0
    yield
    ws._inference_total = 0
    ws._inference_latency_sum = 0.0
    ws._inference_tokens_sum = 0.0


@pytest.fixture
def wandb_enabled():
    cfg = get_config()
    old_enabled = cfg.tracking.wandb_server_enabled
    old_interval = cfg.tracking.wandb_server_interval
    cfg.tracking.wandb_server_enabled = True
    cfg.tracking.wandb_server_interval = 0.01
    yield
    cfg.tracking.wandb_server_enabled = old_enabled
    cfg.tracking.wandb_server_interval = old_interval


# ── Inference counter no-op when disabled ────────────────────────────────────

def test_record_inference_noop_when_disabled():
    ws.record_inference_call(0.5, 10)
    assert ws._inference_total == 0
    assert ws._drain_inference_snapshot() == {}


def test_record_inference_multiple_noop_when_disabled():
    ws.record_inference_call(0.1, 1)
    ws.record_inference_call(0.2, 2)
    ws.record_inference_call(0.3, 3)
    assert ws._inference_total == 0


# ── Inference counter accumulation when enabled ──────────────────────────────

def test_record_inference_accumulates_when_enabled(wandb_enabled):
    ws.record_inference_call(0.2, 5)
    ws.record_inference_call(0.4, 15)
    assert ws._inference_total == 2
    assert ws._inference_latency_sum == pytest.approx(0.6)
    assert ws._inference_tokens_sum == pytest.approx(20)


def test_record_inference_single_call(wandb_enabled):
    ws.record_inference_call(1.0, 100)
    assert ws._inference_total == 1
    assert ws._inference_latency_sum == pytest.approx(1.0)
    assert ws._inference_tokens_sum == pytest.approx(100)


def test_record_inference_zero_latency(wandb_enabled):
    ws.record_inference_call(0.0, 5)
    assert ws._inference_total == 1
    assert ws._inference_latency_sum == pytest.approx(0.0)


def test_record_inference_zero_tokens(wandb_enabled):
    ws.record_inference_call(0.5, 0.0)
    assert ws._inference_total == 1
    assert ws._inference_tokens_sum == pytest.approx(0.0)


def test_record_inference_large_values(wandb_enabled):
    ws.record_inference_call(999.9, 100000.0)
    assert ws._inference_total == 1
    assert ws._inference_latency_sum == pytest.approx(999.9)
    assert ws._inference_tokens_sum == pytest.approx(100000.0)


# ── Drain snapshot ───────────────────────────────────────────────────────────

def test_drain_snapshot_metrics(wandb_enabled):
    ws.record_inference_call(0.2, 5)
    ws.record_inference_call(0.4, 15)
    snap = ws._drain_inference_snapshot()
    assert snap["inference/requests_since_flush"] == 2
    assert snap["inference/mean_latency_s"] == pytest.approx(0.3)
    assert snap["inference/mean_approx_tokens"] == pytest.approx(10)


def test_drain_resets_counters(wandb_enabled):
    ws.record_inference_call(0.2, 5)
    ws._drain_inference_snapshot()
    assert ws._drain_inference_snapshot() == {}
    assert ws._inference_total == 0


def test_drain_empty_when_no_records(wandb_enabled):
    snap = ws._drain_inference_snapshot()
    assert snap == {}


def test_drain_after_single_record(wandb_enabled):
    ws.record_inference_call(0.5, 10)
    snap = ws._drain_inference_snapshot()
    assert snap["inference/requests_since_flush"] == 1
    assert snap["inference/mean_latency_s"] == pytest.approx(0.5)
    assert snap["inference/mean_approx_tokens"] == pytest.approx(10)


def test_drain_multiple_sequential(wandb_enabled):
    ws.record_inference_call(0.1, 2)
    ws.record_inference_call(0.3, 8)
    snap1 = ws._drain_inference_snapshot()
    assert snap1["inference/requests_since_flush"] == 2

    ws.record_inference_call(0.5, 20)
    snap2 = ws._drain_inference_snapshot()
    assert snap2["inference/requests_since_flush"] == 1
    assert snap2["inference/mean_latency_s"] == pytest.approx(0.5)


def test_drain_snapshot_keys(wandb_enabled):
    ws.record_inference_call(0.1, 1)
    snap = ws._drain_inference_snapshot()
    assert "inference/requests_since_flush" in snap
    assert "inference/mean_latency_s" in snap
    assert "inference/mean_approx_tokens" in snap
    assert len(snap) == 3


def test_drain_snapshot_values_are_float(wandb_enabled):
    ws.record_inference_call(0.1, 1)
    snap = ws._drain_inference_snapshot()
    for v in snap.values():
        assert isinstance(v, float)


def test_drain_correct_mean_with_many_records(wandb_enabled):
    for i in range(10):
        ws.record_inference_call(float(i), float(i * 10))
    snap = ws._drain_inference_snapshot()
    assert snap["inference/requests_since_flush"] == 10
    assert snap["inference/mean_latency_s"] == pytest.approx(4.5)
    assert snap["inference/mean_approx_tokens"] == pytest.approx(45.0)


# ── Background start when disabled / missing wandb ──────────────────────────

def test_start_background_returns_none_when_disabled():
    result = asyncio.run(ws.start_wandb_server_background(object()))
    assert result is None


def test_start_background_returns_none_when_wandb_missing(wandb_enabled, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    result = asyncio.run(ws.start_wandb_server_background(object()))
    assert result is None


# ── wandb log/init/finish ───────────────────────────────────────────────────

def test_wandb_log_payload_calls_wandb_log(monkeypatch):
    dummy = types.ModuleType("wandb")
    calls = []
    dummy.log = lambda payload, step: calls.append((payload, step))
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    ws._wandb_log_payload({"a": 1}, 3)
    assert calls == [({"a": 1}, 3)]


def test_wandb_log_payload_multiple_calls(monkeypatch):
    dummy = types.ModuleType("wandb")
    calls = []
    dummy.log = lambda payload, step: calls.append((payload, step))
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    ws._wandb_log_payload({"x": 1}, 0)
    ws._wandb_log_payload({"y": 2}, 1)
    assert len(calls) == 2
    assert calls[0] == ({"x": 1}, 0)
    assert calls[1] == ({"y": 2}, 1)


def test_wandb_init_server_run_uses_config(monkeypatch, wandb_enabled):
    dummy = types.ModuleType("wandb")
    init_calls = []
    dummy.init = lambda **kw: init_calls.append(kw)
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    cfg = get_config()
    cfg.tracking.wandb_project = "proj-x"
    cfg.tracking.wandb_entity = "ent-y"
    cfg.tracking.wandb_mode = "offline"
    ws._wandb_init_server_run()
    assert init_calls
    assert init_calls[0]["project"] == "proj-x"
    assert init_calls[0]["entity"] == "ent-y"
    assert init_calls[0]["mode"] == "offline"
    assert init_calls[0]["job_type"] == "server"


def test_wandb_init_server_run_tags(monkeypatch, wandb_enabled):
    dummy = types.ModuleType("wandb")
    init_calls = []
    dummy.init = lambda **kw: init_calls.append(kw)
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    ws._wandb_init_server_run()
    assert "sloughgpt" in init_calls[0]["tags"]
    assert "fastapi" in init_calls[0]["tags"]
    assert "server" in init_calls[0]["tags"]


def test_wandb_finish_run_swallows_errors(monkeypatch):
    dummy = types.ModuleType("wandb")
    dummy.run = object()

    def boom():
        raise RuntimeError("no run")

    dummy.finish = boom
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    ws._wandb_finish_run()  # must not raise


def test_wandb_finish_run_skips_when_no_run(monkeypatch):
    dummy = types.ModuleType("wandb")
    dummy.run = None
    finished = []

    def finish():
        finished.append(1)

    dummy.finish = finish
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    ws._wandb_finish_run()
    assert finished == []


def test_wandb_finish_run_calls_finish(monkeypatch):
    dummy = types.ModuleType("wandb")
    dummy.run = object()
    finished = []
    dummy.finish = lambda: finished.append(True)
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    ws._wandb_finish_run()
    assert finished == [True]


# ── Background loop flushes ──────────────────────────────────────────────────

def test_background_loop_flushes(wandb_enabled, monkeypatch):
    dummy = types.ModuleType("wandb")
    dummy.log = MagicMock()
    dummy.init = MagicMock()
    dummy.run = None
    dummy.finish = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    init = MagicMock()
    finish = MagicMock()
    log_payload = MagicMock()
    monkeypatch.setattr(ws, "_wandb_init_server_run", init)
    monkeypatch.setattr(ws, "_wandb_finish_run", finish)
    monkeypatch.setattr(ws, "_wandb_log_payload", log_payload)

    class HttpMetrics:
        def wandb_aggregate(self):
            return {"http/requests": 1}

    async def run():
        task = await ws.start_wandb_server_background(HttpMetrics())
        assert task is not None
        ws.record_inference_call(0.5, 10)
        await _wait_until_logged(log_payload)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())

    assert init.called
    assert log_payload.called
    payloads = [call.args[0] for call in log_payload.call_args_list]
    assert any("http/requests" in p for p in payloads)
    assert any(p.get("inference/requests_since_flush") == 1 for p in payloads)
    assert any(p.get("inference/mean_latency_s") == pytest.approx(0.5) for p in payloads)
    assert finish.called


def test_background_loop_extra_metrics_failure_ignored(wandb_enabled, monkeypatch):
    dummy = types.ModuleType("wandb")
    dummy.log = MagicMock()
    dummy.init = MagicMock()
    dummy.run = None
    dummy.finish = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    monkeypatch.setattr(ws, "_wandb_init_server_run", MagicMock())
    monkeypatch.setattr(ws, "_wandb_finish_run", MagicMock())
    log_payload = MagicMock()
    monkeypatch.setattr(ws, "_wandb_log_payload", log_payload)

    def bad_extra():
        raise RuntimeError("psutil exploded")

    async def run():
        task = await ws.start_wandb_server_background(
            type("HM", (), {"wandb_aggregate": lambda self: {"http/requests": 1}})(),
            extra_metrics=bad_extra,
        )
        await _wait_until_logged(log_payload)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert log_payload.called
    payload = log_payload.call_args[0][0]
    assert "http/requests" in payload


def test_background_loop_skips_empty_payload(wandb_enabled, monkeypatch):
    dummy = types.ModuleType("wandb")
    dummy.log = MagicMock()
    dummy.init = MagicMock()
    dummy.run = None
    dummy.finish = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    monkeypatch.setattr(ws, "_wandb_init_server_run", MagicMock())
    monkeypatch.setattr(ws, "_wandb_finish_run", MagicMock())
    log_payload = MagicMock()
    monkeypatch.setattr(ws, "_wandb_log_payload", log_payload)

    async def run():
        task = await ws.start_wandb_server_background(
            type("HM", (), {"wandb_aggregate": lambda self: {}})()
        )
        await asyncio.sleep(0.08)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert not log_payload.called


def test_background_loop_with_no_http_metrics(wandb_enabled, monkeypatch):
    dummy = types.ModuleType("wandb")
    dummy.log = MagicMock()
    dummy.init = MagicMock()
    dummy.run = None
    dummy.finish = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    monkeypatch.setattr(ws, "_wandb_init_server_run", MagicMock())
    monkeypatch.setattr(ws, "_wandb_finish_run", MagicMock())
    log_payload = MagicMock()
    monkeypatch.setattr(ws, "_wandb_log_payload", log_payload)

    class NoHttpMetrics:
        pass

    async def run():
        task = await ws.start_wandb_server_background(NoHttpMetrics())
        ws.record_inference_call(0.1, 5)
        await _wait_until_logged(log_payload)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert log_payload.called
    payload = log_payload.call_args[0][0]
    assert "inference/requests_since_flush" in payload


def test_background_loop_step_increments(wandb_enabled, monkeypatch):
    dummy = types.ModuleType("wandb")
    dummy.log = MagicMock()
    dummy.init = MagicMock()
    dummy.run = None
    dummy.finish = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    monkeypatch.setattr(ws, "_wandb_init_server_run", MagicMock())
    monkeypatch.setattr(ws, "_wandb_finish_run", MagicMock())
    log_payload = MagicMock()
    monkeypatch.setattr(ws, "_wandb_log_payload", log_payload)

    class HttpMetrics:
        def wandb_aggregate(self):
            return {"http/requests": 1}

    async def run():
        task = await ws.start_wandb_server_background(HttpMetrics())
        ws.record_inference_call(0.1, 5)
        await _wait_until_logged(log_payload)
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    steps = [call.args[1] for call in log_payload.call_args_list]
    assert len(steps) >= 1
    assert steps[0] == 0


def test_background_loop_finishes_on_cancel(wandb_enabled, monkeypatch):
    dummy = types.ModuleType("wandb")
    dummy.log = MagicMock()
    dummy.init = MagicMock()
    dummy.run = None
    dummy.finish = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    monkeypatch.setattr(ws, "_wandb_init_server_run", MagicMock())
    finish_mock = MagicMock()
    monkeypatch.setattr(ws, "_wandb_finish_run", finish_mock)
    monkeypatch.setattr(ws, "_wandb_log_payload", MagicMock())

    async def run():
        task = await ws.start_wandb_server_background(
            type("HM", (), {"wandb_aggregate": lambda self: {}})()
        )
        await asyncio.sleep(0.03)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert finish_mock.called


# ── Thread safety of counters ────────────────────────────────────────────────

def test_counter_concurrent_increments(wandb_enabled):
    import threading

    def record_many():
        for _ in range(100):
            ws.record_inference_call(0.001, 1)

    threads = [threading.Thread(target=record_many) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ws._inference_total == 500
    assert ws._inference_latency_sum == pytest.approx(0.5)


# ── Drain thread safety ─────────────────────────────────────────────────────

def test_drain_concurrent_with_record(wandb_enabled):
    import threading

    results = []

    def drain():
        snap = ws._drain_inference_snapshot()
        results.append(snap)

    ws.record_inference_call(0.1, 5)
    t = threading.Thread(target=drain)
    t.start()
    t.join()
    assert len(results) == 1


# ── Config interaction ───────────────────────────────────────────────────────

def test_wandb_init_uses_run_name(monkeypatch, wandb_enabled):
    dummy = types.ModuleType("wandb")
    init_calls = []
    dummy.init = lambda **kw: init_calls.append(kw)
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    cfg = get_config()
    cfg.tracking.wandb_server_run_name = "test-run-42"
    ws._wandb_init_server_run()
    assert init_calls[0]["name"] == "test-run-42"


def test_wandb_init_default_run_name(monkeypatch, wandb_enabled):
    dummy = types.ModuleType("wandb")
    init_calls = []
    dummy.init = lambda **kw: init_calls.append(kw)
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    cfg = get_config()
    cfg.tracking.wandb_server_run_name = ""
    ws._wandb_init_server_run()
    assert "api-" in init_calls[0]["name"]


def test_wandb_init_entity_none_when_empty(monkeypatch, wandb_enabled):
    dummy = types.ModuleType("wandb")
    init_calls = []
    dummy.init = lambda **kw: init_calls.append(kw)
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    cfg = get_config()
    cfg.tracking.wandb_entity = ""
    ws._wandb_init_server_run()
    assert init_calls[0]["entity"] is None


# ── Extra edge cases ─────────────────────────────────────────────────────────

def test_drain_after_disable_reenable(wandb_enabled):
    ws.record_inference_call(0.1, 5)
    snap = ws._drain_inference_snapshot()
    assert snap["inference/requests_since_flush"] == 1
    ws.record_inference_call(0.2, 10)
    snap2 = ws._drain_inference_snapshot()
    assert snap2["inference/requests_since_flush"] == 1


def test_drain_snapshot_returns_float_values(wandb_enabled):
    ws.record_inference_call(1, 2)
    snap = ws._drain_inference_snapshot()
    assert isinstance(snap["inference/requests_since_flush"], float)
    assert isinstance(snap["inference/mean_latency_s"], float)
    assert isinstance(snap["inference/mean_approx_tokens"], float)


def test_record_many_calls_then_drain(wandb_enabled):
    for i in range(100):
        ws.record_inference_call(0.01 * i, float(i))
    snap = ws._drain_inference_snapshot()
    assert snap["inference/requests_since_flush"] == 100
    expected_latency = sum(0.01 * i for i in range(100)) / 100
    assert snap["inference/mean_latency_s"] == pytest.approx(expected_latency)


def test_drain_twice_returns_empty_second(wandb_enabled):
    ws.record_inference_call(0.5, 10)
    ws._drain_inference_snapshot()
    second = ws._drain_inference_snapshot()
    assert second == {}


def test_wandb_log_payload_empty_dict(monkeypatch):
    dummy = types.ModuleType("wandb")
    calls = []
    dummy.log = lambda payload, step: calls.append((payload, step))
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    ws._wandb_log_payload({}, 0)
    assert calls == [({}, 0)]


def test_wandb_finish_run_raises_runtime_error(monkeypatch):
    dummy = types.ModuleType("wandb")
    dummy.run = object()
    dummy.finish = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    ws._wandb_finish_run()  # must not raise


def test_wandb_init_server_run_mode_default(monkeypatch, wandb_enabled):
    dummy = types.ModuleType("wandb")
    init_calls = []
    dummy.init = lambda **kw: init_calls.append(kw)
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    cfg = get_config()
    cfg.tracking.wandb_mode = "online"
    ws._wandb_init_server_run()
    assert init_calls[0]["mode"] == "online"


def test_record_inference_negative_latency(wandb_enabled):
    ws.record_inference_call(-0.5, 10)
    assert ws._inference_total == 1
    assert ws._inference_latency_sum == pytest.approx(-0.5)


def test_record_inference_negative_tokens(wandb_enabled):
    ws.record_inference_call(0.5, -10)
    assert ws._inference_total == 1
    assert ws._inference_tokens_sum == pytest.approx(-10)


def test_drain_single_record_means_are_identity(wandb_enabled):
    ws.record_inference_call(3.14, 42)
    snap = ws._drain_inference_snapshot()
    assert snap["inference/mean_latency_s"] == pytest.approx(3.14)
    assert snap["inference/mean_approx_tokens"] == pytest.approx(42.0)


def test_background_loop_two_flush_cycles(wandb_enabled, monkeypatch):
    dummy = types.ModuleType("wandb")
    dummy.log = MagicMock()
    dummy.init = MagicMock()
    dummy.run = None
    dummy.finish = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    monkeypatch.setattr(ws, "_wandb_init_server_run", MagicMock())
    monkeypatch.setattr(ws, "_wandb_finish_run", MagicMock())
    log_payload = MagicMock()
    monkeypatch.setattr(ws, "_wandb_log_payload", log_payload)

    async def run():
        task = await ws.start_wandb_server_background(
            type("HM", (), {"wandb_aggregate": lambda self: {"http/requests": 1}})()
        )
        ws.record_inference_call(0.1, 5)
        ws.record_inference_call(0.2, 10)
        await _wait_until_logged(log_payload)
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert log_payload.call_count >= 1
    payloads = [call.args[0] for call in log_payload.call_args_list]
    assert any("inference/requests_since_flush" in p for p in payloads)


def test_counter_resets_after_drain(wandb_enabled):
    ws.record_inference_call(1.0, 100)
    ws._drain_inference_snapshot()
    assert ws._inference_total == 0
    assert ws._inference_latency_sum == 0.0
    assert ws._inference_tokens_sum == 0.0


def test_thread_safety_drain_concurrent(wandb_enabled):
    import threading
    ws.record_inference_call(0.1, 5)
    results = []
    barrier = threading.Barrier(5)

    def drain():
        barrier.wait()
        snap = ws._drain_inference_snapshot()
        results.append(snap)

    threads = [threading.Thread(target=drain) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    non_empty = [r for r in results if r]
    assert len(non_empty) == 1


def test_wandb_init_server_run_api_key(monkeypatch, wandb_enabled):
    dummy = types.ModuleType("wandb")
    init_calls = []
    dummy.init = lambda **kw: init_calls.append(kw)
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    cfg = get_config()
    cfg.tracking.wandb_api_key = "sk-test-123"
    ws._wandb_init_server_run()
    assert init_calls[0]["api_key"] == "sk-test-123"


def test_wandb_init_server_run_no_api_key(monkeypatch, wandb_enabled):
    dummy = types.ModuleType("wandb")
    init_calls = []
    dummy.init = lambda **kw: init_calls.append(kw)
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    cfg = get_config()
    cfg.tracking.wandb_api_key = ""
    ws._wandb_init_server_run()
    assert init_calls[0]["api_key"] is None


def test_drain_empty_metrics_dict(wandb_enabled):
    snap = ws._drain_inference_snapshot()
    assert snap == {}
    assert isinstance(snap, dict)


def test_record_zero_tokens_and_zero_latency(wandb_enabled):
    ws.record_inference_call(0.0, 0.0)
    assert ws._inference_total == 1
    snap = ws._drain_inference_snapshot()
    assert snap["inference/mean_latency_s"] == pytest.approx(0.0)
    assert snap["inference/mean_approx_tokens"] == pytest.approx(0.0)
