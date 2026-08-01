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


def test_record_inference_noop_when_disabled():
    ws.record_inference_call(0.5, 10)
    assert ws._inference_total == 0
    assert ws._drain_inference_snapshot() == {}


def test_record_inference_accumulates_when_enabled(wandb_enabled):
    ws.record_inference_call(0.2, 5)
    ws.record_inference_call(0.4, 15)
    assert ws._inference_total == 2
    assert ws._inference_latency_sum == pytest.approx(0.6)
    assert ws._inference_tokens_sum == pytest.approx(20)


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


def test_start_background_returns_none_when_disabled():
    result = asyncio.run(ws.start_wandb_server_background(object()))
    assert result is None


def test_start_background_returns_none_when_wandb_missing(wandb_enabled, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    result = asyncio.run(ws.start_wandb_server_background(object()))
    assert result is None


def test_wandb_log_payload_calls_wandb_log(monkeypatch):
    dummy = types.ModuleType("wandb")
    calls = []
    dummy.log = lambda payload, step: calls.append((payload, step))
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    ws._wandb_log_payload({"a": 1}, 3)
    assert calls == [({"a": 1}, 3)]


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
        await asyncio.sleep(0.08)
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
        await asyncio.sleep(0.08)
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
