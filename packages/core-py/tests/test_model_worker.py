"""Tests for domains.infrastructure.model_worker — WorkerHealth, _new_session_id,
WorkerStreamStalledError, constants.

Covers: dataclass creation, session ID generation, error class, timeout constants.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.infrastructure.model_worker import (
    WorkerHealth,
    _new_session_id,
    WorkerStreamStalledError,
    _STREAM_PUT_TIMEOUT_S,
    _STALL_TIMEOUT_S,
    ModelWorkerProcess,
    _session_ids,
)


class TestWorkerHealth:
    def test_defaults(self):
        h = WorkerHealth()
        assert h.pid is None
        assert h.alive is False
        assert h.started_at == 0.0
        assert h.last_heartbeat == 0.0
        assert h.requests_served == 0
        assert h.errors == 0
        assert h.crashed is False
        assert h.crash_count == 0

    def test_custom(self):
        h = WorkerHealth(pid=1234, alive=True, requests_served=50)
        assert h.pid == 1234
        assert h.alive is True
        assert h.requests_served == 50

    def test_with_errors(self):
        h = WorkerHealth(errors=5)
        assert h.errors == 5

    def test_crashed_state(self):
        h = WorkerHealth(crashed=True, crash_count=3)
        assert h.crashed is True
        assert h.crash_count == 3

    def test_heartbeat_timestamp(self):
        ts = 1234567890.0
        h = WorkerHealth(last_heartbeat=ts)
        assert h.last_heartbeat == ts

    def test_started_at_timestamp(self):
        ts = 1234567890.0
        h = WorkerHealth(started_at=ts)
        assert h.started_at == ts

    def test_all_fields_settable(self):
        h = WorkerHealth(
            pid=9999,
            alive=True,
            started_at=1.0,
            last_heartbeat=2.0,
            requests_served=100,
            errors=5,
            crashed=True,
            crash_count=2,
        )
        assert h.pid == 9999
        assert h.alive is True
        assert h.started_at == 1.0
        assert h.last_heartbeat == 2.0
        assert h.requests_served == 100
        assert h.errors == 5
        assert h.crashed is True
        assert h.crash_count == 2

    def test_zero_pid(self):
        h = WorkerHealth(pid=0)
        assert h.pid == 0

    def test_negative_crash_count(self):
        h = WorkerHealth(crash_count=-1)
        assert h.crash_count == -1


class TestNewSessionId:
    def test_format(self):
        sid = _new_session_id()
        assert sid.startswith("req-")
        assert str(os.getpid()) in sid

    def test_unique(self):
        ids = {_new_session_id() for _ in range(10)}
        assert len(ids) == 10

    def test_contains_pid(self):
        sid = _new_session_id()
        parts = sid.split("-")
        assert len(parts) == 3
        assert parts[0] == "req"
        assert parts[1] == str(os.getpid())

    def test_monotonic_increasing(self):
        id1 = _new_session_id()
        id2 = _new_session_id()
        num1 = int(id1.split("-")[-1])
        num2 = int(id2.split("-")[-1])
        assert num2 > num1

    def test_format_after_many(self):
        for _ in range(100):
            sid = _new_session_id()
            assert sid.startswith("req-")
            assert str(os.getpid()) in sid

    def test_session_id_counter_increments(self):
        initial = next(_session_ids)
        _new_session_id()
        after = next(_session_ids)
        assert after >= initial


class TestWorkerStreamStalledError:
    def test_is_runtime_error(self):
        e = WorkerStreamStalledError("stalled")
        assert isinstance(e, RuntimeError)
        assert "stalled" in str(e)

    def test_error_message(self):
        e = WorkerStreamStalledError("Worker stalled for 30s")
        assert "Worker stalled for 30s" in str(e)

    def test_can_be_caught(self):
        try:
            raise WorkerStreamStalledError("test")
        except RuntimeError as e:
            assert "test" in str(e)

    def test_with_empty_message(self):
        e = WorkerStreamStalledError("")
        assert isinstance(e, RuntimeError)
        assert str(e) == ""

    def test_with_long_message(self):
        msg = "x" * 10000
        e = WorkerStreamStalledError(msg)
        assert len(str(e)) == 10000

    def test_exception_chain(self):
        try:
            try:
                raise ValueError("original")
            except ValueError as orig:
                raise WorkerStreamStalledError("stalled") from orig
        except WorkerStreamStalledError as e:
            assert e.__cause__ is not None


class TestConstants:
    def test_stream_put_timeout(self):
        assert _STREAM_PUT_TIMEOUT_S == 30.0

    def test_stall_timeout(self):
        assert _STALL_TIMEOUT_S == 30.0

    def test_timeout_is_float(self):
        assert isinstance(_STREAM_PUT_TIMEOUT_S, float)
        assert isinstance(_STALL_TIMEOUT_S, float)

    def test_timeouts_equal(self):
        assert _STREAM_PUT_TIMEOUT_S == _STALL_TIMEOUT_S

    def test_positive_values(self):
        assert _STREAM_PUT_TIMEOUT_S > 0
        assert _STALL_TIMEOUT_S > 0


class TestModelWorkerProcess:
    def test_requires_model_path(self):
        with pytest.raises(ValueError, match="requires either"):
            ModelWorkerProcess(worker_id="test")

    def test_slo_backend(self):
        w = ModelWorkerProcess(slnc_path="test.slnc", model_id="test")
        assert w.backend == "slo"
        assert w._use_slo is True

    def test_hf_backend(self):
        w = ModelWorkerProcess(model_cls_path="transformers.AutoModel")
        assert w.backend == "hf"
        assert w._use_slo is False

    def test_worker_id(self):
        w = ModelWorkerProcess(slnc_path="test.slnc", worker_id="my_worker")
        assert w.worker_id == "my_worker"

    def test_default_worker_id(self):
        w = ModelWorkerProcess(slnc_path="test.slnc")
        assert w.worker_id == "worker"

    def test_quantize_params(self):
        w = ModelWorkerProcess(
            slnc_path="test.slnc",
            quantize=True,
            quant_bits=4,
            quant_mode="asymmetric",
            quant_clip=0.95,
        )
        assert w._quantize is True
        assert w._quant_bits == 4
        assert w._quant_mode == "asymmetric"
        assert w._quant_clip == 0.95

    def test_default_quantize_params(self):
        w = ModelWorkerProcess(slnc_path="test.slnc")
        assert w._quantize is False
        assert w._quant_bits == 8
        assert w._quant_mode == "symmetric"
        assert w._quant_clip == 0.999

    def test_generate_timeout(self):
        w = ModelWorkerProcess(slnc_path="test.slnc", generate_timeout=60.0)
        assert w._generate_timeout == 60.0

    def test_startup_timeout(self):
        w = ModelWorkerProcess(slnc_path="test.slnc", startup_timeout=10.0)
        assert w._startup_timeout == 10.0

    def test_stall_timeout(self):
        w = ModelWorkerProcess(slnc_path="test.slnc", stall_timeout=15.0)
        assert w._stall_timeout == 15.0

    def test_extra_sys_paths(self):
        w = ModelWorkerProcess(slnc_path="test.slnc", extra_sys_paths=["/tmp"])
        assert w._extra_sys_paths == ["/tmp"]

    def test_default_extra_sys_paths(self):
        w = ModelWorkerProcess(slnc_path="test.slnc")
        assert w._extra_sys_paths == []

    def test_hf_model_kwargs(self):
        w = ModelWorkerProcess(
            model_cls_path="transformers.AutoModel",
            model_kwargs={"pretrained_model_name_or_path": "gpt2"},
        )
        assert w._model_kwargs == {"pretrained_model_name_or_path": "gpt2"}

    def test_default_hf_model_kwargs(self):
        w = ModelWorkerProcess(model_cls_path="transformers.AutoModel")
        assert w._model_kwargs == {}

    def test_split_resp_2tuple(self):
        session_id, data = ModelWorkerProcess._split_resp("result", ["data1"])
        assert session_id is None
        assert data == "data1"

    def test_split_resp_3tuple(self):
        session_id, data = ModelWorkerProcess._split_resp("result", ["session1", "data1"])
        assert session_id == "session1"
        assert data == "data1"

    def test_split_resp_many_elements(self):
        session_id, data = ModelWorkerProcess._split_resp("result", ["session1", "data1", "extra"])
        assert session_id == "session1"
        assert data == "data1"

    def test_health_not_alive_initially(self):
        w = ModelWorkerProcess(slnc_path="test.slnc")
        assert w.alive is False

    def test_health_check_returns_worker_health(self):
        w = ModelWorkerProcess(slnc_path="test.slnc")
        h = w.health_check()
        assert isinstance(h, WorkerHealth)
        assert h.alive is False

    def test_stop_when_not_started(self):
        w = ModelWorkerProcess(slnc_path="test.slnc")
        w.stop()  # Should not raise

    def test_context_manager_not_started(self):
        w = ModelWorkerProcess(slnc_path="test.slnc")
        # Just test that the class can be instantiated
        assert w.backend == "slo"
