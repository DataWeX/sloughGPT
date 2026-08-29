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


class TestNewSessionId:
    def test_format(self):
        sid = _new_session_id()
        assert sid.startswith("req-")
        assert str(os.getpid()) in sid

    def test_unique(self):
        ids = {_new_session_id() for _ in range(10)}
        assert len(ids) == 10


class TestWorkerStreamStalledError:
    def test_is_runtime_error(self):
        e = WorkerStreamStalledError("stalled")
        assert isinstance(e, RuntimeError)
        assert "stalled" in str(e)


class TestConstants:
    def test_stream_put_timeout(self):
        assert _STREAM_PUT_TIMEOUT_S == 30.0

    def test_stall_timeout(self):
        assert _STALL_TIMEOUT_S == 30.0
