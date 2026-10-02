"""Characterization tests for the health read model.

``get_detailed_health()`` is projected three different ways — by
``/health/debug``, ``/health/summary`` and the ``/health/stream`` SSE
snapshot. Each projection hand-picks keys and flattens some of them, so the
three lists and the producer can silently drift apart.

These tests pin the EXACT current wire shape of all three projections plus
the produced key set, so the Field-table refactor can be proven to be a
zero-behavior-change change. If one fails after a refactor, the refactor
changed the wire shape — revert, or consciously update both together.

The phantom-read guard at the bottom catches the drift class this file was
written for: a projection reading a key no producer ever emits (such reads
silently fall back to their default forever).
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_all_handlers

from apps.api.server.controllers.health import HealthController
from apps.api.server.routers.health import router

SERVER_DIR = str(Path(__file__).resolve().parents[2] / "apps" / "api" / "server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

# Keys every real ``get_detailed_health()`` call must emit.
REQUIRED_PRODUCED = frozenset(
    {
        "avg_latency_ms",
        "avg_tokens_per_request",
        "bandwidth",
        "degraded",
        "device",
        "error_count",
        "gpu",
        "health_history",
        "health_score",
        "inference",
        "inference_count",
        "kv_sessions",
        "lifecycle",
        "memory_history",
        "memory_pressure",
        "model_events",
        "model_loaded",
        "model_loading",
        "model_metrics",
        "model_type",
        "mps_monitor",
        "p95_latency_ms",
        "path_latencies",
        "process_guard",
        "quantization",
        "rate_violations",
        "recent_errors",
        "registry",
        "request_count",
        "requests_per_minute",
        "resource_allocation",
        "soul",
        "startup_progress",
        "status",
        "status_message",
        "system",
        "timestamp",
        "tokens_per_sec",
        "total_tokens",
        "training_pool",
        "uptime_seconds",
        "versions",
    }
)

# Read by a projection but NEVER emitted by get_detailed_health().
# Both are emitted by get_basic_health() instead (controller lines 500/525/529)
# — the SSE snapshot projection was evidently written against the basic payload.
# Both preserve current behaviour by decision; follow-up card filed.
PHANTOM_READS_ALLOWED = frozenset({"is_inferencing", "num_parameters"})


def make_full_detailed() -> dict:
    """Deterministic payload covering every key the controller emits.

    Includes ``num_parameters`` (a phantom read — see PHANTOM_READS_ALLOWED) so
    every projection read path is exercised both present and absent.
    """
    return {
        "status": "healthy",
        "timestamp": "2026-01-02T03:04:05Z",
        "uptime_seconds": 12.5,
        "request_count": 10,
        "error_count": 1,
        "avg_latency_ms": 25.5,
        "p95_latency_ms": 40.0,
        "requests_per_minute": 3,
        "path_latencies": [{"path": "/chat", "p50": 1.0}],
        "recent_errors": [{"error": "boom"}],
        "inference_count": 7,
        "total_tokens": 1234,
        "tokens_per_sec": 2.5,
        "avg_tokens_per_request": 3.0,
        "health_score": {
            "score": 85,
            "status": "healthy",
            "summary": "All good",
            "diagnoses": [{"check": "model", "score": 100}],
        },
        "model_metrics": [{"name": "m"}],
        "model_events": [{"event": "e"}],
        "health_history": [{"score": 85}],
        "memory_history": [{"rss_mb": 1.0}],
        "rate_violations": [{"rule": "r"}],
        "degraded": ["disk"],
        "system": {"cpu_percent": 45.0, "memory_percent": 60.0, "memory_available_mb": 1024},
        "gpu": {"backend": "cuda"},
        "mps_monitor": {"active": False},
        "model_loaded": True,
        "model_loading": False,
        "model_type": "gpt2",
        "device": "cpu",
        "soul": "sage",
        "inference": {"is_inferencing": True, "inference_count": 7},
        "registry": {"models": []},
        "quantization": {"method": "none"},
        "kv_sessions": {"enabled": False},
        "lifecycle": {"is_running": True, "phase": "running"},
        "startup_progress": {
            "stage": "ready",
            "stage_value": 100,
            "elapsed_seconds": 3.0,
            "model_progress": 1.0,
            "model_progress_message": "done",
            "hooks": {},
        },
        "training_pool": {"active": 1},
        "resource_allocation": {"mode": "auto"},
        "process_guard": {"active": False},
        "memory_pressure": {"level": "ok"},
        "versions": {"app": "1.0"},
        "bandwidth": None,
        "num_parameters": 124_000_000,
    }


# --- pinned outputs (captured from the pre-refactor implementation) ----------

EXPECTED_DEBUG_INFO = {
    "avg_latency_ms": 25.5,
    "avg_tokens_per_request": 3.0,
    "cpu_percent": 45.0,
    "error_count": 1,
    "gpu_backend": "cuda",
    "health_history": [{"score": 85}],
    "health_score": {
        "diagnoses": [{"check": "model", "score": 100}],
        "score": 85,
        "status": "healthy",
        "summary": "All good",
    },
    "inference_count": 7,
    "memory_history": [{"rss_mb": 1.0}],
    "memory_percent": 60.0,
    "model_events": [{"event": "e"}],
    "model_loaded": True,
    "model_metrics": [{"name": "m"}],
    "model_type": "gpt2",
    "path_latencies": [{"p50": 1.0, "path": "/chat"}],
    "rate_violations": [{"rule": "r"}],
    "recent_errors": [{"error": "boom"}],
    "request_count": 10,
    "requests_per_minute": 3,
    "soul": "sage",
    "tokens_per_sec": 2.5,
    "total_tokens": 1234,
    "uptime_seconds": 12.5,
}

EXPECTED_HEALTH_SUMMARY = {
    "cpu_percent": 45.0,
    "diagnoses": [{"check": "model", "score": 100}],
    "error_count": 1,
    "memory_percent": 60.0,
    "model_loaded": True,
    "model_loading": False,
    "model_type": "gpt2",
    "request_count": 10,
    "score": 85,
    "soul": "sage",
    "status": "healthy",
    "summary": "All good",
    "tokens_per_sec": 2.5,
    "uptime_seconds": 12.5,
}

EXPECTED_SSE_SNAPSHOT = {
    "avg_latency_ms": 25.5,
    "avg_tokens_per_request": 3.0,
    "bandwidth": None,
    "cpu_percent": 45.0,
    "diagnoses": [{"check": "model", "score": 100}],
    "error_count": 1,
    "health_history": [{"score": 85}],
    "health_score": 85,
    "health_status": "healthy",
    "health_summary": "All good",
    "inference_count": 7,
    # Pinned on purpose: reads the flat key, which the detailed payload never
    # emits, so this stays False even while generating. See the follow-up card.
    "is_inferencing": False,
    "memory_history": [{"rss_mb": 1.0}],
    "memory_percent": 60.0,
    "model_events": [{"event": "e"}],
    "model_loaded": True,
    "model_loading": False,
    "model_metrics": [{"name": "m"}],
    "model_type": "gpt2",
    "num_parameters": 124_000_000,
    "path_latencies": [{"p50": 1.0, "path": "/chat"}],
    "quantization": {"method": "none"},
    "rate_violations": [{"rule": "r"}],
    "recent_errors": [{"error": "boom"}],
    "request_count": 10,
    "requests_per_minute": 3,
    "soul": "sage",
    "startup_elapsed": 3.0,
    "startup_hooks": {},
    "startup_model_progress": 1.0,
    "startup_model_progress_message": "done",
    "startup_progress": {
        "elapsed_seconds": 3.0,
        "hooks": {},
        "model_progress": 1.0,
        "model_progress_message": "done",
        "stage": "ready",
        "stage_value": 100,
    },
    "startup_stage": "ready",
    "startup_stage_value": 100,
    "tokens_per_sec": 2.5,
    "total_tokens": 1234,
    "training_pool": {"active": 1},
    "uptime_seconds": 12.5,
}


# --- harness ----------------------------------------------------------------


@pytest.fixture
def pinned():
    """Client whose controller patch stays active for the whole test."""
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(router)

    ctrl = MagicMock()
    ctrl.get_detailed_health.return_value = make_full_detailed()

    with patch("apps.api.server.routers.health.get_health_controller", return_value=ctrl):
        yield TestClient(app, raise_server_exceptions=False)


def _sse_envelope(client: TestClient) -> dict:
    with (
        patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[False, True])),
        patch("asyncio.sleep", new=AsyncMock(return_value=None)),
    ):
        with client.stream("GET", "/health/stream") as resp:
            assert resp.status_code == 200
            body = resp.read()
    lines = [ln for ln in body.decode().split("\r\n") if ln]
    first = next(ln for ln in lines if ln.startswith("data: "))
    return json.loads(first[6:])


# --- characterization -------------------------------------------------------


class TestDebugInfoProjection:
    def test_exact_output(self, pinned):
        assert pinned.get("/health/debug").json()["data"] == EXPECTED_DEBUG_INFO

    def test_key_count(self, pinned):
        assert len(pinned.get("/health/debug").json()["data"]) == 23


class TestHealthSummaryProjection:
    def test_exact_output(self, pinned):
        assert pinned.get("/health/summary").json()["data"] == EXPECTED_HEALTH_SUMMARY

    def test_key_count(self, pinned):
        assert len(pinned.get("/health/summary").json()["data"]) == 14


class TestSseSnapshotProjection:
    def test_exact_data(self, pinned):
        assert _sse_envelope(pinned)["data"] == EXPECTED_SSE_SNAPSHOT

    def test_key_count(self, pinned):
        assert len(_sse_envelope(pinned)["data"]) == 38

    def test_envelope_shape(self, pinned):
        env = _sse_envelope(pinned)
        assert set(env) == {"stream", "phase", "status", "data", "meta", "message"}

    def test_is_inferencing_stays_false_despite_nested_true(self, pinned):
        """Guards the preserved bug: flat read, detailed nests it."""
        assert make_full_detailed()["inference"]["is_inferencing"] is True
        assert _sse_envelope(pinned)["data"]["is_inferencing"] is False


class TestProducedKeySet:
    def test_required_keys_always_emitted(self):
        produced = set(HealthController().get_detailed_health())
        missing = REQUIRED_PRODUCED - produced
        assert not missing, f"produced payload lost keys: {sorted(missing)}"

    def test_no_undeclared_extra_keys(self):
        produced = set(HealthController().get_detailed_health())
        extra = produced - REQUIRED_PRODUCED
        assert not extra, f"new undeclared keys (update the schema): {sorted(extra)}"

    def test_num_parameters_not_produced_by_detailed(self):
        """SSE reads it, but only get_basic_health() emits it (phantom read)."""
        assert "num_parameters" not in HealthController().get_detailed_health()


class TestPhantomReadGuard:
    """A projection reading a key nobody produces silently sends the default."""

    @staticmethod
    class _Spy(dict):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.accessed: set[str] = set()

        def get(self, key, default=None):  # noqa: D102
            self.accessed.add(key)
            return super().get(key, default)

    def _phantom_reads(self, detailed: dict, call) -> set[str]:
        spy = self._Spy(detailed)
        ctrl = MagicMock()
        ctrl.get_detailed_health.return_value = spy
        app = FastAPI()
        register_all_handlers(app)
        app.include_router(router)
        with patch("apps.api.server.routers.health.get_health_controller", return_value=ctrl):
            client = TestClient(app, raise_server_exceptions=False)
            call(client, spy)
        return spy.accessed - set(detailed)

    def test_debug_info_reads_only_produced_keys(self):
        produced = HealthController().get_detailed_health()
        phantom = self._phantom_reads(produced, lambda c, s: c.get("/health/debug"))
        assert not phantom - PHANTOM_READS_ALLOWED, sorted(phantom)

    def test_health_summary_reads_only_produced_keys(self):
        produced = HealthController().get_detailed_health()
        phantom = self._phantom_reads(produced, lambda c, s: c.get("/health/summary"))
        assert not phantom - PHANTOM_READS_ALLOWED, sorted(phantom)

    def test_sse_reads_only_produced_keys(self):
        produced = HealthController().get_detailed_health()

        def call(client, spy):
            with (
                patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[False, True])),
                patch("asyncio.sleep", new=AsyncMock(return_value=None)),
            ):
                with client.stream("GET", "/health/stream") as resp:
                    resp.read()

        phantom = self._phantom_reads(produced, call)
        assert not phantom - PHANTOM_READS_ALLOWED, sorted(phantom)
