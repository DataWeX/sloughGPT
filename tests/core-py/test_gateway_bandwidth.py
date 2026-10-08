"""Tests for domain/infrastructure/_internal/gateway_bandwidth.py.

Covers the mirror's four contracts:
  1. Delta math — baseline, idle silence, counter resets, expansion.
  2. Absent-tolerant polling — silent on closed ports, transition logging.
  3. Schema integration — op → domain payload → dashboard category.
  4. Passthrough — snapshot property, metrics gauges, daemon thread.
"""

from __future__ import annotations

import http.server
import json
import logging
import threading
import time

import pytest

from domain.infrastructure._internal.gateway_bandwidth import (
    DEFAULT_GATEWAY_URL,
    GatewayBandwidthMirror,
    get_bandwidth_mirror,
)
from domain.logging._internal.config import (
    _collect_domain_payload,
    _derive_op,
)
from domain.logging._internal.dashboard_filter import (
    _WATCHED_OPS,
    _summarize_from_op,
)

LOGGER_NAME = "slo.infrastructure.gateway"


def _bandwidth(
    identity: int = 0,
    wire: int = 0,
    compressed: int = 0,
    identity_responses: int = 0,
    zstd: int = 0,
    gzip: int = 0,
) -> dict:
    """Build a cumulative snapshot in the gateway's key shape."""
    saved = identity - wire
    return {
        "identity_bytes": identity,
        "wire_bytes": wire,
        "saved_bytes": saved,
        "saved_pct": (saved * 100.0 / identity) if identity else 0.0,
        "compressed_responses": compressed,
        "identity_responses": identity_responses,
        "zstd_responses": zstd,
        "gzip_responses": gzip,
    }


def _capinfo(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER_NAME]


# ── loopback gateway fixture ─────────────────────────────────────────────────


class _GatewayHandler(http.server.BaseHTTPRequestHandler):
    """Serves GET /health/detailed from a class-level payload dict."""

    payload: dict = {}

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path == "/health/detailed":
            body = json.dumps(type(self).payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, *args) -> None:  # silence request logging
        pass


class _LoopbackGateway:
    def __init__(self) -> None:
        _GatewayHandler.payload = {}
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), _GatewayHandler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def set_payload(self, payload: dict) -> None:
        _GatewayHandler.payload = payload

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


@pytest.fixture()
def gateway():
    gw = _LoopbackGateway()
    yield gw
    gw.shutdown()


# ── 1. delta math ────────────────────────────────────────────────────────────


class TestDeltaMath:
    def test_first_sample_is_baseline_only(self):
        mirror = GatewayBandwidthMirror(base_url="http://unused:1")
        assert mirror._ingest(_bandwidth(identity=1000, wire=900)) is None
        assert mirror.snapshot == _bandwidth(identity=1000, wire=900)

    def test_delta_of_two_samples_is_logged_exactly(self, caplog):
        mirror = GatewayBandwidthMirror(base_url="http://unused:1")
        mirror._ingest(_bandwidth(identity=1000, wire=900, compressed=2))
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            delta = mirror._ingest(
                _bandwidth(
                    identity=1000 + 4096,
                    wire=900 + 1024,
                    compressed=4,
                    identity_responses=1,
                )
            )
        assert delta is not None
        assert delta["identity_bytes"] == 4096
        assert delta["wire_bytes"] == 1024
        assert delta["saved_bytes"] == 3072
        assert delta["compressed_responses"] == 2

        recs = _capinfo(caplog)
        assert len(recs) == 1
        rec = recs[0]
        assert _derive_op(rec) == "bandwidth.interval"
        assert rec.identity_bytes == 4096
        assert rec.wire_bytes == 1024
        assert rec.saved_bytes == 3072
        assert rec.responses == 3  # (4-2) compressed + (1-0) identity
        assert rec.saved_pct == pytest.approx(75.0)
        assert "edge" in rec.getMessage() and "saved" in rec.getMessage()

    def test_idle_interval_emits_nothing(self, caplog):
        mirror = GatewayBandwidthMirror(base_url="http://unused:1")
        snap = _bandwidth(identity=500, wire=400)
        mirror._ingest(snap)
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            assert mirror._ingest(dict(snap)) is None
        assert _capinfo(caplog) == []

    def test_counter_reset_rebaselines_silently(self, caplog):
        mirror = GatewayBandwidthMirror(base_url="http://unused:1")
        mirror._ingest(_bandwidth(identity=100_000, wire=90_000))
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            # Gateway restarted: counters dropped to near zero.
            assert mirror._ingest(_bandwidth(identity=100, wire=90)) is None
            # Next interval is measured from the reset baseline.
            delta = mirror._ingest(_bandwidth(identity=1100, wire=590))
        assert delta is not None
        assert delta["identity_bytes"] == 1000
        assert delta["wire_bytes"] == 500
        # No negative-identity record was ever emitted.
        assert not any(getattr(r, "identity_bytes", 0) < 0 for r in _capinfo(caplog))

    def test_wire_expansion_reports_negative_savings(self, caplog):
        """An incompressible payload can grow on the wire — savings must not lie."""
        mirror = GatewayBandwidthMirror(base_url="http://unused:1")
        mirror._ingest(_bandwidth(identity=100, wire=90))
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            delta = mirror._ingest(_bandwidth(identity=200, wire=210))
        assert delta is not None
        # Deltas: identity +100, wire +120 → saved -20, -20%.
        assert delta["identity_bytes"] == 100
        assert delta["wire_bytes"] == 120
        rec = _capinfo(caplog)[0]
        assert rec.saved_bytes == -20
        assert rec.saved_pct == pytest.approx(-20.0)


# ── 2. absent-tolerant polling ───────────────────────────────────────────────


class TestAbsentGateway:
    def test_closed_port_is_silent_on_every_level(self, caplog):
        # Port 9 (discard) — connection refused, no server ever.
        mirror = GatewayBandwidthMirror(base_url="http://127.0.0.1:9")
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            assert mirror.poll_once() is None
            assert mirror.poll_once() is None
        assert _capinfo(caplog) == []  # fresh mirror: not even a debug line

    def test_disconnect_logs_debug_once_on_transition(self, caplog, gateway):
        gateway.set_payload({"bandwidth": _bandwidth(identity=100, wire=50)})
        mirror = GatewayBandwidthMirror(base_url=gateway.base_url)
        assert mirror.poll_once() is not None  # connected

        gateway.shutdown()  # gateway goes away
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            assert mirror.poll_once() is None
            assert mirror.poll_once() is None  # already idle — no repeat
        debugs = [r for r in _capinfo(caplog) if r.levelno == logging.DEBUG]
        assert len(debugs) == 1
        assert "unreachable" in debugs[0].getMessage()
        assert not any(r.levelno >= logging.WARNING for r in _capinfo(caplog))

    def test_payload_without_bandwidth_block_is_ignored(self, caplog, gateway):
        # Simulate a gateway that predates the counters: /health/detailed
        # answers, but carries no bandwidth block.
        gateway.set_payload({"status": "ok"})
        mirror = GatewayBandwidthMirror(base_url=gateway.base_url)
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            assert mirror.poll_once() is None
        assert mirror.snapshot is None
        # Connect transition may log, but no delta record may fire.
        assert not any(getattr(r, "op", None) == "bandwidth.interval" for r in _capinfo(caplog))

    def test_empty_url_disables_polling(self, monkeypatch, caplog):
        monkeypatch.setenv("MAN_GATEWAY_URL", "")
        mirror = GatewayBandwidthMirror()  # env-driven construction
        assert mirror.enabled is False
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            assert mirror.poll_once() is None
            assert mirror.start() is False
        assert _capinfo(caplog) == []

    def test_env_url_and_default(self, monkeypatch):
        monkeypatch.setenv("MAN_GATEWAY_URL", "http://gw.local:8080/")
        assert GatewayBandwidthMirror().base_url == "http://gw.local:8080"
        monkeypatch.delenv("MAN_GATEWAY_URL", raising=False)
        assert GatewayBandwidthMirror().base_url == DEFAULT_GATEWAY_URL


# ── 3. schema integration ────────────────────────────────────────────────────


class TestSchemaIntegration:
    def _emit(self, caplog) -> logging.LogRecord:
        from domain.infrastructure._internal.structured_log import StructuredLogger

        mirror_log = StructuredLogger(LOGGER_NAME)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            mirror_log.info(
                "edge 1.0 MB → 256.0 KB on the wire (75.0% saved)",
                op="bandwidth.interval",
                identity_bytes=1_048_576,
                wire_bytes=262_144,
                saved_bytes=786_432,
                saved_pct=75.0,
                responses=3,
            )
        return _capinfo(caplog)[0]

    def test_domain_payload_collects_registered_keys_only(self, caplog):
        rec = self._emit(caplog)
        assert _derive_op(rec) == "bandwidth.interval"
        payload = _collect_domain_payload(rec, "bandwidth")
        assert payload == {
            "identity_bytes": 1_048_576,
            "wire_bytes": 262_144,
            "saved_bytes": 786_432,
            "saved_pct": 75.0,
            "responses": 3,
        }

    def test_unregistered_keys_are_not_in_payload(self, caplog):
        rec = self._emit(caplog)
        rec.extra_field = "should-not-appear"  # not in _DOMAIN_KEYS["bandwidth"]
        payload = _collect_domain_payload(rec, "bandwidth")
        assert "extra_field" not in payload

    def test_dashboard_category_is_infra(self):
        assert _WATCHED_OPS["bandwidth"] == "INFRA"

    def test_dashboard_summary_is_punchy(self):
        rec = logging.LogRecord("t", logging.INFO, "", 0, "msg", (), None)
        rec.identity_bytes = 2 * 1024 * 1024
        rec.wire_bytes = 512 * 1024
        rec.saved_pct = 75.0
        category, message = _summarize_from_op(rec, "bandwidth.interval")
        assert category == "INFRA"
        assert message.startswith("Edge 2.0 MB → 512.0 KB")
        assert "75.0% saved" in message

    def test_legacy_tag_maps_to_bandwidth_op(self):
        from domain.logging._internal.config import _LEGACY_TAG_TO_OP

        assert _LEGACY_TAG_TO_OP["BANDWIDTH"] == "bandwidth.interval"


# ── 4. passthrough — snapshot, metrics, thread ───────────────────────────────


class TestPassthrough:
    def test_snapshot_follows_latest_cumulative_sample(self, gateway):
        gateway.set_payload({"bandwidth": _bandwidth(identity=100, wire=50)})
        mirror = GatewayBandwidthMirror(base_url=gateway.base_url)
        assert mirror.snapshot is None
        mirror.poll_once()
        assert mirror.snapshot == _bandwidth(identity=100, wire=50)

        gateway.set_payload({"bandwidth": _bandwidth(identity=300, wire=120)})
        mirror.poll_once()
        assert mirror.snapshot["identity_bytes"] == 300
        assert mirror.snapshot["wire_bytes"] == 120

    def test_metrics_gauges_render_after_sample(self, gateway):
        from domain.infrastructure._internal.metrics import get_metrics_collector

        gateway.set_payload({"bandwidth": _bandwidth(identity=4096, wire=1024)})
        mirror = GatewayBandwidthMirror(base_url=gateway.base_url)
        mirror.poll_once()
        text = get_metrics_collector().render()
        assert "sloughgpt_gateway_identity_bytes 4096" in text
        assert "sloughgpt_gateway_wire_bytes 1024" in text

    def test_daemon_thread_polls_and_stops(self, gateway):
        gateway.set_payload({"bandwidth": _bandwidth(identity=77, wire=33)})
        mirror = GatewayBandwidthMirror(base_url=gateway.base_url)
        try:
            assert mirror.start() is True
            assert mirror.start() is False  # idempotent
            deadline = time.time() + 3.0
            while mirror.snapshot is None and time.time() < deadline:
                time.sleep(0.02)
            assert mirror.snapshot is not None
            assert mirror.snapshot["identity_bytes"] == 77
        finally:
            mirror.stop()
        assert mirror._thread is None

    def test_singleton_is_stable(self):
        assert get_bandwidth_mirror() is get_bandwidth_mirror()
