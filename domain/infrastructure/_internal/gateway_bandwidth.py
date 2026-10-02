"""Gateway bandwidth mirror: edge byte counters → infra logger deltas.

The Rust gateway is the single source of truth for bandwidth. It counts
bytes at the encode point (``apps/gateway/src/compression.rs``) with relaxed
atomics and exposes cumulative counters on ``GET /health/detailed`` under a
``bandwidth`` block. Python never re-counts bytes — this module polls that
block on a daemon thread, keeps the last cumulative snapshot, and emits
*delta-only* records through the standard infra logger schema
(``op="bandwidth.interval"`` → domain ``bandwidth``).

Environment:
    MAN_GATEWAY_URL — gateway base URL (default ``http://127.0.0.1:8080``).
                      Empty string disables the poller entirely.

Behaviour:
    * Absent-tolerant: connection failures log at DEBUG and only on state
      transitions — a Python-only deployment (:8000, no gateway) stays
      silent instead of spamming errors every interval.
    * Counter resets (gateway restart) re-baseline silently — no negative
      deltas are ever logged.
    * Idle intervals (all deltas zero) emit nothing: the log only records
      bytes that actually crossed the wire since the previous poll.

Usage::

    from domain.infrastructure._internal.gateway_bandwidth import (
        get_bandwidth_snapshot,
        start_bandwidth_mirror,
    )

    start_bandwidth_mirror()          # daemon thread, idempotent
    snap = get_bandwidth_snapshot()   # cumulative, or None if never polled
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from typing import Any

from .metrics import get_metrics_collector
from .structured_log import StructuredLogger

#: Where the gateway listens when ``MAN_GATEWAY_URL`` is not set.
DEFAULT_GATEWAY_URL = "http://127.0.0.1:8080"

#: Poll cadence — matches the health-poll cadence of the monitoring UI.
POLL_INTERVAL_S = 5.0

#: Short timeout so a hung gateway can never stall the mirror thread.
POLL_TIMEOUT_S = 2.0

#: Cumulative counters the gateway publishes (deltas are computed on these).
_COUNTER_KEYS = (
    "identity_bytes",
    "wire_bytes",
    "compressed_responses",
    "identity_responses",
    "zstd_responses",
    "gzip_responses",
)


def _fmt_bytes(n: int) -> str:
    """Human-readable byte size for log messages."""
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


class GatewayBandwidthMirror:
    """Polls the gateway's ``bandwidth`` block and logs interval deltas.

    One instance per process (see :func:`get_bandwidth_mirror`). All state
    is lock-protected so ``poll_once`` is safe to call from the daemon
    thread and from tests concurrently.
    """

    def __init__(self, base_url: str | None = None) -> None:
        if base_url is None:
            base_url = os.environ.get("MAN_GATEWAY_URL", DEFAULT_GATEWAY_URL)
        # "" (or whitespace) disables the poller entirely.
        self._enabled = base_url.strip() != ""
        self._base_url = base_url.rstrip("/") or DEFAULT_GATEWAY_URL
        self._log = StructuredLogger("slo.infrastructure.gateway")

        self._lock = threading.Lock()
        self._prev: dict[str, Any] | None = None  # last cumulative snapshot
        self._snapshot: dict[str, Any] | None = None  # for health passthrough
        self._connected = False  # transition tracking (silent when absent)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Introspection ────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def snapshot(self) -> dict[str, Any] | None:
        """Last cumulative ``bandwidth`` block from the gateway, or None."""
        with self._lock:
            return dict(self._snapshot) if self._snapshot is not None else None

    # ── Polling ──────────────────────────────────────────────────────

    def poll_once(self) -> dict[str, Any] | None:
        """Fetch one cumulative snapshot and ingest its delta.

        Returns the gateway's ``bandwidth`` block on success, or None when
        disabled, unreachable, or the payload carries no bandwidth block.
        Failures are debug-only and logged on state transitions alone.
        """
        if not self._enabled:
            return None
        try:
            url = f"{self._base_url}/health/detailed"
            with urllib.request.urlopen(url, timeout=POLL_TIMEOUT_S) as resp:
                payload = json.load(resp)
        except Exception as exc:  # absent gateway is the normal dev case
            if self._connected:
                self._connected = False
                self._log.debug(
                    "gateway unreachable — bandwidth mirror idle",
                    reason=str(exc),
                )
            return None

        if not self._connected:
            self._connected = True
            self._log.info("gateway reachable — bandwidth mirror active")

        band = payload.get("bandwidth") if isinstance(payload, dict) else None
        if not isinstance(band, dict) or "identity_bytes" not in band:
            return None
        self._ingest(band)
        return band

    def _ingest(self, snap: dict[str, Any]) -> dict[str, Any] | None:
        """Store the cumulative snapshot and log a delta record when bytes moved.

        Returns the delta dict when a record was emitted, otherwise None
        (baseline sample, counter reset, or idle interval).
        """
        with self._lock:
            prev = self._prev
            self._prev = snap
            self._snapshot = snap

        # Cumulative gauges for /metrics — refreshed on every sample.
        get_metrics_collector().record_bandwidth(
            int(snap.get("identity_bytes") or 0),
            int(snap.get("wire_bytes") or 0),
        )

        if prev is None:
            return None  # first sample — establishes the baseline only

        # Counter reset (gateway restarted mid-interval): re-baseline
        # silently rather than logging a negative delta.
        if int(snap.get("identity_bytes") or 0) < int(prev.get("identity_bytes") or 0):
            return None

        delta = {key: int(snap.get(key) or 0) - int(prev.get(key) or 0) for key in _COUNTER_KEYS}
        identity = delta["identity_bytes"]
        wire = delta["wire_bytes"]
        if identity <= 0 and wire <= 0:
            return None  # idle interval — silence, not zero-noise

        saved = identity - wire
        saved_pct = (saved * 100.0 / identity) if identity > 0 else 0.0
        responses = delta["compressed_responses"] + delta["identity_responses"]
        delta["saved_bytes"] = saved
        delta["saved_pct"] = round(saved_pct, 2)
        delta["responses"] = responses
        self._log.info(
            f"edge {_fmt_bytes(identity)} → {_fmt_bytes(wire)} on the wire "
            f"({saved_pct:.1f}% saved)",
            op="bandwidth.interval",
            identity_bytes=identity,
            wire_bytes=wire,
            saved_bytes=saved,
            saved_pct=round(saved_pct, 2),
            responses=responses,
        )
        return delta

    # ── Thread lifecycle ─────────────────────────────────────────────

    def start(self) -> bool:
        """Start the daemon poller. Idempotent; False when disabled/running."""
        if not self._enabled or self._thread is not None:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gateway-bandwidth", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the daemon poller and wait for it to exit."""
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=timeout)
        self._thread = None

    def _loop(self) -> None:
        self.poll_once()  # baseline immediately, then every interval
        while not self._stop.wait(POLL_INTERVAL_S):
            self.poll_once()


# ── Singleton ────────────────────────────────────────────────────────

_mirror: GatewayBandwidthMirror | None = None
_mirror_lock = threading.Lock()


def get_bandwidth_mirror() -> GatewayBandwidthMirror:
    """The process-wide mirror (construction only — does not start it)."""
    global _mirror
    if _mirror is None:
        with _mirror_lock:
            if _mirror is None:
                _mirror = GatewayBandwidthMirror()
    return _mirror


def start_bandwidth_mirror() -> bool:
    """Start the process-wide mirror. Idempotent; False when disabled."""
    return get_bandwidth_mirror().start()


def stop_bandwidth_mirror() -> None:
    """Stop the process-wide mirror (no-op when not running)."""
    global _mirror
    with _mirror_lock:
        if _mirror is not None:
            _mirror.stop()


def get_bandwidth_snapshot() -> dict[str, Any] | None:
    """Cumulative bandwidth block for ``/health/detailed`` passthrough.

    Mirrors the gateway's own key shape so the UI cannot tell whether the
    payload came from the gateway directly (:8080) or from this mirror
    (Python-only :8000). None until the first successful poll.
    """
    return get_bandwidth_mirror().snapshot
