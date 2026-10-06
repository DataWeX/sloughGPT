"""SSE probe — watchdog on ``/health/stream`` (+ best-effort ``/errors/stream``).

Detects the frame-oversize / cadence-stall bug class observed in this
stack: ``/health/stream`` pushed 10,566,666-byte frames every ~3.3s
instead of compact snapshots at the nominal 3.0s cadence.

Thresholds:
    oversized frame   payload > 256 KiB (262144 B) → critical
    stall             inter-frame gap > 8.0s (2× cadence + slack) → critical
    ERROR frame       ``phase: "ERROR", status: "error"`` → critical
    < 2 frames        in the window → warn (cadence check inconclusive)
    otherwise         one info finding: "SSE cadence nominal (N frames, max X B)"

Stream reader injection: ``fetch_stream(url, window_s)`` returns an
iterator whose items are either frame strings (the probe stamps wall-clock
arrival times — trailing-silence gaps are then measured too) or explicit
``(timestamp, frame)`` tuples (tests fabricate timing directly; only
inter-frame gaps are evaluated for such streams).

``/errors/stream`` is probed best-effort with a short window: 401/403
becomes an info finding ("requires auth"), open failures a warn, observed
``ERROR``/``CLIENT_ERROR`` events a warn (server-side ``status:"error"``
frames a critical). A quiet error stream (zero frames) is *good news* and
produces no finding.

Chat-stream probe: **deferred to Phase B** — it requires an authenticated
session plus live model state (which model, which session) that this
read-only, no-auth Phase A surface cannot establish. Follow-up: add
``probes/chat.py`` once auth/session seams exist.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from domain.core._internal.mole.models import Finding
from domain.infrastructure._internal.health_flow import Severity

from . import ProbeResult

USER_AGENT = "sloughgpt-Mole/1.0"
MAX_FRAME_BYTES = 262144  # 256 KiB — the original bug was ~10.5 MB/frame
STALL_GAP_S = 8.0  # 2× the 3.0s cadence + slack
EXPECTED_CADENCE_S = 3.0
ERRORS_STREAM_WINDOW_S = 4.0


class StreamOpenError(Exception):
    """The SSE endpoint could not be opened (non-auth failure)."""


class StreamAuthError(StreamOpenError):
    """The SSE endpoint rejected the request (401/403)."""


def _default_reader(url: str, window_s: float):
    """Yield frame payloads from an SSE endpoint for ``window_s`` seconds.

    Yields plain strings; the probe stamps arrival times. A stalled read
    ends via the socket timeout (window + slack), a closed stream ends
    when iteration stops — both are handled as normal completion.
    """
    req = urllib.request.Request(
        url,
        headers={"Accept": "text/event-stream", "User-Agent": USER_AGENT},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=window_s + 2.0)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise StreamAuthError(f"HTTP {exc.code}") from exc
        raise StreamOpenError(f"HTTP {exc.code}") from exc
    except Exception as exc:
        raise StreamOpenError(str(exc)) from exc

    deadline = time.monotonic() + window_s
    data_lines: list[str] = []
    try:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line == "" and data_lines:
                frame = "\n".join(data_lines)
                data_lines = []
                yield frame
                if time.monotonic() >= deadline:
                    break
    except OSError:
        # Socket timeout or half-closed stream — trailing silence is
        # evaluated by the caller from the last frame's timestamp.
        pass
    finally:
        resp.close()


def _collect(reader, url: str, window_s: float) -> tuple[list, str]:
    """Iterate a reader into (items, error). Never raises."""
    items: list = []
    try:
        for item in reader(url, window_s):
            items.append(item)
    except StreamAuthError as exc:
        return items, f"auth:{exc}"
    except Exception as exc:
        return items, str(exc)
    return items, ""


def _normalize(items: list) -> tuple[list[tuple[float, str]], bool]:
    """Split into (timestamp, frame) pairs; flag explicit timestamps."""
    frames: list[tuple[float, str]] = []
    explicit = False
    for item in items:
        if isinstance(item, tuple):
            explicit = True
            ts, frame = item
        else:
            ts, frame = time.monotonic(), item
        frames.append((float(ts), str(frame)))
    return frames, explicit


def _parse(frame: str):
    try:
        evt = json.loads(frame)
    except ValueError:
        return None
    return evt if isinstance(evt, dict) else None


def _health_stream_findings(
    frames: list[tuple[float, str]],
    explicit: bool,
    window_s: float,
    end_ts: float,
    findings: list[Finding],
) -> None:
    sizes = [len(frame.encode("utf-8", errors="replace")) for _, frame in frames]

    oversized = [size for size in sizes if size > MAX_FRAME_BYTES]
    if oversized:
        findings.append(
            Finding(
                source="sse",
                check="stream.frame_size",
                severity=Severity.CRITICAL,
                score=0.0,
                message=(
                    f"SSE frame oversized: max {max(oversized):,} B across "
                    f"{len(oversized)} frame(s) (limit {MAX_FRAME_BYTES:,} B)"
                ),
                component="api",
            )
        )

    max_gap, gap_after = 0.0, 0
    for i in range(1, len(frames)):
        gap = frames[i][0] - frames[i - 1][0]
        if gap > max_gap:
            max_gap, gap_after = gap, i
    if frames and not explicit:
        trailing = end_ts - frames[-1][0]
        if trailing > max_gap:
            max_gap, gap_after = trailing, len(frames)
    if max_gap > STALL_GAP_S:
        findings.append(
            Finding(
                source="sse",
                check="stream.stall",
                severity=Severity.CRITICAL,
                score=10.0,
                message=(
                    f"SSE stream stalled {max_gap:.1f}s after frame {gap_after} "
                    f"(limit {STALL_GAP_S:.0f}s)"
                ),
                component="api",
            )
        )

    error_frames = []
    malformed = 0
    for _, frame in frames:
        evt = _parse(frame)
        if evt is None:
            malformed += 1
            continue
        if evt.get("phase") == "ERROR" and evt.get("status") == "error":
            data = evt.get("data") or {}
            error_frames.append(str(data.get("error") or evt.get("message") or "unknown error"))
    if error_frames:
        findings.append(
            Finding(
                source="sse",
                check="stream.error_frame",
                severity=Severity.CRITICAL,
                score=10.0,
                message=(
                    f"/health/stream emitted {len(error_frames)} ERROR frame(s): "
                    f"{error_frames[0][:160]}"
                ),
                component="api",
            )
        )
    if malformed:
        findings.append(
            Finding(
                source="sse",
                check="stream.parse",
                severity=Severity.WARN,
                score=60.0,
                message=f"{malformed} malformed frame(s) on /health/stream",
                component="api",
            )
        )

    if any(f.severity == Severity.CRITICAL for f in findings):
        return
    if len(frames) < 2:
        findings.append(
            Finding(
                source="sse",
                check="stream.cadence",
                severity=Severity.WARN,
                score=60.0,
                message=(
                    f"Only {len(frames)} frame(s) in {window_s:.0f}s "
                    f"(expected ~{max(1, int(window_s / EXPECTED_CADENCE_S))}) — "
                    "cadence check inconclusive"
                ),
                component="api",
            )
        )
    elif not any(f.severity == Severity.WARN for f in findings):
        findings.append(
            Finding(
                source="sse",
                check="stream.cadence",
                severity=Severity.INFO,
                score=90.0,
                message=f"SSE cadence nominal ({len(frames)} frames, max {max(sizes):,} B)",
                component="api",
            )
        )


def _errors_stream_findings(reader, api: str, window_s: float, findings: list[Finding]) -> None:
    url = f"{api}/errors/stream"
    items, error = _collect(reader, url, window_s)
    if error.startswith("auth:"):
        findings.append(
            Finding(
                source="sse",
                check="errors_stream.auth",
                severity=Severity.INFO,
                score=90.0,
                message="/errors/stream requires auth (skipped)",
                detail=error[5:],
                component="api",
            )
        )
        return
    if error:
        findings.append(
            Finding(
                source="sse",
                check="errors_stream.open",
                severity=Severity.WARN,
                score=60.0,
                message="/errors/stream unavailable",
                detail=error,
                component="api",
            )
        )
        return

    server_errors: list[str] = []
    client_errors: list[str] = []
    for item in items:
        frame = item[1] if isinstance(item, tuple) else item
        evt = _parse(str(frame))
        if evt is None:
            continue
        if evt.get("status") == "error":
            data = evt.get("data") or {}
            server_errors.append(str(data.get("error") or evt.get("message") or "unknown"))
        elif evt.get("phase") in ("ERROR", "CLIENT_ERROR"):
            data = evt.get("data") or {}
            client_errors.append(str(data.get("message") or ""))
    if server_errors:
        findings.append(
            Finding(
                source="sse",
                check="errors_stream.frame",
                severity=Severity.CRITICAL,
                score=20.0,
                message=f"/errors/stream failure frame: {server_errors[0][:160]}",
                component="api",
            )
        )
    if client_errors:
        findings.append(
            Finding(
                source="sse",
                check="errors_stream.events",
                severity=Severity.WARN,
                score=60.0,
                message=f"{len(client_errors)} live error event(s) observed",
                detail=client_errors[0][:160],
                component="api",
            )
        )
    # Zero frames = quiet error stream = good; no finding (strict-safe).


def run_probe(window_s: float = 9.0, fetch_stream=None) -> ProbeResult:
    """Watch ``/health/stream`` for ``window_s`` seconds and triage it."""
    os.environ.setdefault("SLO_API_URL", "http://localhost:8000")
    api = os.environ["SLO_API_URL"]
    findings: list[Finding] = []
    reader = fetch_stream or _default_reader

    start_ts = time.monotonic()
    items, error = _collect(reader, f"{api}/health/stream", window_s)
    end_ts = time.monotonic()

    if error.startswith("auth:"):
        findings.append(
            Finding(
                source="sse",
                check="stream.auth",
                severity=Severity.INFO,
                score=90.0,
                message="/health/stream requires auth (skipped)",
                detail=error[5:],
                component="api",
            )
        )
        return ProbeResult(name="sse", ok=False, findings=findings, error=error[5:])

    if error:
        findings.append(
            Finding(
                source="sse",
                check="stream.open",
                severity=Severity.WARN,
                score=60.0,
                message="/health/stream unavailable",
                detail=error,
                component="api",
            )
        )
        return ProbeResult(name="sse", ok=False, findings=findings, error=error)

    frames, explicit = _normalize(items)
    if not frames:
        elapsed = max(0.0, time.monotonic() - start_ts)
        findings.append(
            Finding(
                source="sse",
                check="stream.frames",
                severity=Severity.WARN,
                score=60.0,
                message=f"No frames from /health/stream in {elapsed:.1f}s (window {window_s:.0f}s)",
                component="api",
            )
        )
        return ProbeResult(name="sse", ok=False, findings=findings, error="no frames received")

    _health_stream_findings(frames, explicit, window_s, end_ts, findings)
    _errors_stream_findings(reader, api, min(window_s, ERRORS_STREAM_WINDOW_S), findings)

    return ProbeResult(
        name="sse",
        ok=True,
        findings=findings,
        raw={
            "frames": len(frames),
            "max_bytes": max(len(f.encode("utf-8", errors="replace")) for _, f in frames),
            "window_s": window_s,
        },
    )
