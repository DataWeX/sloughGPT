"""The site doctor must separate its own artefacts from real defects.

A probe that reports "108 broken routes" when most of them are query-required
calls, SSE streams that never finish, or auth gates behaving correctly, is
worse than no probe: the number is large, true-ish, and points nowhere. The
classification is the product here, so it is what gets pinned.

Real run 2026-10-11, before the classification: 108 of 266 "failures", none of
which were defects. After: 4, of which two are hard 500s with named error codes
(`/phoneme/languages` -> `'list' object is not callable`, `/providers` ->
`E_SERIALIZATION`), one is an 11.8s route mistaken for a dead one by the
client's own timeout, and one genuinely hangs.

Phase 2 (2026-10-12) is the same lesson applied to the routes phase 1 skipped.
Streaming GETs were skipped wholesale because reading SSE to completion turns
the client's own `IncompleteRead` into a false failure — correct, but it made a
real defect invisible: a stream that accepts the connection and then never
emits is indistinguishable from a healthy one until a consumer waits. So they
are now judged on their *first* read instead.

That judgement produced its own false failure, and it is the sharpest thing in
this file: a 4s window reported `/consciousness/stream` (10s heartbeat) and
`/errors/stream` (30s heartbeat) as silent when both were heartbeating exactly
as written. The observation was shorter than the thing observed, so correct
behaviour read as a defect — phase 1's original sin wearing phase 2's clothes.
Hence `test_the_observation_window_outlasts_the_slowest_heartbeat`, which reads
the cadences out of the routers rather than copying them, and the
`heartbeat-only` class, which refuses to call a correctly-keepaliveing route
either healthy or broken.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import site_doctor_probe as doctor  # noqa: E402


def _ops(*parameters: dict, responses: dict | None = None) -> dict:
    return {"get": {"parameters": list(parameters), "responses": responses or {}}}


def test_a_required_query_param_makes_a_route_uncallable_bare() -> None:
    """Such a route returns 422 for want of its query, not for being unhealthy."""
    ops = _ops({"name": "q", "in": "query", "required": True})

    assert doctor._required_query_params(ops) == ["q"]


def test_optional_parameters_do_not_disqualify_a_route() -> None:
    """OpenAPI defaults `required` to false — absent must not mean skipped."""
    ops = _ops({"name": "limit", "in": "query"})

    assert doctor._required_query_params(ops) == [], (
        "an optional param must not get a route skipped, or the probe would cover almost nothing"
    )


def test_streaming_is_detected_by_path_when_openapi_stays_silent() -> None:
    """Our SSE routes under-declare their content type — the path must catch them."""
    ops = _ops(responses={"200": {"content": {"application/json": {}}}})

    for path in ("/consciousness/stream", "/dashboard/stream", "/registry/events"):
        assert doctor._is_streaming(ops, path), f"{path} was not treated as streaming"


def test_streaming_is_detected_by_content_type_when_the_path_is_innocuous() -> None:
    ops = _ops(responses={"200": {"content": {"text/event-stream": {}}}})

    assert doctor._is_streaming(ops, "/models/watch")


def test_an_ordinary_route_is_not_skipped() -> None:
    ops = _ops(responses={"200": {"content": {"application/json": {}}}})

    assert not doctor._is_streaming(ops, "/providers")
    assert doctor._required_query_params(ops) == []


def test_an_unreachable_host_reads_as_none_instead_of_raising() -> None:
    """The probe must survive every endpoint it touches.

    `IncompleteRead` is an `http.client.HTTPException`, not an `OSError`, so
    the first version of the probe crashed on a truncated chunked response and
    reported nothing at all — the one outcome a diagnostic must never produce.
    An unreachable host is the milder cousin of that case and must likewise
    come back as a status, not an exception.
    """
    status, body, _elapsed = doctor.get("http://127.0.0.1:1/nope", timeout=0.3)

    assert status is None, f"an unreachable host must read as None, got {status!r}"
    assert body is None


# --- phase 2: streaming routes are judged on their first read ---


def test_a_heartbeat_is_not_an_event_and_not_a_silence() -> None:
    """SSE comments are keepalives; a route that only sends them delivers nothing.

    Measured 2026-10-12: `/errors/stream` answered 200 `text/event-stream` and
    emitted exactly 13 bytes in 30.6s — its `: heartbeat` cadence — and no
    events. Reading that as "ok" hides a consumer that receives nothing; reading
    it as "silent" blames a route that is behaving exactly as written. It needs
    its own class, because the two defects it sits between have different owners.
    """
    assert doctor._is_heartbeat_only(b": heartbeat\n\n")
    assert doctor._is_heartbeat_only(b"   : keepalive\n\n"), "leading space is still a comment"

    assert not doctor._is_heartbeat_only(b'data: {"x": 1}\n\n')
    assert not doctor._is_heartbeat_only(b""), "no bytes is closed-empty, not heartbeat-only"
    # A comment followed by a real event means the route DOES deliver.
    assert not doctor._is_heartbeat_only(b": heartbeat\n\nid: 1\ndata: {}\n\n")


def test_the_observation_window_outlasts_the_slowest_heartbeat() -> None:
    """Silence is only meaningful once the route's own keepalive was due.

    The first cut of phase 2 used a 4s window and reported /consciousness/stream
    (10s heartbeat) and /errors/stream (30s heartbeat) as silent when both were
    heartbeating correctly. That is phase 1's original failure in a new costume:
    the observation was shorter than the thing observed, so correct behaviour
    read as a defect. The bound is read from the routers themselves rather than
    copied, so tightening a heartbeat anywhere in the tree trips this first.
    """
    import re

    cadences: list[tuple[float, str]] = []
    routers = REPO / "apps" / "api" / "server" / "routers"
    for router in routers.glob("*.py"):
        text = router.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"(?:heartbeat|_last_yield)[^=\n]*>=\s*([0-9.]+)", text):
            cadences.append((float(match.group(1)), router.name))

    assert cadences, "no heartbeat cadence found — the probe's window invariant lost its source"
    slowest, source = max(cadences)

    assert doctor.STREAM_GRACE > slowest, (
        f"STREAM_GRACE={doctor.STREAM_GRACE}s must exceed the slowest heartbeat "
        f"({slowest}s in {source}), or every correctly-behaved keepalive route "
        f"will be reported as silent"
    )


def test_streaming_routes_that_need_a_query_are_not_probed() -> None:
    """A query-required stream 422s before it ever streams, so it says nothing."""
    spec = {
        "paths": {
            "/stream/needs-q": _ops({"name": "q", "in": "query", "required": True}),
            "/stream/{id}": _ops(),
            "/plain": _ops(),
        }
    }

    assert doctor.check_streams("http://unused", spec) == [
        {
            "severity": "ok",
            "check": "stream-coverage",
            "message": "the gateway advertises no streaming GETs",
        }
    ], "parameterised and query-required paths must not be counted as streams"


def test_each_stream_failure_gets_its_own_finding(monkeypatch) -> None:
    """Failures are separated by cause, not summed into one number.

    This is the whole point of the card: a single "N streams broken" figure has
    no owner. Dropped, heartbeat-only and silent have three different fixes in
    three different places, so they must stay separable at the report.
    """
    seen: dict[str, str] = {}

    def fake_probe(url: str) -> tuple[str, dict]:
        name = url.rsplit("/", 1)[-1]
        return seen[name], {"url": url, "status": 200}

    monkeypatch.setattr(doctor, "probe_stream", fake_probe)
    spec = {
        "paths": {
            # `/sse` is one of the path shapes _is_streaming recognises.
            "/sse/ok": _ops(),
            "/sse/quiet": _ops(),
            "/sse/beat": _ops(),
            "/sse/dead": _ops(),
        }
    }
    seen.update({"ok": "ok", "quiet": "silent", "beat": "heartbeat-only", "dead": "dropped"})

    checks = {f["check"]: f for f in doctor.check_streams("http://unused", spec)}

    assert "stream-silent" in checks, "a route that says nothing must be named"
    assert "stream-heartbeat-only" in checks, "a route that only keepalives must be named"
    assert "stream-dropped" in checks, "a route that dies mid-stream must be named"
    assert "stream-health" in checks, "the healthy route must still be counted"
    assert "stream-error" not in checks


def test_a_stream_probe_survives_an_unreachable_host(monkeypatch) -> None:
    """Same contract as `get`: a diagnostic must report, never crash."""
    monkeypatch.setattr(doctor.urllib.request, "urlopen", _raise_unreachable)

    kind, detail = doctor.probe_stream("http://127.0.0.1:1/sse")

    assert kind == "error", f"an unreachable host must classify, not raise, got {kind!r}"
    assert detail["url"] == "http://127.0.0.1:1/sse"


def _raise_unreachable(request, timeout=0):  # noqa: ANN001, ANN202
    raise OSError("connection refused")


# -----------------------------------------------------------------------
# Phase 3 (2026-10-13): the journey report is READ, never launched, and the
# inference engine is triaged from the health surface. The probe stays a
# consumer: it never starts a browser sweep, so `--strict` can gate CI on a
# report someone else produced.
# -----------------------------------------------------------------------


def test_a_missing_journey_report_is_an_info_finding_not_a_crash(tmp_path) -> None:
    """No report yet (no sweep has run) is information, not a defect."""
    missing = tmp_path / "nope" / "ux-flows-report.json"
    findings = doctor.check_journeys(str(missing))
    assert len(findings) == 1, findings
    assert findings[0]["severity"] == "info"
    assert findings[0]["check"] == "journeys"


def test_a_failed_flow_reads_as_a_high_finding_naming_the_flow(tmp_path) -> None:
    """A failed journey is the site doctor's core signal — high, with the error."""
    report = {
        "passed": 1,
        "failed": 1,
        "console_error_count": 2,
        "console_errors_sample": ["Uncaught TypeError: x is not a function"],
        "flows": [
            {"id": "2-write", "status": "passed", "label": "Write a note"},
            {"id": "3-chat", "status": "failed", "error": "selector not found"},
        ],
    }
    path = tmp_path / "ux-flows-report.json"
    path.write_text(json.dumps(report))
    checks = {f["check"]: f for f in doctor.check_journeys(str(path))}
    assert "journeys-failed" in checks, checks
    finding = checks["journeys-failed"]
    assert finding["severity"] == "high"
    assert "3-chat" in finding["message"]
    assert "selector not found" in finding["message"]


def test_console_errors_alone_are_a_medium_finding(tmp_path) -> None:
    """A passing sweep can still be broken in the browser console."""
    report = {
        "passed": 3,
        "failed": 0,
        "console_error_count": 5,
        "console_errors_sample": ["Refused to load image"],
        "flows": [{"id": "1-read", "status": "passed"}],
    }
    path = tmp_path / "ux-flows-report.json"
    path.write_text(json.dumps(report))
    checks = {f["check"]: f for f in doctor.check_journeys(str(path))}
    assert "journeys-console" in checks, checks
    assert checks["journeys-console"]["severity"] == "medium"


def test_the_probe_never_launches_a_browser_sweep() -> None:
    """Read-only: importing avion/running a sweep would make the probe a writer."""
    src = open(doctor.__file__, encoding="utf-8").read()
    assert "import avion" not in src and "from avion" not in src, (
        "the probe must read the journey report, never launch a sweep"
    )
    assert "playwright" not in src, "playwright import would break the read-only contract"


def test_inference_triage_reports_a_stuck_or_error_engine(monkeypatch) -> None:
    """An erroring inference service is a defect the health surface already knows."""
    detailed = {
        "data": {
            "status": "degraded",
            "health_score": {"status": "degraded"},
            "services": {"inference": {"status": "error", "error": "engine timeout"}},
            "model_loaded": True,
            "device": "cpu",
            "inference_count": 42,
        }
    }

    def fake_get(url, timeout=5.0):
        if url.endswith("/health/detailed"):
            return 200, detailed, 0.1
        return 200, {"data": {"status": "healthy"}}, 0.1

    monkeypatch.setattr(doctor, "get", fake_get)
    checks = {f["check"]: f for f in doctor.check_inference("http://unused")}
    assert "inference-engine" in checks, checks
    finding = checks["inference-engine"]
    assert finding["severity"] == "high"
    assert "engine timeout" in finding["message"]
