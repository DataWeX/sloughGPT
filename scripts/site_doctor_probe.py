"""Site doctor — read-only probe phase.

Card e8fd60d4. Phase 1 of the site doctor: probe the running stack **as a
consumer** and emit a severity-ranked worklist. It orders tests; it does not
operate. Every request here is a GET with no body, so a probe can never change
what it is measuring.

Three things are checked, in the order they matter:

1. **Self-consistency of the health surface.** `/health` and `/health/detailed`
   are two views of one process. If they disagree about whether the site is
   healthy, the operator-facing one is lying, and that is worse than being
   broken — a broken site gets fixed, a lying health check does not.
2. **Lifecycle progress.** A server that is still in phase `init` after N
   minutes of uptime never finished starting, however many requests it serves.
3. **Route reachability.** Every *parameterless* GET advertised by the
   gateway's openapi document is fetched and its status recorded. Paths with
   `{parameters}` are reported as skipped, not guessed at: inventing an id to
   probe would produce noise, and a wrong id reads as a 404 that isn't a
   defect.
4. **Streaming health.** Phase 1 skipped every SSE route, because reading a
   stream to completion turns the client's own `IncompleteRead` into a false
   failure. Phase 2 judges them on the *first* read instead — bounded in bytes
   and in seconds — and separates four outcomes that need four different fixes:
   emits (`ok`), keepalives and nothing else (`heartbeat-only`), holds the
   connection and says nothing (`silent`), and dies (`dropped`/`closed-empty`).
   The window must outlast the slowest heartbeat in the tree or correctly
   behaving routes read as broken, so it is pinned against the routers by
   `tests/contract/test_site_doctor_probe.py`.

Findings are ranked critical > high > medium > low, and the run exits non-zero
only under `--strict`, so it can gate CI once the baseline is clean.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

TIMEOUT = 6.0

# A stream is judged on its FIRST read, bounded in bytes and seconds.
STREAM_BYTES = 512
# The window MUST exceed the slowest heartbeat the tree actually uses,
# or a correctly-behaving stream is indistinguishable from a hung one and the
# probe reports false silences — the same mistake as phase 1's 108 failures.
# Slowest cadence measured in apps/api/server/routers:
#   errors.py:451      heartbeat every 30.0s
#   consciousness.py   heartbeat every 10s, loop sleeps 5
#   health.py:49       HEALTH_STREAM_INTERVAL 3.0s, emits every 1.0s
STREAM_GRACE = 35.0

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "ok": 4}


def get(url: str, timeout: float = TIMEOUT) -> tuple[int | None, Any, float]:
    """GET a URL. Returns (status|None, decoded body|None, elapsed ms).

    ``IncompleteRead`` is deliberately not fatal: a route that advertises a
    chunked body and then stops sending is a *finding*, and a probe that
    crashes on it would report nothing at all — the one outcome a diagnostic
    must never produce. It surfaces as status ``"truncated"``.
    """
    started = time.perf_counter()
    request = urllib.request.Request(url, method="GET")
    raw = b""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status: int | str | None = response.status
    except urllib.error.HTTPError as error:
        raw = error.read()
        status = error.code
    except http.client.HTTPException:
        # IncompleteRead / RemoteDisconnected / BadStatusLine and friends.
        status = "truncated" if raw else None
    except (urllib.error.URLError, TimeoutError, OSError):
        status = None
    elapsed = (time.perf_counter() - started) * 1000
    try:
        return status, json.loads(raw) if raw else None, elapsed
    except (json.JSONDecodeError, UnicodeDecodeError):
        return status, None, elapsed


def check_health_consistency(api: str) -> list[dict[str, Any]]:
    """The two health views must agree about whether the site is healthy."""
    findings: list[dict[str, Any]] = []
    status, brief, _ = get(f"{api}/health")
    if brief is None:
        return [
            {
                "severity": "critical",
                "check": "health-reachable",
                "message": f"/health did not answer ({status}) at {api}",
            }
        ]

    _, detailed, _ = get(f"{api}/health/detailed")
    brief_status = (brief.get("data") or {}).get("status")
    if not isinstance(detailed, dict):
        findings.append(
            {
                "severity": "high",
                "check": "health-detailed",
                "message": "/health/detailed did not return JSON",
            }
        )
        return findings

    data = detailed.get("data") or {}
    detailed_status = data.get("status")
    score = (data.get("health_score") or {}).get("status")

    if brief_status == "healthy" and score and score != "healthy":
        findings.append(
            {
                "severity": "critical",
                "check": "health-contradiction",
                "message": (
                    f"/health reports {brief_status!r} while /health/detailed "
                    f"reports status={detailed_status!r} health_score={score!r} "
                    f"— the operator-facing check is green on a degraded site, "
                    f"so a real outage would be invisible"
                ),
            }
        )

    uptime = data.get("uptime_seconds") or 0
    phase = ((data.get("lifecycle") or {}) if "lifecycle" in data else {}).get("phase")
    if phase is None:
        phase = (brief.get("data") or {}).get("lifecycle", {}).get("phase")
    if phase == "init" and uptime > 60:
        findings.append(
            {
                "severity": "high",
                "check": "lifecycle-stuck",
                "message": (
                    f"lifecycle phase is still {phase!r} after {uptime:.0f}s "
                    f"of uptime — startup never completed, however many "
                    f"requests are being served"
                ),
            }
        )

    requests_seen = data.get("request_count")
    inferences = data.get("inference_count")
    if isinstance(requests_seen, int) and isinstance(inferences, int):
        if requests_seen == 0 and inferences > 0:
            findings.append(
                {
                    "severity": "high",
                    "check": "metrics-unwired",
                    "message": (
                        f"request_count=0 while inference_count={inferences} — "
                        f"inference is happening but the request metrics are "
                        f"not recording it, so latency/percentile panels are "
                        f"meaningless"
                    ),
                }
            )

    throughput = (data.get("health_score") or {}).get("diagnoses") or []
    for diagnosis in throughput:
        if diagnosis.get("severity") in ("critical", "high"):
            findings.append(
                {
                    "severity": diagnosis.get("severity", "medium"),
                    "check": f"self-reported:{diagnosis.get('check')}",
                    "message": str(diagnosis.get("message")),
                }
            )
    return findings


def _required_query_params(operations: dict[str, Any]) -> list[str]:
    """Names of parameters a GET needs before it can return anything but 422."""
    required = []
    for parameter in operations.get("get", {}).get("parameters", []) or []:
        if not isinstance(parameter, dict):
            continue
        # `required` defaults to false for query/header params (OpenAPI 3).
        if parameter.get("required", False):
            required.append(parameter.get("name", "?"))
    return required


def _is_streaming(operations: dict[str, Any], path: str) -> bool:
    """True when the GET is a stream and therefore never really finishes.

    Two signals, because OpenAPI alone under-reports: many of our streaming
    routes advertise no `text/event-stream` response at all, so a content-type
    check passes them and the probe then blocks on a body that is never
    closed. The path shape catches those. Guessing wrong here costs a skipped
    route; guessing wrong the other way costs the whole run.
    """
    if any(token in path for token in ("/stream", "/events", "/sse", "/ws")):
        return True
    responses = operations.get("get", {}).get("responses", {}) or {}
    for _code, body in responses.items():
        if not isinstance(body, dict):
            continue
        for content_type in body.get("content") or {}:
            if "event-stream" in str(content_type):
                return True
    return False


def _is_heartbeat_only(chunk: bytes) -> bool:
    """True when every line received so far is an SSE comment, i.e. a heartbeat.

    SSE reserves lines beginning with ``:`` as comments, which is how this tree
    sends keepalives (``yield ": heartbeat\\n\\n"``). A stream that sends only
    comments is *alive* but delivers nothing: the connection is healthy, the
    status line is 200, and the consumer still receives no events. Reporting
    that as "ok" would hide it; reporting it as "silent" would misattribute it.
    """
    lines = [line for line in chunk.decode("utf-8", "replace").splitlines() if line.strip()]
    return bool(lines) and all(line.lstrip().startswith(":") for line in lines)


def probe_stream(url: str) -> tuple[str, dict[str, Any]]:
    """Bounded first-read of a streaming route. Returns (classification, detail).

    Phase 1 skipped every streaming route, and that was right for its reason:
    an SSE body never finishes, so reading it to completion turns the client's
    own ``IncompleteRead`` into a false failure. But skipping them wholesale
    hid a defect that only a consumer ever sees — a route that accepts the
    connection and then never emits a byte is indistinguishable from a healthy
    one until someone waits on it.

    The observation that separates them is not the whole body, it is the
    *first* read, capped in bytes and seconds: connect, take what has already
    arrived, and close. Classification:

      ``ok``          bytes arrived and they are events — the stream delivers;
      ``heartbeat-only``bytes arrived but every line is an SSE comment (``:``)
                      — the route is alive and heartbeating, yet a consumer
                      receives no events at all; a distinct failure from
                      silence, because the connection looks healthy;
      ``silent``      connected, but nothing arrived even within STREAM_GRACE
                      — longer than the slowest heartbeat in the tree, so the
                      route holds the connection open and says nothing;
      ``dropped``     the connection died before any byte (RemoteDisconnected
                      / IncompleteRead at zero);
      ``closed-empty``the route answered 200 and closed without emitting;
      ``auth-gated``  401/403 — correct behaviour unauthenticated;
      ``error``       any other >= 400, or no answer at all.

    Still read-only: a GET with no body, exactly as in phase 1.
    """
    request = urllib.request.Request(url, method="GET")
    # An SSE consumer's own headers, so the route is seen as the consumer sees it.
    request.add_header("Accept", "text/event-stream")
    request.add_header("Cache-Control", "no-cache")
    detail: dict[str, Any] = {"url": url}
    started = time.perf_counter()
    try:
        # The window is deliberately long: a stream that emits returns the
        # instant its first bytes arrive, so the long window costs nothing on a
        # healthy route and only the genuinely silent ones wait it out. Cost
        # therefore tracks defects, not route count.
        with urllib.request.urlopen(request, timeout=STREAM_GRACE) as response:
            detail["status"] = response.status
            detail["content_type"] = response.headers.get("Content-Type", "")
            try:
                # read1(), not read(): read(n) blocks until it has n bytes or
                # EOF, so a route that only heartbeats (14 bytes every 30s)
                # looks starved and then "drops" minutes later. read1() returns
                # whatever one buffer already holds — the true first read.
                reader = getattr(response, "read1", None)
                chunk = reader(STREAM_BYTES) if reader else response.read(STREAM_BYTES)
            except http.client.HTTPException:
                # RemoteDisconnected / IncompleteRead — the stream died.
                detail["elapsed_ms"] = (time.perf_counter() - started) * 1000
                return "dropped", detail
            except (TimeoutError, OSError):
                # Connected and holding the connection open without sending.
                detail["elapsed_ms"] = (time.perf_counter() - started) * 1000
                return "silent", detail
            detail["elapsed_ms"] = (time.perf_counter() - started) * 1000
            detail["bytes"] = len(chunk)
            if not chunk:
                return "closed-empty", detail
            if _is_heartbeat_only(chunk):
                return "heartbeat-only", detail
            return "ok", detail
    except urllib.error.HTTPError as error:
        detail["status"] = error.code
        if error.code in (401, 403):
            return "auth-gated", detail
        return "error", detail
    except (urllib.error.URLError, TimeoutError, OSError):
        detail["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return "error", detail


def check_streams(gateway: str, spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Classify every streaming GET the gateway advertises.

    These are the routes phase 1 skipped as ``skipped_stream``. They get their
    own check rather than joining ``route-reachability`` because their failure
    modes are different: a stream does not go 500 when it breaks, it goes
    quiet, and quiet reads as success to anything that only inspects the
    status line.
    """
    findings: list[dict[str, Any]] = []
    if spec is None:
        status, spec, _ = get(f"{gateway}/openapi.json")
    else:
        status = 200
    if not isinstance(spec, dict):
        return [
            {
                "severity": "high",
                "check": "openapi-reachable",
                "message": f"gateway openapi.json did not answer ({status})",
            }
        ]
    paths = spec.get("paths") or {}
    seen: dict[str, list[str]] = {}
    for path, operations in sorted(paths.items()):
        if not isinstance(operations, dict) or "get" not in operations:
            continue
        if "{" in path:
            continue
        # A query-required stream would 422 before it ever streamed.
        if _required_query_params(operations):
            continue
        if not _is_streaming(operations, path):
            continue
        classification, detail = probe_stream(f"{gateway}{path}")
        entry = path if classification == "ok" else f"{path} [{detail}]"
        seen.setdefault(classification, []).append(entry)

    total = sum(len(v) for v in seen.values())
    if not total:
        findings.append(
            {
                "severity": "ok",
                "check": "stream-coverage",
                "message": "the gateway advertises no streaming GETs",
            }
        )
        return findings

    findings.append(
        {
            "severity": "ok",
            "check": "stream-coverage",
            "message": f"{total} streaming route(s) probed with a bounded first read",
        }
    )

    healthy = len(seen.get("ok", [])) + len(seen.get("auth-gated", []))
    if healthy:
        findings.append(
            {
                "severity": "ok",
                "check": "stream-health",
                "message": (
                    f"{healthy} streaming route(s) emitted bytes or are "
                    f"correctly auth-gated: " + ", ".join(seen.get("ok", [])[:6])
                ),
            }
        )

    silent = seen.get("silent", [])
    if silent:
        findings.append(
            {
                "severity": "high",
                "check": "stream-silent",
                "message": (
                    f"{len(silent)} stream(s) accepted the connection and then "
                    f"said nothing within {STREAM_GRACE:.0f}s — longer than the "
                    f"slowest heartbeat in the tree, so this is not a slow "
                    f"route, it is a quiet one: the status line reads 200 so "
                    f"every shallow check passes, and only a consumer waiting "
                    f"on an event notices: " + "; ".join(silent[:6])
                ),
            }
        )

    for kind, severity, why in (
        ("heartbeat-only", "medium", "heartbeats and nothing else — alive but event-free"),
        ("dropped", "high", "the connection died mid-stream"),
        ("closed-empty", "high", "answered 200 and closed without emitting"),
        ("error", "medium", "did not return a streamable response"),
    ):
        bad = seen.get(kind, [])
        if bad:
            findings.append(
                {
                    "severity": severity,
                    "check": f"stream-{kind}",
                    "message": (
                        f"{len(bad)} streaming route(s) where {why}: " + "; ".join(bad[:6])
                    ),
                }
            )
    return findings


def check_routes(gateway: str, spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Fetch every genuinely-callable GET the gateway advertises.

    The OpenAPI document advertises far more than a bare GET can answer, so
    three classes are separated rather than lumped into one failure count.
    Reporting "108 broken routes" when most are query-required or SSE would be
    a number without a cause, and a cause-less number is not actionable:

      * GETs with **required query/header parameters** are skipped — calling
        them without their query is a guaranteed 422 that says nothing about
        the route's health;
      * **streaming (SSE)** GETs are skipped — they do not end, so reading the
        body to completion is the wrong observation, and the resulting
        IncompleteRead is an artefact of the client, not the server;
      * **401/403** are counted as auth-gated, which is correct behaviour and
        not a defect.
    """
    findings: list[dict[str, Any]] = []
    if spec is None:
        status, spec, _ = get(f"{gateway}/openapi.json")
    else:
        status = 200
    if not isinstance(spec, dict):
        return [
            {
                "severity": "high",
                "check": "openapi-reachable",
                "message": f"gateway openapi.json did not answer ({status})",
            }
        ]

    paths = spec.get("paths") or {}
    probed = skipped_path = skipped_query = skipped_stream = 0
    auth_gated: list[str] = []
    failures: list[str] = []
    for path, operations in sorted(paths.items()):
        if not isinstance(operations, dict) or "get" not in operations:
            continue
        if "{" in path:
            skipped_path += 1
            continue
        if _required_query_params(operations):
            skipped_query += 1
            continue
        if _is_streaming(operations, path):
            skipped_stream += 1
            continue
        code, _body, _elapsed = get(f"{gateway}{path}")
        probed += 1
        if code in (401, 403):
            auth_gated.append(f"{path} -> {code}")
        elif code == "truncated":
            failures.append(f"{path} -> truncated")
        elif code is None or (isinstance(code, int) and code >= 400):
            failures.append(f"{path} -> {code}")

    findings.append(
        {
            "severity": "ok",
            "check": "probe-coverage",
            "message": (
                f"{probed} routes actually probed; skipped "
                f"{skipped_path} parameterised, {skipped_query} query-required, "
                f"{skipped_stream} streaming"
            ),
        }
    )
    if auth_gated:
        findings.append(
            {
                "severity": "ok",
                "check": "auth-gated",
                "message": (
                    f"{len(auth_gated)} route(s) require auth (401/403), which "
                    f"is correct behaviour unauthenticated: " + ", ".join(auth_gated[:6])
                ),
            }
        )
    if failures:
        findings.append(
            {
                "severity": "high",
                "check": "route-reachability",
                "message": (
                    f"{len(failures)} of {probed} probed routes are not "
                    f"serving 2xx/3xx: " + "; ".join(failures[:12])
                ),
            }
        )
    else:
        findings.append(
            {
                "severity": "ok",
                "check": "route-reachability",
                "message": (
                    f"all {probed} probed routes answered < 400 "
                    f"({skipped_path + skipped_query + skipped_stream} skipped)"
                ),
            }
        )
    return findings


def check_edge(gateway: str, api: str, web: str) -> list[dict[str, Any]]:
    """The three hops must all answer, and the edge must not be the weak one."""
    findings: list[dict[str, Any]] = []
    latencies: dict[str, float] = {}
    for name, url in (("edge", gateway), ("web", web), ("api", api)):
        code, _body, elapsed = get(f"{url}/health" if name != "web" else f"{url}/health")
        if code is None:
            findings.append(
                {
                    "severity": "critical",
                    "check": f"{name}-reachable",
                    "message": f"{name} did not answer at {url}",
                }
            )
        else:
            latencies[name] = elapsed
            if isinstance(code, int) and code >= 400:
                findings.append(
                    {
                        "severity": "high",
                        "check": f"{name}-status",
                        "message": f"{name} /health returned {code}",
                    }
                )
    if len(latencies) == 3:
        findings.append(
            {
                "severity": "ok",
                "check": "three-hops",
                "message": "all three hops answered: "
                + ", ".join(f"{k}={v:.0f}ms" for k, v in latencies.items()),
            }
        )
    return findings


def report(findings: list[dict[str, Any]], out: Any = sys.stdout) -> int:
    ordered = sorted(findings, key=lambda f: SEVERITY_ORDER.get(f["severity"], 9))
    counts: dict[str, int] = {}
    for finding in ordered:
        counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1
    print("  severity counts:", counts, file=out)
    print(file=out)
    for finding in ordered:
        marker = {"critical": "!!", "high": " !", "medium": " -", "low": " .", "ok": "ok"}.get(
            finding["severity"], "  "
        )
        print(f"  {marker} [{finding['check']}] {finding['message']}", file=out)
    return sum(v for k, v in counts.items() if k in ("critical", "high", "medium", "low"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--gateway", default="http://127.0.0.1:8080")
    parser.add_argument("--web", default="http://127.0.0.1:5173")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 while any non-ok finding remains",
    )
    args = parser.parse_args()

    findings: list[dict[str, Any]] = []
    findings += check_edge(args.gateway, args.api, args.web)
    findings += check_health_consistency(args.api)
    # One openapi fetch for both route checks — the document is the gateway's
    # own description, so reading it twice would only double the chance of the
    # two checks disagreeing about what is a stream.
    _status, spec, _ = get(f"{args.gateway}/openapi.json")
    findings += check_routes(args.gateway, spec)
    findings += check_streams(args.gateway, spec)

    problems = report(findings)
    if args.strict and problems:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
