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


def check_routes(gateway: str) -> list[dict[str, Any]]:
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
    status, spec, _ = get(f"{gateway}/openapi.json")
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
    findings += check_routes(args.gateway)

    problems = report(findings)
    if args.strict and problems:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
