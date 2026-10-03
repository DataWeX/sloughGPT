"""HTTP probe — read-only GETs against the API self-diagnosis surface.

Endpoints probed (all GET, unwrap ``{"status": "success", "data": ...}``):

    /health                   reachability (first probe — failure is fatal)
    /health/detailed          health_score, degraded, recent_errors,
                              rate_violations, traffic counters
    /errors/grouped           fingerprint groups (>=10 warn, >=50 critical)
    /errors/trends?hours=24   recorded for raw evidence
    /errors/recent?limit=20   recorded for raw evidence

``fetch`` is injectable for hermetic tests: ``fetch(url) -> (status,
body_bytes)``. The default uses stdlib urllib with a timeout. ``/errors/*``
is not auth-allowlisted — a 401/403 becomes an info finding, not a
failure.
"""

from __future__ import annotations

import json
import os
import urllib.request

from domain.core._internal.doctor.models import Finding
from domain.infrastructure._internal.health_flow import Severity

from . import ProbeResult

USER_AGENT = "sloughgpt-site-doctor/1.0"
TIMEOUT_S = 8.0
GROUP_WARN_COUNT = 10
GROUP_CRITICAL_COUNT = 50


def _default_fetch(url: str, timeout: float = TIMEOUT_S) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def _get(fetch, url: str) -> tuple[int, object]:
    """Fetch + JSON-decode + unwrap the success envelope (raw if absent)."""
    status, body = fetch(url)
    if status >= 400:
        return status, None
    try:
        payload = json.loads(body)
    except ValueError:
        return status, None
    if isinstance(payload, dict) and payload.get("status") == "success":
        return status, payload.get("data")
    return status, payload


def _warn(check: str, message: str, detail: str = "") -> Finding:
    return Finding(
        source="http",
        check=check,
        severity=Severity.WARN,
        score=60.0,
        message=message,
        detail=detail,
        component="api",
    )


def _secondary_endpoint(fetch, url: str, label: str, findings: list[Finding], raw: dict):
    """Probe a non-critical endpoint; failures become findings, never raise."""
    try:
        status, payload = _get(fetch, url)
    except Exception as exc:  # urllib errors and injected fetches alike
        findings.append(_warn("api.probe_failed", f"{label} probe failed", str(exc)))
        return None
    if status in (401, 403):
        findings.append(
            Finding(
                source="http",
                check="api.auth_required",
                severity=Severity.INFO,
                score=90.0,
                message=f"{label} requires auth (skipped)",
                detail=f"HTTP {status}",
                component="api",
            )
        )
        return None
    if status >= 400 or payload is None:
        findings.append(_warn("api.probe_failed", f"{label} returned HTTP {status}", url))
        return None
    raw[label] = payload
    return payload


def _health_findings(detail: dict, findings: list[Finding]) -> None:
    score = float((detail.get("health_score") or {}).get("score") or 0)
    status = str((detail.get("health_score") or {}).get("status") or "").lower()
    summary = str((detail.get("health_score") or {}).get("summary") or "")
    if status == "unhealthy":
        severity = Severity.CRITICAL
    elif status == "degraded":
        severity = Severity.WARN
    elif status == "healthy":
        severity = Severity.OK
    else:
        severity = Severity.INFO
        summary = summary or f"status={status or 'unknown'}"
    # CRITICAL/WARN keep the server's own band-aligned score; OK/INFO never
    # dip below 80 so ranking stays consistent with the models.py convention.
    score_value = score if severity in (Severity.CRITICAL, Severity.WARN) else max(score, 80.0)
    findings.append(
        Finding(
            source="http",
            check="api.health_score",
            severity=severity,
            score=score_value,
            message=f"API health score {score:.0f} — {status or 'unknown'}",
            detail=summary,
            component="api",
        )
    )

    degraded = detail.get("degraded") or []
    if degraded:
        findings.append(
            _warn(
                "api.degraded",
                "Degraded subsystems: " + ", ".join(str(d) for d in degraded),
            )
        )

    request_count = int(detail.get("request_count") or 0)
    error_count = int(detail.get("error_count") or 0)
    p95 = detail.get("p95_latency_ms")
    findings.append(
        Finding(
            source="http",
            check="api.traffic",
            severity=Severity.INFO if error_count else Severity.OK,
            score=90.0 if error_count else 100.0,
            message=f"Traffic: {request_count} requests, {error_count} errors, p95 {p95} ms",
            component="api",
        )
    )

    for rec in (detail.get("recent_errors") or [])[:5]:
        findings.append(
            Finding(
                source="http",
                check="api.recent_error",
                severity=Severity.WARN,
                score=55.0,
                message=(
                    f"{rec.get('method', '?')} {rec.get('path', '?')} → "
                    f"{rec.get('status', '?')}: {str(rec.get('message', ''))[:120]}"
                ),
                detail=f"type={rec.get('error_type', '')} ts={rec.get('ts', '')}",
                component="api",
            )
        )

    violations = detail.get("rate_violations") or []
    if violations:
        findings.append(
            _warn(
                "api.rate_violations",
                f"{len(violations)} path(s) hit rate limits",
                "\n".join(
                    f"{v.get('path')} ×{v.get('count')} (limit {v.get('limit')})"
                    for v in violations[:5]
                ),
            )
        )


def _group_findings(grouped: dict, findings: list[Finding]) -> None:
    for group in grouped.get("groups") or []:
        count = int(group.get("count") or 0)
        if count >= GROUP_CRITICAL_COUNT:
            severity, score = Severity.CRITICAL, 25.0
        elif count >= GROUP_WARN_COUNT:
            severity, score = Severity.WARN, 60.0
        else:
            continue
        findings.append(
            Finding(
                source="http",
                check="api.error_group",
                severity=severity,
                score=score,
                message=f"Error group ×{count}: {str(group.get('message', ''))[:100]}",
                detail=(
                    f"fingerprint={str(group.get('fingerprint', ''))[:12]} "
                    f"source={group.get('source', '')} latest={group.get('latest', '')}"
                ),
                component="api",
            )
        )


def run_probe(fetch=None) -> ProbeResult:
    """Probe the API diagnosis surface. ``fetch(url) -> (status, bytes)``."""
    fetch = fetch or _default_fetch
    os.environ.setdefault("SLO_API_URL", "http://localhost:8000")
    api = os.environ["SLO_API_URL"]
    findings: list[Finding] = []
    raw: dict = {}

    # Reachability — the only endpoint whose failure fails the whole probe.
    try:
        status, payload = _get(fetch, f"{api}/health")
    except Exception as exc:
        return ProbeResult(
            name="http",
            ok=False,
            findings=[
                Finding(
                    source="http",
                    check="api.reachable",
                    severity=Severity.CRITICAL,
                    score=0.0,
                    message="API unreachable",
                    detail=f"{api}/health → {exc}",
                    component="api",
                )
            ],
            error=str(exc),
            raw=raw,
        )
    if status >= 400:
        return ProbeResult(
            name="http",
            ok=False,
            findings=[
                Finding(
                    source="http",
                    check="api.reachable",
                    severity=Severity.CRITICAL,
                    score=max(0.0, 100.0 - status),
                    message=f"API unreachable: /health returned HTTP {status}",
                    component="api",
                )
            ],
            error=f"HTTP {status}",
            raw=raw,
        )
    raw["/health"] = payload

    detail = _secondary_endpoint(fetch, f"{api}/health/detailed", "/health/detailed", findings, raw)
    if isinstance(detail, dict):
        _health_findings(detail, findings)

    grouped = _secondary_endpoint(fetch, f"{api}/errors/grouped", "/errors/grouped", findings, raw)
    if isinstance(grouped, dict):
        _group_findings(grouped, findings)

    _secondary_endpoint(fetch, f"{api}/errors/trends?hours=24", "/errors/trends", findings, raw)
    _secondary_endpoint(fetch, f"{api}/errors/recent?limit=20", "/errors/recent", findings, raw)

    return ProbeResult(name="http", ok=True, findings=findings, raw=raw)
