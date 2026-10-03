"""Journey probe — reads the UX sweep report written by ``domain.journeys``.

The journey runner freezes ``SLO_WEB_URL`` / ``SLO_API_URL`` as module
globals at *import* time, and its web default is stale (``:5175``; the
live vite dev server is ``:5173``). This probe therefore calls
``os.environ.setdefault`` for the live truth BEFORE importing anything
from ``domain.journeys`` — the fix lives doctor-side; journey files are
never edited.

Findings (component ``journeys``):
    failed flows        critical when >= 25% of flows fail or >= 3 fail,
                        otherwise warn; per-flow detail in the message
    console errors      warn (samples in detail)
    network errors      warn (samples in detail)
    report missing      probe ok=False, info finding, error on the result
    report stale        info (older than 24h)
"""

from __future__ import annotations

import json
import os
import time

from domain.core._internal.doctor.models import Finding
from domain.infrastructure._internal.health_flow import Severity

from . import ProbeResult

STALE_AFTER_S = 24 * 60 * 60
_FAIL_CRITICAL_FRACTION = 0.25
_FAIL_CRITICAL_COUNT = 3


def _set_env_defaults() -> None:
    """Freeze live-stack env BEFORE ``domain.journeys`` import freezes it."""
    os.environ.setdefault("SLO_WEB_URL", "http://localhost:5173")
    os.environ.setdefault("SLO_API_URL", "http://localhost:8000")


def _report_path() -> str:
    """Mirror runner.REPORT resolution without importing the runner."""
    cache = os.environ.get("SLO_JOURNEY_CACHE", os.path.expanduser("~/.cache/slog-journeys"))
    return os.environ.get("SLO_JOURNEY_REPORT", os.path.join(cache, "ux-flows-report.json"))


def _flow_label(flow: dict) -> str:
    return str(flow.get("label") or flow.get("task") or flow.get("name") or "?")


def _load_report(path: str) -> tuple[dict | None, str]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None, "no report — run with sweep"
    except (OSError, ValueError) as exc:
        return None, f"unreadable report: {exc}"
    if not isinstance(data, dict):
        return None, "unreadable report: not a JSON object"
    return data, ""


def _sweep() -> tuple[str, str]:
    """Run the live sweep; returns (report_path, error)."""
    import asyncio
    import importlib

    # importlib returns whatever sys.modules holds (stub-friendly) and
    # runs domain.journeys.__init__ only if it has not been imported yet.
    runner = importlib.import_module("domain.journeys.runner")
    journeys = importlib.import_module("domain.journeys")
    try:
        asyncio.run(runner.run(list(journeys.FLOWS), headed=False, strict_errors=False))
    except Exception as exc:  # a crashed sweep must still yield a probe result
        return runner.REPORT, f"sweep crashed: {exc}"
    return runner.REPORT, ""


def run_probe(run_sweep: bool) -> ProbeResult:
    """Read (or produce) the journey sweep report and triage it."""
    _set_env_defaults()
    sweep_error = ""
    path = _report_path()
    if run_sweep:
        path, sweep_error = _sweep()

    report, load_error = _load_report(path)
    if report is None:
        error = sweep_error or load_error
        findings = [
            Finding(
                source="journey",
                check="journey.report",
                severity=Severity.INFO,
                score=90.0,
                message="Journey report missing — no sweep has been recorded",
                detail=error,
                component="journeys",
            )
        ]
        return ProbeResult(name="journey", ok=False, findings=findings, error=error)

    findings: list[Finding] = []
    flows = report.get("flows") or []
    failed_flows = [f for f in flows if f.get("status") != "passed"]
    failed = len(failed_flows)
    total = len(flows)

    if failed:
        fraction = failed / total if total else 1.0
        critical = fraction >= _FAIL_CRITICAL_FRACTION or failed >= _FAIL_CRITICAL_COUNT
        severity = Severity.CRITICAL if critical else Severity.WARN
        score = 25.0 if critical else 60.0
        labels = [_flow_label(f) for f in failed_flows]
        message = f"{failed}/{total} journeys failed: {', '.join(labels[:3])}"
        if len(labels) > 3:
            message += f" (+{len(labels) - 3} more)"
        detail = "\n".join(
            f"- {_flow_label(f)} [{f.get('status')}] {f.get('error') or ''}".rstrip()
            for f in failed_flows
        )
        findings.append(
            Finding(
                source="journey",
                check="journey.flows",
                severity=severity,
                score=score,
                message=message,
                detail=detail,
                component="journeys",
            )
        )
    elif total:
        findings.append(
            Finding(
                source="journey",
                check="journey.flows",
                severity=Severity.OK,
                score=100.0,
                message=f"All {total} journeys passed",
                component="journeys",
            )
        )

    console_count = int(report.get("console_error_count") or 0)
    if console_count:
        samples = report.get("console_errors_sample") or []
        findings.append(
            Finding(
                source="journey",
                check="journey.console_errors",
                severity=Severity.WARN,
                score=60.0,
                message=f"{console_count} console errors during sweep",
                detail="\n".join(str(s) for s in samples[:5]),
                component="journeys",
            )
        )

    network_count = int(report.get("network_error_count") or 0)
    if network_count:
        samples = report.get("network_errors_sample") or []
        findings.append(
            Finding(
                source="journey",
                check="journey.network_errors",
                severity=Severity.WARN,
                score=60.0,
                message=f"{network_count} network errors during sweep",
                detail="\n".join(str(s) for s in samples[:5]),
                component="journeys",
            )
        )

    try:
        age_s = max(0.0, time.time() - os.path.getmtime(path))
    except OSError:
        age_s = 0.0
    if age_s > STALE_AFTER_S:
        findings.append(
            Finding(
                source="journey",
                check="journey.stale",
                severity=Severity.INFO,
                score=90.0,
                message=f"Journey report is stale ({age_s / 3600:.1f}h old)",
                detail=path,
                component="journeys",
            )
        )

    return ProbeResult(
        name="journey",
        ok=True,
        findings=findings,
        error=sweep_error,
        raw={"report_path": path, "passed": report.get("passed"), "failed": report.get("failed")},
    )
