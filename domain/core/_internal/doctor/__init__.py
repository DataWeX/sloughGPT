"""site-doctor — system manager/monitor that works through the site process.

Placement (user directive): no new domain — the wiring layer lives in
``domain/core`` ("the body of the system": infra/system-level ops), with
probe seams integrating the core components (journeys, api, inference,
training, system).

Phase A is REPORT-ONLY: probes observe the live stack (read-only GETs,
SSE consumption, one opt-in browser sweep), triage what they see through
the shared health-flow severity vocabulary (``domain/infrastructure/
_internal/health_flow.py``), write a JSON report, and set an exit code.
No remediation, no writes to the site.

Probe seams:
    http      API self-diagnosis surface (/health, /errors/*)
    sse       /health/stream — payload-size + cadence watchdog
    journey   UX sweep report (domain.journeys) — report-only unless swept

The journey runner freezes ``SLO_WEB_URL``/``SLO_API_URL`` at *import*
time with a stale ``:5175`` web default; this package sets
``SLO_WEB_URL=http://localhost:5173`` (setdefault) doctor-side before any
journey import. Journey files are never edited.

Chat-stream probe: deferred to Phase B — it needs an authenticated
session plus live model state that Phase A's read-only, no-auth surface
does not provide (follow-up noted in ``probes/sse.py``).

Usage:
    .venv/bin/python -m domain.core._internal.doctor --no-sweep
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Iterable

from domain.core._internal.doctor.models import Finding, band, rank, worst
from domain.core._internal.doctor.probes import PROBES, ProbeResult
from domain.core._internal.doctor.report import (
    DoctorReport,
    default_report_path,
    default_targets,
    merge,
)
from domain.infrastructure._internal.health_flow import Severity

__all__ = [
    "PROBES",
    "DoctorReport",
    "Finding",
    "ProbeResult",
    "band",
    "default_report_path",
    "default_targets",
    "merge",
    "rank",
    "run_doctor",
    "worst",
]

_PREFLIGHT_TIMEOUT_S = 3.0


def _preflight(api: str) -> ProbeResult | None:
    """Quick API reachability check. None → reachable, else a critical finding.

    A dead target must surface as exit code 2 even if every other probe
    is skipped or crashes on its own way to the same unreachable server.
    """
    detail = ""
    try:
        with urllib.request.urlopen(f"{api}/health", timeout=_PREFLIGHT_TIMEOUT_S) as resp:
            if 200 <= resp.status < 300:
                return None
            detail = f"HTTP {resp.status}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        detail = str(exc)
    detail = detail or "no response"
    return ProbeResult(
        name="preflight",
        ok=False,
        findings=[
            Finding(
                source="preflight",
                check="api.reachable",
                severity=Severity.CRITICAL,
                score=0.0,
                message="API unreachable",
                detail=f"{api}/health → {detail}",
                component="api",
            )
        ],
        error=detail,
    )


def run_doctor(
    *,
    run_sweep: bool = False,
    window_s: float = 9.0,
    skip: Iterable[str] = (),
    report_path: str | None = None,
    write: bool = True,
    preflight: bool = True,
) -> DoctorReport:
    """Run the registered probes, triage findings, optionally write the report.

    Args:
        run_sweep: run the browser journey sweep (default: read the
            existing journey report file only).
        window_s: SSE observation window in seconds.
        skip: probe names to skip (``http``, ``sse``, ``journey``).
        report_path: report destination (default ``$SLO_DOCTOR_REPORT`` or
            ``~/.cache/slog-doctor/findings-report.json``).
        write: persist the report as JSON.
        preflight: probe API reachability first; when it fails, ``http``
            and ``sse`` are skipped (their target is down) and the
            critical finding still forces exit code 2.

    Returns:
        The merged :class:`DoctorReport` (ranked findings, worst-first).
    """
    skipped = {s.strip() for s in skip if s.strip()}
    # Freeze targets (and the doctor-side :5173 env default) BEFORE probes
    # — the journey probe imports domain.journeys, which freezes env at import.
    targets = default_targets()
    results: list[ProbeResult] = []

    if preflight and "http" not in skipped:
        pre = _preflight(targets["api"])
        if pre is not None:
            results.append(pre)
            skipped.update({"http", "sse"})

    executed: list[str] = []
    for entry in PROBES:
        name = entry["name"]
        if name in skipped:
            continue
        executed.append(name)
        try:
            if name == "journey":
                res = entry["run"](run_sweep=run_sweep)
            elif name == "sse":
                res = entry["run"](window_s=window_s)
            else:
                res = entry["run"]()
        except Exception as exc:  # a probe crash must not take the doctor down
            res = ProbeResult(
                name=name,
                ok=False,
                findings=[
                    Finding(
                        source=name,
                        check="doctor.probe_crash",
                        severity=Severity.WARN,
                        score=60.0,
                        message=f"Probe {name} crashed: {exc}",
                        component="system",
                    )
                ],
                error=str(exc),
            )
        results.append(res)

    report = merge(results, targets=targets, skipped=sorted(skipped - set(executed)))
    if write:
        report.write(report_path)
    return report
