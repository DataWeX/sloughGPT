"""Mole — the one monitoring app: read-only probe sweep + always-on watch.

One package, one brand (2026-10 rebrand; the legacy doctor package merged
in — module paths, URLs, and CLI all say Mole; grandfathered literals keep
``$SLO_DOCTOR_*`` / ``~/.cache/slog-doctor/``):

- **Probes + report** (report-only): read-only observers over the live
  stack — gates, benchmarks, http, sse, journey. Triage through the shared
  health-flow severity vocabulary (``domain/infrastructure/_internal/
  health_flow.py``), one JSON report, one exit code. No remediation, no
  writes to the site.
- **Watch** (always-on): the same sweep on a cadence — findings
  fingerprint (delta dedupe), append-only journal, change-only events.
  Read-only by construction; suggests, never applies.

Placement (user directive): no new domain — the wiring layer lives in
``domain/core`` ("the body of the system": infra/system-level ops), with
probe seams integrating the core components (journeys, api, inference,
training, system).

Probe seams:
    gates       newest suite*.out vs the known drift baseline (local files)
    benchmarks  stored benchmark records vs thresholds — the verdict comes
                from scripts/benchmark_results.py itself (local files)
    http        API self-diagnosis surface (/health, /errors/*)
    sse         /health/stream — payload-size + cadence watchdog
    journey     UX sweep report (domain.journeys) — report-only unless swept

The journey runner freezes ``SLO_WEB_URL``/``SLO_API_URL`` at *import*
time with a stale ``:5175`` web default; this package sets
``SLO_WEB_URL=http://localhost:5173`` (setdefault) mole-side before any
journey import. Journey files are never edited.

Chat-stream probe: deferred to Phase B — it needs an authenticated
session plus live model state that the read-only, no-auth surface does
not provide (follow-up noted in ``probes/sse.py``).

Usage:
    .venv/bin/python -m domain.core._internal.mole            # one-pass sweep
    .venv/bin/python -m domain.core._internal.mole --watch    # cadence + journal
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Iterable

from domain.core._internal.mole.models import Finding, band, rank, worst
from domain.core._internal.mole.probes import PROBES, ProbeResult
from domain.core._internal.mole.report import (
    MoleReport,
    default_report_path,
    default_targets,
    merge,
)
from domain.infrastructure._internal.health_flow import Severity

__all__ = [
    "PROBES",
    "Finding",
    "MoleReport",
    "ProbeResult",
    "band",
    "default_journal_path",
    "default_report_path",
    "default_targets",
    "fingerprint",
    "load_context",
    "merge",
    "rank",
    "run_mole",
    "run_watch",
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


def run_mole(
    *,
    run_sweep: bool = False,
    window_s: float = 9.0,
    skip: Iterable[str] = (),
    report_path: str | None = None,
    write: bool = True,
    preflight: bool = True,
) -> MoleReport:
    """Run the registered probes, triage findings, optionally write the report.

    Args:
        run_sweep: run the browser journey sweep (default: read the
            existing journey report file only).
        window_s: SSE observation window in seconds.
        skip: probe names to skip (``gates``, ``benchmarks``, ``http``,
            ``sse``, ``journey``).
        report_path: report destination (default ``$SLO_DOCTOR_REPORT`` or
            ``~/.cache/slog-doctor/findings-report.json``).
        write: persist the report as JSON.
        preflight: probe API reachability first; when it fails, ``http``
            and ``sse`` are skipped (their target is down) and the
            critical finding still forces exit code 2.

    Returns:
        The merged :class:`MoleReport` (ranked findings, worst-first).
    """
    skipped = {s.strip() for s in skip if s.strip()}
    # Freeze targets (and the mole-side :5173 env default) BEFORE probes
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
        except Exception as exc:  # a probe crash must not take Mole down
            res = ProbeResult(
                name=name,
                ok=False,
                findings=[
                    Finding(
                        source=name,
                        check="mole.probe_crash",
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


# Watch last: the watch module must never import this package at module
# scope (it is imported from here — a package<->module cycle otherwise).
from domain.core._internal.mole.watch import (  # noqa: E402
    default_journal_path,
    fingerprint,
    load_context,
    run_watch,
)
