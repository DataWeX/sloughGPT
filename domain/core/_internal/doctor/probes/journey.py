"""Journey probe — reads the UX sweep report written by ``ux_journeys``.

The harness lives in ``tests/ux_journeys`` behind a process boundary: this
probe reads the report file, and only crosses into the harness by spawning it.
The live vite dev server serves from ``:5173``, so this probe calls
``os.environ.setdefault`` for the live truth before handing env over — the
fix lives doctor-side rather than in the harness.

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
import pathlib
import time

from domain.core._internal.doctor.models import Finding
from domain.infrastructure._internal.health_flow import Severity

from . import ProbeResult

STALE_AFTER_S = 24 * 60 * 60
_FAIL_CRITICAL_FRACTION = 0.25
_FAIL_CRITICAL_COUNT = 3


def _set_env_defaults() -> None:
    """Freeze this process's defaults so the sweep subprocess inherits them.

    The harness lives in ``tests/ux_journeys`` and runs behind a process
    boundary, so nothing it does can mutate our environment any more — these
    are defaults we choose to hand over, not a race we order imports around.
    """
    os.environ.setdefault("SLO_WEB_URL", "http://localhost:5173")
    os.environ.setdefault("SLO_API_URL", "http://localhost:8000")


def _journeys_dir() -> pathlib.Path | None:
    """Locate the repo root by finding ``tests/ux_journeys`` from this file.

    Keys off this file's own path rather than the process cwd, so it works
    whether the doctor runs from the source tree or from an installed layout.
    """
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "tests" / "ux_journeys").is_dir():
            return parent
    return None


def _report_path() -> str:
    """Mirror runner.REPORT resolution — the report file is the whole contract."""
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
    """Run the live sweep in its own process; returns (report_path, error).

    A process boundary, not an import: ``ux_journeys`` resolves the stack URLs
    at import time, so importing it used to race this probe's own env defaults
    (hence the ordering comment that lived on ``_set_env_defaults``). A
    subprocess inherits a plain snapshot of ``os.environ`` and cannot mutate
    ours — the race is gone rather than ordered around. The report file stays
    the entire contract between us.
    """
    import subprocess
    import sys

    root = _journeys_dir()
    if root is None:
        return _report_path(), "cannot locate tests/ux_journeys"

    env = dict(os.environ)
    # `tests` is shadowed by a stray user-site package (AGENTS.md), and we
    # invoke the harness by that name — so the child must skip user site.
    env["PYTHONNOUSERSITE"] = "1"
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(root), prior) if p)

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "tests.ux_journeys"],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except (
        OSError,
        subprocess.SubprocessError,
    ) as exc:  # a crashed sweep must still yield a probe result
        return _report_path(), f"sweep failed to start: {exc}"

    if proc.returncode == 1:
        # Flows failed — the report says so, and triage below is where that
        # belongs. Treating it as an error would hide the per-flow detail.
        return _report_path(), ""
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        return _report_path(), f"sweep exited {proc.returncode}: {tail}"
    return _report_path(), ""


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
