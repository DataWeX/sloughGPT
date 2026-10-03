"""Doctor report — merge probe results into one ranked, serializable report.

Exit codes mirror the journeys convention: ``0`` ok/info, ``1`` warn,
``2`` critical. Report path: ``${SLO_DOCTOR_REPORT:-~/.cache/slog-doctor/
findings-report.json}`` (parent dirs created; written tmp+rename so a
concurrent reader never sees a half-written file).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from domain.core._internal.doctor.models import Finding, rank, worst
from domain.core._internal.doctor.probes import ProbeResult
from domain.infrastructure._internal.health_flow import Severity

SCHEMA_VERSION = 1


def default_report_path() -> str:
    """``$SLO_DOCTOR_REPORT`` or ``~/.cache/slog-doctor/findings-report.json``."""
    return os.path.expanduser(
        os.environ.get("SLO_DOCTOR_REPORT", "~/.cache/slog-doctor/findings-report.json")
    )


def default_targets() -> dict:
    """Probe targets from env — also applies the doctor-side :5173 fix.

    Must run BEFORE ``domain.journeys`` is imported anywhere: the journey
    runner freezes these values at import and its own web default is
    stale (``:5175``).
    """
    web = os.environ.setdefault("SLO_WEB_URL", "http://localhost:5173")
    api = os.environ.setdefault("SLO_API_URL", "http://localhost:8000")
    targets = {"web": web, "api": api}
    gateway = os.environ.get("SLO_GATEWAY_URL")
    if gateway:
        targets["gateway"] = gateway
    return targets


@dataclass
class DoctorReport:
    """Merged doctor output: ranked findings + probe health + summary."""

    findings: list[Finding] = field(default_factory=list)
    probes: list[dict] = field(default_factory=list)
    targets: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION

    @property
    def summary(self) -> dict:
        counts = {severity.value: 0 for severity in Severity}
        for finding in self.findings:
            counts[finding.severity.value] += 1
        return {"total": len(self.findings), "by_severity": counts}

    @property
    def overall(self) -> Severity:
        return worst(self.findings)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "ts": self.ts,
            "targets": self.targets,
            "probes": self.probes,
            "findings": [
                {
                    "source": f.source,
                    "check": f.check,
                    "severity": str(f.severity),
                    "score": f.score,
                    "message": f.message,
                    "detail": f.detail,
                    "component": f.component,
                }
                for f in self.findings
            ],
            "summary": self.summary,
            "overall": str(self.overall),
        }

    def write(self, path: str | None = None) -> str:
        """Write JSON atomically (tmp + rename in the target dir). Returns the path."""
        target = path or default_report_path()
        parent = os.path.dirname(target) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=".findings-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return target

    def exit_code(self, strict: bool = False) -> int:
        """0 → ok/info, 1 → warn (or info when strict), 2 → critical."""
        overall = self.overall
        if overall == Severity.CRITICAL:
            return 2
        if overall == Severity.WARN or (strict and overall == Severity.INFO):
            return 1
        return 0


def merge(
    probe_results: Iterable[ProbeResult],
    targets: dict | None = None,
    skipped: Iterable[str] = (),
) -> DoctorReport:
    """Fold probe results into a ranked :class:`DoctorReport`."""
    results = list(probe_results)
    findings = rank(f for result in results for f in result.findings)
    probes = [{"name": r.name, "ok": r.ok, "error": r.error} for r in results]
    probes += [{"name": name, "ok": None, "error": "skipped"} for name in skipped]
    return DoctorReport(
        findings=findings,
        probes=probes,
        targets=targets if targets is not None else default_targets(),
    )
