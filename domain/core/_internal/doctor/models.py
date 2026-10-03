"""Severity triage primitives for the site doctor.

The vocabulary is shared with the API health flow — ``Severity`` and
``Diagnosis`` are imported from ``domain/infrastructure/_internal/
health_flow.py``, never redefined, so a doctor finding projects straight
into the ``health_score.diagnoses`` shape the frontend already renders.

Score banding (mirrors health_flow's healthy/degraded/unhealthy cut lines)::

    score >= 80   → Severity.OK         healthy / nominal
    50 <= score < 80 → Severity.WARN    degraded — needs attention
    score < 50    → Severity.CRITICAL   unhealthy — blocks the journey

``band()`` never returns ``Severity.INFO``: INFO is assigned *explicitly*
to informational findings (context worth recording, no action required —
"report missing", "stream requires auth"). Convention: INFO findings carry
score 90, OK findings 100, WARN 50-79, CRITICAL < 50, so ``rank()`` stays
consistent both across and within severities.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from domain.infrastructure._internal.health_flow import Diagnosis, Severity

_SEVERITY_RANK = {
    Severity.OK: 0,
    Severity.INFO: 1,
    Severity.WARN: 2,
    Severity.CRITICAL: 3,
}


@dataclass
class Finding:
    """One doctor observation, produced by exactly one probe."""

    source: str  # probe that produced it: http | sse | journey | preflight
    check: str  # stable check id, e.g. "api.health_score"
    severity: Severity
    score: float  # 0-100, same scale as Diagnosis.score
    message: str  # human-readable, one line
    detail: str = ""  # supporting evidence (samples, lists, paths)
    component: str = ""  # core component: journeys | api | inference | training | system

    def to_diagnosis(self) -> Diagnosis:
        """Project into the health-flow Diagnosis shape (shared vocabulary)."""
        return Diagnosis(
            check=self.check,
            severity=self.severity,
            score=float(self.score),
            message=self.message,
            detail=self.detail,
        )


def band(score: float) -> Severity:
    """Map a 0-100 score onto the health-flow severity bands.

    ``Severity.INFO`` is deliberately never returned — see module docstring.
    """
    if score >= 80:
        return Severity.OK
    if score >= 50:
        return Severity.WARN
    return Severity.CRITICAL


def rank(findings: Iterable[Finding]) -> list[Finding]:
    """Sort worst-first: critical > warn > info > ok, then score ascending.

    Score ascending inside a severity means "worst instance first" (the
    lowest-scoring critical leads the report).
    """
    return sorted(findings, key=lambda f: (-_SEVERITY_RANK[f.severity], f.score))


def worst(findings: Iterable[Finding]) -> Severity:
    """Highest severity present; empty input → ``Severity.OK``.

    Drives the report summary line and the process exit code.
    """
    highest = Severity.OK
    for finding in findings:
        if _SEVERITY_RANK[finding.severity] > _SEVERITY_RANK[highest]:
            highest = finding.severity
    return highest
