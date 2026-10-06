"""Gates probe — newest full-suite gate artifact vs the known drift baseline.

Reads (never re-runs) the newest parseable ``suite*.out`` under
``$SLO_GATES_DIR`` (default ``~/.cache/slog-gates``), pulls the trailing
pytest summary line (``= N failed, M passed, ... E errors in Xs =``) from
the file tail, and compares failed+errors against the known-drift
baseline ``$SLO_GATES_DRIFT_MAX`` (default 838 — the measured main-suite
baseline of 2026-10-03, kanban 56b49cf1). Read-only, deterministic,
no AI: it only inspects what the gate runner already left behind.

Only runs shaped like the canonical full suite (``passed >=
$SLO_GATES_FULL_PASSED``, default 40000 — the canonical run passes ~46k)
are comparable to that baseline; scope-limited runs (subsets, extracts)
get an honest info finding instead of a misleading verdict.

Findings (component ``ci``):
    baseline  ok within baseline · warn over · critical at >= 2x baseline
    scope     info for scope-limited artifacts, probe ok=False
    parse     warn when artifacts exist but none parse, probe ok=False
    artifact  info when none exist, probe ok=False (journey precedent)
    stale     info past ``$SLO_GATES_STALE_H`` (default 24h)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from domain.core._internal.mole.models import Finding
from domain.infrastructure._internal.health_flow import Severity

from . import ProbeResult

_DEFAULT_BASELINE = 838  # measured main-suite baseline, 2026-10-03 (card 56b49cf1)
_DEFAULT_FULL_PASSED = 40000  # canonical suite passes ~46k; subsets sit far below
_DEFAULT_STALE_H = 24.0
_TAIL_BYTES = 4096
_SUMMARY = re.compile(r"^=+\s+(.+?)\s+=+$")


def _dir() -> Path:
    return Path(os.environ.get("SLO_GATES_DIR") or "~/.cache/slog-gates").expanduser()


def _baseline() -> int:
    try:
        return max(1, int(os.environ.get("SLO_GATES_DRIFT_MAX", _DEFAULT_BASELINE)))
    except ValueError:
        return _DEFAULT_BASELINE


def _full_min() -> int:
    try:
        return max(1, int(os.environ.get("SLO_GATES_FULL_PASSED", _DEFAULT_FULL_PASSED)))
    except ValueError:
        return _DEFAULT_FULL_PASSED


def _stale_s() -> float:
    try:
        return float(os.environ.get("SLO_GATES_STALE_H", _DEFAULT_STALE_H)) * 3600
    except ValueError:
        return _DEFAULT_STALE_H * 3600


def _finding(
    check: str, severity: Severity, score: float, message: str, detail: str = ""
) -> Finding:
    return Finding(
        source="gates",
        check=check,
        severity=severity,
        score=score,
        message=message,
        detail=detail,
        component="ci",
    )


def _parse(path: Path) -> dict | None:
    """Counts from the LAST pytest summary line in the file tail, else None."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - _TAIL_BYTES))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    counts: dict | None = None
    for line in tail.splitlines():
        m = _SUMMARY.match(line.strip())
        if m is None:
            continue
        body = m.group(1)
        passed = re.search(r"(\d+) passed", body)
        if passed is None:
            continue  # incomplete line or "no tests ran" — keep scanning
        failed = re.search(r"(\d+) failed", body)
        errors = re.search(r"(\d+) errors?", body)
        counts = {
            "failed": int(failed.group(1)) if failed else 0,
            "passed": int(passed.group(1)),
            "errors": int(errors.group(1)) if errors else 0,
        }
    return counts


def _missing(d: Path, err: str = "") -> ProbeResult:
    return ProbeResult(
        name="gates",
        ok=False,
        findings=[
            _finding(
                "gates.artifact",
                Severity.INFO,
                90.0,
                "No gate artifacts yet",
                f"no suite*.out in {d}",
            )
        ],
        error=err or f"no suite*.out in {d}",
    )


def _scope(counts: dict) -> ProbeResult:
    full_min = _full_min()
    return ProbeResult(
        name="gates",
        ok=False,
        findings=[
            _finding(
                "gates.scope",
                Severity.INFO,
                90.0,
                f"Gate artifact is scope-limited ({counts['passed']} passed < "
                f"{full_min} full-suite threshold) — drift baseline not comparable",
                f"{counts['failed']} failed, {counts['errors']} errors in this scope",
            )
        ],
        error="no full-suite artifact",
        raw={**counts, "full": False, "baseline": _baseline()},
    )


def _verdict(path: Path, mtime: float, counts: dict) -> ProbeResult:
    baseline = _baseline()
    bad = counts["failed"] + counts["errors"]
    age_h = round((time.time() - mtime) / 3600, 1)
    raw = {
        "file": path.name,
        "age_h": age_h,
        "baseline": baseline,
        "bad": bad,
        "full": True,
        **counts,
    }
    numbers = f"{counts['failed']} failed, {counts['errors']} errors, {counts['passed']} passed"
    if bad <= baseline:
        finding = _finding(
            "gates.baseline",
            Severity.OK,
            100.0,
            f"Gate run within baseline ({bad} bad of {baseline} allowed: {numbers})",
            path.name,
        )
    elif bad >= 2 * baseline:
        finding = _finding(
            "gates.baseline",
            Severity.CRITICAL,
            0.0,
            f"Gate run far over baseline ({bad} bad >= {2 * baseline}: {numbers})",
            path.name,
        )
    else:
        finding = _finding(
            "gates.baseline",
            Severity.WARN,
            60.0,
            f"Gate run over baseline ({bad} bad > {baseline} allowed: {numbers})",
            path.name,
        )
    findings = [finding]
    if age_h * 3600 > _stale_s():
        findings.append(
            _finding(
                "gates.stale",
                Severity.INFO,
                90.0,
                f"Gate artifact {age_h:.0f}h old (limit {_stale_s() / 3600:.0f}h) — rerun the suite",
                path.name,
            )
        )
    return ProbeResult(name="gates", ok=True, findings=findings, raw=raw)


def run_probe() -> ProbeResult:
    """Probe the local gate artifacts. No network, no subprocess, no AI."""
    d = _dir()
    if not d.is_dir():
        return _missing(d)
    try:
        candidates = sorted(
            ((p, p.stat().st_mtime) for p in d.glob("suite*.out")),
            key=lambda pm: pm[1],
            reverse=True,
        )
    except OSError as exc:  # dir vanished mid-scan
        return _missing(d, str(exc))
    if not candidates:
        return _missing(d)
    subset: dict | None = None
    for path, mtime in candidates:
        counts = _parse(path)
        if counts is None:
            continue
        if counts["passed"] < _full_min():
            if subset is None:  # remember the NEWEST subset for the info finding
                subset = counts
            continue
        return _verdict(path, mtime, counts)
    if subset is not None:
        return _scope(subset)
    return ProbeResult(
        name="gates",
        ok=False,
        findings=[
            _finding(
                "gates.parse",
                Severity.WARN,
                60.0,
                "Gate artifacts unreadable — no pytest summary in any suite*.out",
                f"newest: {candidates[0][0].name} ({len(candidates)} file(s))",
            )
        ],
        error="no parsable pytest summary",
        raw={"files": [p.name for p, _ in candidates[:5]]},
    )
