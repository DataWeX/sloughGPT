"""Benchmarks probe — latest recorded benchmark vs its predecessor.

Reads ``$SLO_BENCH_DATA_DIR`` (default
``~/.local/share/sloughgpt/benchmark_results`` — the store written by
``scripts/benchmark_results.py record``; the SAME env var and default as
that script, so this probe never carries a second default path to drift)
and, for every kind that script tracks in ``REGRESSION_THRESHOLDS``,
compares the two newest records with THAT script's own ``is_regression``
— one regression definition, never a second copy. Kinds with fewer than
two records have no baseline and stay silent; untracked kind dirs are
ignored. Read-only, deterministic, no AI.

Findings (component ``benchmarks``):
    store              info when the results dir is missing (fresh
                       checkout), warn when the benchmark script cannot
                       load; probe ok=False
    benchmarks.{kind}  warn when regressed vs the previous record (all
                       deltas in detail), ok when within thresholds
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from domain.core._internal.mole.models import Finding
from domain.infrastructure._internal.health_flow import Severity

from . import ProbeResult

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCRIPT = _REPO_ROOT / "scripts" / "benchmark_results.py"
_mod = None  # one load per process, not per tick
_default_dir: Path | None = None  # the SCRIPT's canonical RESULTS_DIR, captured at load


def _bench():
    """Load scripts/benchmark_results.py once (importlib; it has a main guard)."""
    global _mod, _default_dir
    if _mod is None:
        spec = importlib.util.spec_from_file_location("_mole_benchmark_results", _SCRIPT)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {_SCRIPT}")
        mod = importlib.util.module_from_spec(spec)
        # exec with the store env cleared: captures the script's own
        # canonical RESULTS_DIR independent of any per-call override, so a
        # later env-unset tick can never inherit a stale dir from an
        # earlier env-set one (deterministic per call, one default only)
        saved = os.environ.pop("SLO_BENCH_DATA_DIR", None)
        try:
            spec.loader.exec_module(mod)
        finally:
            if saved is not None:
                os.environ["SLO_BENCH_DATA_DIR"] = saved
        _default_dir = mod.RESULTS_DIR
        _mod = mod
    return _mod


def _finding(
    check: str, severity: Severity, score: float, message: str, detail: str = ""
) -> Finding:
    return Finding(
        source="benchmarks",
        check=check,
        severity=severity,
        score=score,
        message=message,
        detail=detail,
        component="benchmarks",
    )


def run_probe() -> ProbeResult:
    """Probe the stored benchmark history. No network, no subprocess, no AI."""
    try:
        mod = _bench()
    except Exception as exc:
        return ProbeResult(
            name="benchmarks",
            ok=False,
            findings=[
                _finding(
                    "benchmarks.store",
                    Severity.WARN,
                    60.0,
                    "Benchmark history unreadable",
                    str(exc),
                )
            ],
            error=f"load {_SCRIPT.name}: {exc}",
        )

    # per-call env wins; otherwise the script's own canonical default
    # (captured at load) — never a second default path here to drift from
    # the store owner. This replaces the old SLO_BENCH_RESULTS_DIR +
    # <repo>/data fallback, which pointed at a dir the store stopped
    # writing when it moved to SLO_BENCH_DATA_DIR (probe reported "No
    # benchmark records yet" forever).
    env = os.environ.get("SLO_BENCH_DATA_DIR")
    d = Path(env).expanduser() if env else _default_dir
    mod.RESULTS_DIR = d  # deterministic per call; no stale override survives
    if not d.is_dir():
        return ProbeResult(
            name="benchmarks",
            ok=False,
            findings=[
                _finding(
                    "benchmarks.store", Severity.INFO, 90.0, "No benchmark records yet", str(d)
                )
            ],
            error=f"no results dir: {d}",
        )

    findings: list[Finding] = []
    raw: dict = {}
    for kind in mod.REGRESSION_THRESHOLDS:
        records = mod.collect_records(kind)
        raw[kind] = len(records)
        if len(records) < 2:
            continue  # no baseline yet — nothing actionable, no nagging
        try:
            new = mod.load_result(records[0])  # collect_records is newest-first
            old = mod.load_result(records[1])
        except (OSError, ValueError) as exc:
            findings.append(
                _finding(
                    f"benchmarks.{kind}",
                    Severity.WARN,
                    60.0,
                    f"{kind} benchmark record unreadable",
                    str(exc),
                )
            )
            continue
        deltas = mod._regression_deltas(kind, new, old)  # noqa: SLF001 — display only
        detail = ", ".join(f"{m}={delta:+.4g}" for m, delta in deltas.items())
        if mod.is_regression(kind, new, old):
            findings.append(
                _finding(
                    f"benchmarks.{kind}",
                    Severity.WARN,
                    60.0,
                    f"{kind} regressed vs previous record",
                    detail,
                )
            )
        else:
            findings.append(
                _finding(
                    f"benchmarks.{kind}",
                    Severity.OK,
                    100.0,
                    f"{kind} within thresholds vs previous record",
                    detail,
                )
            )
    return ProbeResult(name="benchmarks", ok=True, findings=findings, raw=raw)
