"""Mole — the always-on monitor: dig the running system for change deltas.

Mole reuses the probe contract wholesale (``run_mole``,
``MoleReport``, the ``PROBES`` registry) and adds *time*: a cadence, a
findings fingerprint, an append-only journal, and change-only events so
the same finding never alerts twice. No AI anywhere — the probes are
deterministic read-only observers and the loop is plain stdlib, runnable
standalone (CLI in ``__main__``, scripts, tests — no event loop, no
model). Traversal follows contracts, not code: ``PROBES`` → report →
journal lines; a new mole probe is picked up on the next tick.

Scope guardrails (per the Mole card): suggest, never apply — Mole only
observes and journals; load context is recorded alongside every tick so
benchmarks/findings carry the machine state they were taken on.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import TYPE_CHECKING, Any, Callable

# Annotations only: this module must never import the package at module
# scope — ``domain.core._internal.mole`` imports *this* module (cycle).
if TYPE_CHECKING:
    from domain.core._internal.mole.report import MoleReport

__all__ = ["default_journal_path", "fingerprint", "load_context", "run_watch"]

_ERROR_EXIT = 2


def default_journal_path() -> str:
    """``$SLO_MOLE_JOURNAL`` or ``~/.cache/slog-doctor/mole-journal.jsonl``."""
    return os.path.expanduser(
        os.environ.get("SLO_MOLE_JOURNAL", "~/.cache/slog-doctor/mole-journal.jsonl")
    )


def fingerprint(report: MoleReport) -> str:
    """Stable hash of the findings' *identity* set — noise-insensitive.

    Identity = (source, check, severity, component): structural, not the
    payload. Messages and scores carry live values (p95, health score,
    frame sizes) that change every sweep; hashing those would re-alert
    on every tick. A band flip, a new finding, or a removed one changes
    the identity set and *does* flip the fingerprint. Sorted, so ranked
    reordering between sweeps cannot flip it either.
    """
    identities = sorted(
        json.dumps([f.source, f.check, str(f.severity), f.component]) for f in report.findings
    )
    blob = "\n".join(identities).encode("utf-8", "replace")
    return hashlib.sha1(blob).hexdigest()[:16]


def load_context() -> dict[str, Any]:
    """Host context for a tick (recorded, never a finding — context only)."""
    try:
        loadavg: list[float] | None = [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):  # non-POSIX: context degrades, watch continues
        loadavg = None
    return {"loadavg": loadavg, "cpu_count": os.cpu_count()}


def _append(path: str, line: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, sort_keys=True) + "\n")


def run_watch(
    *,
    interval_s: float = 30.0,
    max_ticks: int | None = None,
    skip: tuple[str, ...] = (),
    journal_path: str | None = None,
    run: Callable[..., MoleReport] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_event: Callable[[dict, MoleReport | None], None] | None = None,
    context: Callable[[], dict] | None = None,
    strict: bool = False,
) -> int:
    """Run ``run`` on a cadence; journal every tick; emit events only on change.

    Sleeps happen *between* ticks (``max_ticks`` runs cost
    ``max_ticks - 1`` sleeps). Tick 1 is always a change (it establishes
    the baseline). A failing tick is journaled and contained — the loop
    never dies mid-run — and yields exit code ``2``; otherwise the exit
    code is the last report's (mole convention, ``strict`` aware).

    All seams (``run``, ``sleep``, ``on_event``, ``context``) are
    injectable so the loop is fully testable without network or time.
    """
    if run is None:  # late import: the package imports this module first
        from domain.core._internal.mole import run_mole

        run = run_mole
    context = context or load_context
    path = journal_path or default_journal_path()

    prev: str | None = None
    tick = 0
    exit_code = 0
    while max_ticks is None or tick < max_ticks:
        tick += 1
        ts = time.time()
        try:
            report = run(skip=set(skip), write=True)
            fp = fingerprint(report)
            changed = fp != prev
            prev = fp
            exit_code = report.exit_code(strict=strict)
            line = {
                "ts": ts,
                "tick": tick,
                "context": context(),
                "fingerprint": fp,
                "changed": changed,
                "overall": str(report.overall),
                "exit": exit_code,
                "summary": report.summary,
            }
            payload: MoleReport | None = report
        except Exception as exc:  # noqa: BLE001 — containment is the point
            exit_code = _ERROR_EXIT
            line = {
                "ts": ts,
                "tick": tick,
                "context": context(),
                "error": f"{type(exc).__name__}: {exc}",
                "changed": True,
                "exit": _ERROR_EXIT,
            }
            payload = None
        _append(path, line)
        if on_event is not None and line.get("changed"):
            on_event(line, payload)
        if max_ticks is None or tick < max_ticks:
            sleep(interval_s)
    return exit_code
