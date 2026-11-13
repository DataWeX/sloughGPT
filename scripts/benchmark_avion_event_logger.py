#!/usr/bin/env python3
"""Benchmark — avion event-logger write path (journal + sink fan-out).

What it measures (per-op cost of ``EventLogger.log``, the path every
avion event goes through):

1. **journal append** — JSONL append + flush-per-event (the durability
   contract), with and without ``fsync``;
2. **fan-out** — journal-only vs +1 healthy sink vs +5 sinks (cost of
   the registration API);
3. **isolation overhead** — one always-failing sink next to a healthy
   one (price of per-sink try/except + failure counting);
4. **reopen** — seq recovery over an already-populated journal.

Usage::

    .venv/bin/python scripts/benchmark_avion_event_logger.py [--events 5000]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "avion" / "src"))

from avion.events.journal import EventJournal  # noqa: E402
from avion.events.logger import EventLogger  # noqa: E402
from avion.events.models import Event, EventType  # noqa: E402


class _Sink:
    """Healthy no-op sink (fan-out target)."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def emit(self, event: Event) -> None:
        pass


class _BrokenSink:
    """Always-failing sink (isolation overhead target)."""

    @property
    def name(self) -> str:
        return "broken"

    def emit(self, event: Event) -> None:
        raise RuntimeError("sink down")


def _run(label: str, n: int, fn) -> None:
    """Time fn(n) and print a one-line summary."""
    start = time.perf_counter()
    fn(n)
    elapsed = time.perf_counter() - start
    per_op_us = (elapsed / n) * 1e6
    rate = n / elapsed
    print(f"  {label:<44} {rate:>10,.0f} ev/s   {per_op_us:>8.2f} us/event")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=5000, help="events per scenario")
    args = parser.parse_args()
    n = args.events

    workdir = Path(tempfile.mkdtemp(prefix="avion_logger_bench_"))
    print(f"avion event-logger benchmark — {n:,} events per scenario")
    try:
        # 1. journal append (durability contract: flush per event)
        journal = EventJournal(workdir / "bench.jsonl")
        _run("journal append (flush per event)", n, lambda k: [journal.append(_ev()) for _ in range(k)])
        journal.close()

        # 2. fan-out
        _run("journal-only (no sinks)", n, lambda k: _fanout(workdir / "f0.jsonl", [], k))
        _run("journal + 1 sink", n, lambda k: _fanout(workdir / "f1.jsonl", [_Sink("s1")], k))
        _run("journal + 5 sinks", n, lambda k: _fanout(workdir / "f5.jsonl", [_Sink(f"s{i}") for i in range(5)], k))

        # 3. isolation overhead: 1 failing + 1 healthy
        logger = EventLogger(
            EventJournal(workdir / "fiso.jsonl"), [_BrokenSink(), _Sink("healthy")]
        )
        _run("journal + 1 broken + 1 healthy sink", n, lambda k: _log(logger, k))
        assert logger.sink_failures("broken") > 0, "isolation must count failures"

        # 4. reopen / seq recovery over a populated journal
        reopen_path = workdir / "bench.jsonl"
        lines = reopen_path.read_text(encoding="utf-8").count("\n")
        start = time.perf_counter()
        reps = 50
        for _ in range(reps):
            EventJournal(reopen_path).close()
        elapsed = (time.perf_counter() - start) / reps
        print(
            f"  {'reopen + seq recovery':<44} {elapsed * 1e3:>10,.2f} ms/reopen"
            f"   ({lines:,} lines scanned)"
        )

        # 5. fsync LAST, on purpose: a long fsync run congests the page
        # writeback and would throttle every scenario measured after it.
        fsync_journal = EventJournal(workdir / "bench_fsync.jsonl", fsync=True)
        _run("journal append (+ fsync)", min(n, 1000), lambda k: [fsync_journal.append(_ev()) for _ in range(k)])
        fsync_journal.close()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


def _ev() -> Event:
    return Event(
        type=EventType.CLICK,
        name="text='Start'",
        data={"target": "button", "attempt": 1},
        duration_ms=1.5,
    )


def _fanout(path: Path, sinks: list, n: int) -> None:
    logger = EventLogger(EventJournal(path), sinks)
    _log(logger, n)
    logger.close()


def _log(logger: EventLogger, n: int) -> None:
    event = _ev()
    for _ in range(n):
        logger.log(event)


if __name__ == "__main__":
    raise SystemExit(main())
