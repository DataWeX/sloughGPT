"""Consciousness hot-path benchmark: the inner life's per-event costs.

Measures what the consciousness lane added to the per-request path so
refactors have a repeatable before/after gate (AGENTS.md: benchmark after
every inference implementation — this is the lane's benchmark):

1. reflect_cold    -- propose-only reflection with the fingerprint cache
                      cleared: full narrative rebuild (SelfModel.reflect).
2. reflect_cached  -- cache hit on unchanged state: the card 0cc54e7e
                      promise — injection cost collapses to a fingerprint
                      compare (sanity-asserts one object is served).
                      speedup = cold / cached is REPORTED, not
                      thresholded: a ratio of two noisy numbers.
3. mindlog_append  -- one MindRecord JSONL append (the substrate's
                      marginal per-message cost: json.dumps + one write,
                      torn-tail heal check included in the path).
4. query_1000_records -- query_mind(limit=100) over a 1000-record log:
                      projection render cost (full scan + parse + filter).
5. feedback_observe -- observe_feedback('thumbs_up'): qualia + calibrated
                      belief deltas + self_model.json save (real per-event
                      disk cost, calibration card 449e1c81).

GOLD (deliberately generous: catch order-of-magnitude regressions —
accidental O(n) scans, fsync-per-write, cache bypass — not scheduler jitter):
  - reflect_cached    <= 500 us (the per-request injection budget; cache
                       bypass is caught stronger by the identity sanity:
                       one object must serve unchanged state — card
                       0cc54e7e's determinism promise)
  - mindlog_append     <= 500 us
  - query 1000 records <= 100 ms
  - feedback_observe   <= 25 ms

cache_speedup (cold/cached) is REPORTED for context, never thresholded —
both sides are microseconds at this scale and the ratio is noisy; the
identity check, not the ratio, detects a broken cache.

Usage:
    python scripts/benchmark_consciousness_hotpaths.py [--n 2000] [--reps 7] [--json FILE]

Record:
    python scripts/benchmark_consciousness_hotpaths.py --json /tmp/opencode/consciousness.json
    python scripts/benchmark_results.py record --kind consciousness --json-file /tmp/opencode/consciousness.json

Exit: 0 when all GOLD checks pass, 1 otherwise. Measure the exit through a
file redirect (`... --json f.json > /tmp/out; echo $?`) — piping to tail
reports tail's status, not this script's.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from domain.cognition._internal.consciousness.config import ConsciousnessConfig  # noqa: E402
from domain.cognition._internal.consciousness.engine import ConsciousnessEngine  # noqa: E402
from domain.cognition._internal.consciousness.mind_state import (  # noqa: E402
    MindLog,
    MindRecord,
    query_mind,
)
from domain.cognition._internal.consciousness.self_model import SelfEpisode  # noqa: E402

GOLD = {
    "max_reflect_cached_us": 500.0,
    "max_mindlog_append_us": 500.0,
    "max_query_us": 100_000.0,
    "max_feedback_observe_us": 25_000.0,
}


def _median_us(fn, n: int, reps: int) -> float:
    """Median us per-call over interleaved reps (robust to load spikes)."""
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        samples.append((time.perf_counter() - t0) / n * 1e6)
    return statistics.median(samples)


def _build_engine(store: Path) -> ConsciousnessEngine:
    """Engine at the production window (save() trims to 100 episodes)."""
    engine = ConsciousnessEngine(ConsciousnessConfig(level=1, store_path=str(store)))
    for i in range(100):
        engine.self_model.episodes.append(
            SelfEpisode(
                timestamp=time.time() - (10 - i),
                input_text=f"question {i}",
                response=f"answer {i}",
                qualia={"valence": 0.1},
                self_insight="benchmark seed",
                growth_delta=0.02,
            )
        )
    return engine


def main() -> int:
    parser = argparse.ArgumentParser(description="Consciousness hot-path benchmark")
    parser.add_argument("--n", type=int, default=2000, help="calls per timed metric")
    parser.add_argument("--reps", type=int, default=7, help="interleaved repetitions")
    parser.add_argument("--json", type=str, default=None, help="write results JSON here")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="bench_consciousness_") as tmp:
        store = Path(tmp)
        engine = _build_engine(store)

        # Sanity: unchanged state serves ONE cached object.
        assert engine.reflect(apply=False) is engine.reflect(apply=False), "cache miss?"

        def _cold() -> None:
            engine._reflection_cache = None  # noqa: SLF001 — force rebuild
            engine.reflect(apply=False)

        def _warm() -> None:
            engine.reflect(apply=False)

        reflect_cold_us = _median_us(_cold, max(100, args.n // 10), args.reps)
        reflect_cached_us = _median_us(_warm, args.n, args.reps)
        speedup = reflect_cold_us / max(reflect_cached_us, 1e-9)

        log = MindLog(store / "append")
        record = MindRecord(
            type="episode",
            payload={"input_text": "x" * 200, "response": "y" * 400},
            provenance="experience",
        )
        mindlog_append_us = _median_us(lambda: log.append(record), args.n, args.reps)

        qlog = MindLog(store / "query")
        for i in range(1000):
            qlog.append(
                MindRecord(type="episode", payload={"i": i}, provenance="experience", t=float(i))
            )

        def _query() -> None:
            got = query_mind(qlog.path.parent, record_type="episode", limit=100)
            assert len(got) == 100

        query_us = _median_us(_query, 20, args.reps)

        feedback_us = _median_us(
            lambda: engine.observe_feedback("thumbs_up"), max(20, args.n // 20), args.reps
        )

    checks = [
        (
            "reflect_cached <= 500us (injection budget)",
            reflect_cached_us <= GOLD["max_reflect_cached_us"],
        ),
        ("mindlog_append <= 500us", mindlog_append_us <= GOLD["max_mindlog_append_us"]),
        ("query_1000_records <= 100ms", query_us <= GOLD["max_query_us"]),
        ("feedback_observe <= 25ms", feedback_us <= GOLD["max_feedback_observe_us"]),
    ]
    passed = all(ok for _, ok in checks)

    print("── consciousness hot paths ──────────────────────────────")
    for label, value, unit in (
        ("reflect_cold", reflect_cold_us, "us"),
        ("reflect_cached", reflect_cached_us, "us"),
        ("cache_speedup", speedup, "x"),
        ("mindlog_append", mindlog_append_us, "us"),
        ("query_1000_records", query_us, "us"),
        ("feedback_observe", feedback_us, "us"),
    ):
        print(f"  {label:24} {value:>12.1f} {unit}")
    print("── GOLD ─────────────────────────────────────────────────")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "kind": "consciousness",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "metrics": {
                        "reflect_cold_us": round(reflect_cold_us, 2),
                        "reflect_cached_us": round(reflect_cached_us, 2),
                        "cache_speedup": round(speedup, 1),
                        "mindlog_append_us": round(mindlog_append_us, 2),
                        "query_1000_records_us": round(query_us, 2),
                        "feedback_observe_us": round(feedback_us, 2),
                    },
                    "gold": GOLD,
                    "passed": passed,
                },
                indent=2,
            )
            + "\n"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
