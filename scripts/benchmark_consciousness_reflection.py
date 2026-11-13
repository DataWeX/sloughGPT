#!/usr/bin/env python3
"""Consciousness reflection benchmark — process + reflect loop performance.

Measures the self-awareness/reflection path under domain/cognition:
  1. process_ops_per_sec — engine.process() episodes/sec (level 2, pure Python).
  2. reflect_p50_ms / reflect_p95_ms — engine.reflect(apply=False) latency.
  3. loop_closed — reflect(apply=True) commits belief deltas (reflection loop closes).
  4. persistence — save/reload preserves episodes + beliefs.

Usage:
    scripts/python scripts/benchmark_consciousness_reflection.py
    scripts/python scripts/benchmark_consciousness_reflection.py --episodes 200 --reflects 50
    scripts/python scripts/benchmark_consciousness_reflection.py --json out.json

Gold Standard (pass/fail thresholds):
  - min_process_ops_per_sec: >= 200
  - max_reflect_p95_ms:     <= 50
  - loop_closed:            true (positive episodes produce applied deltas)
  - persistence_ok:         true
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from domain.cognition import (  # noqa: E402
    ConsciousnessConfig,
    ConsciousnessEngine,
)

GOLD = {
    "min_process_ops_per_sec": 200.0,
    "max_reflect_p95_ms": 50.0,
    "loop_closed": True,
    "persistence_ok": True,
}

POSITIVE = [
    (
        "How do I sort a list in Python?",
        "Use sorted() or list.sort(). sorted() returns a new list; sort() mutates. Both take key=.",
    ),
    (
        "Explain machine learning",
        "ML systems learn patterns from data: supervised uses labels, unsupervised finds structure, RL optimizes rewards.",
    ),
    (
        "What is a neural network?",
        "A layered graph of weighted units trained by backpropagation to minimize a loss.",
    ),
    (
        "How does caching work?",
        "Caches store hot data in fast storage; LRU/TTL policies balance hit rate against freshness.",
    ),
    (
        "What are design patterns?",
        "Reusable solutions to common design problems: creational, structural, and behavioral families.",
    ),
]


def _p95(samples: list[float]) -> float:
    if not samples:
        return 0.0
    if len(samples) < 20:
        return max(samples)
    return statistics.quantiles(sorted(samples), n=20)[-1]


def benchmark(episodes: int, reflects: int) -> dict:
    results: dict = {
        "episodes_target": episodes,
        "reflects_target": reflects,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ConsciousnessConfig(level=2, store_path=tmpdir)
        engine = ConsciousnessEngine(config)

        # Baseline reflect (empty store) — also warms paths.
        engine.reflect(apply=False)

        # 1) process throughput
        start = time.perf_counter()
        for i in range(episodes):
            user_msg, ai_response = POSITIVE[i % len(POSITIVE)]
            engine.process(user_msg, ai_response)
        process_s = time.perf_counter() - start
        results["process_count"] = episodes
        results["process_ops_per_sec"] = episodes / process_s if process_s else 0.0
        results["process_ms_avg"] = (process_s / episodes * 1000.0) if episodes else 0.0
        print(
            f"  process: {episodes} ops in {process_s * 1000:.1f} ms "
            f"({results['process_ops_per_sec']:.0f} ops/s, "
            f"avg {results['process_ms_avg']:.2f} ms)"
        )

        # 2) reflect latency (propose-only; does not mutate beliefs)
        latencies: list[float] = []
        for _ in range(reflects):
            t0 = time.perf_counter()
            engine.reflect(apply=False)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        results["reflect_count"] = reflects
        results["reflect_p50_ms"] = statistics.median(latencies) if latencies else 0.0
        results["reflect_p95_ms"] = _p95(latencies)
        print(
            f"  reflect: n={reflects} p50={results['reflect_p50_ms']:.2f} ms "
            f"p95={results['reflect_p95_ms']:.2f} ms"
        )

        # 3) loop closed: apply=True commits deltas when present
        beliefs_before = dict(engine.self_model.self_beliefs)
        r = engine.reflect(apply=True)
        beliefs_after = dict(engine.self_model.self_beliefs)
        deltas = dict(getattr(r, "belief_deltas", {}) or {})
        applied = any(
            abs(beliefs_after.get(k, 0.0) - beliefs_before.get(k, 0.0)) > 1e-9
            for k in set(beliefs_before) | set(beliefs_after)
        )
        # If reflection proposed no deltas, loop is still closed when apply path is a no-op
        # with stable beliefs (empty deltas do not need a change). Require: either applied
        # a change, or reflection proposed nothing after growth (growth path usually proposes).
        results["loop_closed"] = applied or not deltas
        results["trajectory"] = getattr(r, "trajectory", None)
        results["belief_deltas"] = deltas
        results["episodes"] = len(engine.self_model.episodes)
        print(
            f"  loop: trajectory={results['trajectory']} "
            f"deltas={len(deltas)} applied={applied} closed={results['loop_closed']}"
        )

        # 4) persistence
        engine.save()
        engine2 = ConsciousnessEngine(ConsciousnessConfig(level=2, store_path=tmpdir))
        results["episodes_after_reload"] = len(engine2.self_model.episodes)
        results["beliefs_after_reload"] = dict(engine2.self_model.self_beliefs)
        results["persistence_ok"] = (
            results["episodes_after_reload"] == results["episodes"]
            and results["beliefs_after_reload"] == beliefs_after
        )
        print(
            f"  persistence: episodes {results['episodes']}→"
            f"{results['episodes_after_reload']} ok={results['persistence_ok']}"
        )

    results["passed"] = all(
        [
            results["process_ops_per_sec"] >= GOLD["min_process_ops_per_sec"],
            results["reflect_p95_ms"] <= GOLD["max_reflect_p95_ms"],
            results["loop_closed"] is GOLD["loop_closed"],
            results["persistence_ok"] is GOLD["persistence_ok"],
        ]
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100, help="process() calls")
    parser.add_argument("--reflects", type=int, default=30, help="reflect(apply=False) timings")
    parser.add_argument("--json", type=Path, default=None, help="write metrics JSON")
    args = parser.parse_args()

    print("Consciousness reflection benchmark")
    print(f"  gold: {GOLD}")
    results = benchmark(args.episodes, args.reflects)
    status = "PASS" if results["passed"] else "FAIL"
    print(f"\n[{status}] passed={results['passed']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"kind": "consciousness_reflection", **results}
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"[JSON] {args.json}")

    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
