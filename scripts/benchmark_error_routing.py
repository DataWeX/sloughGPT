#!/usr/bin/env python3
"""Error-path benchmark: registry resolver + envelope cost, old vs new.

No benchmark covered the error path, so this measures exactly what the
error-taxonomy work (card 99994ad6) touched:

    resolver      raise_error()'s registry lookup — the only per-raise code the
                  change altered. "old" is the previous implementation
                  (value-only ``ErrorCode(code)`` + silent (ValueError, KeyError)
                  swallow), "new" is name-first lookup with a loud fallback.
    alias         the legacy alias path (``E_CREATE_FAILED``), which used to
                  miss the value-only lookup entirely and raise ValueError.
    unknown       an unregistered code: the new path logs before falling back,
                  so it is expected to be slower — this is the defect path, and
                  the drift gate makes it unreachable.
    envelope      ``error_response()`` body construction (unchanged by the work,
                  reported for context).
    full raise    AppError construction through ``raise_error()`` — what a
                  failing request actually pays.

Usage:
    python scripts/benchmark_error_routing.py            # print the table
    python scripts/benchmark_error_routing.py --json     # machine-readable
    python scripts/benchmark_error_routing.py --ci       # exit 1 on >20% regression
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "apps" / "api" / "server"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from schemas.common import error_response, raise_error  # noqa: E402

from domain.infrastructure._internal.errors import (  # noqa: E402
    ERROR_REGISTRY,
    AppError,
    ErrorCode,
)

ITERATIONS = 20_000
REGRESSION_THRESHOLD = 0.20  # 20%, same gate benchmark_latency.py uses


def old_resolver(code: str, status_code: int | None = None) -> tuple[str, int]:
    """The previous lookup: value-only, swallowing (ValueError, KeyError)."""
    try:
        ec = ErrorCode(code)
        class_name, default_status, _recoverable, _user_msg = ERROR_REGISTRY[ec]
    except (ValueError, KeyError):
        class_name = "AppError"
        default_status = status_code or 400
    return class_name, default_status


def new_resolver(code: str, status_code: int | None = None) -> tuple[str, int]:
    """The current lookup: name first (aliases work), loud when unregistered."""
    ec = ErrorCode.__members__.get(code)
    if ec is None:
        try:
            ec = ErrorCode(code)
        except ValueError:
            ec = None
    info = ERROR_REGISTRY.get(ec) if ec is not None else None
    if info is None:
        logging.getLogger("slo.errors").warning(
            "raise_error: unregistered code %r — falling back to generic AppError/400.",
            code,
            stack_info=True,
        )
        return "AppError", status_code or 400
    return info[0], info[1]


def _time(fn, iterations: int) -> float:
    """Seconds per operation — median of 5 rounds, GC paused.

    Timing a dict lookup against an Enum call while the box runs other test
    suites is otherwise pure noise: a single gen-2 collection inside a round
    moves the number by tens of percent.
    """
    samples: list[float] = []
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        fn()  # warm-up: first call pays import/attribute caches
        for _ in range(5):
            start = time.perf_counter()
            for _i in range(iterations):
                fn()
            samples.append((time.perf_counter() - start) / iterations)
    finally:
        if was_enabled:
            gc.enable()
    return sorted(samples)[len(samples) // 2]


def _time_pair(fn_old, fn_new, iterations: int) -> tuple[float, float]:
    """Interleaved old/new timing so clock drift and load cancel out.

    Both implementations are timed in the same rounds, alternately, and each
    reports the median of its own samples.
    """
    samples_old: list[float] = []
    samples_new: list[float] = []
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        fn_old()
        fn_new()
        for _ in range(5):
            start = time.perf_counter()
            for _i in range(iterations):
                fn_old()
            samples_old.append((time.perf_counter() - start) / iterations)
            start = time.perf_counter()
            for _i in range(iterations):
                fn_new()
            samples_new.append((time.perf_counter() - start) / iterations)
    finally:
        if was_enabled:
            gc.enable()
    samples_old.sort()
    samples_new.sort()
    mid = len(samples_old) // 2
    return samples_old[mid], samples_new[mid]


def _measure(label: str, fn, iterations: int) -> dict:
    return _measure_seconds(label, _time(fn, iterations), iterations)


def _measure_seconds(label: str, per_op: float, iterations: int) -> dict:
    return {
        "label": label,
        "seconds_per_op": per_op,
        "ns_per_op": per_op * 1e9,
        "ops_per_sec": 1.0 / per_op if per_op else float("inf"),
        "iterations": iterations,
    }


def run() -> dict:
    # Keep the unknown-code measurement honest: production handlers format and
    # write the record (dominated by I/O); a NullHandler measures the code path.
    slog = logging.getLogger("slo.errors")
    slog.handlers = [logging.NullHandler()]
    slog.propagate = False

    results: list[dict] = []

    # The two A/B comparisons are timed interleaved, round by round.
    for label, code in (("resolver.registered", "E_BUSY"), ("resolver.alias", "E_CREATE_FAILED")):
        old_s, new_s = _time_pair(
            lambda c=code: old_resolver(c), lambda c=code: new_resolver(c), ITERATIONS
        )
        results.append(_measure_seconds(f"{label}.old", old_s, ITERATIONS))
        results.append(_measure_seconds(f"{label}.new", new_s, ITERATIONS))

    results += [
        _measure("resolver.unknown.old", lambda: old_resolver("E_TOTALLY_NEW"), ITERATIONS),
        _measure("resolver.unknown.new", lambda: new_resolver("E_TOTALLY_NEW"), 2_000),
        _measure("envelope.error_response", lambda: error_response("boom", "E_BUSY"), ITERATIONS),
        _measure("raise.full", lambda: _swallow_raise("boom", "E_BUSY"), 5_000),
    ]
    return {"results": results, "iterations": {"default": ITERATIONS}}


def _swallow_raise(message: str, code: str) -> None:
    try:
        raise_error(message, code)
    except AppError:
        pass


def _by_label(run_result: dict) -> dict[str, dict]:
    return {r["label"]: r for r in run_result["results"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--ci", action="store_true", help="exit 1 on a >20% regression")
    args = parser.parse_args()

    run_result = run()
    by_label = _by_label(run_result)

    if args.json:
        print(json.dumps(run_result, indent=2))
        return 0

    width = max(len(r["label"]) for r in run_result["results"])
    print(f"{'case':<{width}}  {'ns/op':>12}  {'ops/sec':>14}")
    print("-" * (width + 30))
    for r in run_result["results"]:
        print(f"{r['label']:<{width}}  {r['ns_per_op']:>12,.1f}  {r['ops_per_sec']:>14,.0f}")

    print("\nold → new deltas (negative = the new resolver is faster):")
    regressions = []
    for pair in (
        ("resolver.registered.old", "resolver.registered.new"),
        ("resolver.alias.old", "resolver.alias.new"),
    ):
        old_ns = by_label[pair[0]]["ns_per_op"]
        new_ns = by_label[pair[1]]["ns_per_op"]
        delta = (new_ns - old_ns) / old_ns
        print(f"  {pair[0].split('.')[1]:<10} {old_ns:>9,.1f} → {new_ns:>9,.1f} ns  ({delta:+.1%})")
        if delta > REGRESSION_THRESHOLD:
            regressions.append((pair[1], delta))

    alias_old = by_label["resolver.alias.old"]["ns_per_op"]
    alias_new = by_label["resolver.alias.new"]["ns_per_op"]
    print(
        f"\nalias lookup: the old resolver missed E_CREATE_FAILED entirely "
        f"({alias_old:,.1f} ns → ValueError + generic 400); the new one resolves it in "
        f"{alias_new:,.1f} ns."
    )
    print(
        "unknown code: the new resolver logs a warning before falling back "
        f"({by_label['resolver.unknown.new']['ns_per_op']:,.1f} ns with a NullHandler; a real "
        "handler adds I/O). The drift gate keeps this path unreachable."
    )

    if regressions and args.ci:
        print(f"\nREGRESSION: {regressions}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
