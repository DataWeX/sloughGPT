"""OOP hot-path benchmark: per-call object cost (RAM + CPU).

Measures the KANBAN card f5c4cf80 ("OOP perf sweep") hot-path metrics so
refactors of per-request/per-token Python objects have a repeatable
before/after gate. Three metrics, all load-robust (interleaved medians):

1. construct  -- GenerationContext construction: current class vs an
                  IDENTICAL un-slotted twin (the pre-refactor shape).
                  Slots-refactor proof: new must stay <= old.
2. step_expr  -- LSTM fallback token-step windowing in SloEngine.generate:
                  ``generated_ids[len(idx.flatten()):]`` (was: O(n) flatten
                  + allocation EVERY token step) vs the hoisted
                  ``generated_ids[prompt_len:]`` (O(1) slice).
3. memory     -- per-instance footprint (instance + __dict__ for the twin,
                  inline slots for the current class) x a per-10k projection.

Usage:
    python scripts/benchmark_oop_hotpaths.py [--n 5000] [--reps 7] [--json FILE]

Record:
    python scripts/benchmark_oop_hotpaths.py --json /tmp/opencode/oop.json
    python scripts/benchmark_results.py record --kind oop --json-file /tmp/opencode/oop.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from domain.core._internal.soul import GenerationContext  # noqa: E402


@dataclass
class _UnslottedContext:
    """Pre-refactor twin: same 16 fields, plain __dict__ (the BEFORE shape)."""

    prompt: str
    prompt_tokens: np.ndarray
    system_prompt: str = ""
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    max_tokens: int = 2048
    stop_tokens: list = field(default_factory=list)
    reasoning_depth: str = "balanced"
    cognitive_boost: bool = True
    emotional_context: dict = field(default_factory=dict)
    soul_overrides: dict = field(default_factory=dict)
    reasoning_chain: list = field(default_factory=list)
    repetition_penalty: float = 1.2
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


def _median_us(fn, n: int, reps: int) -> float:
    """Median ns->us per-call over interleaved reps (robust to load spikes)."""
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        samples.append((time.perf_counter() - t0) / n * 1e6)
    return statistics.median(samples)


def run(n: int, reps: int) -> dict:
    arr = np.array([[0]], dtype=np.int64)
    kw = dict(prompt="t", prompt_tokens=arr)

    # 1. construction: interleave old/new inside the same rep to cancel drift
    old_c, new_c = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n):
            _UnslottedContext(**kw)
        old_c.append((time.perf_counter() - t0) / n * 1e6)
        t0 = time.perf_counter()
        for _ in range(n):
            GenerationContext(**kw)
        new_c.append((time.perf_counter() - t0) / n * 1e6)
    construct_old = statistics.median(old_c)
    construct_new = statistics.median(new_c)

    # 2. token-step windowing expression (LSTM fallback loop, per token)
    idx = np.arange(128, dtype=np.int64)[np.newaxis, :]
    generated_ids = list(range(300))
    step_old = _median_us(lambda: generated_ids[len(idx.flatten()) :], n, reps)
    prompt_len = idx.size
    step_new = _median_us(lambda: generated_ids[prompt_len:], n, reps)

    # 3. memory footprint
    twin = _UnslottedContext(**kw)
    mem_old = sys.getsizeof(twin) + sys.getsizeof(twin.__dict__)
    slot = GenerationContext(**kw)
    mem_new = sys.getsizeof(slot)

    return {
        "kind": "oop",
        "n": n,
        "reps": reps,
        "construct_us_old": round(construct_old, 3),
        "construct_us_new": round(construct_new, 3),
        "construct_pct_faster": round((1 - construct_new / construct_old) * 100, 1),
        "step_expr_us_old": round(step_old, 3),
        "step_expr_us_new": round(step_new, 3),
        "step_pct_faster": round((1 - step_new / step_old) * 100, 1),
        "mem_bytes_old": mem_old,
        "mem_bytes_new": mem_new,
        "mem_pct_smaller": round((1 - mem_new / mem_old) * 100, 1),
        "mem_kb_per_10k_calls": round((mem_old - mem_new) * 10000 / 1024, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5000, help="calls per rep")
    ap.add_argument("--reps", type=int, default=7, help="repetitions (median)")
    ap.add_argument("--json", type=Path, help="write results JSON here")
    args = ap.parse_args()

    res = run(args.n, args.reps)
    try:
        res["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT, check=True,
        ).stdout.strip()
    except Exception:
        res["git_commit"] = "unknown"

    print("OOP hot-path benchmark (interleaved medians, load-robust)")
    print(f"  construct : {res['construct_us_old']:.3f} -> {res['construct_us_new']:.3f} us "
          f"({res['construct_pct_faster']}% faster)")
    print(f"  tok step  : {res['step_expr_us_old']:.3f} -> {res['step_expr_us_new']:.3f} us "
          f"({res['step_pct_faster']}% faster, was O(n) flatten/token)")
    print(f"  memory    : {res['mem_bytes_old']} -> {res['mem_bytes_new']} B/instance "
          f"({res['mem_pct_smaller']}% smaller, {res['mem_kb_per_10k_calls']} KB per 10k calls)")

    ok = res["construct_us_new"] <= res["construct_us_old"] * 1.05
    print(f"  gate: construct(new) <= construct(old)*1.05 -> {'PASS' if ok else 'FAIL'}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
