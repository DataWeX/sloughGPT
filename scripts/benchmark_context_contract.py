"""
Context-contract benchmark: measures the chat-seam selection+render overhead
that `ContextPart`/`budget_parts`/`render_context_block` add per request
relative to the legacy string path they replace.

What it measures (per call, microseconds):
  - legacy:    KnowledgeProcessor bullet rendering over joined strings
    (the pre-contract path: facts joined + "Knowledge context:\\n- ..." build)
  - contract:  budget_parts (sort + token estimate + whole-part selection)
    + render_context_block (attributed numbered block)
  across realistic part counts (5 / 20 / 50) with and without omitted parts.

Why it exists: the contract runs on EVERY chat request, so the budgeter is a
new hot-path cost. This pins it as negligible vs the model call (ms-s range)
and catches accidental regressions (e.g. tokenizer binding falling back to
O(n^2) dedup or text slicing inside the loop).

Usage:
    ./scripts/python scripts/benchmark_context_contract.py            # run
    ./scripts/python scripts/benchmark_context_contract.py --json out.json
    ./scripts/python scripts/benchmark_results.py record --kind context_contract --json-file out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domain.inference._internal.context_contract import (  # noqa: E402
    ContextPart,
    budget_parts,
    render_context_block,
)
from domain.models._internal.provider.processors.knowledge import (  # noqa: E402
    KnowledgeProcessor,
)

REPEATS = 2000
SIZES = (5, 20, 50)


def _make_parts(n: int) -> list[ContextPart]:
    """n realistic parts: mixed layers, scores, locators."""
    layers = ["memory", "rag", "rag", "knowledge", "rag"]
    return [
        ContextPart(
            text=f"Retrieved fact number {i}: the capital of country {i} is city {i}. " * 3,
            source=layers[i % len(layers)],
            layer="rag" if i % 2 else "memory",
            score=round(0.95 - (i % 10) * 0.08, 2),
            rank=i + 1,
            locator=f"chunk:{i:04d}",
        )
        for i in range(n)
    ]


def _make_legacy(n: int) -> list[str]:
    """The pre-contract input: flat strings (augmenter-style, [:300])."""
    return [f"fact number {i}: the capital of country {i} is city {i}. "[:300] for i in range(n)]


def _time(fn, repeats: int = REPEATS) -> dict:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)  # µs
    samples.sort()
    return {
        "mean_us": round(statistics.fmean(samples), 3),
        "p50_us": round(statistics.median(samples), 3),
        "p95_us": round(samples[int(len(samples) * 0.95) - 1], 3),
    }


def run() -> dict:
    results: dict = {"repeats": REPEATS, "sizes": {}}
    # Shared loop: KnowledgeProcessor.process is async; creating a loop per
    # iteration would drown the render cost we're actually measuring.
    loop = asyncio.new_event_loop()

    for n in SIZES:
        parts = _make_parts(n)
        legacy = _make_legacy(n)

        def legacy_fn(legacy=legacy):
            return loop.run_until_complete(
                KnowledgeProcessor(knowledge=legacy).process([{"role": "user", "content": "q"}])
            )

        # Contract path as wired in inference.py: one budget + one render
        # (processor construction is trivial; include it for parity).
        def contract_fn(parts=parts):
            b = budget_parts(parts, 512)
            return loop.run_until_complete(
                KnowledgeProcessor(knowledge=b.parts, omitted_count=b.omitted_count).process(
                    [{"role": "user", "content": "q"}]
                )
            )

        entry = {
            "legacy": _time(legacy_fn),
            "contract": _time(contract_fn),
            "budget_only": _time(lambda parts=parts: budget_parts(parts, 512)),
            "render_only": _time(
                lambda parts=parts: render_context_block(parts[:5], omitted_count=1)
            ),
        }
        entry["overhead_x"] = round(entry["contract"]["mean_us"] / entry["legacy"]["mean_us"], 2)
        results["sizes"][str(n)] = entry

    # Correctness spot-check under budget: nothing destroyed, omission counted.
    b = budget_parts(_make_parts(50), 512)
    results["sanity"] = {
        "selected_of_50": len(b.parts),
        "omitted": b.omitted_count,
        "tokens_used": b.total_tokens,
        "budget": b.max_tokens,
    }
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, help="write raw results to this file")
    args = ap.parse_args()

    print("context-contract benchmark — per-request selection+render cost")
    print(f"repeats per case: {REPEATS}")
    results = run()

    for size, entry in results["sizes"].items():
        print(f"\n{size} parts:")
        print(
            f"  legacy    mean={entry['legacy']['mean_us']:>8.2f}µs  p95={entry['legacy']['p95_us']:>8.2f}µs"
        )
        print(
            f"  contract  mean={entry['contract']['mean_us']:>8.2f}µs  p95={entry['contract']['p95_us']:>8.2f}µs"
            f"  (x{entry['overhead_x']} vs legacy)"
        )
        print(
            f"  budget only: {entry['budget_only']['mean_us']:.2f}µs   render only: {entry['render_only']['mean_us']:.2f}µs"
        )

    s = results["sanity"]
    print(
        f"\nsanity: budget_parts(50 parts, 512 tok) -> selected={s['selected_of_50']} "
        f"omitted={s['omitted']} used={s['tokens_used']}/{s['budget']} tok"
    )

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")

    # Regression guard: the contract path must stay under 1ms mean at 50
    # parts (it is per-request; the model call it fronts is 1000x that).
    worst = results["sizes"]["50"]["contract"]["mean_us"]
    if worst > 1000:
        print(f"REGRESSION: contract path {worst:.1f}µs exceeds 1000µs budget at 50 parts")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
