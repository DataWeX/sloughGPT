"""World workers — parallel evaluation of independent world runs (Stage B).

The seam follows AGENTS.md execution philosophy: the sim core stays
synchronous and single-threaded *per world* (no asyncio in this stack); the
pool fans out across *independent* runs — seeds, parameter sweeps, evolution
shards — each of which builds its own ``WorldDriver``.

Why ``ProcessPoolExecutor`` and not the ThreadPool precedent
(``training/executor.py``): ``np.random`` is **process-global**. Two
threaded runs would interleave draws on the shared MT19937 stream, so a
run's result would depend on thread scheduling — breaking the seeded-parity
contract every world test leans on (``test_deterministic_on_seed``). A
process pool gives each run a private RNG (fork), so seed S produces the
identical snapshot solo or in a pool, *and* real parallelism for the
CPU-bound numpy sim that threads would serialize behind the GIL. For
``--brain llm`` runs the model call blocks inside the worker — process
isolation keeps those runs' RNG streams clean too.

Contract: :func:`run_world_runs` returns results **in input order**;
``max_workers == 1`` (or a single run) executes inline on the caller — the
deterministic baseline the pooled path must match exactly.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from .world_driver import WorldDriver


def _run_one(spec: dict[str, Any]) -> dict[str, Any]:
    """Execute one independent seeded run (top-level: picklable by the pool).

    Args:
        spec: ``{"kind": "ticks"|"evolution", "seed": int, "params":
            WorldParams}`` plus ``"ticks"`` or the evolution knobs
            ``"generations"``/``"population"``/``"ticks_per_generation"``.

    Returns:
        ``{"seed", "kind", "snapshot"}`` for ticks runs, ``{"seed",
        "kind", "summary"}`` for evolution runs.

    Raises:
        ValueError: on an unknown run kind — a typo'd spec must never
            silently become a different run.
    """
    params = spec["params"]
    seed = int(spec["seed"])
    driver = WorldDriver(params, seed=seed)
    kind = spec.get("kind", "ticks")
    if kind == "evolution":
        summary = driver.run_evolution(
            generations=int(spec.get("generations", 5)),
            population=int(spec.get("population", 8)),
            ticks_per_generation=int(spec.get("ticks_per_generation", 20)),
        )
        return {"seed": seed, "kind": "evolution", "summary": summary}
    if kind != "ticks":
        raise ValueError(f"unknown world run kind {kind!r} (known: ticks, evolution)")
    snaps = driver.run_ticks(int(spec.get("ticks", 50)))
    return {"seed": seed, "kind": "ticks", "snapshot": snaps[-1] if snaps else driver.snapshot()}


def run_world_runs(runs: list[dict[str, Any]], max_workers: int = 0) -> list[dict[str, Any]]:
    """Run independent world runs serially or across a worker-process pool.

    Args:
        runs: spec dicts for :func:`_run_one`; result order follows this
            input order (``pool.map`` preserves it).
        max_workers: pool size. ``1`` executes inline (the parity
            baseline); ``0`` (default) sizes the pool from the run count
            and CPU count.

    Returns:
        One result dict per run, in input order.

    Raises:
        Whatever a failing run raises — pool exceptions propagate (fail
        loud; a silently missing sweep row would fake a benchmark).
    """
    if not runs:
        return []
    if max_workers == 1 or len(runs) == 1:
        return [_run_one(spec) for spec in runs]
    workers = (
        min(len(runs), max_workers) if max_workers > 1 else min(len(runs), os.cpu_count() or 1)
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one, runs, chunksize=1))
