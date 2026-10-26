#!/usr/bin/env python
"""Benchmark the checkpoint save path — the stall a save costs the training loop.

Three questions, all in milliseconds a trainer can budget against:

1. **What does a save cost end-to-end?** ``save_soul`` for several model sizes,
   reporting wall time, bytes written and throughput. This is the stall: the
   loop is blocked for exactly this long.

2. **What did adding durability cost?** The same payload written through the
   atomic tier and the durable tier, so the artifact's flush is a measured
   share of the save rather than a guess. For a multi-MB artifact the flush
   should be a rounding error against the write — if it is not, batching the
   saves is the lever (see atomic_write's module docstring).

3. **Does the sidecar stay cheap?** The ``.meta.json`` companion is written
   second and non-durably by design; this reports its share so that ordering
   never silently becomes the expensive half.

Run:  ./scripts/python scripts/benchmark_checkpoint_save.py [--results out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from domain.inference._internal.slo_format import save_soul
from domain.infrastructure._internal.atomic_write import atomic_write

# (name, total params) — tiny matches the gate config used in CI.
SIZES = [
    ("tiny", 55_000),
    ("1M", 1_000_000),
    ("8M", 8_000_000),
]
REPS = 5


class _Model:
    """Model-shaped object: a state_dict of float32 arrays, nothing else."""

    def __init__(self, total_params: int) -> None:
        # Three arrays so the per-key conversion loop is exercised, not just a
        # single bulk write.
        thirds = [total_params // 3] * 2
        thirds.append(total_params - sum(thirds))
        self._sd = {f"w{i}": np.random.rand(n).astype(np.float32) for i, n in enumerate(thirds)}

    def state_dict(self) -> dict[str, np.ndarray]:
        return dict(self._sd)


def _timeit(fn, reps: int = REPS) -> tuple[float, float]:
    """(median, min) wall seconds over ``reps`` runs; the first warms caches.

    Min matters on a shared machine: system load can only add latency, never
    subtract it, so the fastest observed run is the closest estimate of the
    true cost. When min and median diverge wildly the environment — not the
    code — is what is being measured, and the caller should distrust both.
    """
    fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), min(samples)


def _bench_sizes(tmp: Path) -> list[dict]:
    rows = []
    for name, n_params in SIZES:
        model = _Model(n_params)
        target = tmp / f"bench_{name}.soul"

        def save(target=target, model=model):
            save_soul(model, str(target))

        secs, secs_min = _timeit(save)
        soul_bytes = target.stat().st_size
        meta_bytes = Path(str(target) + ".meta.json").stat().st_size
        total = soul_bytes + meta_bytes
        rows.append(
            {
                "size": name,
                "params": n_params,
                "ms_per_save": round(secs * 1000, 3),
                "ms_per_save_min": round(secs_min * 1000, 3),
                "soul_kb": round(soul_bytes / 1024, 1),
                "meta_kb": round(meta_bytes / 1024, 2),
                "mb_per_s": round(total / 1024 / 1024 / secs_min, 1) if secs_min else 0.0,
            }
        )
    return rows


def _bench_tiers(tmp: Path) -> list[dict]:
    """Atomic vs durable on identical payloads — the flush's real share.

    Interleaved, with the measurement order alternating per repetition. Both
    tiers must see the same system load: this machine's fsync latency swings
    2-3x, and measuring one tier to completion before starting the other
    produced a run where the *durable* path came out faster than the atomic
    one — pure load bias, not an effect.
    """
    rows = []
    for name, n_params in SIZES:
        payload = np.random.rand(n_params).astype(np.float32).tobytes()
        target = tmp / f"tier_{name}.bin"
        atomic_s, durable_s = [], []

        def timed(fn) -> float:
            t0 = time.perf_counter()
            fn()
            return time.perf_counter() - t0

        atomic_write(target, payload)  # warm both paths
        atomic_write(target, payload, durable=True)
        for rep in range(REPS):
            pair = (
                [(atomic_write, False), (atomic_write, True)]
                if rep % 2 == 0
                else [(atomic_write, True), (atomic_write, False)]
            )
            for fn, durable in pair:
                secs = timed(lambda: fn(target, payload, durable=durable))
                (durable_s if durable else atomic_s).append(secs)

        atomic = statistics.median(atomic_s)
        durable = statistics.median(durable_s)
        rows.append(
            {
                "size": name,
                "payload_kb": round(len(payload) / 1024, 1),
                "atomic_ms": round(atomic * 1000, 3),
                "atomic_ms_min": round(min(atomic_s) * 1000, 3),
                "durable_ms": round(durable * 1000, 3),
                "durable_ms_min": round(min(durable_s) * 1000, 3),
                "flush_share_pct": round(100 * (durable - atomic) / durable, 1) if durable else 0.0,
            }
        )
    return rows


def _warn_if_noisy(pairs: list[tuple[float, float]], threshold: float = 3.0) -> None:
    """Flag rows where median/min exceeds ``threshold`` — an unreliable run.

    A quiet machine keeps the ratio near 1. A large ratio means load spikes
    dominated the measurement, and the medians describe the machine rather
    than the code. Trust the min columns and re-run when quiet.
    """
    worst = max((m / f for m, f in pairs if f > 0), default=1.0)
    if worst > threshold:
        print(
            f"\n  [unstable] median/min reached {worst:.1f}x — this host is under load, "
            "so medians describe the machine, not the code. Use the min columns, "
            "and re-run on a quiet host for publishable numbers."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", help="JSON file to merge results into")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="ckptbench_") as td:
        tmp = Path(td)
        sizes = _bench_sizes(tmp)
        tiers = _bench_tiers(tmp)

    print("\nCheckpoint save end-to-end (save_soul, durable artifact + cheap sidecar)")
    print("-" * 96)
    print(
        f"{'size':6s} {'params':>10s} {'ms median':>10s} {'ms min':>9s} "
        f"{'soul KB':>9s} {'meta KB':>9s} {'MB/s':>8s}"
    )
    for r in sizes:
        print(
            f"{r['size']:6s} {r['params']:>10,} {r['ms_per_save']:>10.3f} "
            f"{r['ms_per_save_min']:>9.3f} {r['soul_kb']:>9.1f} "
            f"{r['meta_kb']:>9.2f} {r['mb_per_s']:>8.1f}"
        )
    _warn_if_noisy([(r["ms_per_save"], r["ms_per_save_min"]) for r in sizes])

    print("\nTier split on identical payloads (what the artifact's flush adds)")
    print("-" * 96)
    print(
        f"{'size':6s} {'payload KB':>11s} {'atomic med':>11s} {'atomic min':>11s} "
        f"{'durable med':>12s} {'durable min':>12s} {'flush share':>12s}"
    )
    for r in tiers:
        print(
            f"{r['size']:6s} {r['payload_kb']:>11.1f} {r['atomic_ms']:>11.3f} "
            f"{r['atomic_ms_min']:>11.3f} {r['durable_ms']:>12.3f} "
            f"{r['durable_ms_min']:>12.3f} {r['flush_share_pct']:>11.1f}%"
        )
    _warn_if_noisy(
        [(r["atomic_ms"], r["atomic_ms_min"]) for r in tiers]
        + [(r["durable_ms"], r["durable_ms_min"]) for r in tiers]
    )

    if args.results:
        path = Path(args.results)
        prior = json.loads(path.read_text()) if path.exists() else {}
        prior["checkpoint_save"] = {"sizes": sizes, "tiers": tiers}
        path.write_text(json.dumps(prior, indent=2))
        print(f"\nResults merged into {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
