"""Benchmark: VectorBE pool cost across start methods (fork vs forkserver).

Card e69915c1 — `mp.Pool` moved off the bare default context (`fork`) because
fork-into-multithreaded (OpenBLAS threads alive in the parent) deadlocks
workers: the gate-wedge class. This measures what the forkserver switch costs
through the real `VectorBE.from_weights()` + `pool.map` dispatch path:

  - construct_ms          from_weights() -> pool live (worker processes ready
                          to accept, no task dispatched yet)
  - first_dispatch_ms     cold start: for forkserver this includes the fresh
                          worker interpreters importing the domain chain
                          (numpy + domain.__init__ + infrastructure.__init__);
                          for fork it is an instant fork of this process
  - steady_*_ms           warm dispatch latency per op (matmul / softmax /
                          rmsnorm), workers already imported

The fork arm runs under a SIGALRM watchdog: if a fork arm deadlocks (the
hazard this card exists to remove — parent is multithreaded once numpy has
spun BLAS threads), it is reported as DEADLOCKED instead of hanging the
benchmark.

Run from the repo root: ./scripts/python scripts/benchmark_vector_backend_pool.py
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time

import numpy as np

ENV_METHOD = "SLO_VECTOR_START_METHOD"


def _make_arch():
    from domain.infrastructure._internal.arch_config import ArchConfig

    arch = ArchConfig(
        name="bench_vector",
        norm="rms_norm",
        positional="rope",
        activation="swiglu",
        attention="gqa",
    )
    arch.n_layers = 1
    arch.n_heads = 4
    arch.n_kv_heads = 2
    arch.n_embed = 32
    arch.head_dim = 8
    arch.vocab_size = 64
    arch.rope_base = 10000.0
    arch.tied_weights = False
    arch.n_head = 4
    arch.n_kv_head = 2
    return arch


def _make_weights(arch) -> dict[str, np.ndarray]:
    E, H, KV, D, V = arch.n_embed, arch.n_head, arch.n_kv_head, arch.head_dim, arch.vocab_size
    rng = np.random.default_rng(0)
    weights: dict[str, np.ndarray] = {
        "embed_tokens.weight": rng.standard_normal((V, E)).astype(np.float32) * 0.02,
        "model.layers.0.self_attn.q_proj.weight": rng.standard_normal((H * D, E)).astype(np.float32)
        * 0.02,
        "model.layers.0.self_attn.k_proj.weight": rng.standard_normal((KV * D, E)).astype(
            np.float32
        )
        * 0.02,
        "model.layers.0.self_attn.v_proj.weight": rng.standard_normal((KV * D, E)).astype(
            np.float32
        )
        * 0.02,
        "model.layers.0.self_attn.o_proj.weight": rng.standard_normal((E, H * D)).astype(np.float32)
        * 0.02,
        "model.layers.0.mlp.gate_proj.weight": rng.standard_normal((E, 4 * E)).astype(np.float32)
        * 0.02,
        "model.layers.0.mlp.down_proj.weight": rng.standard_normal((4 * E, E)).astype(np.float32)
        * 0.02,
        "model.layers.0.input_layernorm.weight": np.ones(E, dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.ones(E, dtype=np.float32),
        "model.norm.weight": np.ones(E, dtype=np.float32),
        "lm_head.weight": rng.standard_normal((V, E)).astype(np.float32) * 0.02,
    }
    return weights


class _WatchdogTimeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise _WatchdogTimeout()


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def bench_method(method: str, iters: int, watchdog_s: float) -> dict:
    from domain.infrastructure._internal.vector_backend import VectorBE

    prev = os.environ.get(ENV_METHOD)
    os.environ[ENV_METHOD] = method
    signal.signal(signal.SIGALRM, _alarm)
    result: dict = {"start_method": method}
    try:
        arch = _make_arch()
        weights = _make_weights(arch)

        t0 = time.perf_counter()
        be = VectorBE.from_weights(weights, arch)
        result["construct_ms"] = round(_ms(t0), 3)

        a = np.random.randn(256, 128).astype(np.float32)
        b = np.random.randn(128, 128).astype(np.float32)
        x = np.random.randn(64, 128).astype(np.float32)
        w = np.ones(128, dtype=np.float32)

        signal.setitimer(signal.ITIMER_REAL, watchdog_s)
        try:
            t0 = time.perf_counter()
            be.matmul(a, b)
            result["first_dispatch_ms"] = round(_ms(t0), 3)
        except _WatchdogTimeout:
            result["first_dispatch_ms"] = f"DEADLOCKED (>{watchdog_s:.0f}s)"
            result["note"] = "fork-into-multithreaded worker deadlock — the hazard class"
            # reap the stuck fork children so the benchmark process doesn't leak them
            try:
                for proc in getattr(be._pool, "_pool", []):
                    proc.terminate()
            except Exception:
                pass
            return result
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

        # warm steady-state dispatches (workers already imported)
        for name, fn in (
            ("steady_matmul_ms", lambda: be.matmul(a, b)),
            ("steady_softmax_ms", lambda: be.softmax(x)),
            ("steady_rmsnorm_ms", lambda: be.rmsnorm(x, w)),
        ):
            fn()  # warm
            samples = []
            for _ in range(iters):
                t0 = time.perf_counter()
                fn()
                samples.append(_ms(t0))
            samples.sort()
            result[name] = round(samples[len(samples) // 2], 3)
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        try:
            be.__del__()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        if prev is None:
            os.environ.pop(ENV_METHOD, None)
        else:
            os.environ[ENV_METHOD] = prev


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=10, help="steady-state samples per op (median)")
    ap.add_argument("--watchdog", type=float, default=30.0, help="per-arm deadlock watchdog (s)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument(
        "--methods", default="fork,forkserver", help="comma-separated start methods to compare"
    )
    args = ap.parse_args()

    results = []
    for method in args.methods.split(","):
        method = method.strip()
        if not method:
            continue
        print(f"--- {method} ---", flush=True)
        results.append(bench_method(method, args.iters, args.watchdog))

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    cols = [
        "start_method",
        "construct_ms",
        "first_dispatch_ms",
        "steady_matmul_ms",
        "steady_softmax_ms",
        "steady_rmsnorm_ms",
    ]
    widths = {c: max(len(c), *(len(str(r.get(c, "-"))) for r in results)) for c in cols}
    print()
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in results:
        print("  ".join(str(r.get(c, "-")).ljust(widths[c]) for c in cols))
    for r in results:
        if "note" in r:
            print(f"\n{r['start_method']}: {r['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
