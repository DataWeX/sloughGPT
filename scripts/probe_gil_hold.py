#!/usr/bin/env python
"""GIL-hold probe for numba kernels (kanban card f30be275).

Answers one question empirically: when a Python thread calls an
``@njit(nogil=True)`` function, is the GIL actually released for the
duration of the call?  Without ``nogil`` the call holds the GIL the whole
time (a control thread starves — this was the demonstrated outage
mechanism); with ``nogil`` a control thread must keep its heartbeat.

Usage:
    python scripts/probe_gil_hold.py           # standalone twin kernels
    python scripts/probe_gil_hold.py --real     # probe fused_attention_multi
                                                 # from slonet_kernels (needs
                                                 # PYTHONPATH=<repo root>)

Exit code 0 = GIL released (control-thread stall far below call duration),
exit code 1 = GIL held (or probe could not run).

Requires numba: the repo-standard ``PYTHONNOUSERSITE=1`` hides the
user-site numba install, so run with user-site enabled (or put numba's
directory on PYTHONPATH).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

# Call stalls above this fraction of the call duration count as "held".
# Heartbeat jitter is ~5-80 ms against calls of ~2000 ms, so 0.2 is
# generous while still far from a full stall.
RELEASED_THRESHOLD = 0.2
# For the idle baseline (no call duration to divide by) use an absolute cap.
BASELINE_MAX_GAP_S = 0.05


def _heartbeat(stop: threading.Event, gaps: list[float]) -> None:
    """Control thread: record the gap between fixed 5 ms wakeups."""
    prev = time.perf_counter()
    while not stop.is_set():
        time.sleep(0.005)
        now = time.perf_counter()
        gaps.append(now - prev)
        prev = now


def measure(fn, args: tuple, label: str) -> dict:
    """Run ``fn(*args)`` in a worker thread; heartbeat from the main thread."""
    stop = threading.Event()
    gaps: list[float] = []
    th = threading.Thread(target=_heartbeat, args=(stop, gaps))
    th.start()
    time.sleep(0.05)  # let the heartbeat settle

    t0 = time.perf_counter()
    worker = threading.Thread(target=fn, args=args)
    worker.start()
    worker.join()
    dur = time.perf_counter() - t0

    stop.set()
    th.join()
    ratio = max(gaps) / dur if dur > 1e-3 else float("inf")
    print(
        f"  {label:<30} dur={dur * 1000:8.1f}ms  "
        f"max_gap={max(gaps) * 1000:8.1f}ms  ({ratio * 100:6.1f}% of dur)"
    )
    return {"dur": dur, "max_gap": max(gaps), "ratio": ratio}


def probe_standalone() -> bool:
    """Twin kernels (same body, nogil on/off) called directly from Python."""
    from numba import njit

    def make(nogil: bool):
        @njit(cache=False, nogil=nogil)
        def k(a, iters):
            acc = np.float32(0)
            for _ in range(iters):
                for i in range(a.shape[0]):
                    acc = acc + a[i] * np.float32(1.000001)
            return acc

        return k

    a = np.ones(1_000_000, dtype=np.float32)
    k_plain, k_nogil = make(False), make(True)
    k_plain(a, 1)  # compile outside measurement
    k_nogil(a, 1)

    # calibrate to a ~2 s call so a full stall is unmistakable
    iters = 5
    while True:
        t0 = time.perf_counter()
        k_plain(a, iters)
        d = time.perf_counter() - t0
        if d >= 1.5 or iters > 10_000:
            break
        iters = max(iters + 1, int(iters * 2.0 / max(d, 1e-3)))

    print(f"standalone twins (iters={iters}):")
    base = measure(lambda: None, (), "baseline (idle)")
    held = measure(k_plain, (a, iters), "@njit (no nogil)")
    free = measure(k_nogil, (a, iters), "@njit(nogil=True)")

    ok = (
        base["max_gap"] < BASELINE_MAX_GAP_S
        and held["ratio"] > 0.8  # control reproduces the outage mechanism
        and free["ratio"] < RELEASED_THRESHOLD
    )
    print(
        f"  verdict: nogil "
        f"{'RELEASES' if free['ratio'] < RELEASED_THRESHOLD else 'HOLDS'} the GIL "
        f"(control held={held['ratio'] * 100:.0f}%, "
        f"nogil={free['ratio'] * 100:.1f}%) -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def probe_real() -> bool:
    """Probe the repo's real hot kernel: fused_attention_multi."""
    from domain.training._internal import slonet_kernels as sk

    sk._ensure_fused()
    if sk._nb_fused_attention_multi is None:
        print("compiled kernel unavailable (numba not visible?) -> FAIL")
        return False

    # wrapper signature: q (seq, H, E), k/v (H, new, E), scale, H, E_head
    # (128x128: ~2-3 s per call at current kernels — big enough for a
    # decisive stall signal, small enough that the probe stays quick)
    H, E, seq, new = 8, 64, 128, 128
    rng = np.random.default_rng(0)
    q = rng.standard_normal((seq, H, E)).astype(np.float32)
    k = rng.standard_normal((H, new, E)).astype(np.float32)
    v = rng.standard_normal((H, new, E)).astype(np.float32)
    scale = np.float32(1.0 / np.sqrt(E))

    sk.fused_attention_multi(q, k, v, scale, H, E)  # warm/compile outside

    # calibrate to a ~1.5 s batch of repeated calls (production shape)
    iters = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(iters):
            sk.fused_attention_multi(q, k, v, scale, H, E)
        d = time.perf_counter() - t0
        if d >= 1.5 or iters > 10_000:
            break
        iters = max(iters + 1, int(iters * 1.5 / max(d, 1e-3)))

    print(f"real kernel fused_attention_multi ({iters} calls, production decorator):")
    res = measure(
        lambda: [sk.fused_attention_multi(q, k, v, scale, H, E) for _ in range(iters)],
        (),
        "fused_attention_multi",
    )
    ok = res["ratio"] < RELEASED_THRESHOLD
    print(f"  verdict: {'PASS' if ok else 'FAIL'} (GIL {'released' if ok else 'held'})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--real",
        action="store_true",
        help="probe fused_attention_multi from slonet_kernels instead of twins",
    )
    args = ap.parse_args()
    try:
        import numba  # noqa: F401
    except ImportError:
        print("numba not visible (repo protocol hides user-site) -> cannot probe")
        return 1
    ok = probe_real() if args.real else probe_standalone()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
