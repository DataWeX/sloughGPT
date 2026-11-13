#!/usr/bin/env python
"""Benchmark: NPU device dispatch overhead (ioctl/call round-trip latency).

Measures the DEVICE LAYER only — the cost of routing a command through
``NPUDevice.ioctl`` / ``NPUDevice.call`` / the ``NPUVMDevice`` VM face — not
model math. The model slot holds a stub provider whose methods return
instantly, so every nanosecond reported is our dispatch: normalization,
validation, SyscallResult boxing, DeviceFault translation.

Why this exists (card 20261104_090736): the two-NPU unification merges two
ioctl vocabularies behind one dispatch. This gates that the merge does not
make the device seam measurably slower (dispatch is on the DEV_CALL path of
every VM program).

Usage:
    ./scripts/python scripts/benchmark_npu_device.py [--iterations N] [--json PATH]

Reports p50/p95/mean per operation. Compare runs:
    ./scripts/python scripts/benchmark_npu_device.py --json /tmp/npu_after.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from domain.shell._internal.kernel_npu import NPUDevice  # noqa: E402
from domain.shell._internal.vm_devices import NPUVMDevice  # noqa: E402


class _StubProvider:
    """Zero-cost provider: dispatch is what we measure, not the model."""

    def tokenize(self, text: str):
        return list(range(min(len(text), 8)))

    def detokenize(self, ids):
        return "x"

    def generate_numpy(self, ids, max_new_tokens=4, **kw):
        return np.asarray(ids)

    def forward_numpy(self, ids):
        return np.zeros((1, 2, 4))


def _time_op(fn, iterations: int) -> dict:
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)  # µs
    samples.sort()
    return {
        "p50_us": round(statistics.median(samples), 3),
        "p95_us": round(samples[int(len(samples) * 0.95) - 1], 3),
        "mean_us": round(statistics.fmean(samples), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    npu = NPUDevice(name="bench_npu")
    npu._models["stub"] = _StubProvider()  # noqa: SLF001 - benchmark rig
    npu._default_model = "stub"  # noqa: SLF001
    vm_face = NPUVMDevice(npu)

    n = args.iterations
    results = {
        "iterations": n,
        "ioctl_INFO": _time_op(lambda: npu.ioctl("INFO"), n),
        "ioctl_CALL_text": _time_op(lambda: npu.ioctl("CALL", "stub", "hi"), n),
        "call_info_guest_face": _time_op(lambda: npu.call("info"), n),
        "vmface_generate": _time_op(lambda: vm_face.call("generate", "stub", "hi", 4), n),
        "ioctl_LIST_COMMANDS": _time_op(lambda: npu.ioctl("LIST_COMMANDS"), max(n // 10, 100)),
    }

    width = max(len(k) for k in results if k != "iterations")
    print(f"NPU device dispatch benchmark ({n} iterations)")
    for key, stats in results.items():
        if key == "iterations":
            continue
        print(
            f"  {key:<{width}}  p50={stats['p50_us']:>9.3f}µs  "
            f"p95={stats['p95_us']:>9.3f}µs  mean={stats['mean_us']:>9.3f}µs"
        )

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
