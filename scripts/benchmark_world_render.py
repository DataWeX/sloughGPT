#!/usr/bin/env python3
"""World render benchmark — Gateway image latency after the batch tracer.

Measures the two render paths the Gateway serves:
  1. CyclesDevice at Gateway defaults (160x120, 16spp) — sphere+light scene
  2. RenderBridge.render_tick over a live world scene (the /world/render/
     world frame path: voxel mesh rebuild + BVH build + trace per frame)

Gates (both must pass):
  - device frame < 120s at Gateway defaults
  - world frame < 240s (rebuilds the voxel mesh every frame, by design)

Baseline (pre-vectorization, measured 2026-10-16): one 128x96 @ 2spp frame
took 140-320s through the per-pixel Python loops — extrapolated to the
Gateway default of 160x120 @ 16spp that is ~2900s/frame. The batch tracer
(frontier BVH walk + vectorized emission/BSDF) replaces every per-ray
Python loop; real-time streaming is the follow-up GPU card.

Usage:
    python scripts/benchmark_world_render.py [--frames 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core-py"))

import numpy as np  # noqa: E402

from domain.shell._internal.cycles_device import CyclesDevice  # noqa: E402
from domain.shell._internal.realm_view import make_live_scene  # noqa: E402
from domain.shell._internal.world_render import RenderBridge, RenderConfig  # noqa: E402

GATE_DEVICE_S = 120.0
GATE_WORLD_S = 240.0


def bench_device(frames: int) -> float:
    """Mean seconds per Gateway-default device frame (sphere + light)."""
    dev = CyclesDevice(width=160, height=120, samples=16)
    dev.call("add_sphere", 1.0, 0.0, 0.5, 0.0, 0, 24)
    dev.call("add_light", 2.0, 5.0, 2.0, 1.0, 0.95, 0.9, 6.0)
    dev.call("render")  # warmup + BVH build
    times = []
    for _ in range(frames):
        t0 = time.perf_counter()
        slot = dev.call("render")
        times.append(time.perf_counter() - t0)
        img = dev.arena.get(slot)
        assert img.shape == (120, 160, 3) and np.isfinite(img).all()
        dev.call("free_slot", slot)
    return float(np.mean(times))


def bench_world(frames: int) -> float:
    """Mean seconds per live-world frame through RenderBridge.render_tick."""
    scene = make_live_scene(
        grid=(24, 12, 24), population=8, organic_pools=3, day_ticks=24, seed=7
    )
    bridge = RenderBridge(RenderConfig(width=160, height=120, samples=16))
    times = []
    for _ in range(frames):
        t0 = time.perf_counter()
        bridge.render_tick(scene.world, babies=scene.babies, nests=scene.nests)
        times.append(time.perf_counter() - t0)
        img = bridge.last_image
        assert img is not None and np.isfinite(img).all()
    return float(np.mean(times))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=3, help="timed frames per path")
    args = parser.parse_args()

    dev_s = bench_device(args.frames)
    print(f"device frame (160x120 @16spp): {dev_s:8.2f}s   gate < {GATE_DEVICE_S:.0f}s")
    world_s = bench_world(args.frames)
    print(f"world frame (RenderBridge):    {world_s:8.2f}s   gate < {GATE_WORLD_S:.0f}s")

    baseline_equiv_s = 2875.0  # scalar tracer extrapolated to this config
    print(f"speedup vs scalar baseline:    {baseline_equiv_s / dev_s:8.1f}x "
          f"(baseline {baseline_equiv_s:.0f}s/frame, extrapolated)")

    ok = dev_s < GATE_DEVICE_S and world_s < GATE_WORLD_S
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
