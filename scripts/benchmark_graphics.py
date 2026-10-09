"""Benchmark the terminal graphics engine.

Measures: frame render time, diff render, layer compositing, resize,
snapshot/restore, and hardware state comparison.
"""

from __future__ import annotations

import statistics
import time

from domain.shell._internal.graphics import (
    BoxStyle,
    Cell,
    Color,
    Framebuffer,
    GraphicsEngine,
    HardwareState,
    Layer,
    Pattern,
    SnapshotManager,
    compare_hardware_outputs,
    compare_hardware_states,
)


def bench(label: str, fn, iterations: int = 1000) -> dict:
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return {
        "label": label,
        "iterations": iterations,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": sorted(times)[int(len(times) * 0.95)],
        "min_ms": min(times),
        "max_ms": max(times),
    }


def print_result(r: dict) -> None:
    print(
        f"  {r['label']:40s}  "
        f"mean={r['mean_ms']:8.3f}ms  "
        f"median={r['median_ms']:8.3f}ms  "
        f"p95={r['p95_ms']:8.3f}ms  "
        f"min={r['min_ms']:8.3f}ms  "
        f"max={r['max_ms']:8.3f}ms  "
        f"({r['iterations']} iters)"
    )


def main() -> None:
    print("=" * 72)
    print("Graphics Engine Benchmark")
    print("=" * 72)

    # ── Framebuffer operations ─────────────────────────────────────────
    print("\nFramebuffer (80x24):")
    fb = Framebuffer(24, 80)

    print_result(bench("put cell", lambda: fb.put(12, 40, "X", fg=Color.GREEN, bg=Color.RED)))
    print_result(bench("get cell", lambda: fb.get(12, 40)))
    print_result(
        bench("write_str (short)", lambda: fb.write_str(5, 0, "Hello World", fg=Color.CYAN))
    )
    print_result(
        bench("write_str (full line)", lambda: fb.write_str(0, 0, "A" * 80, fg=Color.WHITE))
    )
    print_result(bench("fill", lambda: fb.fill(".", fg=Color.GREEN)))
    print_result(bench("fill_pattern (DOTS)", lambda: fb.fill_pattern(Pattern.DOTS, fg=Color.CYAN)))
    print_result(bench("draw_rect", lambda: fb.draw_rect(2, 2, 20, 10, char="*")))
    print_result(
        bench("draw_box (SINGLE)", lambda: fb.draw_box(0, 0, 80, 24, style=BoxStyle.SINGLE))
    )
    print_result(bench("clear", lambda: fb.clear()))
    print_result(bench("snapshot", lambda: fb.snapshot()))
    print_result(bench("diff (identical)", lambda: fb.diff(fb)))

    # ── Layer operations ───────────────────────────────────────────────
    print("\nLayer (80x24):")
    layer = Layer("test", 24, 80, z=0)

    print_result(bench("layer.write", lambda: layer.write(5, 10, "Hello World", fg=Color.GREEN)))
    print_result(bench("layer.fill_pattern", lambda: layer.fill_pattern(Pattern.CROSS)))
    print_result(bench("layer.clear", lambda: layer.clear()))
    print_result(bench("layer.resize (same)", lambda: layer.resize(24, 80)))

    # ── Engine operations ──────────────────────────────────────────────
    print("\nGraphicsEngine:")
    engine = GraphicsEngine()
    engine._rows = 24
    engine._cols = 80
    engine._front_buffer = Framebuffer(24, 80)
    engine._back_buffer = Framebuffer(24, 80)

    print_result(bench("engine.create_layer", lambda: engine.create_layer("bench", z=0)))
    # Remove created layers to keep list small
    engine._layers.clear()

    print_result(bench("engine.render (cold)", lambda: _bench_render_cold(engine)))
    print_result(bench("engine.render (warm)", lambda: _bench_render_warm(engine)))
    print_result(bench("engine.write + render", lambda: _bench_write_render(engine)))
    print_result(bench("engine.box + render", lambda: _bench_box_render(engine)))

    # ── Snapshot manager ───────────────────────────────────────────────
    print("\nSnapshotManager:")
    mgr = SnapshotManager(max_history=50)
    fb_for_snap = Framebuffer(24, 80)
    fb_for_snap.write_str(0, 0, "test")

    print_result(bench("save", lambda: mgr.save(fb_for_snap)))
    # Pre-fill for undo/redo benchmarks
    mgr2 = SnapshotManager(max_history=50)
    for i in range(20):
        fb_t = Framebuffer(24, 80)
        fb_t.write_str(0, 0, str(i))
        mgr2.save(fb_t)

    print_result(bench("undo", lambda: mgr2.undo()))
    mgr2.push_redo(mgr2.undo())
    print_result(bench("redo", lambda: mgr2.redo()))

    # ── Hardware state comparison ───────────────────────────────────────
    print("\nHardware state comparison (80x24):")
    cells_a = [[Cell("A", fg=Color.GREEN) for _ in range(80)] for _ in range(24)]
    cells_b = [
        [
            Cell("B", fg=Color.RED) if r == 12 and c == 40 else Cell("A", fg=Color.GREEN)
            for c in range(80)
        ]
        for r in range(24)
    ]
    hs_a = HardwareState(rows=24, cols=80, cells=cells_a)
    hs_b = HardwareState(rows=24, cols=80, cells=cells_b)
    hs_same = HardwareState(rows=24, cols=80, cells=cells_a)

    print_result(
        bench("compare_hardware_states (identical)", lambda: compare_hardware_states(hs_a, hs_same))
    )
    print_result(
        bench("compare_hardware_states (1 diff)", lambda: compare_hardware_states(hs_a, hs_b))
    )

    # ── Hardware output comparison ──────────────────────────────────────
    print("\nHardware output comparison:")
    out_a = b"\x1b[2J\x1b[H" + b"Hello World" * 100
    out_b = b"\x1b[2J\x1b[H" + b"Hello World" * 99 + b"Hello Worlx"
    print_result(
        bench(
            "compare_hardware_outputs (identical)",
            lambda: compare_hardware_outputs(
                type("O", (), {"raw_bytes": out_a})(), type("O", (), {"raw_bytes": out_a})()
            ),
        )
    )
    print_result(
        bench(
            "compare_hardware_outputs (1 byte diff)",
            lambda: compare_hardware_outputs(
                type("O", (), {"raw_bytes": out_a})(), type("O", (), {"raw_bytes": out_b})()
            ),
        )
    )

    print("\n" + "=" * 72)
    print("Done.")


def _bench_render_cold(engine: GraphicsEngine) -> None:
    engine._back_buffer.clear()
    engine._front_buffer.clear()
    engine.render()


def _bench_render_warm(engine: GraphicsEngine) -> None:
    engine._front_buffer.restore(engine._back_buffer)
    engine.render()


def _bench_write_render(engine: GraphicsEngine) -> None:
    engine._front_buffer.write_str(0, 0, "bench", fg=Color.GREEN)
    engine.render()


def _bench_box_render(engine: GraphicsEngine) -> None:
    engine._front_buffer.draw_box(0, 0, 80, 24, style=BoxStyle.SINGLE)
    engine.render()


if __name__ == "__main__":
    main()
