#!/usr/bin/env python3
"""Benchmark the TUI paint pass — `_render_all`, the path the overlay fix moved into.

Measures, in milliseconds per call over `iterations` reps:

* **paint pass, overlay helpers stubbed** — what `_render_all` cost before the
  fix, when it did not know overlays existed. This is the baseline.
* **paint pass, no overlay open** — the every-tick cost with the fix in place.
  This is the common case: the loop calls it on every 100 ms poll tick and on
  every consumed keypress, whether or not a prompt is up.
* **paint pass, ask overlay open** — the same call with a prompt on screen.
* `_draw_overlay` and `_sync_overlay_cursor` in isolation, both idle and active.

Why this exists: the fix moved overlay drawing from a second, separate
write-after-the-flush into the paint pass itself. Before it, a box written after
`engine.render()` landed in a layer buffer that the *next* pane repaint cleared
before anyone composited it — correct-looking code that never reached the
terminal. The fix's whole cost is the two helpers it added to the hot path, and
no existing harness measures `_render_all` at all: `benchmark_graphics.py` stops
at the engine primitives underneath it.

    PYTHONNOUSERSITE=1 PYTHONPATH="$PWD" python scripts/benchmark_tui_render.py

Engine escape output is routed to /dev/null for the duration of each timed run —
`_emit` writes to `sys.stdout`, and the alternate-screen sequences are not
something to aim at a real terminal from a benchmark.
"""

from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain.shell._internal.graphics import GraphicsEngine, Layer
from domain.shell._internal.log_buffer import LogBuffer, LogEntry
from domain.shell._internal.pane import Border, Pane, PaneLayout
from domain.shell._internal.tui_repl import TuiRepl


class _FakeRepl:
    """The minimum `_render_all` touches outside the render state.

    Mirrors `_MainFakeRepl` in the tests: a dispatch sink and a console with an
    `_io` attribute, so `TuiRepl.__init__` can bind and later restore I/O.
    """

    COMMANDS: list[str] = []

    def __init__(self) -> None:
        self.io = "old-io"
        self.console = type("C", (), {"_io": "old-console"})()
        self._history: list[str] = []
        self.dispatched: list[str] = []

    def _dispatch(self, cmd: str) -> None:
        self.dispatched.append(cmd)


class _Devnull:
    """Route `sys.stdout` to /dev/null so engine escapes go nowhere.

    Used only around timed regions. `_emit` writes to `sys.stdout` and `open()`
    emits an alternate-screen entry sequence, so a benchmark that left stdout
    alone would print control codes into the terminal it is being read from.
    """

    def __enter__(self) -> _Devnull:
        self._real = sys.stdout
        self._sink = open(os.devnull, "w")
        sys.stdout = self._sink
        return self

    def __exit__(self, *exc: object) -> bool:
        sys.stdout = self._real
        self._sink.close()
        return False


def bench(label: str, fn, iterations: int = 2000) -> dict:
    """Time *fn* for *iterations* reps, discarding a warm-up pass first."""
    return bench_arms({label: fn}, iterations)[label]


def bench_arms(arms: dict, iterations: int = 2000) -> dict:
    """Time several callables over shared reps, interleaved arm by arm.

    Consecutive runs measure drift as readily as they measure code: three
    end-to-end runs of *identical* code on this shared box gave +2.3%, +5.2%
    and +14.8%, because the machine is shared and the load average sits near 8.
    A verdict that flips between PASS and REGRESSION on unchanged code is
    measuring the weather. Interleaving within a rep — one call per arm, in
    rotation, for `iterations` rounds — means cache state, GC pressure and
    scheduler noise land on every arm about equally, so the *difference*
    between them survives.

    The same reasoning, and the same shape, as `.splitwork/bench_interactive.py`.
    """
    for fn in arms.values():
        fn()  # warm the code paths and the layer buffers before timing anything

    times: dict[str, list[float]] = {label: [] for label in arms}
    for _ in range(iterations):
        for label, fn in arms.items():
            t0 = time.perf_counter()
            fn()
            times[label].append((time.perf_counter() - t0) * 1000)

    out: dict[str, dict] = {}
    for label, samples in times.items():
        out[label] = {
            "label": label,
            "iterations": iterations,
            "mean_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[int(len(samples) * 0.95)],
            "min_ms": min(samples),
            "max_ms": max(samples),
        }
    return out


def print_result(r: dict) -> None:
    print(
        f"  {r['label']:44s}  "
        f"mean={r['mean_ms']:8.4f}ms  "
        f"median={r['median_ms']:8.4f}ms  "
        f"p95={r['p95_ms']:8.4f}ms  "
        f"min={r['min_ms']:8.4f}ms  "
        f"max={r['max_ms']:8.4f}ms  "
        f"({r['iterations']} iters)"
    )


def build_tui() -> tuple[TuiRepl, dict]:
    """A TuiRepl with the production 4-pane layout, a real engine and layers.

    This is the same wiring `_setup_layers` does in the tests, minus the mock:
    the engine is actually `open()`ed so `render()` composites real framebuffers
    instead of returning early on a missing back buffer. Without that the
    benchmark would time a no-op and say nothing about the paint pass.
    """
    # A real LogBuffer, not a stub: `_render_all` calls `_log_surface.render()`,
    # which reads `self._buffer.get()`. Stubbing it would skip the log pane's
    # scroll-and-format work — the bulk of what the paint pass actually does.
    tui = TuiRepl(_FakeRepl(), LogBuffer())
    tui._log_surface._buffer.append(
        LogEntry(time.time(), "INFO", "bench", "log line for the diff to move")
    )
    tui._input_buf = []
    tui._input_cursor = 0
    tui._rows = 24
    tui._out_scroll = 0
    tui._log_scroll = 0
    tui._scroll_target = 0
    tui._active_cmd = None
    tui._searching = False
    tui._search_fwd = False
    tui._search_q = ""
    tui._search_failed = False
    tui._out_searching = False
    tui._out_search_q = ""
    tui._out_search_failed = False

    layout = PaneLayout(
        [
            Pane("console", 0.6, border=Border("horizontal")),
            Pane("output", 0.4, border=Border("top")),
            Pane("status", fixed=1),
            Pane("input", fixed=1),
        ]
    )
    tui._layout = layout
    tui._output_surface.write("output line for the diff to have to move")
    regions = layout.compute(24, 80)

    engine = GraphicsEngine()
    engine._rows = 24
    engine._cols = 80
    tui._engine = engine
    # One layer per pane, z-ordered bottom to top, matching `_setup_layers`.
    spec = (
        ("console", "_layer_console", 1),
        ("output", "_layer_output", 3),
        ("status", "_layer_status", 4),
        ("input", "_layer_input", 5),
    )
    for key, attr, z in spec:
        r = regions.get(key)
        if r is None:
            continue
        setattr(tui, attr, Layer(key, r.rows, r.cols, z=z))
    tui._place_layers(regions)
    return tui, regions


def main() -> None:
    print("=" * 78)
    print("TUI paint pass benchmark — _render_all (the overlay fix's path)")
    print("=" * 78)

    tui, regions = build_tui()

    # `open()` allocates the front/back buffers render() requires, and emits an
    # alternate-screen entry sequence that must not reach a real terminal.
    with _Devnull():
        tui._engine.open()
        real_draw, real_sync = tui._draw_overlay, tui._sync_overlay_cursor
        # Built once: creating these inside the timed call would charge lambda
        # allocation to the baseline every rep and understate the delta.
        stub_draw = lambda regions: None
        stub_sync = lambda open_: None

        # ── Baseline vs. the fix, on one object, interleaved ──
        # The baseline stubs the two helpers the fix added, standing in for a
        # paint pass that did not know overlays existed. All three arms run
        # against the *same* TuiRepl and interleave within each rep, so the only
        # variable is the code under test — an across-version comparison could
        # not say that, because the layers and layout would differ too.
        def baseline_call():
            # Each arm establishes its full state: the arms interleave, so an
            # arm that relies on the previous one's state silently measures the
            # wrong thing. An earlier version never cleared `_ask_event`, and
            # because `open_call` set it, every arm from rep 2 onward ran with a
            # prompt up — idle and open came back nearly identical, which is
            # what a mislabelled arm looks like.
            tui._ask_event.clear()
            tui._draw_overlay = stub_draw
            tui._sync_overlay_cursor = stub_sync
            tui._render_all(regions)

        def idle_call():
            tui._ask_event.clear()
            tui._draw_overlay, tui._sync_overlay_cursor = real_draw, real_sync
            tui._render_all(regions)

        tui._ask_message = "name"
        tui._ask_default = "dflt"
        tui._ask_buf = list("one two")
        tui._ask_cursor = 4

        def open_call():
            tui._draw_overlay, tui._sync_overlay_cursor = real_draw, real_sync
            tui._ask_event.set()
            tui._render_all(regions)

        arms = bench_arms(
            {
                "paint pass, overlay helpers stubbed": baseline_call,
                "paint pass, no overlay open": idle_call,
                "paint pass, ask overlay open": open_call,
            }
        )
        baseline = arms["paint pass, overlay helpers stubbed"]
        idle = arms["paint pass, no overlay open"]
        with_open = arms["paint pass, ask overlay open"]

        # Leave the object in a sane state for the helper measurements below.
        tui._draw_overlay, tui._sync_overlay_cursor = real_draw, real_sync
        tui._ask_event.clear()
        helper_idle_draw = bench("_draw_overlay (nothing open)", lambda: tui._draw_overlay(regions))
        helper_idle_sync = bench(
            "_sync_overlay_cursor (idle)", lambda: tui._sync_overlay_cursor(False)
        )
        tui._ask_event.set()
        helper_open_draw = bench("_draw_overlay (ask open)", lambda: tui._draw_overlay(regions))

    print("\nPaint pass (24x80, 4 panes, real composite):")
    print_result(baseline)
    print_result(idle)
    print_result(with_open)

    print("\nHelpers the fix added to the hot path:")
    print_result(helper_idle_draw)
    print_result(helper_idle_sync)
    print_result(helper_open_draw)

    # The verdict the benchmark exists to produce. Both deltas are against the
    # stubbed baseline on the *same* object, so the only variable is the code
    # the fix added.
    d_idle = idle["mean_ms"] - baseline["mean_ms"]
    d_open = with_open["mean_ms"] - baseline["mean_ms"]
    print("\nVerdict (delta vs. helpers-stubbed baseline, same object):")
    print(
        f"  no overlay open : {d_idle:+.4f}ms per call  ({d_idle / baseline['mean_ms'] * 100:+.2f}%)"
    )
    print(
        f"  overlay open    : {d_open:+.4f}ms per call  ({d_open / baseline['mean_ms'] * 100:+.2f}%)"
    )
    print(
        "\nThe loop calls this on every 100 ms poll tick and every consumed key.\n"
        "Idle is the steady state; the open figure is what a prompt costs while up."
    )

    with _Devnull():
        tui._engine.close()

    print("\n" + "=" * 78)
    print("Done.")


if __name__ == "__main__":
    main()
