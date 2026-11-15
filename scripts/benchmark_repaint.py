"""What the repaint fix costs: the rewind every keystroke now pays.

Eleven prompts in `domain/shell/_internal/interactive_prompt.py` used to paint
like

    sys.stdout.write(f"\\r{_ERASE_LINE}\\r{_render()}")

which erases the one row under the cursor and writes the whole frame from
there, stacking a copy below the last on every keystroke, and on exit runs
`up` before `erase` so the erasure lands on the row *above* the frame. The fix
introduces `_rewind(rows)`, called once before every paint and once on exit.

That is the only new cost in the path, and this measures it: microseconds and
bytes, across the frame heights prompts actually paint. The shape to watch is
that it stays linear in rows and stays far below the paint it accompanies —
the paint renders the frame, joins it, and writes it; the rewind writes a few
tens of bytes of escape.

Screen correctness — that a prompt leaves no copy of itself behind — is not
this benchmark's job. It is asserted exactly by
`tests/test_interactive.py::TestRepaintLeavesNothingBehind`, which drives real
prompts through a terminal grid and requires the screen to come back empty.

    PYTHONPATH="$PWD/packages/core-py:$PWD" python scripts/benchmark_repaint.py
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "core-py"))
sys.path.insert(0, str(ROOT))

from domain.shell._internal.interactive_prompt import _rewind

# Frame heights seen while measuring the defect: one-row inputs, two-row
# confirmations, and the stacked filter lists that climbed to 30 rows.
HEIGHTS = (1, 2, 4, 8, 16, 24, 32, 40)
ITERS = 4000


class Sink(io.StringIO):
    """Captures the escape traffic so it can be counted rather than emitted.

    `_rewind` writes to `sys.stdout` by name, so the sink only sees anything
    once `redirect_stdout` has pointed that name at it — assigning a sink to a
    local variable would measure zero bytes while the terminal filled with
    cursor-up sequences.
    """


def time_rewind(rows: int, iters: int = ITERS) -> tuple[float, int]:
    """`(microseconds per call, bytes written per call)` for `rows` high."""
    sink = Sink()
    with contextlib.redirect_stdout(sink):
        _rewind(rows)  # warm
        sink.seek(0)
        sink.truncate(0)

        t0 = time.perf_counter()
        for _ in range(iters):
            _rewind(rows)
        elapsed = time.perf_counter() - t0
        per_call = sink.tell() / iters
    return (elapsed / iters) * 1e6, round(per_call, 1)


def paint_bytes(rows: int) -> int:
    """Bytes a frame of this height would write, for a like-for-like compare."""
    line = f"\x1b[2K\r  {'x' * 72}"
    return len(line) * rows


def main() -> int:
    print("=" * 74)
    print("Repaint fix — cost of `_rewind(rows)` per keystroke")
    print("=" * 74)
    print(f"\n  {ITERS} iterations per height\n")
    print(f"  {'rows':>5} {'rewind us':>10} {'bytes':>7} {'paint bytes':>12} {'share':>8}")
    print("  " + "-" * 54)

    measured: dict[int, tuple[float, int]] = {}
    worst_bytes = 0.0
    for rows in HEIGHTS:
        us, nbytes = time_rewind(rows)
        measured[rows] = (us, nbytes)
        pb = paint_bytes(rows)
        share = (nbytes / pb) * 100 if pb else 0.0
        worst_bytes = max(worst_bytes, share)
        print(f"  {rows:>5} {us:>10.3f} {nbytes:>7.1f} {pb:>12} {share:>7.2f}%")

    # Reuse the table rather than sampling again: a second pass over 1 row
    # lands somewhere else, and a summary that disagrees with the numbers
    # printed above it is worse than no summary.
    us1, _b1 = measured[1]
    us40, b40 = measured[40]
    repaint_ms = 12.5  # one keystroke's full repaint, per benchmark_filter_box

    print("\n" + "=" * 74)
    print("  shape")
    print(
        f"    cost tracks height:  {us40 / max(us1, 1e-9):>6.1f}x for 40x the rows"
        f"   (1 row {us1:.3f}us -> 40 rows {us40:.3f}us)"
    )
    print(
        f"    against a repaint:   {us40 / (repaint_ms * 1000) * 100:>6.3f}% of one"
        f" 12.5ms keystroke repaint, at the tallest frame measured"
    )
    print(
        f"    bytes:               {b40:>5.0f} at the tallest frame"
        f"   ({worst_bytes:.1f}% of what the frame itself writes)"
    )
    print("=" * 74)
    print(
        """
  reading:
    rewind us   time to clear the previous frame and return to its first row
    bytes       escape sequence written doing so — the only new traffic
    paint bytes  what a frame of the same height writes for comparison
    share       rewind bytes as a percentage of that frame's own output

  A prompt painting a 40-row filter list writes ~3KB of frame; the rewind
  that stops it stacking adds tens of bytes beside it. Linear in rows, with
  no allocation beyond the loop, which is why the height is measured off the
  frame rather than cached.

  Correctness of the result — one frame on screen, not N — is asserted in
  tests/test_interactive.py::TestRepaintLeavesNothingBehind."""
    )
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
