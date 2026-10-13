"""What `diff` costs to render, and what the geometry fix did to that cost.

`InteractivePrompt.diff` draws two columns side by side. Three things about
its width were wrong, and all three are fixed:

    header      2*col_w + 7 columns, but col_w was (width - 5) // 2,
                so the first row ran one column past an 80-column terminal
    separator   `"  \u2500" * col_w` repeats the two spaces along with the
                dash, so each half ran three times too wide — 232 columns
    body        padded to col_w + 10 against the header's two-space gutter,
                putting its divider ten columns right of the header's, and
                padded *after* colouring, so a cell carrying ESC[31m/ESC[0m
                counted those nine bytes as content and moved nine more

This measures two things. The whole render, because that is what a viewer
pays for; and the three expressions that actually changed, old against new,
so the fix can be shown not to have traded width for time.

Screen geometry — one divider column for every row, and no row wider than
the terminal showing it — is not this benchmark's job. It is asserted by
`tests/test_interactive.py::TestDiffFitsTheTerminal`, which drives the real
method through a terminal grid.

    PYTHONPATH="$PWD/packages/core-py:$PWD" python scripts/benchmark_diff_render.py
"""

from __future__ import annotations

import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "core-py"))
sys.path.insert(0, str(ROOT))

from domain.shell._internal.interactive import InteractivePrompt
from domain.shell._internal.io import MemoryIO

WIDTH = 80
OLD_COL_W = (WIDTH - 5) // 2
NEW_COL_W = max(1, (WIDTH - 7) // 2)

# These arms build one line, so they land well under a microsecond. At 4000
# iterations a whole arm is ~2ms of wall clock — scheduler noise moves that by
# tens of percent, and the first draft of this benchmark duly reported a
# +61% "regression" that vanished on a second run. 200k puts an arm at ~120ms,
# where the jitter is a fraction of the smallest delta being reported.
EXPRESSION_ITERS = 200_000
FULL_ITERS = 20_000
WARMUP = 2_000

# Below this the two arms are indistinguishable on a shared machine.
NOISE_FLOOR = 5.0

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RED, GREEN, RESET = "\x1b[31m", "\x1b[32m", "\x1b[0m"

LABEL = "alpha bravo charlie"
LEFT = [f"line {i}" for i in range(12)]
RIGHT = [f"line {i}" for i in range(12) if i != 3]

# The three expressions the fix rewrote, kept here in their pre-fix shape so
# the comparison survives the removal of the snapshot they were read from.


def old_separator(col_w: int) -> str:
    left = "  ─" * col_w
    right = "─  " * col_w
    return f"  {left}─  ┼─  {right}─"


def new_separator(col_w: int) -> str:
    rule = "─" * col_w
    return f"  {rule}  ┼  {rule}"


def old_header(col_w: int) -> str:
    return f"  {LABEL:^{col_w}}  │  {LABEL:^{col_w}}"


def new_header(col_w: int) -> str:
    cell = f"{LABEL[:col_w]:^{col_w}}"
    return f"  {cell}  │  {cell}"


def old_row(col_w: int, text: str, colour: str, right: str) -> str:
    """Pads after colouring, to a field ten wider than the header's gutter."""
    cell = f"{colour}{text}{RESET}" if colour else text
    return f"  {cell:<{col_w + 10}}  │  {right}"


def new_row(col_w: int, text: str, colour: str, right: str) -> str:
    """Pads before colouring, to exactly col_w — the header's own cell width."""
    padded = f"{text:<{col_w}}"
    cell = f"{colour}{padded}{RESET}" if colour else padded
    return f"  {cell}  │  {right}"


class Discard:
    """A sink that keeps the arithmetic and drops the bytes.

    `diff` writes through `self._io` four times and does nothing else with it,
    so the work being compared is the string building underneath the write.
    Storing the output would measure the sink's list growth instead.
    """

    def write(self, *args, **kwargs) -> None:
        pass


def bench(fn, iterations: int) -> float:
    """Microseconds per call, after a warm-up long enough to get the code hot."""
    for _ in range(WARMUP):
        fn()
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    return ((time.perf_counter() - t0) / iterations) * 1e6


def visible(text: str) -> str:
    return ANSI.sub("", text)


def shown_width(text: str) -> int:
    return len(visible(text))


def divider_column(text: str) -> int | None:
    flat = visible(text)
    for glyph in ("│", "┼"):
        if glyph in flat:
            return flat.index(glyph)
    return None


def main() -> int:
    inst = InteractivePrompt(MemoryIO())
    inst._io = Discard()

    full_us = bench(lambda: inst.diff(LABEL, LEFT, LABEL, RIGHT, title="changes"), FULL_ITERS)

    pairs = [
        (
            "separator",
            lambda col: old_separator(col),
            lambda col: new_separator(col),
            OLD_COL_W,
            NEW_COL_W,
        ),
        (
            "header",
            lambda col: old_header(col),
            lambda col: new_header(col),
            OLD_COL_W,
            NEW_COL_W,
        ),
        (
            "body (plain)",
            lambda col: old_row(col, "line 3", "", "line 3"),
            lambda col: new_row(col, "line 3", "", "line 3"),
            OLD_COL_W,
            NEW_COL_W,
        ),
        (
            "body (coloured)",
            lambda col: old_row(col, "line 3", RED, "line 3"),
            lambda col: new_row(col, "line 3", RED, "line 3"),
            OLD_COL_W,
            NEW_COL_W,
        ),
    ]

    print("=" * 78)
    print("`diff` render — cost of the geometry fix, and the geometry it fixed")
    print("=" * 78)
    print(f"\n  terminal width {WIDTH}: col_w {OLD_COL_W} (old) -> {NEW_COL_W} (new)")
    print(
        f"  {EXPRESSION_ITERS:,} iterations per expression, {FULL_ITERS:,} for the whole render\n"
    )
    print(
        f"  {'expression':<18} {'old us':>8} {'new us':>8} {'delta':>9}"
        f" {'old cols':>9} {'new cols':>9} {'old div':>8} {'new div':>8}"
    )
    print("  " + "-" * 74)

    worst_delta = -float("inf")
    for label, old_fn, new_fn, old_col, new_col in pairs:
        old_s, new_s = old_fn(old_col), new_fn(new_col)
        old_us = bench(lambda: old_fn(old_col), EXPRESSION_ITERS)
        new_us = bench(lambda: new_fn(new_col), EXPRESSION_ITERS)
        delta = ((new_us - old_us) / old_us) * 100 if old_us else 0.0
        worst_delta = max(worst_delta, delta)
        od, nd = divider_column(old_s), divider_column(new_s)
        print(
            f"  {label:<18} {old_us:>8.3f} {new_us:>8.3f} {delta:>+8.1f}%"
            f" {shown_width(old_s):>9} {shown_width(new_s):>9}"
            f" {('-' if od is None else od):>8} {('-' if nd is None else nd):>8}"
        )

    # Reuse the printed rows rather than sampling again: a second pass lands
    # somewhere else, and a summary that disagrees with the table above it is
    # worse than no summary at all.
    print("\n" + "=" * 78)
    print("  shape")
    print(f"    whole render:      {full_us:>7.3f} us per `diff()` call")
    print(
        f"    widest row:        {shown_width(old_separator(OLD_COL_W))} cols (old)"
        f" -> {shown_width(new_separator(NEW_COL_W))} cols (new), against {WIDTH}"
    )
    print(
        f"    divider column:    {divider_column(old_row(OLD_COL_W, 'line 3', RED, ''))}"
        f" (old, coloured) / {divider_column(old_row(OLD_COL_W, 'line 3', '', ''))} (old, plain)"
        f" -> {divider_column(new_row(NEW_COL_W, 'line 3', RED, ''))} (new)"
    )
    verdict = (
        "within noise, nothing measurably slower" if worst_delta <= NOISE_FLOOR else "REGRESSION"
    )
    print(f"    slowest arm:       {worst_delta:+.1f}% — {verdict}")
    print("=" * 78)
    print(
        """
  reading:
    old/new us   time to build that one line, nothing else
    delta        new against old; negative means the fix got cheaper
    old/new cols visible width after escape sequences are stripped — a cell
                 padded while coloured counts ESC[31m as nine columns
    old/new div  visible column of the divider glyph; every row agreeing on
                 one number is the whole point of the fix

  The separator got cheaper because it is one repeat instead of two multiplies
  and a five-part join; the body row is a wash, since the pad moved rather than
  grew. The render they belong to runs once per view, not once per keystroke —
  that path is benchmark_repaint.py, which measures the _rewind the picker now
  pays on every key.

  Geometry itself is asserted, not timed: tests/test_interactive.py::
  TestDiffFitsTheTerminal requires every row to fit the terminal and every
  divider to land in the same column."""
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
