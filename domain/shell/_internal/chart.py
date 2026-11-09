"""Charts as engine primitives — no dependencies, stdlib only.

Why this exists
---------------
Bars were hand-rolled eight separate times: ``"█" * n + "░" * (w - n)`` appears
in ``repl.py``, ``console.py``, ``interactive_progress.py`` (six sites),
``interactive_screen.py`` and ``realm_view.py``.  Each re-decided bar width and
what an empty cell looks like, so no two progress bars in the shell look
alike.  Same duplication debt as the training loops: one primitive, call sites
adapt.

Sub-cell precision
------------------
A terminal cell is square, so a bar drawn one cell at a time has only eight
possible heights per row.  Bars are indexed by eighth, so a bar can land
anywhere in 1/8-cell steps and a trend reads as a curve rather than a
staircase.

Direction
---------
Text is one character per cell, which is correct for Hebrew, Arabic, CJK and
Latin alike -- none need two cells for a base letter.  Nothing here reorders
text: characters are written in the order given and the terminal's own bidi
pass decides how they display.  A chart that reordered Hebrew itself would
disagree with every other program on the same terminal.  The consequence to
know: a label mixing Hebrew words with Latin digits is a mixed-direction run,
so the digits can land left of the word rather than right.
"""

from __future__ import annotations

# Eight partial blocks then full. Index = how many eighths of a cell it fills.
EIGHTHS = " ▏▎▍▌▋▊▉█"

# Eight vertical heights, bottom-anchored.
RAMP = " ▁▂▃▄▅▆▇█"

FULL = "█"
EMPTY = " "
TRACK = "░"


def bar(fraction: float, width: int, *, filled: str = FULL, track: str = TRACK) -> str:
    """One horizontal bar *width* cells long.

    Args:
        fraction: 0.0-1.0. Clamped, so a caller passing an over-100% value
            gets a full bar rather than one that runs past its track.
        width: total cells including the empty part.
        filled: glyph for the filled part.
        track: glyph for the empty part.
    """
    if width <= 0:
        return ""
    fraction = 0.0 if fraction < 0 else (1.0 if fraction > 1 else fraction)
    n = int(round(fraction * width))
    return filled * n + track * (width - n)


def column(fraction: float, height: int, *, glyph: str = FULL) -> list[str]:
    """One vertical bar *height* rows tall, returned top row first.

    Grows from the bottom, with the topmost partial row using an eighth block
    so the height lands anywhere in 1/8-cell steps.
    """
    if height <= 0:
        return []
    fraction = 0.0 if fraction < 0 else (1.0 if fraction > 1 else fraction)
    whole, remainder = divmod(int(round(fraction * height * 8)), 8)
    whole = min(whole, height)
    # Build bottom-up: full rows, then the partial cap, then the empty head.
    cells = [EMPTY] * (height - whole)
    if remainder and whole < height:
        # The partial row sits *above* the whole ones, since the bar stops part
        # way up its last full cell.
        cells[height - whole - 1] = EIGHTHS[remainder]
    cells.extend(glyph for _ in range(whole))
    return cells[:height]


def sparkline(values: list[float], width: int, *, glyphs: str = RAMP) -> str:
    """The most recent *width* samples as one row of cells."""
    if width <= 0 or not values:
        return ""
    series = values[-width:]
    lo, hi = min(series), max(series)
    # A flat series has no range; give it one so heights stay finite.
    span = (hi - lo) or 1.0
    top = len(glyphs) - 1
    return "".join(glyphs[round((v - lo) / span * top)] for v in series)


def bar_chart(
    values: list[float],
    labels: list[str] | None = None,
    *,
    width: int = 40,
    height: int = 10,
    title: str = "",
    floor: float | None = None,
) -> list[str]:
    """Vertical bars, one per category, as a list of text rows.

    Args:
        values: numbers to plot, in category order.
        labels: category names. Defaults to indices.
        width: total cells per row.
        height: rows available, including the title and label rows.
        title: drawn on the first row if given.
        floor: value drawn as an empty bar. Defaults to 0 when every value is
            non-negative, so bars grow from a real zero rather than an
            arbitrary baseline.

    Labels go on their own row under the bars rather than beside them, and the
    label gutter is capped at a third of the width so a long name can never
    squeeze the plot out. Within that cap a longer label does narrow the bars,
    so callers redrawing every frame should keep their label lengths steady.
    """
    if not values or width <= 0 or height <= 0:
        return []

    labels = list(labels) if labels else [str(i) for i in range(len(values))]
    rows: list[str] = []
    if title:
        rows.append(title[:width])
        height -= 1
    if height <= 0:
        return rows

    # Need at least one bar row above the labels.
    bar_h = height - 1 if height >= 2 else height
    label_w = min(max((len(x) for x in labels), default=0), width // 3)
    gutter = label_w + 1 if label_w else 0
    plot_w = max(len(values), width - gutter)

    lo = floor if floor is not None else (0.0 if min(values) >= 0 else min(values))
    hi = max(values)
    span = (hi - lo) or 1.0

    slot = max(1, plot_w // len(values))
    bar_w = max(1, slot - (1 if slot >= 3 else 0))
    grid = [[EMPTY] * plot_w for _ in range(bar_h)]
    for i, value in enumerate(values):
        x0 = min(i * slot, max(0, plot_w - 1))
        cells = column((value - lo) / span, bar_h)
        for r in range(bar_h):
            for x in range(x0, min(x0 + bar_w, plot_w)):
                grid[r][x] = cells[r]

    pad = " " * gutter
    for r in range(bar_h):
        rows.append(pad + "".join(grid[r])[: max(0, width - gutter)])

    if bar_h < height:
        cells = [" "] * plot_w
        for i, label in enumerate(labels):
            x0 = min(i * slot, max(0, plot_w - 1))
            for k, ch in enumerate(label[: max(0, plot_w - x0)]):
                cells[x0 + k] = ch
        rows.append(pad + "".join(cells)[: max(0, width - gutter)])

    return rows
