"""
pane — pure layout engine for the shell TUI.

The "arranger" from the split-window model: given a total geometry it
computes the region for each pane.  It has no curses dependency, no
rendering logic, and no knowledge of pane content — exactly like a
minimal window manager on a headless display.

A ``Pane`` is just a named region with a preferred height.  ``PaneLayout``
assigns regions greedily:

  1. The top panes take their fixed/proportional share of rows.
  2. The bottom pane absorbs the remainder (like a bottom bar).
  3. All panes span the full width unless ``width_ratio`` is set.

Region is a ``Rect(top, left, rows, cols)`` — the raw geometry handed to
whatever content surface is bound to the pane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Rect:
    """A screen region: top row, left col, height (rows), width (cols)."""

    top: int
    left: int
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.rows < 0 or self.cols < 0 or self.top < 0 or self.left < 0:
            raise ValueError(f"negative region component: {self}")

    def __bool__(self) -> bool:
        return self.rows > 0 and self.cols > 0


@dataclass(frozen=True)
class Pane:
    """A named region request. ``ratio`` is fractional share of rows
    (0.3 = 30%); ``min_rows`` floors it; ``fixed`` pins an exact height."""

    name: str
    ratio: float = 0.0
    min_rows: int = 0
    fixed: int | None = None
    width_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.ratio < 0.0 or self.ratio > 1.0:
            raise ValueError(f"ratio out of range 0..1: {self.ratio}")
        if self.fixed is not None and self.fixed < 0:
            raise ValueError(f"fixed height cannot be negative: {self.fixed}")


@dataclass
class PaneLayout:
    """Assigns regions to panes from a total geometry."""

    panes: List[Pane] = field(default_factory=list)

    def compute(self, rows: int, cols: int) -> dict[str, Rect]:
        """Return ``{pane_name: Rect}`` for the given terminal size.

        Fixed panes keep their exact height wherever they sit.  The flex
        region (``rows - fixed_total``) is shared among proportional panes
        by ``ratio``; the *final proportional pane* absorbs whatever is
        left over, so the layout always fills the screen.
        """
        if rows <= 0 or cols <= 0:
            return {}

        fixed_total = sum(p.fixed or 0 for p in self.panes)
        flex = max(rows - fixed_total, 0)
        total_ratio = sum(p.ratio for p in self.panes)
        proportional = [i for i, p in enumerate(self.panes) if p.fixed is None]
        final_prop = proportional[-1] if proportional else -1

        regions: dict[str, Rect] = {}
        y = 0
        used_flex = 0.0

        for i, pane in enumerate(self.panes):
            if pane.fixed is not None:
                h = pane.fixed
            elif i == final_prop:
                # Absorb the remainder of the flex region.
                h = int(round(flex - used_flex))
                h = max(h, 0)
            else:
                share = pane.ratio * flex if total_ratio else 0.0
                h = int(round(share))
                h = max(h, pane.min_rows)
                h = min(h, max(int(flex - used_flex), 0))

            h = min(h, max(rows - y, 0))
            w = int(cols * pane.width_ratio)
            left = (cols - w) // 2 if pane.width_ratio < 1.0 else 0
            regions[pane.name] = Rect(top=y, left=left, rows=h, cols=w)
            if pane.fixed is None:
                used_flex += h
            y += h

        return regions


def split(
    rows: int,
    cols: int,
    ratios: list[float],
    min_rows: list[int] | None = None,
) -> list[Rect]:
    """Convenience: split a geometry into horizontal bands.

    ``ratios`` are fractional shares; ``min_rows`` optionally floors each.
    The last band absorbs rounding remainder.
    """
    min_rows = min_rows or [0] * len(ratios)
    panes = [
        Pane(name=str(i), ratio=r, min_rows=m)
        for i, (r, m) in enumerate(zip(ratios, min_rows))
    ]
    layout = PaneLayout(panes)
    out = layout.compute(rows, cols)
    return [out[str(i)] for i in range(len(panes))]
