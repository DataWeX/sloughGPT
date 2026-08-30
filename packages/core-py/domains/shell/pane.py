"""
pane — pure layout engine for the shell TUI.

The "arranger" from the split-window model: given a total geometry it
computes the region for each pane.  It has no curses dependency, no
rendering logic, and no knowledge of pane content — exactly like a
minimal window manager on a headless display.

A ``Pane`` is a named region with sizing constraints.  ``PaneLayout``
assigns regions greedily:

  1. Fixed panes take their exact height.
  2. Border panes consume 1 row each (top/bottom separator).
  3. The flex region is shared among proportional panes by ``ratio``.
  4. The final proportional pane absorbs rounding remainder.

Region is a ``Rect(top, left, rows, cols)`` — the raw geometry handed to
whatever content surface is bound to the pane.  ``content_rect()`` strips
borders and padding so the surface knows its drawable area.

PaneLayout also tracks **focus** (which pane has the cursor) and
**visibility** (which panes are shown).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ── Rect ──────────────────────────────────────────────────────────────


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

    def inset(self, top: int = 0, left: int = 0, bottom: int = 0, right: int = 0) -> Rect:
        """Return a new Rect shrunk by the given offsets (content area)."""
        new_top = self.top + top
        new_left = self.left + left
        new_rows = max(self.rows - top - bottom, 0)
        new_cols = max(self.cols - left - right, 0)
        return Rect(new_top, new_left, new_rows, new_cols)


# ── Border ────────────────────────────────────────────────────────────


class Border:
    """Border configuration for a pane.

    ``kind`` controls which edges get a separator line:
      - ``"top"`` / ``"bottom"`` / ``"left"`` / ``"right"`` — single edge
      - ``"horizontal"`` — top + bottom
      - ``"vertical"`` — left + right
      - ``"all"`` — all four edges
      - ``"none"`` — no borders (default)

    Each border consumes 1 cell on its side.  ``ch`` is the character
    drawn (default ``"─"`` for horizontal, ``"│"`` for vertical).
    """

    __slots__ = ("kind", "ch")

    NONE = "none"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    ALL = "all"

    def __init__(self, kind: str = "none", ch: str = "") -> None:
        self.kind = kind
        self.ch = ch

    @property
    def top(self) -> bool:
        return self.kind in (self.TOP, self.HORIZONTAL, self.ALL)

    @property
    def bottom(self) -> bool:
        return self.kind in (self.BOTTOM, self.HORIZONTAL, self.ALL)

    @property
    def left(self) -> bool:
        return self.kind in (self.LEFT, self.VERTICAL, self.ALL)

    @property
    def right(self) -> bool:
        return self.kind in (self.RIGHT, self.VERTICAL, self.ALL)

    @property
    def is_empty(self) -> bool:
        return self.kind == self.NONE


# ── Pane ──────────────────────────────────────────────────────────────


@dataclass
class Pane:
    """A named region request.

    Sizing (mutually exclusive for height):
      ``ratio``  — fractional share of flex rows (0.3 = 30%).
      ``fixed``  — exact height in rows.

    Constraints:
      ``min_rows`` — floor for proportional panes.
      ``max_rows`` — ceiling for any pane (0 = no limit).
      ``width_ratio`` — fractional share of columns (1.0 = full width).

    Visual:
      ``border``  — ``Border`` instance controlling separator lines.
      ``padding`` — inner cells to reserve on each side ``(top, left, bottom, right)``.
      ``visible`` — if False, pane is skipped during layout.

    Focus:
      ``focusable`` — if False, pane cannot receive focus via navigation.
    """

    name: str
    ratio: float = 0.0
    min_rows: int = 0
    max_rows: int = 0
    fixed: int | None = None
    width_ratio: float = 1.0
    border: Border = field(default_factory=Border)
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    visible: bool = True
    focusable: bool = True

    def __post_init__(self) -> None:
        if self.ratio < 0.0 or self.ratio > 1.0:
            raise ValueError(f"ratio out of range 0..1: {self.ratio}")
        if self.fixed is not None and self.fixed < 0:
            raise ValueError(f"fixed height cannot be negative: {self.fixed}")
        if self.max_rows < 0:
            raise ValueError(f"max_rows cannot be negative: {self.max_rows}")

    @property
    def border_top(self) -> int:
        return 1 if self.border.top else 0

    @property
    def border_bottom(self) -> int:
        return 1 if self.border.bottom else 0

    @property
    def border_left(self) -> int:
        return 1 if self.border.left else 0

    @property
    def border_right(self) -> int:
        return 1 if self.border.right else 0

    @property
    def pad_top(self) -> int:
        return self.padding[0]

    @property
    def pad_left(self) -> int:
        return self.padding[1]

    @property
    def pad_bottom(self) -> int:
        return self.padding[2]

    @property
    def pad_right(self) -> int:
        return self.padding[3]

    def outer_height(self, content_rows: int) -> int:
        """Total rows consumed including borders."""
        return content_rows + self.border_top + self.border_bottom

    def outer_width(self, content_cols: int) -> int:
        """Total cols consumed including borders."""
        return content_cols + self.border_left + self.border_right

    def content_rect(self, rect: Rect) -> Rect:
        """Shrink *rect* to the drawable area inside borders and padding."""
        return rect.inset(
            top=self.border_top + self.pad_top,
            left=self.border_left + self.pad_left,
            bottom=self.border_bottom + self.pad_bottom,
            right=self.border_right + self.pad_right,
        )


# ── PaneLayout ────────────────────────────────────────────────────────


@dataclass
class PaneLayout:
    """Assigns regions to panes from a total geometry.

    Tracks focus and visibility.  Only visible panes participate in
    layout; only focusable panes receive focus.
    """

    panes: List[Pane] = field(default_factory=list)
    _focus_idx: int = field(default=0, init=False, repr=False)

    # ── Focus ──────────────────────────────────────────────────────────

    @property
    def focus_index(self) -> int:
        """Index of the currently focused pane (among all panes)."""
        return self._focus_idx

    @property
    def focus_name(self) -> str | None:
        """Name of the currently focused pane, or None."""
        if 0 <= self._focus_idx < len(self.panes):
            return self.panes[self._focus_idx].name
        return None

    def set_focus(self, name: str) -> bool:
        """Move focus to the pane with *name*.  Returns False if not found or not focusable."""
        for i, p in enumerate(self.panes):
            if p.name == name:
                if not p.focusable or not p.visible:
                    return False
                self._focus_idx = i
                return True
        return False

    def focus_next(self) -> str | None:
        """Advance focus to the next focusable, visible pane.  Returns the name."""
        if not self.panes:
            return None
        n = len(self.panes)
        for offset in range(1, n + 1):
            idx = (self._focus_idx + offset) % n
            p = self.panes[idx]
            if p.focusable and p.visible:
                self._focus_idx = idx
                return p.name
        return None

    def focus_prev(self) -> str | None:
        """Move focus to the previous focusable, visible pane.  Returns the name."""
        if not self.panes:
            return None
        n = len(self.panes)
        for offset in range(1, n + 1):
            idx = (self._focus_idx - offset) % n
            p = self.panes[idx]
            if p.focusable and p.visible:
                self._focus_idx = idx
                return p.name
        return None

    # ── Visibility ─────────────────────────────────────────────────────

    def set_visible(self, name: str, visible: bool) -> bool:
        """Show or hide a pane by name.  Returns False if not found."""
        for p in self.panes:
            if p.name == name:
                p.visible = visible
                return True
        return False

    def is_visible(self, name: str) -> bool:
        """Check if a pane is visible."""
        for p in self.panes:
            if p.name == name:
                return p.visible
        return False

    # ── Layout ─────────────────────────────────────────────────────────

    def compute(self, rows: int, cols: int) -> dict[str, Rect]:
        """Return ``{pane_name: Rect}`` for the given terminal size.

        Algorithm:
          1. Hidden panes are skipped entirely.
          2. Fixed panes take their exact height.
          3. Border rows are subtracted from available space.
          4. The flex region is shared among proportional panes by ratio.
          5. The final proportional pane absorbs rounding remainder.
          6. max_rows clamps any pane.
        """
        if rows <= 0 or cols <= 0:
            return {}

        visible = [p for p in self.panes if p.visible]
        if not visible:
            return {}

        # Pass 1: compute content heights for fixed panes and borders.
        fixed_content = 0
        border_total = 0
        proportional: list[Pane] = []

        for p in visible:
            bt, bb = p.border_top, p.border_bottom
            border_total += bt + bb
            if p.fixed is not None:
                fixed_content += p.fixed
            else:
                proportional.append(p)

        # Subtract border rows from total available space.
        available = max(rows - border_total, 0)
        flex = max(available - fixed_content, 0)
        total_ratio = sum(p.ratio for p in proportional)

        # Pass 2: assign content heights.
        content_heights: dict[str, int] = {}
        used_flex = 0.0

        for idx, p in enumerate(visible):
            if p.fixed is not None:
                h = p.fixed
            elif proportional and p is proportional[-1]:
                h = int(round(flex - used_flex))
                h = max(h, p.min_rows)
                # Clamp to remaining space even after min_rows.
                h = min(h, max(int(flex - used_flex), 0))
            else:
                share = p.ratio * flex if total_ratio else 0.0
                h = int(round(share))
                h = max(h, p.min_rows)
                h = min(h, max(int(flex - used_flex), 0))

            # Clamp to max_rows.
            if p.max_rows > 0:
                h = min(h, p.max_rows)

            content_heights[p.name] = h
            if p.fixed is None:
                used_flex += h

        # Pass 3: build Rects with borders.
        regions: dict[str, Rect] = {}
        y = 0

        for p in visible:
            ch = content_heights[p.name]
            # Total outer width for this pane (full cols or width_ratio share).
            outer_w = int(cols * p.width_ratio)
            # Content width = outer width minus border columns.
            content_w = max(outer_w - p.border_left - p.border_right, 0)
            # Centering when width_ratio < 1.
            left = (cols - outer_w) // 2 if p.width_ratio < 1.0 else 0

            # The outer rect includes borders; content_w is the drawable area.
            outer = Rect(top=y, left=left, rows=p.outer_height(ch), cols=outer_w)
            regions[p.name] = outer

            # Advance y past the outer rect.
            y += outer.rows

        # Fix centering for nested width_ratio panes.
        for p in visible:
            r = regions[p.name]
            if p.width_ratio < 1.0:
                outer_w = int(cols * p.width_ratio)
                new_left = (cols - outer_w) // 2
                regions[p.name] = Rect(top=r.top, left=new_left, rows=r.rows, cols=r.cols)

        return regions

    def content_regions(self, rows: int, cols: int) -> dict[str, Rect]:
        """Return ``{pane_name: Rect}`` for the drawable content area.

        Each rect is shrunk by the pane's borders and padding so
        surfaces can draw directly into it without knowing about
        borders.
        """
        outer = self.compute(rows, cols)
        return {name: self._pane_by_name(name).content_rect(rect) for name, rect in outer.items()}

    def _pane_by_name(self, name: str) -> Pane:
        for p in self.panes:
            if p.name == name:
                return p
        raise KeyError(name)


# ── Convenience ───────────────────────────────────────────────────────


def split(
    rows: int,
    cols: int,
    ratios: list[float],
    min_rows: list[int] | None = None,
    borders: str = "none",
) -> list[Rect]:
    """Split a geometry into horizontal bands.

    ``ratios`` are fractional shares; ``min_rows`` optionally floors each.
    ``borders`` adds separator lines between bands (``"top"``, ``"bottom"``,
    ``"both"``).  The last band absorbs rounding remainder.
    """
    min_rows = min_rows or [0] * len(ratios)
    b = Border(borders)
    panes = [
        Pane(name=str(i), ratio=r, min_rows=m, border=b)
        for i, (r, m) in enumerate(zip(ratios, min_rows))
    ]
    layout = PaneLayout(panes)
    out = layout.compute(rows, cols)
    return [out[str(i)] for i in range(len(panes))]


def vsplit(
    rows: int,
    cols: int,
    ratios: list[float],
    min_cols: list[int] | None = None,
) -> list[Rect]:
    """Split a geometry into vertical bands (side by side).

    ``ratios`` are fractional shares of width; ``min_cols`` floors each.
    The last band absorbs rounding remainder.
    """
    min_cols = min_cols or [0] * len(ratios)
    band_widths: list[int] = []
    total_w = 0
    for i, r in enumerate(ratios):
        w = int(round(cols * r))
        w = max(w, min_cols[i])
        w = min(w, cols - total_w)
        band_widths.append(w)
        total_w += w
    # Last band absorbs remainder.
    if band_widths:
        band_widths[-1] = cols - sum(band_widths[:-1])

    rects = []
    x = 0
    for w in band_widths:
        rects.append(Rect(top=0, left=x, rows=rows, cols=w))
        x += w
    return rects
