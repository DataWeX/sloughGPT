"""
graphics — terminal graphics engine for the shell TUI.

Replaces curses-based rendering with a compositing framebuffer model:

  Cell → Framebuffer → Layer → GraphicsEngine → Terminal

Architecture:
  - Cell: single character cell with fg/bg colors and attributes
  - Framebuffer: 2D grid of Cells (the backing store)
  - Layer: named, ordered group of draw operations
  - GraphicsEngine: composites layers, renders to terminal, handles input
  - FontConfig: terminal font/size configuration
  - BoxModel: border/padding/margin for widgets
  - TabManager: focus navigation between focusable regions
  - DisplayMode: pattern-based rendering (reverse, dim, strikethrough, etc.)
  - Snapshot: captures/restores engine state for undo/redo

Usage::

    engine = GraphicsEngine()
    engine.open()

    # Draw directly
    engine.write(0, 0, "Hello World", fg=COLOR_GREEN)
    engine.box(0, 0, 40, 10, border=BOX_DOUBLE)

    # Or use layers
    bg = engine.create_layer("bg", z=0)
    bg.fill(pattern=Pattern.DOTS)
    content = engine.create_layer("content", z=1)
    content.write(2, 2, "Status: OK")

    engine.render()
    engine.close()
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable

# ── Color Constants ────────────────────────────────────────────────────


class Color(IntEnum):
    """ANSI 256-color palette indices for common colors."""

    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7
    DEFAULT = 9
    # Bright variants (16-231 are 6x6x6 color cube)
    BRIGHT_BLACK = 8
    BRIGHT_RED = 9
    BRIGHT_GREEN = 10
    BRIGHT_YELLOW = 11
    BRIGHT_BLUE = 12
    BRIGHT_MAGENTA = 13
    BRIGHT_CYAN = 14
    BRIGHT_WHITE = 15


def rgb_to_ansi256(r: int, g: int, b: int) -> int:
    """Convert RGB (0-255 each) to nearest ANSI 256-color index."""
    if r == g == b:
        if r < 8:
            return 0
        if r > 248:
            return 15
        return int(((r - 8) / 248) * 23 + 23)
    ri = int(((r - 8) / 248) * 5 + 0.5)
    gi = int(((g - 8) / 248) * 5 + 0.5)
    bi = int(((b - 8) / 248) * 5 + 0.5)
    return 16 + 36 * ri + 6 * gi + bi


# ── Attributes ─────────────────────────────────────────────────────────


class Attr(IntEnum):
    """Terminal text attributes (bitmask)."""

    NONE = 0
    BOLD = 1
    DIM = 2
    ITALIC = 4
    UNDERLINE = 8
    BLINK = 16
    REVERSE = 32
    HIDDEN = 64
    STRIKETHROUGH = 128


# ── Cell ───────────────────────────────────────────────────────────────


@dataclass
class Cell:
    """A single character cell in the framebuffer."""

    char: str = " "
    fg: int = Color.DEFAULT
    bg: int = Color.DEFAULT
    attr: int = Attr.NONE

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cell):
            return NotImplemented
        return (
            self.char == other.char
            and self.fg == other.fg
            and self.bg == other.bg
            and self.attr == other.attr
        )

    def copy(self) -> Cell:
        return Cell(self.char, self.fg, self.bg, self.attr)


# ── Display Patterns ──────────────────────────────────────────────────


class Pattern(Enum):
    """Background fill patterns using Unicode/ASCII art."""

    NONE = "none"
    SOLID = "solid"
    DOTS = "dots"
    DOTS_DENSE = "dots_dense"
    DASHES = "dashes"
    DIAGONAL = "diagonal"
    DIAGONAL_THICK = "diagonal_thick"
    CROSS = "cross"
    CROSS_DENSE = "cross_dense"
    CHECKER = "checker"
    GRADIENT_LIGHT = "gradient_light"
    GRADIENT_MED = "gradient_med"
    GRADIENT_HEAVY = "gradient_heavy"
    BRAILLE = "braille"
    BLOCK_UPPER = "block_upper"
    BLOCK_LOWER = "block_lower"
    BLOCK_FULL = "block_full"
    HALF_BLOCK = "half_block"
    SHADE_LIGHT = "shade_light"
    SHADE_MED = "shade_med"
    SHADE_HEAVY = "shade_heavy"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


_PATTERN_CHARS: dict[Pattern, list[str]] = {
    Pattern.SOLID: ["█"],
    Pattern.DOTS: ["·", " "],
    Pattern.DOTS_DENSE: ["•", "·"],
    Pattern.DASHES: ["─", " "],
    Pattern.DIAGONAL: ["\\", " "],
    Pattern.DIAGONAL_THICK: ["\\", "·"],
    Pattern.CROSS: ["┼", " "],
    Pattern.CROSS_DENSE: ["┼", "┤"],
    Pattern.CHECKER: ["░", "▓"],
    Pattern.GRADIENT_LIGHT: ["░"],
    Pattern.GRADIENT_MED: ["▒"],
    Pattern.GRADIENT_HEAVY: ["▓"],
    Pattern.BRAILLE: ["⠁", "⠂", "⠄", "⠈"],
    Pattern.BLOCK_UPPER: ["▀"],
    Pattern.BLOCK_LOWER: ["▄"],
    Pattern.BLOCK_FULL: ["█"],
    Pattern.HALF_BLOCK: ["▌", "▐"],
    Pattern.SHADE_LIGHT: ["░"],
    Pattern.SHADE_MED: ["▒"],
    Pattern.SHADE_HEAVY: ["▓"],
    Pattern.HORIZONTAL: ["─"],
    Pattern.VERTICAL: ["│"],
}


def _pattern_char(pattern: Pattern, row: int, col: int) -> str:
    """Return the pattern character for a given cell position."""
    if pattern == Pattern.NONE:
        return " "
    chars = _PATTERN_CHARS.get(pattern, [" "])
    if len(chars) == 1:
        return chars[0]
    # Checker alternates; dots/diagonal use position hash
    if pattern == Pattern.CHECKER:
        return chars[(row + col) % 2]
    return chars[(row * 3 + col * 7) % len(chars)]


# ── Box Drawing ───────────────────────────────────────────────────────


class BoxStyle(Enum):
    """Box border styles."""

    NONE = "none"
    ASCII = "ascii"
    SINGLE = "single"
    DOUBLE = "double"
    HEAVY = "heavy"
    ROUNDED = "rounded"


# (top-left, top-right, bottom-left, bottom-right,
#  horizontal, vertical, top-t, bottom-t, left-t, right-t, cross)
_BOX_CHARS: dict[BoxStyle, tuple[str, ...]] = {
    BoxStyle.ASCII: ("+", "+", "+", "+", "-", "|", "+", "+", "+", "+", "+"),
    BoxStyle.SINGLE: ("┌", "┐", "└", "┘", "─", "│", "┬", "┴", "├", "┤", "┼"),
    BoxStyle.DOUBLE: ("╔", "╗", "╚", "╝", "═", "║", "╦", "╩", "╠", "╣", "╬"),
    BoxStyle.HEAVY: ("┏", "┓", "┗", "┛", "━", "┃", "┳", "┻", "┣", "┫", "╋"),
    BoxStyle.ROUNDED: ("╭", "╮", "╰", "╯", "─", "│", "┬", "┴", "├", "┤", "┼"),
}


@dataclass
class BoxModel:
    """Border and spacing for a widget region."""

    border: BoxStyle = BoxStyle.NONE
    border_fg: int = Color.DEFAULT
    border_bg: int = Color.DEFAULT
    border_attr: int = Attr.NONE
    padding_top: int = 0
    padding_left: int = 0
    padding_bottom: int = 0
    padding_right: int = 0
    margin_top: int = 0
    margin_left: int = 0
    margin_bottom: int = 0
    margin_right: int = 0

    @property
    def border_top(self) -> int:
        return 1 if self.border != BoxStyle.NONE else 0

    @property
    def border_bottom(self) -> int:
        return 1 if self.border != BoxStyle.NONE else 0

    @property
    def border_left(self) -> int:
        return 1 if self.border != BoxStyle.NONE else 0

    @property
    def border_right(self) -> int:
        return 1 if self.border != BoxStyle.NONE else 0


# ── Framebuffer ───────────────────────────────────────────────────────


class Framebuffer:
    """2D grid of Cells — the backing store for rendering.

    All drawing operations target the framebuffer. The engine composites
    multiple framebuffers (layers) into a final buffer before rendering
    to the terminal.
    """

    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self._cells: list[list[Cell]] = [[Cell() for _ in range(cols)] for _ in range(rows)]

    def resize(self, rows: int, cols: int) -> None:
        """Resize the framebuffer, preserving existing content where possible."""
        new: list[list[Cell]] = [[Cell() for _ in range(cols)] for _ in range(rows)]
        for r in range(min(rows, self.rows)):
            for c in range(min(cols, self.cols)):
                new[r][c] = self._cells[r][c]
        self._cells = new
        self.rows = rows
        self.cols = cols

    def clear(self) -> None:
        """Clear all cells to default."""
        for r in range(self.rows):
            for c in range(self.cols):
                self._cells[r][c] = Cell()

    def get(self, row: int, col: int) -> Cell:
        """Get the cell at (row, col). Returns default Cell if out of bounds."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self._cells[row][col]
        return Cell()

    def set(self, row: int, col: int, cell: Cell) -> None:
        """Set the cell at (row, col). No-op if out of bounds."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self._cells[row][col] = cell

    def put(
        self,
        row: int,
        col: int,
        char: str,
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
    ) -> None:
        """Write a single character at (row, col)."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self._cells[row][col] = Cell(char, fg, bg, attr)

    def write_str(
        self,
        row: int,
        col: int,
        text: str,
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
    ) -> int:
        """Write a string at (row, col). Returns number of cells written."""
        written = 0
        for i, ch in enumerate(text):
            c = col + i
            if c >= self.cols:
                break
            if 0 <= row < self.rows and 0 <= c < self.cols:
                # Handle wide (CJK) characters
                w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                self._cells[row][c] = Cell(ch, fg, bg, attr)
                if w == 2 and c + 1 < self.cols:
                    self._cells[row][c + 1] = Cell("", fg, bg, attr)
                written += w
        return written

    def fill(
        self,
        char: str = " ",
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
    ) -> None:
        """Fill the entire buffer with a character."""
        for r in range(self.rows):
            for c in range(self.cols):
                self._cells[r][c] = Cell(char, fg, bg, attr)

    def fill_pattern(
        self,
        pattern: Pattern,
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
    ) -> None:
        """Fill the buffer with a repeating pattern."""
        for r in range(self.rows):
            for c in range(self.cols):
                ch = _pattern_char(pattern, r, c)
                self._cells[r][c] = Cell(ch, fg, bg, attr)

    def draw_rect(
        self,
        row: int,
        col: int,
        width: int,
        height: int,
        char: str = " ",
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
    ) -> None:
        """Fill a rectangular region."""
        for r in range(row, min(row + height, self.rows)):
            for c in range(col, min(col + width, self.cols)):
                self._cells[r][c] = Cell(char, fg, bg, attr)

    def draw_box(
        self,
        row: int,
        col: int,
        width: int,
        height: int,
        style: BoxStyle = BoxStyle.SINGLE,
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
    ) -> None:
        """Draw a box border around a region."""
        if style == BoxStyle.NONE or width < 2 or height < 2:
            return
        chars = _BOX_CHARS[style]
        tl, tr, bl, br, h, v, tt, bt, lt, rt, cr = chars

        # Corners
        self.put(row, col, tl, fg, bg, attr)
        self.put(row, col + width - 1, tr, fg, bg, attr)
        self.put(row + height - 1, col, bl, fg, bg, attr)
        self.put(row + height - 1, col + width - 1, br, fg, bg, attr)

        # Top/bottom edges
        for c in range(col + 1, col + width - 1):
            self.put(row, c, h, fg, bg, attr)
            self.put(row + height - 1, c, h, fg, bg, attr)

        # Left/right edges
        for r in range(row + 1, row + height - 1):
            self.put(r, col, v, fg, bg, attr)
            self.put(r, col + width - 1, v, fg, bg, attr)

    def composite(
        self, other: Framebuffer, dest_row: int = 0, dest_col: int = 0, alpha: float = 1.0
    ) -> None:
        """Composite another framebuffer onto this one at (dest_row, dest_col).

        Non-empty cells in ``other`` overwrite cells in this buffer.
        """
        for r in range(other.rows):
            dr = dest_row + r
            if dr < 0 or dr >= self.rows:
                continue
            for c in range(other.cols):
                dc = dest_col + c
                if dc < 0 or dc >= self.cols:
                    continue
                src = other._cells[r][c]
                if src.char != " " or src.bg != Color.DEFAULT:
                    self._cells[dr][dc] = src

    def diff(self, other: Framebuffer) -> list[tuple[int, int, Cell]]:
        """Return cells that differ between self and other."""
        changes = []
        rows = min(self.rows, other.rows)
        cols = min(self.cols, other.cols)
        for r in range(rows):
            for c in range(cols):
                if self._cells[r][c] != other._cells[r][c]:
                    changes.append((r, c, self._cells[r][c]))
        return changes

    def snapshot(self) -> Framebuffer:
        """Return a deep copy of this framebuffer."""
        fb = Framebuffer(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                fb._cells[r][c] = self._cells[r][c].copy()
        return fb

    def restore(self, other: Framebuffer) -> None:
        """Restore this buffer from a snapshot."""
        self.resize(other.rows, other.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                self._cells[r][c] = other._cells[r][c].copy()


# ── Layer ──────────────────────────────────────────────────────────────


class Layer:
    """A named rendering layer with its own framebuffer.

    Layers are composited in z-order by the GraphicsEngine. Higher z values
    are drawn on top.
    """

    def __init__(
        self,
        name: str,
        rows: int,
        cols: int,
        z: int = 0,
        visible: bool = True,
        opacity: float = 1.0,
    ) -> None:
        self.name = name
        self.z = z
        self.visible = visible
        self.opacity = opacity
        self.framebuffer = Framebuffer(rows, cols)
        self._draw_ops: list[Callable[[], None]] = []

    def resize(self, rows: int, cols: int) -> None:
        self.framebuffer.resize(rows, cols)

    def clear(self) -> None:
        self.framebuffer.clear()

    def write(
        self,
        row: int,
        col: int,
        text: str,
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
    ) -> int:
        return self.framebuffer.write_str(row, col, text, fg, bg, attr)

    def put(
        self,
        row: int,
        col: int,
        char: str,
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
    ) -> None:
        self.framebuffer.put(row, col, char, fg, bg, attr)

    def fill(
        self,
        char: str = " ",
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
    ) -> None:
        self.framebuffer.fill(char, fg, bg, attr)

    def fill_pattern(
        self,
        pattern: Pattern,
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
    ) -> None:
        self.framebuffer.fill_pattern(pattern, fg, bg, attr)

    def draw_box(
        self,
        row: int,
        col: int,
        width: int,
        height: int,
        style: BoxStyle = BoxStyle.SINGLE,
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
    ) -> None:
        self.framebuffer.draw_box(row, col, width, height, style, fg, bg, attr)

    def draw_rect(
        self,
        row: int,
        col: int,
        width: int,
        height: int,
        char: str = " ",
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
    ) -> None:
        self.framebuffer.draw_rect(row, col, width, height, char, fg, bg, attr)


# ── Font Configuration ────────────────────────────────────────────────


@dataclass
class FontConfig:
    """Terminal font/size configuration.

    Terminals don't expose direct font control via escape codes, but we
    can configure:
    - Character width multiplier (half-width vs full-width)
    - Line spacing (via newline padding)
    - Style hints for the renderer
    """

    size: int = 14
    family: str = "monospace"
    char_width: int = 1  # 1 = normal, 2 = double-width
    line_spacing: int = 0  # extra rows between lines
    bold_supported: bool = True
    italic_supported: bool = True
    underline_supported: bool = True
    strikethrough_supported: bool = True

    def cells_per_char(self) -> int:
        """Number of terminal cells per character."""
        return self.char_width


# ── Snapshot ───────────────────────────────────────────────────────────


@dataclass
class Snapshot:
    """A point-in-time capture of the graphics engine state."""

    framebuffer: Framebuffer
    cursor_row: int = 0
    cursor_col: int = 0
    timestamp: float = field(default_factory=time.time)


class SnapshotManager:
    """Manages undo/redo snapshots of the engine state."""

    def __init__(self, max_history: int = 50) -> None:
        self._max_history = max_history
        self._undo_stack: list[Snapshot] = []
        self._redo_stack: list[Snapshot] = []

    def save(self, fb: Framebuffer, cursor_row: int = 0, cursor_col: int = 0) -> None:
        """Save current state to undo stack."""
        self._undo_stack.append(Snapshot(fb.snapshot(), cursor_row, cursor_col))
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> Snapshot | None:
        """Pop and return the last saved state, pushing current to redo."""
        if not self._undo_stack:
            return None
        return self._undo_stack.pop()

    def redo(self) -> Snapshot | None:
        """Pop and return the last undone state."""
        if not self._redo_stack:
            return None
        return self._redo_stack.pop()

    def push_redo(self, snap: Snapshot) -> None:
        """Push a state onto the redo stack."""
        self._redo_stack.append(snap)

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0


# ── Tab/Focus Manager ─────────────────────────────────────────────────


@dataclass
class FocusRegion:
    """A focusable region in the terminal."""

    name: str
    row: int
    col: int
    width: int
    height: int
    tab_order: int = 0
    focused: bool = False
    on_focus: Callable[[], None] | None = None
    on_blur: Callable[[], None] | None = None


class TabManager:
    """Manages focus navigation between focusable regions."""

    def __init__(self) -> None:
        self._regions: list[FocusRegion] = []
        self._focus_idx: int = -1
        self._tab_key: str = "\t"
        self._shift_tab: str = "\x1b[Z"  # Shift+Tab escape

    def add_region(
        self,
        name: str,
        row: int,
        col: int,
        width: int,
        height: int,
        tab_order: int = 0,
        on_focus: Callable[[], None] | None = None,
        on_blur: Callable[[], None] | None = None,
    ) -> FocusRegion:
        region = FocusRegion(
            name, row, col, width, height, tab_order, on_focus=on_focus, on_blur=on_blur
        )
        self._regions.append(region)
        self._regions.sort(key=lambda r: r.tab_order)
        return region

    def remove_region(self, name: str) -> None:
        self._regions = [r for r in self._regions if r.name != name]

    def focus_next(self) -> FocusRegion | None:
        """Move focus to the next region."""
        if not self._regions:
            return None
        if self._focus_idx >= 0:
            old = self._regions[self._focus_idx]
            old.focused = False
            if old.on_blur:
                old.on_blur()
        self._focus_idx = (self._focus_idx + 1) % len(self._regions)
        new = self._regions[self._focus_idx]
        new.focused = True
        if new.on_focus:
            new.on_focus()
        return new

    def focus_prev(self) -> FocusRegion | None:
        """Move focus to the previous region."""
        if not self._regions:
            return None
        if self._focus_idx >= 0:
            old = self._regions[self._focus_idx]
            old.focused = False
            if old.on_blur:
                old.on_blur()
        self._focus_idx = (self._focus_idx - 1) % len(self._regions)
        new = self._regions[self._focus_idx]
        new.focused = True
        if new.on_focus:
            new.on_focus()
        return new

    def focus(self, name: str) -> FocusRegion | None:
        """Focus a specific region by name."""
        for i, r in enumerate(self._regions):
            if r.name == name:
                if self._focus_idx >= 0:
                    old = self._regions[self._focus_idx]
                    old.focused = False
                    if old.on_blur:
                        old.on_blur()
                self._focus_idx = i
                r.focused = True
                if r.on_focus:
                    r.on_focus()
                return r
        return None

    @property
    def focused(self) -> FocusRegion | None:
        if 0 <= self._focus_idx < len(self._regions):
            return self._regions[self._focus_idx]
        return None

    @property
    def regions(self) -> list[FocusRegion]:
        return list(self._regions)


# ── Display Mode ──────────────────────────────────────────────────────


class DisplayMode(Enum):
    """Display modes that affect how content is rendered."""

    NORMAL = "normal"
    REVERSE = "reverse"
    DIM = "dim"
    BOLD = "bold"
    UNDERLINE = "underline"
    HIGHLIGHT = "highlight"
    SELECTED = "selected"
    DISABLED = "disabled"
    INVERSE_VIDEO = "inverse_video"


_DISPLAY_MODE_ATTRS: dict[DisplayMode, int] = {
    DisplayMode.NORMAL: Attr.NONE,
    DisplayMode.REVERSE: Attr.REVERSE,
    DisplayMode.DIM: Attr.DIM,
    DisplayMode.BOLD: Attr.BOLD,
    DisplayMode.UNDERLINE: Attr.UNDERLINE,
    DisplayMode.HIGHLIGHT: Attr.BOLD,
    DisplayMode.SELECTED: Attr.REVERSE | Attr.BOLD,
    DisplayMode.DISABLED: Attr.DIM,
    DisplayMode.INVERSE_VIDEO: Attr.REVERSE,
}


def get_mode_attr(mode: DisplayMode) -> int:
    """Get the curses attribute for a display mode."""
    return _DISPLAY_MODE_ATTRS.get(mode, Attr.NONE)


# ── Graphics Engine ───────────────────────────────────────────────────


class GraphicsEngine:
    """Terminal graphics engine — compositing framebuffer renderer.

    Manages layers, compositing, cursor, and terminal I/O. Replaces
    curses-based rendering with a clean framebuffer model.
    """

    def __init__(self, font: FontConfig | None = None) -> None:
        self.font = font or FontConfig()
        self._layers: list[Layer] = []
        self._front_buffer: Framebuffer | None = None
        self._back_buffer: Framebuffer | None = None
        self._snapshot_mgr = SnapshotManager()
        self._tab_mgr = TabManager()
        self._cursor_row: int = 0
        self._cursor_col: int = 0
        self._cursor_visible: bool = True
        self._running: bool = False
        self._rows: int = 24
        self._cols: int = 80
        self._alt_screen: bool = False
        self._old_attrs: dict[str, Any] = {}
        self._dirty: bool = True
        self._lock = threading.Lock()
        self._key_handlers: dict[int | str, Callable[[int], bool]] = {}
        self._on_resize: Callable[[int, int], None] | None = None

    # ── Lifecycle ──────────────────────────────────────────────────

    def open(self) -> None:
        """Enter alternate screen buffer and initialize."""
        self._detect_size()
        self._front_buffer = Framebuffer(self._rows, self._cols)
        self._back_buffer = Framebuffer(self._rows, self._cols)
        self._alt_screen = True
        # Enter alternate screen + hide cursor + enable mouse
        sys.stdout.write("\x1b[?1049h")  # alt screen
        sys.stdout.write("\x1b[?25l")  # hide cursor
        sys.stdout.write("\x1b[?1003h")  # mouse tracking
        sys.stdout.write("\x1b[2J")  # clear screen
        sys.stdout.write("\x1b[H")  # home cursor
        sys.stdout.flush()
        self._running = True

    def close(self) -> None:
        """Restore terminal state."""
        self._running = False
        sys.stdout.write("\x1b[?1003l")  # disable mouse
        sys.stdout.write("\x1b[?25h")  # show cursor
        sys.stdout.write("\x1b[0m")  # reset attributes
        sys.stdout.write("\x1b[2J")  # clear
        sys.stdout.write("\x1b[H")  # home
        sys.stdout.write("\x1b[?1049l")  # leave alt screen
        sys.stdout.flush()
        self._alt_screen = False

    def _detect_size(self) -> tuple[int, int]:
        """Detect terminal size."""
        try:
            sz = os.get_terminal_size(0)
            self._rows = sz.lines
            self._cols = sz.columns
        except (ValueError, OSError):
            self._rows = 24
            self._cols = 80
        return self._rows, self._cols

    # ── Layers ─────────────────────────────────────────────────────

    def create_layer(
        self, name: str, z: int = 0, visible: bool = True, opacity: float = 1.0
    ) -> Layer:
        """Create and register a new rendering layer."""
        layer = Layer(name, self._rows, self._cols, z, visible, opacity)
        self._layers.append(layer)
        self._layers.sort(key=lambda l: l.z)
        self._dirty = True
        return layer

    def get_layer(self, name: str) -> Layer | None:
        for layer in self._layers:
            if layer.name == name:
                return layer
        return None

    def remove_layer(self, name: str) -> None:
        self._layers = [l for l in self._layers if l.name != name]
        self._dirty = True

    # ── Drawing Primitives ─────────────────────────────────────────

    def write(
        self,
        row: int,
        col: int,
        text: str,
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
        layer: str | None = None,
    ) -> int:
        """Write text to the specified layer (or default back buffer)."""
        target = self._get_layer_or_back(layer)
        return target.write(row, col, text, fg, bg, attr)

    def put(
        self,
        row: int,
        col: int,
        char: str,
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
        layer: str | None = None,
    ) -> None:
        target = self._get_layer_or_back(layer)
        target.put(row, col, char, fg, bg, attr)

    def fill(
        self,
        char: str = " ",
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
        layer: str | None = None,
    ) -> None:
        target = self._get_layer_or_back(layer)
        target.fill(char, fg, bg, attr)

    def fill_pattern(
        self,
        pattern: Pattern,
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
        layer: str | None = None,
    ) -> None:
        target = self._get_layer_or_back(layer)
        target.fill_pattern(pattern, fg, bg, attr)

    def draw_box(
        self,
        row: int,
        col: int,
        width: int,
        height: int,
        style: BoxStyle = BoxStyle.SINGLE,
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
        layer: str | None = None,
    ) -> None:
        target = self._get_layer_or_back(layer)
        target.draw_box(row, col, width, height, style, fg, bg, attr)

    def draw_rect(
        self,
        row: int,
        col: int,
        width: int,
        height: int,
        char: str = " ",
        fg: int = Color.DEFAULT,
        bg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
        layer: str | None = None,
    ) -> None:
        target = self._get_layer_or_back(layer)
        target.draw_rect(row, col, width, height, char, fg, bg, attr)

    def _get_layer_or_back(self, name: str | None) -> Layer | Framebuffer:
        if name is not None:
            layer = self.get_layer(name)
            if layer is not None:
                return layer
        assert self._back_buffer is not None
        return self._back_buffer

    # ── Rendering ──────────────────────────────────────────────────

    def render(self) -> None:
        """Composite all layers and render to terminal."""
        if self._back_buffer is None or self._front_buffer is None:
            return

        # Start with the back buffer
        self._front_buffer.restore(self._back_buffer)

        # Composite visible layers in z-order
        for layer in self._layers:
            if not layer.visible:
                continue
            self._front_buffer.composite(layer.framebuffer)

        # Diff against previous frame and emit minimal escape codes
        self._render_diff()
        self._dirty = False

    def _render_diff(self) -> None:
        """Render only changed cells to minimize terminal I/O."""
        if self._back_buffer is None or self._front_buffer is None:
            return
        changes = self._front_buffer.diff(self._back_buffer)
        if not changes:
            return

        # Group consecutive cells in same row for batch writes
        out: list[str] = []
        prev_row, prev_col = -1, -1
        last_fg, last_bg, last_attr = -1, -1, -1

        for row, col, cell in changes:
            # Move cursor if not contiguous
            if row != prev_row or col != prev_col + 1:
                out.append(f"\x1b[{row + 1};{col + 1}H")
            prev_row, prev_col = row, col

            # Emit attribute changes only when needed
            if cell.fg != last_fg or cell.bg != last_bg or cell.attr != last_attr:
                out.append(self._make_sgr(cell.fg, cell.bg, cell.attr))
                last_fg, last_bg, last_attr = cell.fg, cell.bg, cell.attr

            out.append(cell.char)

        # Final reset
        out.append("\x1b[0m")
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def _make_sgr(self, fg: int, bg: int, attr: int) -> str:
        """Build SGR (Select Graphic Rendition) escape sequence."""
        parts: list[str] = ["\x1b["]
        codes: list[str] = []

        if attr & Attr.BOLD:
            codes.append("1")
        if attr & Attr.DIM:
            codes.append("2")
        if attr & Attr.ITALIC:
            codes.append("3")
        if attr & Attr.UNDERLINE:
            codes.append("4")
        if attr & Attr.BLINK:
            codes.append("5")
        if attr & Attr.REVERSE:
            codes.append("7")
        if attr & Attr.HIDDEN:
            codes.append("8")
        if attr & Attr.STRIKETHROUGH:
            codes.append("9")

        # Foreground
        if fg != Color.DEFAULT:
            if fg < 8:
                codes.append(str(30 + fg))
            elif fg < 16:
                codes.append(str(90 + fg - 8))
            else:
                codes.extend(["38", "5", str(fg)])

        # Background
        if bg != Color.DEFAULT:
            if bg < 8:
                codes.append(str(40 + bg))
            elif bg < 16:
                codes.append(str(100 + bg - 8))
            else:
                codes.extend(["48", "5", str(bg)])

        parts.append(";".join(codes))
        parts.append("m")
        return "".join(parts)

    # ── Cursor ─────────────────────────────────────────────────────

    def set_cursor(self, row: int, col: int) -> None:
        self._cursor_row = row
        self._cursor_col = col
        sys.stdout.write(f"\x1b[{row + 1};{col + 1}H")
        sys.stdout.flush()

    def show_cursor(self, visible: bool = True) -> None:
        self._cursor_visible = visible
        sys.stdout.write("\x1b[?25h" if visible else "\x1b[?25l")
        sys.stdout.flush()

    # ── Snapshots (Undo/Redo) ─────────────────────────────────────

    def save_snapshot(self) -> None:
        """Save current state for undo."""
        if self._back_buffer is not None:
            self._snapshot_mgr.save(self._back_buffer, self._cursor_row, self._cursor_col)

    def undo(self) -> bool:
        """Restore the previous state. Returns True if successful."""
        snap = self._snapshot_mgr.undo()
        if snap is None:
            return False
        if self._back_buffer is not None:
            # Save current to redo
            self._snapshot_mgr.push_redo(
                Snapshot(self._back_buffer.snapshot(), self._cursor_row, self._cursor_col)
            )
            self._back_buffer.restore(snap.framebuffer)
            self._cursor_row = snap.cursor_row
            self._cursor_col = snap.cursor_col
            self._dirty = True
            return True
        return False

    def redo(self) -> bool:
        """Restore the next state. Returns True if successful."""
        snap = self._snapshot_mgr.redo()
        if snap is None:
            return False
        if self._back_buffer is not None:
            self._snapshot_mgr.push_redo(
                Snapshot(self._back_buffer.snapshot(), self._cursor_row, self._cursor_col)
            )
            self._back_buffer.restore(snap.framebuffer)
            self._cursor_row = snap.cursor_row
            self._cursor_col = snap.cursor_col
            self._dirty = True
            return True
        return False

    # ── Tab/Focus ──────────────────────────────────────────────────

    def add_focus_region(
        self,
        name: str,
        row: int,
        col: int,
        width: int,
        height: int,
        tab_order: int = 0,
        on_focus: Callable[[], None] | None = None,
        on_blur: Callable[[], None] | None = None,
    ) -> FocusRegion:
        return self._tab_mgr.add_region(name, row, col, width, height, tab_order, on_focus, on_blur)

    def focus_next(self) -> FocusRegion | None:
        return self._tab_mgr.focus_next()

    def focus_prev(self) -> FocusRegion | None:
        return self._tab_mgr.focus_prev()

    def focused_region(self) -> FocusRegion | None:
        return self._tab_mgr.focused

    # ── Input ──────────────────────────────────────────────────────

    def poll_key(self, timeout_ms: int = 100) -> int | None:
        """Poll for a key press. Returns key code or None."""
        import select

        r, _, _ = select.select([sys.stdin], [], [], timeout_ms / 1000.0)
        if r:
            return ord(sys.stdin.read(1))
        return None

    def on_key(self, key: int | str, handler: Callable[[int], bool]) -> None:
        """Register a key handler. Handler returns True to consume the key."""
        self._key_handlers[key] = handler

    # ── Resize ─────────────────────────────────────────────────────

    def handle_resize(self) -> None:
        """Handle terminal resize."""
        old_rows, old_cols = self._rows, self._cols
        self._detect_size()
        if self._rows != old_rows or self._cols != old_cols:
            if self._back_buffer is not None:
                self._back_buffer.resize(self._rows, self._cols)
            if self._front_buffer is not None:
                self._front_buffer.resize(self._rows, self._cols)
            for layer in self._layers:
                layer.resize(self._rows, self._cols)
            if self._on_resize:
                self._on_resize(self._rows, self._cols)
            self._dirty = True

    def set_resize_handler(self, handler: Callable[[int, int], None]) -> None:
        self._on_resize = handler

    # ── Hardware Rendering ──────────────────────────────────────────

    def capture_hardware_state(self) -> HardwareState:
        """Capture the current terminal state as a HardwareState.

        This records what the terminal *should* be displaying based on
        all rendered frames. Use compare_hardware_states() to diff two
        captures and detect rendering drift.
        """
        if self._front_buffer is None:
            return HardwareState(rows=self._rows, cols=self._cols)
        return HardwareState(
            rows=self._rows,
            cols=self._cols,
            cells=[
                [self._front_buffer.get(r, c) for c in range(self._cols)] for r in range(self._rows)
            ],
            cursor_row=self._cursor_row,
            cursor_col=self._cursor_col,
            cursor_visible=self._cursor_visible,
            timestamp=time.time(),
        )

    def render_to_hardware(self) -> HardwareOutput:
        """Render the current frame and capture the raw terminal bytes.

        Returns a HardwareOutput containing the exact byte sequence that
        was written to stdout, suitable for replay or comparison.
        """
        if self._back_buffer is None or self._front_buffer is None:
            return HardwareOutput()

        # Build the raw output
        raw_bytes = bytearray()
        raw_bytes.extend(b"\x1b[?1049h")  # alt screen
        raw_bytes.extend(b"\x1b[2J")  # clear
        raw_bytes.extend(b"\x1b[H")  # home

        # Render the front buffer as raw escape sequences
        for row in range(self._rows):
            for col in range(self._cols):
                cell = self._front_buffer.get(row, col)
                # Move cursor
                raw_bytes.extend(f"\x1b[{row + 1};{col + 1}H".encode())
                # Set attributes
                sgr = self._make_sgr(cell.fg, cell.bg, cell.attr)
                raw_bytes.extend(sgr.encode())
                # Write character
                raw_bytes.extend(cell.char.encode("utf-8", errors="replace"))

        raw_bytes.extend(b"\x1b[0m")  # reset

        return HardwareOutput(
            raw_bytes=bytes(raw_bytes),
            rows=self._rows,
            cols=self._cols,
            timestamp=time.time(),
        )


# ── Hardware State & Comparison ───────────────────────────────────────


@dataclass
class HardwareState:
    """A captured snapshot of the terminal's logical state.

    Used for comparing what the engine *thinks* is on screen vs what
    was actually rendered, to detect rendering bugs or drift.
    """

    rows: int = 24
    cols: int = 80
    cells: list[list[Cell]] = field(default_factory=list)
    cursor_row: int = 0
    cursor_col: int = 0
    cursor_visible: bool = True
    timestamp: float = field(default_factory=time.time)

    def get_cell(self, row: int, col: int) -> Cell:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            if row < len(self.cells) and col < len(self.cells[row]):
                return self.cells[row][col]
        return Cell()

    def text_at(self, row: int, col: int, length: int) -> str:
        """Read a string of text from a row starting at col."""
        result = []
        for c in range(col, min(col + length, self.cols)):
            cell = self.get_cell(row, c)
            if cell.char:
                result.append(cell.char)
        return "".join(result)


@dataclass
class CellDiff:
    """A single cell difference between two hardware states."""

    row: int
    col: int
    expected: Cell
    actual: Cell

    def __str__(self) -> str:
        return (
            f"({self.row},{self.col}): "
            f"expected=({self.expected.char!r}, fg={self.expected.fg}, "
            f"bg={self.expected.bg}, attr={self.expected.attr}) "
            f"actual=({self.actual.char!r}, fg={self.actual.fg}, "
            f"bg={self.actual.bg}, attr={self.actual.attr})"
        )


@dataclass
class HardwareDiff:
    """Result of comparing two hardware states."""

    identical: bool = True
    cell_diffs: list[CellDiff] = field(default_factory=list)
    cursor_diff: bool = False
    size_diff: bool = False
    expected_rows: int = 0
    expected_cols: int = 0
    actual_rows: int = 0
    actual_cols: int = 0
    summary: str = ""

    @property
    def diff_count(self) -> int:
        return len(self.cell_diffs) + (1 if self.cursor_diff else 0) + (1 if self.size_diff else 0)


@dataclass
class HardwareOutput:
    """Raw terminal output bytes from a render pass."""

    raw_bytes: bytes = b""
    rows: int = 0
    cols: int = 0
    timestamp: float = field(default_factory=time.time)

    def replay(self) -> None:
        """Replay the raw output to stdout."""
        sys.stdout.buffer.write(self.raw_bytes)
        sys.stdout.buffer.flush()

    def save(self, path: str) -> None:
        """Save raw output to a file for later replay."""
        with open(path, "wb") as f:
            f.write(self.raw_bytes)

    @classmethod
    def load(cls, path: str) -> HardwareOutput:
        """Load a saved output file."""
        with open(path, "rb") as f:
            raw = f.read()
        return cls(raw_bytes=raw)


def compare_hardware_states(expected: HardwareState, actual: HardwareState) -> HardwareDiff:
    """Compare two hardware states and return detailed differences.

    This is the core comparison function for detecting rendering bugs.
    It compares every cell's character, fg, bg, and attributes, plus
    cursor position and terminal size.
    """
    diff = HardwareDiff(
        identical=True,
        expected_rows=expected.rows,
        expected_cols=expected.cols,
        actual_rows=actual.rows,
        actual_cols=actual.cols,
    )

    # Size mismatch
    if expected.rows != actual.rows or expected.cols != actual.cols:
        diff.size_diff = True
        diff.identical = False
        diff.summary = (
            f"Size mismatch: expected {expected.rows}x{expected.cols}, "
            f"got {actual.rows}x{actual.cols}"
        )

    # Cursor mismatch
    if (
        expected.cursor_row != actual.cursor_row
        or expected.cursor_col != actual.cursor_col
        or expected.cursor_visible != actual.cursor_visible
    ):
        diff.cursor_diff = True
        diff.identical = False

    # Cell-by-cell comparison
    rows = min(expected.rows, actual.rows)
    cols = min(expected.cols, actual.cols)
    for r in range(rows):
        for c in range(cols):
            exp_cell = expected.get_cell(r, c)
            act_cell = actual.get_cell(r, c)
            if exp_cell != act_cell:
                diff.cell_diffs.append(CellDiff(r, c, exp_cell, act_cell))
                diff.identical = False

    if diff.identical:
        diff.summary = "States are identical"
    elif not diff.cell_diffs:
        diff.summary = f"Cursor/size only: {diff.diff_count} metadata diffs"
    else:
        diff.summary = (
            f"{len(diff.cell_diffs)} cell diffs, "
            f"cursor={'differs' if diff.cursor_diff else 'ok'}, "
            f"size={'differs' if diff.size_diff else 'ok'}"
        )

    return diff


def compare_hardware_outputs(a: HardwareOutput, b: HardwareOutput) -> dict:
    """Compare two raw hardware outputs byte-by-byte.

    Returns a dict with:
      - identical: bool
      - byte_diff_count: int
      - first_diff_offset: int (or -1 if identical)
      - size_a, size_b: int
    """
    identical = a.raw_bytes == b.raw_bytes
    first_diff = -1
    if not identical:
        for i in range(min(len(a.raw_bytes), len(b.raw_bytes))):
            if a.raw_bytes[i] != b.raw_bytes[i]:
                first_diff = i
                break
        if first_diff == -1 and len(a.raw_bytes) != len(b.raw_bytes):
            first_diff = min(len(a.raw_bytes), len(b.raw_bytes))

    return {
        "identical": identical,
        "byte_diff_count": sum(1 for x, y in zip(a.raw_bytes, b.raw_bytes) if x != y),
        "first_diff_offset": first_diff,
        "size_a": len(a.raw_bytes),
        "size_b": len(b.raw_bytes),
    }
