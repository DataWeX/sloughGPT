"""
widgets — text-only reactive widget system for the shell TUI.

Zero curses dependency.  Everything renders as plain text lines with
box-drawing characters for borders.  When mature, the display layer
swaps text borders for real TUI elements.

Architecture (learned from tui_repl/pane/surface):

  Widget tree  →  compute layout (rows/cols)  →  render to string lines
       ↑                                            ↓
  event bus  ←  key/resize/focus events      stdout (print)

Widget lifecycle:
  1. mount(parent)    — attach to tree, subscribe to events
  2. compute(w, h)    — claim rows/cols from available space
  3. render()         — return list[str] (one per row)
  4. handle(key)      — process input, return True if consumed
  5. unmount()        — detach from tree

Reactive state:
  widget.value = x  →  _dirty = True  →  next render() picks up change
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ── Box-drawing characters (text placeholders, swap for real later) ────

class Box:
    H = "\u2500"   # ─
    V = "\u2502"   # │
    TL = "\u250c"  # ┌
    TR = "\u2510"  # ┐
    BL = "\u2514"  # └
    BR = "\u2518"  # ┘
    T = "\u252c"   # ┬
    B = "\u2534"   # ┴
    L = "\u251c"   # ├
    R = "\u2524"   # ┤
    X = "\u253c"   # ┼
    # Double
    DH = "\u2550"  # ═
    DV = "\u2551"  # ║
    DTL = "\u2554" # ╔
    DTR = "\u2557" # ╗
    DBL = "\u255a" # ╚
    DBR = "\u255d" # ╝
    # Shorthand
    DASH = "-"
    PIPE = "|"
    DOT = "\u00b7"  # ·
    ARROW_R = "\u25b6"  # ▶
    ARROW_D = "\u25bc"  # ▼
    CHECK = "\u2713"    # ✓
    CROSS = "\u2717"    # ✗
    BULLET = "\u2022"   # •


# ── Event types ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class KeyEvent:
    key: str          # "a", "enter", "esc", "tab", "up", "down", "ctrl+c", etc.
    raw: int = 0      # raw byte value (for debug)

@dataclass(frozen=True)
class ResizeEvent:
    rows: int
    cols: int

@dataclass(frozen=True)
class FocusEvent:
    gained: bool

Event = KeyEvent | ResizeEvent | FocusEvent


# ── Event bus ─────────────────────────────────────────────────────────

class EventBus:
    """Pub-sub event bus.  Widgets subscribe to event types."""

    def __init__(self) -> None:
        self._subscribers: dict[type, list[Callable]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: type, callback: Callable) -> None:
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: type, callback: Callable) -> None:
        with self._lock:
            subs = self._subscribers.get(event_type, [])
            if callback in subs:
                subs.remove(callback)

    def publish(self, event: Event) -> None:
        with self._lock:
            subs = list(self._subscribers.get(type(event), []))
        for cb in subs:
            cb(event)


# ── Widget base ───────────────────────────────────────────────────────

class Widget:
    """Base class for all widgets.

    Subclasses override:
      - compute(w, h) → (used_rows, used_cols)  — how much space this widget needs
      - render() → list[str]  — the text lines to display
      - handle(key) → bool   — True if key was consumed
    """

    def __init__(self, name: str = "", visible: bool = True) -> None:
        self.name = name
        self.visible = visible
        self._parent: Widget | None = None
        self._children: list[Widget] = []
        self._dirty = True
        self._focused = False
        self._event_bus: EventBus | None = None
        self._computed_rows = 0
        self._computed_cols = 0
        # Where this widget sits on screen (set by parent during layout)
        self._screen_top = 0
        self._screen_left = 0

    # ── Tree ──────────────────────────────────────────────────────────

    def mount(self, parent: Widget | None = None, event_bus: EventBus | None = None) -> None:
        self._parent = parent
        if parent:
            self._event_bus = event_bus or parent._event_bus
        if self._event_bus:
            self._event_bus.subscribe(KeyEvent, self._on_key)
            self._event_bus.subscribe(ResizeEvent, self._on_resize)
        if parent:
            parent._children.append(self)
            parent._dirty = True

    def unmount(self) -> None:
        if self._event_bus:
            self._event_bus.unsubscribe(KeyEvent, self._on_key)
            self._event_bus.unsubscribe(ResizeEvent, self._on_resize)
        if self._parent:
            self._parent._children.remove(self)
            self._parent._dirty = True
        self._parent = None

    @property
    def root(self) -> Widget:
        w = self
        while w._parent:
            w = w._parent
        return w

    # ── Focus ─────────────────────────────────────────────────────────

    @property
    def focused(self) -> bool:
        return self._focused

    def focus(self) -> None:
        self._focused = True
        self._dirty = True
        if self._event_bus:
            self._event_bus.publish(FocusEvent(gained=True))

    def blur(self) -> None:
        self._focused = False
        self._dirty = True
        if self._event_bus:
            self._event_bus.publish(FocusEvent(gained=False))

    # ── Reactive state ────────────────────────────────────────────────

    def invalidate(self) -> None:
        """Mark this widget as needing re-render."""
        self._dirty = True

    def mark_clean(self) -> None:
        self._dirty = False

    # ── Layout (override in subclasses) ───────────────────────────────

    def compute(self, rows: int, cols: int) -> tuple[int, int]:
        """Compute layout.  Returns (used_rows, used_cols)."""
        self._computed_rows = rows
        self._computed_cols = cols
        return rows, cols

    # ── Render (override in subclasses) ───────────────────────────────

    def render(self) -> list[str]:
        """Return list of text lines (one per row)."""
        self._dirty = False
        return [""]

    def render_to_string(self) -> str:
        """Full render as a single string."""
        return "\n".join(self.render())

    # ── Input (override in subclasses) ────────────────────────────────

    def handle(self, key: KeyEvent) -> bool:
        """Handle a key event.  Return True if consumed."""
        return False

    # ── Internal event routing ────────────────────────────────────────

    def _on_key(self, event: KeyEvent) -> None:
        if self._focused:
            self.handle(event)

    def _on_resize(self, event: ResizeEvent) -> None:
        self._dirty = True

    # ── Helpers ───────────────────────────────────────────────────────

    def _pad_line(self, text: str, width: int, char: str = " ") -> str:
        """Pad or truncate text to width."""
        if len(text) >= width:
            return text[:width]
        return text + char * (width - len(text))

    def _center(self, text: str, width: int, char: str = " ") -> str:
        """Center text within width."""
        if len(text) >= width:
            return text[:width]
        total_pad = width - len(text)
        left = total_pad // 2
        right = total_pad - left
        return char * left + text + char * right

    def _hline(self, width: int, ch: str = Box.H) -> str:
        """Horizontal line of given width."""
        return ch * max(width, 0)

    def _truncate(self, text: str, max_width: int) -> str:
        """Truncate with ellipsis if too long."""
        if len(text) <= max_width:
            return text
        if max_width <= 3:
            return text[:max_width]
        return text[:max_width - 1] + "\u2026"


# ── Container ─────────────────────────────────────────────────────────

class Container(Widget):
    """Base for widgets that hold children."""

    def __init__(self, name: str = "", direction: str = "vertical",
                 gap: int = 0, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.direction = direction  # "vertical" or "horizontal"
        self.gap = gap

    def add(self, child: Widget) -> Widget:
        child.mount(parent=self)
        return child

    def remove(self, child: Widget) -> None:
        child.unmount()

    def compute(self, rows: int, cols: int) -> tuple[int, int]:
        self._computed_rows = rows
        self._computed_cols = cols
        if self.direction == "vertical":
            return self._layout_vertical(rows, cols)
        else:
            return self._layout_horizontal(rows, cols)

    def _layout_vertical(self, rows: int, cols: int) -> tuple[int, int]:
        visible = [c for c in self._children if c.visible]
        if not visible:
            return 0, cols

        total_gap = self.gap * max(len(visible) - 1, 0)
        available = max(rows - total_gap, 0)

        # First pass: discover natural heights (only for content-sized widgets)
        natural = []
        for child in visible:
            if hasattr(child, '_natural_height'):
                natural.append(child._natural_height())
            else:
                natural.append(0)  # 0 = use equal distribution
        total_natural = sum(natural)

        # Check if all children are content-sized
        all_content = all(n > 0 for n in natural)

        if all_content and total_natural <= available:
            # Content-sized: use natural heights directly
            heights = natural[:]
        else:
            # Default: equal distribution
            heights = []
            remaining = available
            for i in range(len(visible)):
                if i == len(visible) - 1:
                    heights.append(max(remaining, 0))
                else:
                    share = max(1, available // len(visible))
                    heights.append(min(share, remaining))
                    remaining -= heights[-1]

        # Second pass: re-compute with allocated heights
        y = 0
        used_cols = 0
        for i, child in enumerate(visible):
            child._screen_top = self._screen_top + y
            child._screen_left = self._screen_left
            child.compute(heights[i], cols)
            y += child._computed_rows + self.gap
            used_cols = max(used_cols, child._computed_cols)

        return min(y - self.gap, rows) if visible else 0, used_cols

    def _layout_horizontal(self, rows: int, cols: int) -> tuple[int, int]:
        visible = [c for c in self._children if c.visible]
        if not visible:
            return rows, 0

        total_gap = self.gap * max(len(visible) - 1, 0)
        available = max(cols - total_gap, 0)
        x = 0
        used_rows = 0

        for child in visible:
            child_w = max(available // len(visible), 0)
            if child is visible[-1]:
                child_w = max(available - x + total_gap, 0) if x < available else 0
            child._screen_top = self._screen_top
            child._screen_left = self._screen_left + x
            child.compute(rows, child_w)
            x += child._computed_cols + self.gap
            used_rows = max(used_rows, child._computed_rows)

        return used_rows, min(x - self.gap, cols) if visible else 0

    def render(self) -> list[str]:
        visible = [c for c in self._children if c.visible]
        if not visible:
            return []

        if self.direction == "vertical":
            lines = []
            for child in visible:
                child_lines = child.render()
                for line in child_lines:
                    lines.append(self._pad_line(line, self._computed_cols))
                if self.gap > 0 and child is not visible[-1]:
                    for _ in range(self.gap):
                        lines.append(" " * self._computed_cols)
            # Pad to computed rows
            while len(lines) < self._computed_rows:
                lines.append(" " * self._computed_cols)
            self._dirty = False
            return lines[:self._computed_rows] if lines else [""]

        else:  # horizontal
            child_renders = [c.render() for c in visible]
            max_h = max((len(r) for r in child_renders), default=0)
            lines = []
            for row_idx in range(max_h):
                parts = []
                for ci, child in enumerate(visible):
                    cr = child_renders[ci]
                    if row_idx < len(cr):
                        parts.append(self._pad_line(cr[row_idx], child._computed_cols))
                    else:
                        parts.append(" " * child._computed_cols)
                    if self.gap > 0 and ci < len(visible) - 1:
                        parts.append(" " * self.gap)
                lines.append("".join(parts)[:self._computed_cols])
            # Pad to computed rows
            while len(lines) < self._computed_rows:
                lines.append(" " * self._computed_cols)
            self._dirty = False
            return lines[:self._computed_rows] if lines else [""]

    def handle(self, key: KeyEvent) -> bool:
        for child in self._children:
            if child.visible and child._focused and child.handle(key):
                return True
        return False


# ── Panel ─────────────────────────────────────────────────────────────

class Panel(Widget):
    """Bordered box with title.  Text borders — replaced with real TUI later."""

    def __init__(self, title: str = "", child: Widget | None = None,
                 border: bool = True, padding: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.title = title
        self.child = child
        self.border = border
        self.padding = padding
        if child:
            child.mount(parent=self)

    def set_child(self, child: Widget) -> None:
        if self.child:
            self.child.unmount()
        self.child = child
        child.mount(parent=self)

    def compute(self, rows: int, cols: int) -> tuple[int, int]:
        self._computed_rows = rows
        self._computed_cols = cols
        if self.child and self.child.visible:
            b = 1 if self.border else 0
            p = self.padding if self.border else 0
            inner_rows = max(rows - 2 * b - 2 * p, 0)
            inner_cols = max(cols - 2 * b - 2 * p, 0)
            self.child._screen_top = self._screen_top + b + p
            self.child._screen_left = self._screen_left + b + p
            self.child.compute(inner_rows, inner_cols)
        return rows, cols

    def render(self) -> list[str]:
        w = self._computed_cols
        h = self._computed_rows
        lines = []

        if self.border:
            # Top border with title
            if self.title:
                title_str = f" {self.title} "
                avail = max(w - 2, 0)
                if len(title_str) < avail:
                    left_h = (avail - len(title_str)) // 2
                    right_h = avail - len(title_str) - left_h
                    top = Box.TL + Box.H * left_h + title_str + Box.H * right_h + Box.TR
                else:
                    top = Box.TL + self._truncate(title_str, avail) + Box.TR
            else:
                top = Box.TL + Box.H * max(w - 2, 0) + Box.TR
            lines.append(top[:w])

            # Child content with side borders
            child_lines = self.child.render() if self.child and self.child.visible else []
            p = self.padding
            content_rows = max(h - 2, 0)

            # Padding top
            for _ in range(p):
                lines.append(Box.V + " " * max(w - 2, 0) + Box.V)

            for i in range(content_rows - 2 * p):
                if i < len(child_lines):
                    content = self._pad_line(child_lines[i], max(w - 2, 0))
                else:
                    content = " " * max(w - 2, 0)
                lines.append(Box.V + content + Box.V)

            # Padding bottom
            for _ in range(p):
                lines.append(Box.V + " " * max(w - 2, 0) + Box.V)

            # Bottom border
            lines.append(Box.BL + Box.H * max(w - 2, 0) + Box.BR)
        else:
            child_lines = self.child.render() if self.child and self.child.visible else []
            for line in child_lines:
                lines.append(self._pad_line(line, w))
            # Fill remaining rows
            while len(lines) < h:
                lines.append(" " * w)

        self._dirty = False
        return lines[:h] if lines else [""]


# ── Text ──────────────────────────────────────────────────────────────

class Text(Widget):
    """Static or dynamic text display."""

    def __init__(self, content: str = "", align: str = "left", **kwargs) -> None:
        super().__init__(**kwargs)
        self._content = content
        self.align = align  # "left", "center", "right"

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        if self._content != value:
            self._content = value
            self._dirty = True

    def _natural_height(self) -> int:
        return len(self._content.split("\n"))

    def compute(self, rows: int, cols: int) -> tuple[int, int]:
        self._computed_rows = rows
        self._computed_cols = cols
        return rows, cols

    def render(self) -> list[str]:
        lines = self._content.split("\n")
        result = []
        for line in lines:
            if self.align == "center":
                result.append(self._center(line, self._computed_cols))
            elif self.align == "right":
                result.append(line.rjust(self._computed_cols))
            else:
                result.append(self._pad_line(line, self._computed_cols))
        # Pad to computed rows
        while len(result) < self._computed_rows:
            result.append(" " * self._computed_cols)
        self._dirty = False
        return result[:self._computed_rows]


# ── Button ────────────────────────────────────────────────────────────

class Button(Widget):
    """Clickable button with label."""

    def __init__(self, label: str = "", on_click: Callable | None = None,
                 style: str = "default", **kwargs) -> None:
        super().__init__(**kwargs)
        self.label = label
        self.on_click = on_click
        self.style = style  # "default", "primary", "danger"
        self._pressed = False

    def handle(self, key: KeyEvent) -> bool:
        if key.key == "enter":
            self._pressed = True
            self._dirty = True
            if self.on_click:
                self.on_click()
            return True
        return False

    def compute(self, rows: int, cols: int) -> tuple[int, int]:
        self._computed_rows = rows
        self._computed_cols = cols
        return rows, cols

    def render(self) -> list[str]:
        if self._computed_rows == 0:
            self._dirty = False
            return []
        w = self._computed_cols
        label = f"[ {self.label} ]"
        if self._focused:
            label = f"> {self.label} <"
        line = self._center(label, w)
        self._dirty = False
        return [line]


# ── Input ─────────────────────────────────────────────────────────────

class Input(Widget):
    """Text input field with cursor."""

    def __init__(self, prompt: str = "", default: str = "",
                 on_submit: Callable | None = None, password: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.prompt = prompt
        self._value = default
        self.on_submit = on_submit
        self.password = password
        self._cursor = len(default)
        self._scroll_offset = 0

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, v: str) -> None:
        if self._value != v:
            self._value = v
            self._cursor = len(v)
            self._dirty = True

    def compute(self, rows: int, cols: int) -> tuple[int, int]:
        self._computed_rows = max(rows, 1)
        self._computed_cols = cols
        return self._computed_rows, self._computed_cols

    def render(self) -> list[str]:
        w = self._computed_cols
        prompt_w = len(self.prompt) + 1  # prompt + space
        display_w = max(w - prompt_w, 0)

        # Build display text
        if self.password:
            display = "\u2022" * len(self._value)
        else:
            display = self._value

        # Handle scroll if display is wider than available space
        vis_start = self._scroll_offset
        vis_end = min(vis_start + display_w, len(display))
        visible = display[vis_start:vis_end]

        # Ensure cursor is visible
        cursor_pos = self._cursor - vis_start
        if cursor_pos < 0:
            self._scroll_offset = max(self._cursor - display_w // 2, 0)
            vis_start = self._scroll_offset
            vis_end = min(vis_start + display_w, len(display))
            visible = display[vis_start:vis_end]
            cursor_pos = self._cursor - vis_start
        elif cursor_pos >= display_w:
            self._scroll_offset = max(self._cursor - display_w + 1, 0)
            vis_start = self._scroll_offset
            vis_end = min(vis_start + display_w, len(display))
            visible = display[vis_start:vis_end]
            cursor_pos = self._cursor - vis_start

        # Pad visible to display_w
        visible = self._pad_line(visible, display_w)

        # Cursor indicator
        cursor_char = Box.V if self._focused else "|"

        line = self.prompt + " " + visible
        self._dirty = False
        return [line[:w]]

    def handle(self, key: KeyEvent) -> bool:
        if key.key == "enter":
            if self.on_submit:
                self.on_submit(self._value)
            return True
        elif key.key == "backspace":
            if self._cursor > 0:
                self._value = self._value[:self._cursor - 1] + self._value[self._cursor:]
                self._cursor -= 1
                self._dirty = True
            return True
        elif key.key == "delete":
            if self._cursor < len(self._value):
                self._value = self._value[:self._cursor] + self._value[self._cursor + 1:]
                self._dirty = True
            return True
        elif key.key == "left":
            if self._cursor > 0:
                self._cursor -= 1
                self._dirty = True
            return True
        elif key.key == "right":
            if self._cursor < len(self._value):
                self._cursor += 1
                self._dirty = True
            return True
        elif key.key == "home":
            self._cursor = 0
            self._dirty = True
            return True
        elif key.key == "end":
            self._cursor = len(self._value)
            self._dirty = True
            return True
        elif len(key.key) == 1 and key.key.isprintable():
            self._value = self._value[:self._cursor] + key.key + self._value[self._cursor:]
            self._cursor += 1
            self._dirty = True
            return True
        return False


# ── List ──────────────────────────────────────────────────────────────

class List(Widget):
    """Scrollable list with selectable items."""

    def __init__(self, items: list[str] | None = None, on_select: Callable | None = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._items: list[str] = items or []
        self._selected = 0
        self._scroll = 0
        self.on_select = on_select
        self._filter = ""

    @property
    def items(self) -> list[str]:
        return self._items

    @items.setter
    def items(self, value: list[str]) -> None:
        self._items = value
        self._selected = 0
        self._scroll = 0
        self._dirty = True

    @property
    def selected(self) -> int:
        return self._selected

    @property
    def selected_item(self) -> str | None:
        if 0 <= self._selected < len(self._items):
            return self._items[self._selected]
        return None

    @property
    def filtered_items(self) -> list[str]:
        if not self._filter:
            return self._items
        f = self._filter.lower()
        return [i for i in self._items if f in i.lower()]

    def compute(self, rows: int, cols: int) -> tuple[int, int]:
        self._computed_rows = rows
        self._computed_cols = cols
        return rows, cols

    def render(self) -> list[str]:
        w = self._computed_cols
        h = self._computed_rows
        filtered = self.filtered_items
        lines = []

        # Filter bar
        if self._filter:
            lines.append(f"  Filter: {self._filter}_")
            h -= 1

        max_visible = max(h, 0)

        for i in range(max_visible):
            idx = self._scroll + i
            if idx >= len(filtered):
                break
            prefix = f" {Box.ARROW_R} " if idx == self._selected else "   "
            text = filtered[idx]
            line = self._pad_line(prefix + text, w)
            lines.append(line)

        # Scroll indicator
        if len(filtered) > max_visible:
            lines.append(f"   {len(filtered)} items ({self._selected + 1}/{len(filtered)})")

        # Pad remaining rows
        while len(lines) < h:
            lines.append(" " * w)

        self._dirty = False
        return lines[:h] if lines else [""]

    def handle(self, key: KeyEvent) -> bool:
        filtered = self.filtered_items
        max_visible = self._computed_rows - (1 if self._filter else 0)

        if key.key == "up":
            if self._selected > 0:
                self._selected -= 1
                if self._selected < self._scroll:
                    self._scroll = self._selected
                self._dirty = True
            return True
        elif key.key == "down":
            if self._selected < len(filtered) - 1:
                self._selected += 1
                if self._selected >= self._scroll + max_visible:
                    self._scroll = self._selected - max_visible + 1
                self._dirty = True
            return True
        elif key.key == "page_up":
            self._selected = max(0, self._selected - max_visible)
            self._scroll = max(0, self._scroll - max_visible)
            self._dirty = True
            return True
        elif key.key == "page_down":
            self._selected = min(len(filtered) - 1, self._selected + max_visible)
            if self._selected >= self._scroll + max_visible:
                self._scroll = self._selected - max_visible + 1
            self._dirty = True
            return True
        elif key.key == "enter":
            if self.on_select and 0 <= self._selected < len(filtered):
                self.on_select(filtered[self._selected])
            return True
        elif key.key == "backspace":
            if self._filter:
                self._filter = self._filter[:-1]
                self._selected = 0
                self._scroll = 0
                self._dirty = True
            return True
        elif len(key.key) == 1 and key.key.isprintable():
            self._filter += key.key
            self._selected = 0
            self._scroll = 0
            self._dirty = True
            return True
        return False


# ── Menu ──────────────────────────────────────────────────────────────

class Menu(Widget):
    """Dropdown-style menu with options."""

    def __init__(self, title: str = "", options: list[str] | None = None,
                 on_select: Callable | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.title = title
        self._options: list[str] = options or []
        self._selected = 0
        self._open = False
        self.on_select = on_select

    @property
    def options(self) -> list[str]:
        return self._options

    @options.setter
    def options(self, value: list[str]) -> None:
        self._options = value
        self._selected = 0
        self._dirty = True

    def toggle(self) -> None:
        self._open = not self._open
        self._dirty = True

    def compute(self, rows: int, cols: int) -> tuple[int, int]:
        self._computed_rows = rows
        self._computed_cols = cols
        return rows, cols

    def render(self) -> list[str]:
        w = self._computed_cols
        lines = []

        # Header
        label = self.title or "Menu"
        header = f" {Box.ARROW_D} {label} " if self._open else f" {label} "
        lines.append(self._pad_line(header, w))

        if self._open:
            lines.append(self._hline(w, Box.H))
            for i, opt in enumerate(self._options):
                prefix = f" {Box.ARROW_R} " if i == self._selected else "   "
                lines.append(self._pad_line(prefix + opt, w))

        self._dirty = False
        return lines

    def handle(self, key: KeyEvent) -> bool:
        if not self._open:
            if key.key == "enter" or key.key == " ":
                self._open = True
                self._dirty = True
                return True
            return False

        if key.key == "esc":
            self._open = False
            self._dirty = True
            return True
        elif key.key == "up":
            if self._selected > 0:
                self._selected -= 1
                self._dirty = True
            return True
        elif key.key == "down":
            if self._selected < len(self._options) - 1:
                self._selected += 1
                self._dirty = True
            return True
        elif key.key == "enter":
            self._open = False
            if self.on_select and 0 <= self._selected < len(self._options):
                self.on_select(self._options[self._selected])
            self._dirty = True
            return True
        return False


# ── Tabs ──────────────────────────────────────────────────────────────

class Tabs(Widget):
    """Tabbed container.  Switches visible child by tab."""

    def __init__(self, tabs: list[tuple[str, Widget]] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tab_names: list[str] = []
        self._tab_widgets: dict[str, Widget] = {}
        self._active = ""
        if tabs:
            for name, widget in tabs:
                self.add_tab(name, widget)

    def add_tab(self, name: str, widget: Widget) -> None:
        self._tab_names.append(name)
        self._tab_widgets[name] = widget
        widget.mount(parent=self)
        if not self._active:
            self._active = name
            widget.visible = True
        else:
            widget.visible = False

    @property
    def active_tab(self) -> str:
        return self._active

    def set_active(self, name: str) -> None:
        if name in self._tab_widgets:
            for n, w in self._tab_widgets.items():
                w.visible = (n == name)
            self._active = name
            self._dirty = True
            # Re-compute the newly active widget so it has valid dimensions
            if self._computed_rows > 0 and self._computed_cols > 0:
                child_rows = max(self._computed_rows - 1, 0)
                w = self._tab_widgets[name]
                w._screen_top = self._screen_top + 1
                w._screen_left = self._screen_left
                w.compute(child_rows, self._computed_cols)

    def compute(self, rows: int, cols: int) -> tuple[int, int]:
        self._computed_rows = rows
        self._computed_cols = cols
        # Tab bar takes 1 row
        child_rows = max(rows - 1, 0)
        for name in self._tab_names:
            w = self._tab_widgets[name]
            if w.visible:
                w._screen_top = self._screen_top + 1
                w._screen_left = self._screen_left
                w.compute(child_rows, cols)
        return rows, cols

    def render(self) -> list[str]:
        w = self._computed_cols
        h = self._computed_rows

        # Tab bar
        tab_parts = []
        for name in self._tab_names:
            if name == self._active:
                tab_parts.append(f" [{name}] ")
            else:
                tab_parts.append(f"  {name}  ")
        tab_bar = "".join(tab_parts)
        lines = [self._pad_line(tab_bar, w)]

        # Active tab content
        active = self._tab_widgets.get(self._active)
        if active and active.visible:
            child_lines = active.render()
            for line in child_lines:
                lines.append(self._pad_line(line, w))
        else:
            lines.append(" " * w)

        # Pad remaining rows
        while len(lines) < h:
            lines.append(" " * w)

        self._dirty = False
        return lines[:h]

    def handle(self, key: KeyEvent) -> bool:
        # Tab switching with left/right arrows
        if key.key == "left":
            idx = self._tab_names.index(self._active) if self._active in self._tab_names else 0
            if idx > 0:
                self.set_active(self._tab_names[idx - 1])
            return True
        elif key.key == "right":
            idx = self._tab_names.index(self._active) if self._active in self._tab_names else 0
            if idx < len(self._tab_names) - 1:
                self.set_active(self._tab_names[idx + 1])
            return True

        # Forward to active tab
        active = self._tab_widgets.get(self._active)
        if active and active.visible:
            return active.handle(key)
        return False


# ── Dialog ────────────────────────────────────────────────────────────

class Dialog(Widget):
    """Modal dialog overlay.  Wraps a child widget in a centered box."""

    def __init__(self, title: str = "", child: Widget | None = None,
                 on_close: Callable | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.title = title
        self.child = child
        self.on_close = on_close
        self._visible = False
        if child:
            child.mount(parent=self)

    def open(self) -> None:
        self._visible = True
        self.visible = True
        self._dirty = True

    def close(self) -> None:
        self._visible = False
        self.visible = False
        if self.on_close:
            self.on_close()
        self._dirty = True

    def compute(self, rows: int, cols: int) -> tuple[int, int]:
        self._computed_rows = rows
        self._computed_cols = cols
        if not self._visible:
            return 0, 0
        # Dialog is centered, 60% width, auto height
        dialog_w = max(int(cols * 0.6), 20)
        dialog_h = max(int(rows * 0.4), 8)
        dialog_top = (rows - dialog_h) // 2
        dialog_left = (cols - dialog_w) // 2

        self._computed_rows = rows
        self._computed_cols = cols

        if self.child and self.child.visible:
            b = 2  # border
            p = 1  # padding
            inner_rows = max(dialog_h - 2 * b - 2 * p, 0)
            inner_cols = max(dialog_w - 2 * b - 2 * p, 0)
            self.child._screen_top = self._screen_top + dialog_top + b + p
            self.child._screen_left = self._screen_left + dialog_left + b + p
            self.child.compute(inner_rows, inner_cols)

        return rows, cols

    def render(self) -> list[str]:
        if not self._visible:
            return []

        w = self._computed_cols
        h = self._computed_rows
        dialog_w = max(int(w * 0.6), 20)
        dialog_h = max(int(h * 0.4), 8)
        dialog_top = (h - dialog_h) // 2
        dialog_left = (w - dialog_w) // 2

        # Build overlay (dim background)
        lines = [" " * w for _ in range(h)]

        # Draw dialog box
        # Top border with title
        if self.title:
            title_str = f" {self.title} "
            avail = max(dialog_w - 2, 0)
            if len(title_str) < avail:
                left_h = (avail - len(title_str)) // 2
                right_h = avail - len(title_str) - left_h
                top_line = Box.DTL + Box.DH * left_h + title_str + Box.DH * right_h + Box.DTR
            else:
                top_line = Box.DTL + self._truncate(title_str, avail) + Box.DTR
        else:
            top_line = Box.DTL + Box.DH * max(dialog_w - 2, 0) + Box.DTR

        # Content
        child_lines = self.child.render() if self.child and self.child.visible else []
        content_rows = max(dialog_h - 2, 0)

        for row in range(dialog_h):
            line_idx = dialog_top + row
            if line_idx >= h:
                break
            if row == 0:
                line = top_line
            elif row == dialog_h - 1:
                line = Box.DBL + Box.DH * max(dialog_w - 2, 0) + Box.DBR
            else:
                content_row = row - 1
                if content_row < len(child_lines):
                    content = self._pad_line(child_lines[content_row], max(dialog_w - 2, 0))
                else:
                    content = " " * max(dialog_w - 2, 0)
                line = Box.DV + content + Box.DV

            # Overlay onto background
            padded = " " * dialog_left + line
            padded = self._pad_line(padded, w)
            lines[line_idx] = padded

        self._dirty = False
        return lines

    def handle(self, key: KeyEvent) -> bool:
        if not self._visible:
            return False
        if key.key == "esc":
            self.close()
            return True
        if self.child and self.child.visible:
            return self.child.handle(key)
        return False


# ── Separator ─────────────────────────────────────────────────────────

class Separator(Widget):
    """Horizontal or vertical line separator."""

    def __init__(self, direction: str = "horizontal", char: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.direction = direction
        self.char = char

    def compute(self, rows: int, cols: int) -> tuple[int, int]:
        self._computed_rows = rows
        self._computed_cols = cols
        if self.direction == "horizontal":
            return 1, cols
        else:
            return rows, 1

    def render(self) -> list[str]:
        if self.direction == "horizontal":
            ch = self.char or Box.H
            return [ch * self._computed_cols]
        else:
            ch = self.char or Box.V
            return [ch] * self._computed_rows


# ── ProgressBar ───────────────────────────────────────────────────────

class ProgressBar(Widget):
    """Text-based progress bar."""

    def __init__(self, value: float = 0.0, label: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._value = max(0.0, min(1.0, value))
        self.label = label

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, v: float) -> None:
        new_v = max(0.0, min(1.0, v))
        if self._value != new_v:
            self._value = new_v
            self._dirty = True

    def compute(self, rows: int, cols: int) -> tuple[int, int]:
        self._computed_rows = max(rows, 1)
        self._computed_cols = cols
        return self._computed_rows, self._computed_cols

    def render(self) -> list[str]:
        w = self._computed_cols
        pct = int(self._value * 100)
        bar_w = max(w - len(self.label) - 8, 10)
        filled = int(self._value * bar_w)
        empty = bar_w - filled
        bar = "[" + "\u2588" * filled + "\u2591" * empty + "]"
        text = f"{self.label} {bar} {pct}%"
        self._dirty = False
        return [self._pad_line(text, w)]


# ── Spinner ───────────────────────────────────────────────────────────

class Spinner(Widget):
    """Animated spinner (frames cycled by caller)."""

    FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827",
              "\u2807", "\u280f"]

    def __init__(self, text: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.text = text
        self._frame = 0

    def tick(self) -> None:
        self._frame = (self._frame + 1) % len(self.FRAMES)
        self._dirty = True

    def compute(self, rows: int, cols: int) -> tuple[int, int]:
        self._computed_rows = max(rows, 1)
        self._computed_cols = cols
        return self._computed_rows, self._computed_cols

    def render(self) -> list[str]:
        frame = self.FRAMES[self._frame]
        text = f" {frame} {self.text}"
        self._dirty = False
        return [self._pad_line(text, self._computed_cols)]


# ── App (root container with event loop) ──────────────────────────────

class App:
    """Root application.  Manages the widget tree and event loop."""

    def __init__(self, title: str = "SloughGPT") -> None:
        self.title = title
        self.event_bus = EventBus()
        self.root_widget: Container | None = None
        self._running = False
        self._size = (24, 80)

    def set_root(self, widget: Container) -> None:
        self.root_widget = widget
        widget._event_bus = self.event_bus

    def resize(self, rows: int, cols: int) -> None:
        self._size = (rows, cols)
        self.event_bus.publish(ResizeEvent(rows=rows, cols=cols))
        if self.root_widget:
            self.root_widget.compute(rows, cols)

    def render(self) -> str:
        if not self.root_widget:
            return ""
        lines = self.root_widget.render()
        return "\n".join(lines)

    def handle_key(self, key: str, raw: int = 0) -> bool:
        event = KeyEvent(key=key, raw=raw)
        self.event_bus.publish(event)
        if self.root_widget:
            return self.root_widget.handle(event)
        return False

    def focus_first(self) -> None:
        """Focus the first focusable widget in the tree."""
        self._focus_next(from_widget=None, forward=True)

    def focus_next(self) -> None:
        """Move focus to the next focusable widget."""
        focused = self._find_focused()
        self._focus_next(from_widget=focused, forward=True)

    def focus_prev(self) -> None:
        """Move focus to the previous focusable widget."""
        focused = self._find_focused()
        self._focus_next(from_widget=focused, forward=False)

    def _find_focused(self) -> Widget | None:
        if not self.root_widget:
            return None
        return self._find_focused_recursive(self.root_widget)

    def _find_focused_recursive(self, widget: Widget) -> Widget | None:
        if widget._focused:
            return widget
        for child in widget._children:
            result = self._find_focused_recursive(child)
            if result:
                return result
        return None

    def _focus_next(self, from_widget: Widget | None, forward: bool) -> None:
        focusable = []
        if self.root_widget:
            self._collect_focusable(self.root_widget, focusable)
        if not focusable:
            return

        if from_widget is None:
            idx = 0 if forward else len(focusable) - 1
        else:
            try:
                idx = focusable.index(from_widget)
                idx = (idx + (1 if forward else -1)) % len(focusable)
            except ValueError:
                idx = 0

        if from_widget:
            from_widget.blur()
        focusable[idx].focus()

    def _collect_focusable(self, widget: Widget, out: list[Widget]) -> None:
        if widget.visible and not widget._children:
            out.append(widget)
        for child in widget._children:
            self._collect_focusable(child, out)
