"""
Terminal window manager for the SloughGPT Shell.

i3/dwm-style tiling + floating window manager with workspaces,
keyboard-driven navigation, and resize mode.

Keybindings:
  Mod+Enter        New pane
  Mod+[hjkl]       Focus left/down/up/right
  Mod+h/v          Split horizontal/vertical
  Mod+Shift+q      Close focused pane
  Mod+Shift+Space  Toggle floating
  Mod+f            Monocle (fullscreen) layout
  Mod+s            Stacked (tabbed) layout
  Mod+e            Toggle split orientation
  Mod+d            Cycle layouts
  Mod+1-9          Switch workspace
  Mod+Shift+1-9    Move pane to workspace
  Mod+r            Enter resize mode
  Mod+Shift+Arrows Move floating window
  Mod+Arrows       Focus direction
  Esc/q            Quit WM
  :<cmd>           Command mode (:q, :close, :split-h, :split-v, :help)

Status bar: [1] [2] [3*] | layout: split-h | 14:23:45 | font: Digital TS Medium
"""

from __future__ import annotations

import curses
import os
import sys
import time
import logging
import subprocess
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("man.shell.wm")

# ── Colour pairs ──────────────────────────────────────────────────────

_PAIR_BORDER = 1
_PAIR_TITLE = 2
_PAIR_STATUS = 3
_PAIR_FOCUS_BORDER = 4
_PAIR_FLOATING_BORDER = 5
_PAIR_TAB = 6
_PAIR_TAB_ACTIVE = 7
_PAIR_FONT_INDICATOR = 8
_PAIR_RESIZE = 9

_MOD_NAME = "Mod"


def _init_colours() -> None:
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(_PAIR_BORDER, curses.COLOR_WHITE, -1)
        curses.init_pair(_PAIR_TITLE, curses.COLOR_CYAN, -1)
        curses.init_pair(_PAIR_STATUS, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(_PAIR_FOCUS_BORDER, curses.COLOR_YELLOW, -1)
        curses.init_pair(_PAIR_FLOATING_BORDER, curses.COLOR_MAGENTA, -1)
        curses.init_pair(_PAIR_TAB, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.init_pair(_PAIR_TAB_ACTIVE, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(_PAIR_FONT_INDICATOR, curses.COLOR_GREEN, -1)
        curses.init_pair(_PAIR_RESIZE, curses.COLOR_RED, curses.COLOR_YELLOW)


# ── Layout types ──────────────────────────────────────────────────────

class LayoutType:
    SPLIT_H = "split-h"
    SPLIT_V = "split-v"
    STACKED = "stacked"
    MONOCLE = "monocle"

    @staticmethod
    def next(current: str) -> str:
        order = [LayoutType.SPLIT_H, LayoutType.SPLIT_V, LayoutType.STACKED, LayoutType.MONOCLE]
        idx = order.index(current) if current in order else 0
        return order[(idx + 1) % len(order)]


# ── Pane ──────────────────────────────────────────────────────────────

class Pane:
    """A single window pane with buffer, scroll, and floating state."""

    def __init__(self, title: str = "term", buffer: list[str] | None = None):
        self.title = title
        self.buffer = list(buffer) if buffer else []
        self.scroll_offset = 0
        self.floating = False
        self.float_x = 0
        self.float_y = 0
        self.float_w = 40
        self.float_h = 12
        self._cmd_history: list[str] = []
        self._history_pos = 0
        self._input_line = ""

    def write(self, text: str) -> None:
        self.buffer.extend(text.split("\n"))
        if len(self.buffer) > 1000:
            self.buffer = self.buffer[-1000:]
        self.scroll_offset = max(0, len(self.buffer) - self._content_height(20))

    def _content_height(self, pane_h: int) -> int:
        return max(pane_h - 2, 1)

    def scroll_up(self, n: int = 1) -> None:
        self.scroll_offset = max(0, self.scroll_offset - n)

    def scroll_down(self, n: int = 1, pane_h: int = 20) -> None:
        max_offset = max(0, len(self.buffer) - self._content_height(pane_h))
        self.scroll_offset = min(max_offset, self.scroll_offset + n)

    def visible_lines(self, pane_h: int) -> list[str]:
        start = self.scroll_offset
        end = start + self._content_height(pane_h)
        return self.buffer[start:end]


# ── Workspace ─────────────────────────────────────────────────────────

class Workspace:
    """A virtual desktop with its own set of panes and layout."""

    def __init__(self, name: str):
        self.name = name
        self.panes: list[Pane] = [Pane(title="main")]
        self.layout = LayoutType.SPLIT_H
        self.focus_idx = 0
        self.pane_counter = 1

    @property
    def focused_pane(self) -> Optional[Pane]:
        if 0 <= self.focus_idx < len(self.panes):
            return self.panes[self.focus_idx]
        return None

    def add_pane(self, title: str | None = None) -> Pane:
        if title is None:
            title = f"pane-{self.pane_counter}"
            self.pane_counter += 1
        p = Pane(title=title)
        # Split with focus pane: insert next to it
        insert_at = min(self.focus_idx + 1, len(self.panes))
        self.panes.insert(insert_at, p)
        self.focus_idx = insert_at
        return p

    def remove_pane(self, idx: int) -> Optional[str]:
        if len(self.panes) <= 1:
            return None
        removed = self.panes.pop(idx)
        self.focus_idx = min(idx, len(self.panes) - 1)
        return removed.title

    def next_layout(self) -> None:
        self.layout = LayoutType.next(self.layout)

    def set_layout(self, layout: str) -> None:
        self.layout = layout

    def toggle_floating(self) -> bool:
        p = self.focused_pane
        if p is None:
            return False
        p.floating = not p.floating
        return p.floating

    def focus_next(self) -> None:
        if len(self.panes) > 0:
            self.focus_idx = (self.focus_idx + 1) % len(self.panes)

    def focus_prev(self) -> None:
        if len(self.panes) > 0:
            self.focus_idx = (self.focus_idx - 1) % len(self.panes)

    def move_floating(self, dx: int, dy: int) -> None:
        p = self.focused_pane
        if p and p.floating:
            p.float_x += dx
            p.float_y += dy

    def resize_floating(self, dw: int, dh: int) -> None:
        p = self.focused_pane
        if p and p.floating:
            p.float_w = max(10, p.float_w + dw)
            p.float_h = max(4, p.float_h + dh)


# ── Layout computation ───────────────────────────────────────────────

def _compute_tiled_rects(panes: list[Pane], layout: str,
                          x: int, y: int, w: int, h: int
                          ) -> list[tuple[int, int, int, int]]:
    """Assign screen rectangles to panes based on layout type."""
    n = len(panes)
    if n == 0:
        return []

    if layout == LayoutType.MONOCLE:
        return [(x, y, w, h)] * n

    if layout == LayoutType.STACKED:
        # All panes get the same area (only focus is visible)
        return [(x, y, w, h)] * n

    # Split layouts
    rects = []
    if layout == LayoutType.SPLIT_H:
        each_h = h // n
        remainder = h % n
        cy = y
        for i in range(n):
            ph = each_h + (1 if i < remainder else 0)
            rects.append((x, cy, w, ph))
            cy += ph
    else:  # SPLIT_V
        each_w = w // n
        remainder = w % n
        cx = x
        for i in range(n):
            pw = each_w + (1 if i < remainder else 0)
            rects.append((cx, y, pw, h))
            cx += pw

    return rects


# ── WindowManager ─────────────────────────────────────────────────────

class WindowManager:
    """i3/dwm-style tiling + floating window manager with workspaces."""

    def __init__(self, parent_shell: Any = None):
        self._parent_shell = parent_shell
        self._running = False
        self._stdscr: Any = None
        self._dirty = True

        # Workspaces
        self._workspaces: list[Workspace] = [Workspace(f"{i}") for i in range(1, 10)]
        self._current_ws = 0

        # State
        self._status_msg = ""
        self._status_timer = 0.0
        self._input_mode = False
        self._input_buf = ""
        self._resize_mode = False
        self._pane_shell = False
        self._pane_input = ""
        self._font_name = ""

        # Read font from parent shell env if available
        if parent_shell and hasattr(parent_shell, '_env'):
            self._font_name = parent_shell._env.get("FONT", "")

    # ── Properties ─────────────────────────────────────────────────

    @property
    def _workspace(self) -> Workspace:
        return self._workspaces[self._current_ws]

    @property
    def pane_count(self) -> int:
        return len(self._workspace.panes)

    @property
    def focused_pane(self) -> Optional[Pane]:
        return self._workspace.focused_pane

    # ── Public API ─────────────────────────────────────────────────

    def split_horizontal(self) -> None:
        ws = self._workspace
        ws.add_pane()
        ws.layout = LayoutType.SPLIT_H
        ws.focus_idx = len(ws.panes) - 1
        self._dirty = True

    def split_vertical(self) -> None:
        ws = self._workspace
        ws.add_pane()
        ws.layout = LayoutType.SPLIT_V
        ws.focus_idx = len(ws.panes) - 1
        self._dirty = True

    def close_pane(self) -> Optional[str]:
        ws = self._workspace
        title = ws.remove_pane(ws.focus_idx)
        if title:
            self._flash(f"  Closed: {title}")
        else:
            self._flash("  Cannot close last pane")
        self._dirty = True
        return title

    def focus_next(self) -> None:
        self._workspace.focus_next()
        self._dirty = True

    def focus_prev(self) -> None:
        self._workspace.focus_prev()
        self._dirty = True

    def next_layout(self) -> None:
        self._workspace.next_layout()
        self._flash(f"  Layout: {self._workspace.layout}")
        self._dirty = True

    def set_layout(self, layout: str) -> None:
        self._workspace.set_layout(layout)
        self._flash(f"  Layout: {layout}")
        self._dirty = True

    def toggle_floating(self) -> None:
        p = self._workspace.focused_pane
        if p:
            p.floating = not p.floating
            self._flash(f"  {'Floating' if p.floating else 'Tiled'}: {p.title}")
            self._dirty = True

    def switch_workspace(self, idx: int) -> None:
        if 0 <= idx < len(self._workspaces) and idx != self._current_ws:
            self._current_ws = idx
            self._dirty = True

    def move_pane_to_workspace(self, ws_idx: int) -> None:
        if ws_idx < 0 or ws_idx >= len(self._workspaces):
            return
        if ws_idx == self._current_ws:
            return
        ws = self._workspace
        p = ws.focused_pane
        if p is None or len(ws.panes) <= 1:
            self._flash("  Cannot move last pane")
            return
        ws.remove_pane(ws.focus_idx)
        target = self._workspaces[ws_idx]
        target.panes.append(p)
        target.focus_idx = len(target.panes) - 1
        self._flash(f"  Moved to workspace {ws_idx + 1}")

    def move_focus(self, dx: int, dy: int) -> None:
        """Move focus in a direction (i3-style arrow key navigation)."""
        ws = self._workspace
        p = ws.focused_pane
        if p and p.floating:
            p.float_x += dx * 2
            p.float_y += dy * 2
            self._dirty = True
            return
        if dx > 0:
            ws.focus_next()
        elif dx < 0:
            ws.focus_prev()
        elif dy > 0:
            # Down: focus next
            ws.focus_next()
        elif dy < 0:
            # Up: focus prev
            ws.focus_prev()
        self._dirty = True

    # ── Resize (tiling) ────────────────────────────────────────────

    def resize_focused(self, dx: int, dy: int) -> None:
        ws = self._workspace
        p = ws.focused_pane
        if p is None:
            return
        if p.floating:
            p.float_w = max(10, p.float_w + dx)
            p.float_h = max(4, p.float_h + dy)
            self._dirty = True
            return
        # For tiled: adjust split ratio by moving adjacent pane boundary
        # Simple approach: swap with neighbor to grow/shrink
        idx = ws.focus_idx
        if dx < 0 and idx > 0:
            ws.panes[idx], ws.panes[idx - 1] = ws.panes[idx - 1], ws.panes[idx]
            ws.focus_idx = idx - 1
        elif dx > 0 and idx < len(ws.panes) - 1:
            ws.panes[idx], ws.panes[idx + 1] = ws.panes[idx + 1], ws.panes[idx]
            ws.focus_idx = idx + 1
        self._dirty = True

    # ── Internal ───────────────────────────────────────────────────

    def _flash(self, msg: str) -> None:
        self._status_msg = msg
        self._status_timer = time.time()

    def _apply_font(self) -> None:
        if self._font_name:
            sys.stdout.write(f"\x1b]50;Set Font={self._font_name}\x07")
            sys.stdout.flush()

    # ── Main run ───────────────────────────────────────────────────

    def run(self) -> None:
        if self.pane_count == 0:
            ws = self._workspace
            ws.panes = [Pane(title="main")]
            ws.focus_idx = 0
        curses.wrapper(self._curses_loop)

    def _curses_loop(self, stdscr: Any) -> None:
        self._stdscr = stdscr
        curses.mousemask(curses.BUTTON1_CLICKED | curses.BUTTON1_RELEASED)
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        if curses.has_key(curses.KEY_RESIZE):
            stdscr.keypad(True)
        _init_colours()

        self._apply_font()

        self._running = True
        self._dirty = True
        self._flash("  i3 WM ready — Mod+? for help | :help for commands")

        while self._running:
            self._handle_resize()
            if self._dirty:
                self._render()
                self._dirty = False

            key = self._get_key()
            if key < 0:
                continue

            if self._input_mode:
                self._handle_input_key(key)
            elif self._resize_mode:
                self._handle_resize_key(key)
            elif key == curses.KEY_MOUSE:
                self._handle_mouse_event()
            else:
                self._handle_key(key)

            time.sleep(0.016)

    def _get_key(self) -> int:
        self._stdscr.timeout(100)
        try:
            return self._stdscr.getch()
        except KeyboardInterrupt:
            return 3

    def _handle_mouse_event(self) -> None:
        """Handle mouse click to focus panes."""
        try:
            _, mx, my, _, bstate = curses.getmouse()
        except curses.error:
            return

        # Only handle button1 clicks
        if not (bstate & curses.BUTTON1_CLICKED):
            return

        ws = self._workspace
        rows, cols = self._stdscr.getmaxyx()
        status_h = 1
        avail_h = rows - status_h - 1

        # Get tiled pane rects
        tiled = [p for p in ws.panes if not p.floating]
        floating = [p for p in ws.panes if p.floating]

        if tiled:
            rects = _compute_tiled_rects(tiled, ws.layout, 0, 0, cols, avail_h)

            # For stacked/monocle, only the focused pane is visible
            if ws.layout in (LayoutType.STACKED, LayoutType.MONOCLE):
                # Click on tab bar (row 0) switches panes
                if my == 0:
                    tab_w = max(1, cols // len(tiled))
                    clicked = mx // tab_w
                    if clicked < len(tiled):
                        ws.focus_idx = ws.panes.index(tiled[clicked])
                        self._dirty = True
                    return
                # Any click in the content area focuses the visible pane
                pane_idx = ws.panes.index(tiled[ws.focus_idx])
                # Already focused — nothing to do
                return

            # For split layouts, find which rect contains (mx, my)
            for pane, rect in zip(tiled, rects):
                rx, ry, rw, rh = rect
                if rx <= mx < rx + rw and ry <= my < ry + rh:
                    target_idx = ws.panes.index(pane)
                    if target_idx != ws.focus_idx:
                        ws.focus_idx = target_idx
                        self._dirty = True
                    return

        # Check floating panes (render in reverse order = on top)
        for p in reversed(floating):
            if p.float_x <= mx < p.float_x + p.float_w and p.float_y <= my < p.float_y + p.float_h:
                target_idx = ws.panes.index(p)
                if target_idx != ws.focus_idx:
                    ws.focus_idx = target_idx
                    self._dirty = True
                return

    def _is_mod(self, key: int) -> bool:
        """Check if key is Mod4 (Alt/Option on Mac)."""
        return key in (curses.KEY_B1, 545, 561, 553, 537) or key == -1

    def _handle_key(self, key: int) -> None:
        ws = self._workspace

        # Pane shell mode: route keys to pane input
        if self._pane_shell:
            self._handle_pane_shell_key(key)
            return

        # Command mode
        if key == ord(":"):
            self._input_mode = True
            self._input_buf = ":"
            self._dirty = True
            return

        # Quit
        if key in (27, ord("q"), ord("Q")):
            self._running = False
            return

        # Tab for focus next
        if key == 9:
            ws.focus_next()
            self._dirty = True
            return
        if key == 353:  # Shift+Tab
            ws.focus_prev()
            self._dirty = True
            return

        # Arrow keys without mod
        if key == curses.KEY_LEFT:
            ws.focus_prev(); self._dirty = True
            return
        if key == curses.KEY_RIGHT:
            ws.focus_next(); self._dirty = True
            return
        if key == curses.KEY_UP:
            p = ws.focused_pane
            if p: p.scroll_up(); self._dirty = True
            return
        if key == curses.KEY_DOWN:
            p = ws.focused_pane
            if p: p.scroll_down(pane_h=20); self._dirty = True
            return

        # hjkl vim scroll (unmodified)
        if key == ord("j"):
            p = ws.focused_pane
            if p: p.scroll_down(pane_h=20); self._dirty = True
            return
        if key == ord("k"):
            p = ws.focused_pane
            if p: p.scroll_up(); self._dirty = True
            return

        # Mod key prefix — wait for next key
        if key in (curses.KEY_B1, 545, 561, 553, 537):
            self._handle_mod_key()
            return

    def _handle_mod_key(self) -> None:
        """Handle key pressed after Mod (Alt/Option)."""
        ws = self._workspace
        nxt = self._get_key()

        # Mod+Enter — new pane
        if nxt == 10 or nxt == ord("\n"):
            p = ws.add_pane()
            p.write(f"  ┌ pane {p.title}")
            p.write("  │")
            p.write("  └ new")
            self._dirty = True
            return

        # Mod+h — split horizontal
        if nxt == ord("h"):
            self.split_horizontal()
            return

        # Mod+v — split vertical
        if nxt == ord("v"):
            self.split_vertical()
            return

        # Mod+j/k/l — focus direction
        if nxt == ord("j"):
            ws.focus_next(); self._dirty = True; return
        if nxt == ord("k"):
            ws.focus_prev(); self._dirty = True; return
        if nxt == ord("l"):
            ws.focus_next(); self._dirty = True; return

        # Mod+Space — toggle floating
        if nxt == ord(" "):
            self.toggle_floating()
            return

        # Mod+Shift+q — close
        if nxt == ord("Q"):
            self.close_pane()
            return

        # Mod+Shift+Space — toggle floating (alt)
        if nxt == 383:  # Shift+Space
            self.toggle_floating()
            return

        # Mod+f — monocle
        if nxt == ord("f"):
            self.set_layout(LayoutType.MONOCLE)
            return

        # Mod+s — stacked
        if nxt == ord("s"):
            self.set_layout(LayoutType.STACKED)
            return

        # Mod+e — toggle split orientation
        if nxt == ord("e"):
            ws.layout = LayoutType.SPLIT_V if ws.layout == LayoutType.SPLIT_H else LayoutType.SPLIT_H
            self._flash(f"  Layout: {ws.layout}")
            self._dirty = True
            return

        # Mod+d — cycle layouts
        if nxt == ord("d"):
            self.next_layout()
            return

        # Mod+r — resize mode
        if nxt == ord("r"):
            self._resize_mode = True
            self._flash("  Resize mode — arrows to resize, Enter/Esc to exit")
            self._dirty = True
            return

        # Mod+1-9 — switch workspace
        if ord("1") <= nxt <= ord("9"):
            idx = nxt - ord("1")
            self.switch_workspace(idx)
            return

        # Mod+Arrows — focus/move
        if nxt == curses.KEY_LEFT:
            self.move_focus(-1, 0); return
        if nxt == curses.KEY_RIGHT:
            self.move_focus(1, 0); return
        if nxt == curses.KEY_UP:
            self.move_focus(0, -1); return
        if nxt == curses.KEY_DOWN:
            self.move_focus(0, 1); return

        # Mod+Shift+1-9 — move pane to workspace
        if nxt in range(ord("!"), ord(")") + 1):
            idx = nxt - ord("!")
            self.move_pane_to_workspace(idx)
            return

        # Mod+Shift+Arrows — move floating window / swap panes
        shift_mapping = {
            393: (0, -1),  # Shift+Up
            402: (0, 1),   # Shift+Down
            393: (0, -1),
            402: (0, 1),
        }

    def _handle_pane_shell_key(self, key: int) -> None:
        """Handle keyboard input when pane is in shell mode."""
        p = self._workspace.focused_pane
        if not p:
            self._pane_shell = False
            self._dirty = True
            return

        # Esc or Ctrl+C — exit pane shell mode
        if key in (27, 3):
            self._pane_shell = False
            self._pane_input = ""
            p._input_line = ""
            p.write("  ^C")
            self._dirty = True
            self._flash("  Exited pane shell")
            return

        # Enter — submit command
        if key in (10, 13):
            cmd = self._pane_input.strip()
            if cmd:
                p._cmd_history.append(cmd)
                p._history_pos = len(p._cmd_history)
                p.write(f"  $ {cmd}")
                self._pane_input = ""
                self._dirty = True
                # Execute via parent shell or subprocess
                if self._parent_shell and hasattr(self._parent_shell, '_execute_single'):
                    try:
                        out = self._parent_shell._execute_single(cmd, "")
                        if out:
                            p.write(out.rstrip("\n"))
                    except Exception as e:
                        p.write(f"  Error: {e}")
                else:
                    # Fallback: run via subprocess
                    try:
                        result = subprocess.run(
                            cmd, shell=True, capture_output=True, text=True, timeout=30
                        )
                        if result.stdout:
                            p.write(result.stdout.rstrip("\n"))
                        if result.stderr:
                            p.write(f"[stderr]\n{result.stderr.rstrip()}")
                    except subprocess.TimeoutExpired:
                        p.write("  Command timed out (30s)")
                    except Exception as e:
                        p.write(f"  Error: {e}")
            self._dirty = True
            return

        # Tab — completion
        if key == 9:
            prefix = self._pane_input
            matches = []
            try:
                cwd = os.getcwd()
                for entry in os.listdir(cwd):
                    if entry.startswith(prefix) and not entry.startswith("."):
                        matches.append(entry)
            except OSError:
                pass
            if len(matches) == 1:
                fp = os.path.join(cwd, matches[0]) if 'cwd' in dir() and cwd else matches[0]
                is_dir = os.path.isdir(os.path.join(os.getcwd(), matches[0]))
                self._pane_input = matches[0] + ("/" if is_dir else "")
                self._dirty = True
            elif len(matches) > 1:
                common = os.path.commonprefix(matches)
                if common and common != prefix:
                    self._pane_input = common
                else:
                    p.write(f"  {'  '.join(matches)}")
                self._dirty = True
            return

        # Up/Down — history
        if key == curses.KEY_UP:
            if p._history_pos > 0:
                p._history_pos -= 1
                self._pane_input = p._cmd_history[p._history_pos]
                self._dirty = True
            return
        if key == curses.KEY_DOWN:
            if p._history_pos < len(p._cmd_history) - 1:
                p._history_pos += 1
                self._pane_input = p._cmd_history[p._history_pos]
                self._dirty = True
            elif p._history_pos >= len(p._cmd_history) - 1:
                p._history_pos = len(p._cmd_history)
                self._pane_input = ""
                self._dirty = True
            return

        # Backspace
        if key in (curses.KEY_BACKSPACE, 127, 8):
            if self._pane_input:
                self._pane_input = self._pane_input[:-1]
                self._dirty = True
            return

        # Printable characters
        if 32 <= key <= 126:
            self._pane_input += chr(key)
            self._dirty = True
            return

    def _handle_resize_key(self, key: int) -> None:
        if key in (10, 13, 27):  # Enter or Esc — exit resize mode
            self._resize_mode = False
            self._flash("  Exited resize mode")
            self._dirty = True
            return

        step = 2
        if key == curses.KEY_LEFT:
            self.resize_focused(-step, 0)
        elif key == curses.KEY_RIGHT:
            self.resize_focused(step, 0)
        elif key == curses.KEY_UP:
            self.resize_focused(0, -step)
        elif key == curses.KEY_DOWN:
            self.resize_focused(0, step)
        elif key == ord("h"):
            self.resize_focused(-step, 0)
        elif key == ord("l"):
            self.resize_focused(step, 0)
        elif key == ord("k"):
            self.resize_focused(0, -step)
        elif key == ord("j"):
            self.resize_focused(0, step)

    def _handle_input_key(self, key: int) -> None:
        if key in (10, 13):
            self._execute_wm_command(self._input_buf)
            self._input_mode = False
            self._input_buf = ""
            self._dirty = True
        elif key in (27,):
            self._input_mode = False
            self._input_buf = ""
            self._dirty = True
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            if len(self._input_buf) > 1:
                self._input_buf = self._input_buf[:-1]
                self._dirty = True
        elif 32 <= key <= 126:
            self._input_buf += chr(key)
            self._dirty = True

    def _execute_wm_command(self, cmd: str) -> None:
        cmd = cmd.strip().lstrip(":")
        parts = cmd.split()
        if not parts:
            return
        verb = parts[0].lower()
        ws = self._workspace

        if verb in ("q", "quit", "exit"):
            self._running = False
        elif verb in ("close", "c"):
            self.close_pane()
        elif verb in ("split-h", "sh"):
            self.split_horizontal()
        elif verb in ("split-v", "sv"):
            self.split_vertical()
        elif verb in ("layout", "lay"):
            w = self._workspace
            self._flash(f"  Layout: {w.layout} | Panes: {len(w.panes)} | Focus: {w.focus_idx}")
        elif verb in ("workspace", "ws"):
            if len(parts) > 1 and parts[1].isdigit():
                idx = int(parts[1]) - 1
                self.switch_workspace(idx)
                self._flash(f"  Workspace {parts[1]}")
            else:
                ws_names = " ".join(f"[{w.name}]" if i == self._current_ws else f" {w.name} " for i, w in enumerate(self._workspaces))
                self._flash(f"  {ws_names}")
        elif verb in ("float", "floating"):
            self.toggle_floating()
        elif verb in ("resize", "rs"):
            self._resize_mode = True
            self._flash("  Resize mode — arrows to resize, Enter/Esc to exit")
        elif verb in ("monocle", "fullscreen", "m"):
            self.set_layout(LayoutType.MONOCLE)
        elif verb in ("stacked", "tab", "st"):
            self.set_layout(LayoutType.STACKED)
        elif verb in ("split", "sp"):
            self.set_layout(LayoutType.SPLIT_H)
        elif verb in ("split-equal", "tile"):
            """Rebalance all panes to equal size by resetting the layout."""
            # Reset to split layout and alternate orientation for multi-pane
            n = len(ws.panes)
            if n > 1:
                if n == 2:
                    ws.layout = LayoutType.SPLIT_V
                elif n <= 4:
                    # 3-4 panes: use split-h
                    ws.layout = LayoutType.SPLIT_H
                else:
                    ws.layout = LayoutType.STACKED
                ws.focus_idx = 0
                self._flash(f"  Tiled {n} panes equally as {ws.layout}")
            else:
                self._flash("  Only one pane — nothing to tile")
            self._dirty = True
        elif verb == "font":
            if len(parts) > 1:
                font_name = " ".join(parts[1:])
                self._font_name = font_name
                sys.stdout.write(f"\x1b]50;Set Font={font_name}\x07")
                sys.stdout.flush()
                self._flash(f"  Font: {font_name}")
            else:
                self._flash(f"  Font: {self._font_name or 'terminal default'}")
        elif verb == "reset":
            ws.panes = [Pane(title="main")]
            ws.focus_idx = 0
            ws.layout = LayoutType.SPLIT_H
            self._dirty = True
            self._flash("  Layout reset")
        elif verb == "write":
            if len(parts) > 1:
                fname = " ".join(parts[1:])
                p = ws.focused_pane
                if p:
                    try:
                        with open(fname, "w") as f:
                            f.write("\n".join(p.buffer))
                        self._flash(f"  Wrote {len(p.buffer)} lines to {fname}")
                    except Exception as e:
                        self._flash(f"  Error writing {fname}: {e}")
            else:
                self._flash("  Usage: :write <filename>")
        elif verb in ("load", "open"):
            if len(parts) > 1:
                fname = " ".join(parts[1:])
                p = ws.focused_pane
                if p:
                    try:
                        with open(fname) as f:
                            content = f.read()
                        p.write(f"\n  ── {fname} ──")
                        p.write(content.rstrip("\n"))
                        self._flash(f"  Loaded {fname}")
                    except Exception as e:
                        self._flash(f"  Error loading {fname}: {e}")
            else:
                self._flash("  Usage: :load <filename>")
        elif verb == "ls":
            p = ws.focused_pane
            if p:
                try:
                    files = os.listdir(".")
                    # Show with dir markers
                    lines = []
                    for f in sorted(files):
                        fp = os.path.join(".", f)
                        if os.path.isdir(fp):
                            lines.append(f"  {f}/")
                        else:
                            sz = os.path.getsize(fp)
                            lines.append(f"  {f} ({sz} bytes)" if sz > 0 else f"  {f}")
                    p.write("\n" + "\n".join(lines))
                except OSError as e:
                    p.write(f"  Error: {e}")
                self._dirty = True
        elif verb == "pwd":
            p = ws.focused_pane
            if p:
                p.write(f"\n  {os.getcwd()}")
                self._dirty = True
        elif verb == "echo":
            p = ws.focused_pane
            if p and len(parts) > 1:
                p.write(" ".join(parts[1:]))
                self._dirty = True
        elif verb == "date":
            p = ws.focused_pane
            if p:
                p.write(f"\n  {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}")
                self._dirty = True
        elif verb in ("rename", "name", "title"):
            if len(parts) > 1:
                p = ws.focused_pane
                if p:
                    new_title = " ".join(parts[1:])
                    p.title = new_title
                    self._flash(f"  Pane renamed: {new_title}")
                    self._dirty = True
            else:
                self._flash("  Usage: :rename <title>")
        elif verb in ("focus", "goto"):
            if len(parts) > 1 and parts[1].lstrip("-").isdigit():
                target = int(parts[1])
                if 1 <= target <= len(ws.panes):
                    ws.focus_idx = target - 1  # 1-indexed
                    self._dirty = True
                    self._flash(f"  Pane {target}")
                else:
                    self._flash(f"  Pane {target} out of range (1-{len(ws.panes)})")
            else:
                names = " ".join(f"[{i}] {p.title}" for i, p in enumerate(ws.panes, 1))
                self._flash(f"  {names}")
        elif verb in ("hist", "history"):
            p = ws.focused_pane
            if p and p._cmd_history:
                lines = []
                for i, c in enumerate(p._cmd_history, 1):
                    lines.append(f"  {i:3d}  {c}")
                p.write("\n" + "\n".join(lines[-50:]))  # last 50
                self._dirty = True
            elif p:
                p.write("  (no command history)")
                self._dirty = True
        elif verb in ("grep", "search"):
            p = ws.focused_pane
            if p and len(parts) > 1:
                pattern = " ".join(parts[1:])
                try:
                    matches = [line for line in p.buffer if pattern in line]
                    if matches:
                        p.write(f"\n  ── {len(matches)} match(es) for \"{pattern}\" ──")
                        for line in matches[-100:]:  # last 100 matches
                            p.write(f"  {line}")
                    else:
                        p.write(f"  No matches for \"{pattern}\"")
                except Exception as e:
                    p.write(f"  Error: {e}")
                self._dirty = True
            elif p:
                self._flash("  Usage: :grep <pattern>")
        elif verb in ("exec", "run", "e"):
            self._exec_in_pane(" ".join(parts[1:]))
        elif verb in ("shell", "sh"):
            self._pane_shell_mode()
        elif verb in ("clear", "cls"):
            p = self._workspace.focused_pane
            if p:
                p.buffer.clear()
                p.scroll_offset = 0
                self._dirty = True
                self._flash("  Pane cleared")
        elif verb in ("ps", "panes"):
            ws = self._workspace
            p = ws.focused_pane
            if p:
                info = (
                    f"  Workspace: {ws.name}  Layout: {ws.layout}\n"
                    f"  Panes: {len(ws.panes)}  Focus: {ws.focus_idx}\n"
                    f"  Title: {p.title}  Floating: {p.floating}\n"
                    f"  Buffer: {len(p.buffer)} lines  Scroll: {p.scroll_offset}"
                )
                p.write(f"\n  {'─'*30}\n{info}\n")
                self._dirty = True
        elif verb in ("help", "h", "?"):
            if len(parts) > 1:
                self._show_topic_help(" ".join(parts[1:]))
            else:
                self._show_help()
        else:
            self._flash(f"  Unknown: {verb} — :help for commands")

    def _exec_in_pane(self, cmd: str) -> None:
        """Run a shell command and write output to the focused pane."""
        p = self._workspace.focused_pane
        if not p or not cmd:
            self._flash("  Usage: :exec <command>")
            return

        # Check if parent shell has an API command
        if self._parent_shell and hasattr(self._parent_shell, '_execute_single'):
            self._flash(f"  Running: {cmd}")
            try:
                out = self._parent_shell._execute_single(cmd, "")
                if out:
                    p.write(out.rstrip("\n"))
                self._dirty = True
            except Exception as e:
                p.write(f"  Error: {e}")
                self._dirty = True
            return

        # Fallback: run via subprocess
        self._flash(f"  Running: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                    text=True, timeout=30)
            if result.stdout:
                p.write(result.stdout.rstrip("\n"))
            if result.stderr:
                p.write(f"[stderr]\n{result.stderr.rstrip()}")
            self._dirty = True
        except subprocess.TimeoutExpired:
            p.write("  Command timed out (30s)")
            self._dirty = True
        except Exception as e:
            p.write(f"  Error: {e}")
            self._dirty = True

    def _pane_shell_mode(self) -> None:
        """Enter interactive shell mode in the focused pane."""
        p = self._workspace.focused_pane
        if not p:
            return
        if self._pane_shell:
            # Toggle off
            self._pane_shell = False
            self._pane_input = ""
            self._flash("  Exited pane shell")
            self._dirty = True
            return
        self._pane_shell = True
        self._pane_input = ""
        p._history_pos = len(p._cmd_history)
        p.write(f"\n  ┌── pane shell (type commands, Esc to exit) ──")
        self._dirty = True
        self._flash("  Pane shell active — type commands directly")

    def _show_help(self) -> None:
        help_text = (
            "  i3-style Window Manager — Keybindings:\n"
            f"    {_MOD_NAME}+Enter     New pane\n"
            f"    {_MOD_NAME}+h/v       Split horizontal / vertical\n"
            f"    {_MOD_NAME}+j/k/l     Focus down/up/next\n"
            f"    {_MOD_NAME}+Space     Toggle floating\n"
            f"    {_MOD_NAME}+Shift+q   Close pane\n"
            f"    {_MOD_NAME}+f         Monocle (fullscreen)\n"
            f"    {_MOD_NAME}+s         Stacked (tabs)\n"
            f"    {_MOD_NAME}+e         Toggle split orientation\n"
            f"    {_MOD_NAME}+d         Cycle layouts\n"
            f"    {_MOD_NAME}+1-9       Switch workspace\n"
            f"    {_MOD_NAME}+Shift+N   Move pane to workspace N\n"
            f"    {_MOD_NAME}+r         Resize mode\n"
            f"    {_MOD_NAME}+Arrows    Focus / move floating\n"
            "    Tab / Shift+Tab       Cycle panes\n"
            "    j/k/Up/Down           Scroll pane content\n"
            "    :<cmd>                Command mode\n"
            "    Esc / q               Quit\n"
            "\n"
            "  Commands:\n"
            "    :close / :split-h / :split-v — pane management\n"
            "    :layout / :workspace N / :float / :resize — layout control\n"
            "    :split-equal / :tile — equalize all pane sizes\n"
            "    :monocle / :stacked / :split — layout shortcuts\n"
            "    :font <name> / :reset — settings\n"
            "    :exec <cmd> / :shell — run commands in pane\n"
            "    :write <file> / :load <file> — file I/O for pane buffer\n"
            "    :ls / :pwd / :echo / :date / :grep — built-in commands\n"
            "    :rename <title> / :focus <N> / :hist — pane management\n"
            "    :clear / :ps — pane functions\n"
            "    :help <topic> — detailed help for a command\n"
            "    :q / :quit — exit WM\n"
        )
        ws = self._workspace
        p = ws.focused_pane
        if p:
            p.write(help_text)
            self._dirty = True

    def _show_topic_help(self, topic: str) -> None:
        """Show detailed help for a specific command or topic."""
        topic = topic.strip().lower()
        help_map = {
            "shell": (
                "  Pane Shell Mode — :shell\n"
                "    Enter interactive shell mode in the focused pane.\n"
                "    Type commands directly (no : prefix needed).\n"
                "    Up/Down arrows for command history.\n"
                "    Tab for filename completion.\n"
                "    Ctrl+C or Esc to exit shell mode.\n"
                "    The 'SHELL' indicator appears in the status bar.\n"
            ),
            "exec": (
                "  Execute Command — :exec <cmd>\n"
                "    Run any shell command and display output in the pane.\n"
                "    Uses the parent ShellREPL's command set if available,\n"
                "    otherwise falls back to subprocess.\n"
                "    Short form: :e <cmd> or :run <cmd>\n"
            ),
            "focus": (
                "  Focus Pane — :focus [N]\n"
                "    Without arguments, lists all panes with indices.\n"
                "    With a number, focuses pane N (1-indexed).\n"
                "    Tab / Shift+Tab also cycles focus.\n"
            ),
            "workspace": (
                "  Workspaces — :workspace [N]  or  Mod+N\n"
                "    9 workspaces (1-9) for organizing panes.\n"
                "    :workspace (no arg) shows all workspaces.\n"
                "    :workspace N or Mod+N to switch.\n"
                "    Mod+Shift+N moves focused pane to workspace N.\n"
            ),
            "layout": (
                "  Layouts — :monocle / :stacked / :split / :split-equal\n"
                "    :monocle (or :fullscreen, Mod+f) — single pane fills screen.\n"
                "    :stacked (or :tab, Mod+s) — tabbed panes.\n"
                "    :split (Mod+e) — horizontal split layout.\n"
                "    :split-equal (or :tile) — rebalance all panes to equal size.\n"
                "    :layout — show current layout.\n"
                "    Mod+d cycles through all layouts.\n"
            ),
            "write": (
                "  Write Buffer — :write <filename>\n"
                "    Save the focused pane's buffer contents to a file.\n"
                "    Useful for logging commands or saving output.\n"
            ),
            "load": (
                "  Load File — :load <filename>\n"
                "    Load a file's contents into the focused pane.\n"
                "    The file is appended to the pane buffer.\n"
            ),
            "rename": (
                "  Rename Pane — :rename <title>\n"
                "    Set a custom title for the focused pane.\n"
                "    The title appears in the pane's title bar.\n"
            ),
            "grep": (
                "  Search Buffer — :grep <pattern>\n"
                "    Search the focused pane's buffer for lines containing\n"
                "    the given text. Shows up to 100 matching lines.\n"
            ),
        }
        help_text = help_map.get(topic)
        if help_text:
            ws = self._workspace
            p = ws.focused_pane
            if p:
                p.write(f"\n{help_text}")
                self._dirty = True
        else:
            self._flash(f"  No help for '{topic}'. Try :help for general help.")

    # ── Rendering ─────────────────────────────────────────────────

    def _render(self) -> None:
        if not self._stdscr:
            return
        rows, cols = self._stdscr.getmaxyx()
        if rows < 5 or cols < 10:
            self._stdscr.clear()
            self._stdscr.addstr(0, 0, "Terminal too small")
            self._stdscr.refresh()
            return

        self._stdscr.clear()

        ws = self._workspace
        status_h = 1
        avail_h = rows - status_h - 1  # -1 for input line if in input mode

        # Separate floating and tiled panes
        tiled = [p for p in ws.panes if not p.floating]
        floating = [p for p in ws.panes if p.floating]

        # Compute tiled rectangles
        if tiled:
            rects = _compute_tiled_rects(tiled, ws.layout, 0, 0, cols, avail_h)
            for pane, rect in zip(tiled, rects):
                # Only draw visible (focused) for stacked/monocle
                if ws.layout in (LayoutType.STACKED, LayoutType.MONOCLE):
                    if pane is not ws.focused_pane:
                        continue
                idx = ws.panes.index(pane)
                is_focus = idx == ws.focus_idx
                rx, ry, rw, rh = rect
                self._draw_pane(pane, rx, ry, rw, rh, is_focus, floating=False)

            # Draw tabs for stacked layout
            if ws.layout == LayoutType.STACKED and len(tiled) > 1:
                self._draw_tab_bar(tiled, ws.focus_idx, 0, 0, cols)

        # Draw floating panes
        for p in floating:
            idx = ws.panes.index(p)
            is_focus = idx == ws.focus_idx
            self._draw_pane(p, p.float_x, p.float_y, p.float_w, p.float_h, is_focus, floating=True)

        # Status bar
        self._draw_status_bar(ws, rows, cols)

        # Input line
        if self._input_mode:
            try:
                self._stdscr.addstr(rows - 1, 0, f" :{self._input_buf[1:]}")
            except curses.error:
                pass

        if self._resize_mode:
            self._draw_resize_indicator(rows, cols)

        self._stdscr.refresh()

    def _draw_pane(self, pane: Pane, x: int, y: int, w: int, h: int,
                   focused: bool, floating: bool = False) -> None:
        if w < 4 or h < 2:
            return

        # Reserve bottom line for input when in pane shell mode
        input_line_h = 1 if (focused and self._pane_shell) else 0
        avail_h = h - input_line_h

        border_pair = _PAIR_FOCUS_BORDER if focused else _PAIR_BORDER
        if floating:
            border_pair = _PAIR_FLOATING_BORDER if focused else _PAIR_BORDER

        try:
            # Corners and borders — use avail_h for bottom edge
            ul = curses.ACS_ULCORNER
            ur = curses.ACS_URCORNER
            ll = curses.ACS_LLCORNER
            lr = curses.ACS_LRCORNER
            hl = curses.ACS_HLINE
            vl = curses.ACS_VLINE

            self._stdscr.addch(y, x, ul, border_pair)
            for cx in range(x + 1, x + w - 1):
                self._stdscr.addch(y, cx, hl, border_pair)
            self._stdscr.addch(y, x + w - 1, ur, border_pair)

            for cy in range(y + 1, y + avail_h - 1):
                self._stdscr.addch(cy, x, vl, border_pair)
                self._stdscr.addch(cy, x + w - 1, vl, border_pair)

            self._stdscr.addch(y + avail_h - 1, x, ll, border_pair)
            for cx in range(x + 1, x + w - 1):
                self._stdscr.addch(y + avail_h - 1, cx, hl, border_pair)
            self._stdscr.addch(y + avail_h - 1, x + w - 1, lr, border_pair)
        except curses.error:
            pass

        # Title bar
        title_attr = curses.color_pair(_PAIR_TITLE) | (curses.A_BOLD if focused else curses.A_NORMAL)
        float_mark = " [F]" if pane.floating else ""
        # Show pane index for focused panes
        try:
            pane_idx = self._workspace.panes.index(pane) + 1
        except (ValueError, AttributeError):
            pane_idx = 0
        shell_mark = " [*]" if (focused and self._pane_shell) else ""
        title = f" [{pane_idx}] {pane.title}{float_mark}{shell_mark} " if pane_idx else f" {pane.title}{float_mark}{shell_mark} "
        try:
            self._stdscr.addstr(y, x + 2, title[:w - 4], title_attr)
        except curses.error:
            pass

        # Content (use avail_h for height)
        content_h = max(avail_h - 2, 0)
        if content_h > 0:
            visible = pane.visible_lines(avail_h)
            for line_idx in range(content_h):
                if line_idx < len(visible):
                    line = visible[line_idx][:w - 2]
                else:
                    line = ""
                try:
                    self._stdscr.addstr(y + 1 + line_idx, x + 1, line.ljust(w - 2))
                except curses.error:
                    pass

        # Scroll indicator
        if len(pane.buffer) > content_h:
            pct = int((pane.scroll_offset / max(1, len(pane.buffer) - content_h)) * 100)
            indicator = f" [{pct}%]"
            try:
                self._stdscr.addstr(y + avail_h - 1, x + w - len(indicator) - 1, indicator, curses.A_DIM)
            except curses.error:
                pass

        # Pane shell input line
        if input_line_h:
            input_y = y + avail_h  # one row below pane border
            prompt = f" $ {self._pane_input}"
            prompt = prompt[:w - 2]
            try:
                # Draw input area as a shaded bar
                self._stdscr.addstr(input_y, x, " " * w, curses.A_REVERSE)
                self._stdscr.addstr(input_y, x + 1, prompt, curses.A_REVERSE)
            except curses.error:
                pass

    def _draw_tab_bar(self, panes: list[Pane], focus_idx: int,
                      x: int, y: int, w: int) -> None:
        """Draw tab bar for stacked layout."""
        cx = x
        for i, p in enumerate(panes):
            active = i == focus_idx
            label = f" {p.title} "
            attr = curses.color_pair(_PAIR_TAB_ACTIVE if active else _PAIR_TAB)
            if active:
                attr |= curses.A_BOLD
            try:
                self._stdscr.addstr(y, cx, label[:w - cx - 1], attr)
                cx += len(label)
            except curses.error:
                break

    def _draw_status_bar(self, ws: Workspace, rows: int, cols: int) -> None:
        if cols < 10:
            return
        try:
            self._stdscr.addstr(rows - 1, 0, " " * cols, curses.color_pair(_PAIR_STATUS))
        except curses.error:
            pass

        # Workspace indicators
        ws_indicators = []
        for i, w in enumerate(self._workspaces):
            if i == self._current_ws:
                ws_indicators.append(f"[{w.name}]")
            elif len(w.panes) > 0:
                ws_indicators.append(f" {w.name} ")
            else:
                ws_indicators.append(f"-{w.name}-")

        ws_str = " ".join(ws_indicators)

        # Layout indicator
        layout_str = f"layout: {ws.layout}"

        # Shell mode indicator
        shell_str = " [SHELL]" if self._pane_shell else ""

        # Font indicator
        font_str = f" font: {self._font_name}" if self._font_name else ""

        # Clock
        clock = datetime.now().strftime("%H:%M:%S")

        # Combine
        left_parts = [ws_str, layout_str]
        if font_str:
            left_parts.append(font_str)
        left = " | ".join(left_parts) + shell_str
        right = f"Pane {ws.focus_idx + 1}/{len(ws.panes)} | {clock}"

        # Status message (temporary)
        if self._status_msg and (time.time() - self._status_timer < 3.0):
            left = f" {self._status_msg}"
        else:
            self._status_msg = ""

        if len(left) + len(right) > cols - 2:
            left = left[:cols - len(right) - 4] + ".."

        try:
            self._stdscr.addstr(rows - 1, 0, left, curses.color_pair(_PAIR_STATUS))
            if right and cols - len(right) - 1 > 0:
                self._stdscr.addstr(rows - 1, cols - len(right) - 1, right,
                                    curses.color_pair(_PAIR_STATUS))
        except curses.error:
            pass

    def _draw_resize_indicator(self, rows: int, cols: int) -> None:
        label = " [RESIZE MODE] "
        try:
            self._stdscr.addstr(0, cols - len(label) - 1, label,
                                curses.color_pair(_PAIR_RESIZE) | curses.A_BOLD)
        except curses.error:
            pass

    def _handle_resize(self) -> None:
        if not self._stdscr:
            return
        try:
            rows, cols = self._stdscr.getmaxyx()
            self._dirty = True
        except curses.error:
            pass


# ── Convenience factory ───────────────────────────────────────────────

_wm_instance: Optional[WindowManager] = None


def get_window_manager(parent_shell: Any = None) -> WindowManager:
    global _wm_instance
    if _wm_instance is None:
        _wm_instance = WindowManager(parent_shell=parent_shell)
    return _wm_instance


def reset_window_manager() -> None:
    global _wm_instance
    _wm_instance = None
