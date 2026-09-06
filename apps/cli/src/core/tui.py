"""
TUI — Pure-ANSI terminal UI library. Zero external dependencies.

Builds beautiful terminal interfaces using only Python stdlib
and ANSI escape codes. No Rich, no curses, no external packages.
"""

from __future__ import annotations

import os
import sys
import time
import signal
import threading
import re
import select
import subprocess
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ── ANSI helpers ──────────────────────────────────────────────────

_STDOUT = sys.stdout


def _scrn() -> tuple[int, int]:
    try:
        import shutil
        return shutil.get_terminal_size()
    except Exception:
        return 80, 24


_COLS, _ROWS = _scrn()


def _refresh_size():
    global _COLS, _ROWS
    _COLS, _ROWS = _scrn()


# ── 256-colour palette ────────────────────────────────────────────

_COLOR_OVERRIDES: dict[str, int] = {}


class _FGMeta(type):
    """Metaclass that lets *FG.XXX* resolve to themed overrides when set."""

    def __getattribute__(cls, name: str) -> int:
        overrides = type.__getattribute__(cls, "_OVERRIDES")
        if name in overrides:
            return overrides[name]
        return type.__getattribute__(cls, name)


class FG(metaclass=_FGMeta):
    _OVERRIDES = _COLOR_OVERRIDES
    BLACK = 0
    MAROON = 1
    GREEN = 2
    OLIVE = 3
    NAVY = 4
    PURPLE = 5
    TEAL = 6
    SILVER = 7
    GREY = 8
    RED = 9
    LIME = 10
    YELLOW = 11
    BLUE = 12
    MAGENTA = 13
    CYAN = 14
    WHITE = 15
    PRIMARY = 45       # cyan
    PRIMARY_DIM = 31   # dark cyan
    SUCCESS = 83       # bright green
    SUCCESS_DIM = 35   # dark green
    WARNING = 221      # warm gold
    ERROR = 203        # soft red
    MUTED = 248        # light grey
    HIGHLIGHT = 75     # light blue
    DIM = 242          # dark grey
    INFO = 45          # cyan
    BORDER = 238       # subtle grey
    BORDER_BRIGHT = 244
    GRADIENT_A = 39    # bright cyan start
    GRADIENT_B = 27    # blue end


# ── theme engine ──────────────────────────────────────────────────

_THEME_NAMES: list[str] = ["default", "retro", "mono", "ocean"]
_THEME_IDX: int = 0

_BUILTIN_THEMES: dict[str, dict[str, int]] = {
    "default": {},
    "retro": {
        "PRIMARY": 214, "SUCCESS": 118, "WARNING": 226, "ERROR": 196,
        "GRADIENT_A": 214, "GRADIENT_B": 130,
        "DIM": 240, "MUTED": 246, "BORDER": 239,
        "INFO": 214,
    },
    "mono": {
        "PRIMARY": 255, "SUCCESS": 255, "WARNING": 248, "ERROR": 244,
        "GRADIENT_A": 255, "GRADIENT_B": 244,
        "DIM": 240, "MUTED": 242, "BORDER": 238,
        "INFO": 255,
        "BLACK": 255, "WHITE": 248,
    },
    "ocean": {
        "PRIMARY": 81, "SUCCESS": 85, "WARNING": 221, "ERROR": 203,
        "GRADIENT_A": 81, "GRADIENT_B": 26,
        "DIM": 243, "MUTED": 249, "BORDER": 240,
        "INFO": 81,
    },
}


def set_theme(name: str) -> str:
    """
    Apply a built-in colour theme by name.

    Returns the theme name (lowercase). Unknown names silently fall
    back to ``"default"``.
    """
    global _THEME_IDX
    theme = _BUILTIN_THEMES.get(name, _BUILTIN_THEMES["default"])
    _COLOR_OVERRIDES.clear()
    _COLOR_OVERRIDES.update(theme)
    _THEME_IDX = _THEME_NAMES.index(name) if name in _THEME_NAMES else 0
    return name if theme else "default"


def cycle_theme() -> str:
    """Advance to the next built-in theme and apply it. Returns the new name."""
    global _THEME_IDX
    _THEME_IDX = (_THEME_IDX + 1) % len(_THEME_NAMES)
    set_theme(_THEME_NAMES[_THEME_IDX])
    return _THEME_NAMES[_THEME_IDX]


def current_theme() -> str:
    return _THEME_NAMES[_THEME_IDX]


# ── ANSI escape sequences ─────────────────────────────────────────


def _sgr(*codes: int) -> str:
    return f"\033[{';'.join(str(c) for c in codes)}m"


def _fg(code: int) -> str:
    return _sgr(38, 5, code)


def _bg(code: int) -> str:
    return _sgr(48, 5, code)


_RESET = _sgr(0)
_BOLD = _sgr(1)
_DIM = _sgr(2)
_ITALIC = _sgr(3)
_UNDERLINE = _sgr(4)
_REVERSE = _sgr(7)
_CLEAR = "\033[2J"
_CLEAR_LINE = "\033[2K"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_MOVE_HOME = "\033[H"


# ── box-drawing characters ────────────────────────────────────────


class Box:
    H = "─"
    V = "│"
    H2 = "━"
    V2 = "┃"
    TL = "┌"
    TR = "┐"
    BL = "└"
    BR = "┘"
    TLB = "┏"
    TRB = "┓"
    BLB = "┗"
    BRB = "┛"
    H_DASH = "╌"
    V_DASH = "╎"
    DOT = "●"
    DOT_SM = "•"
    RING = "◌"
    RING_HALF = "◐"
    ARROW = "➜"
    CHEV = "▸"
    CH = "◆"
    BLOCK = "█"
    BLOCK2 = "▓"
    BLOCK3 = "▒"
    BLOCK4 = "░"
    STAR = "✦"
    TRI_R = "▶"


# ── spinner frames ────────────────────────────────────────────────

_SPINNER_FRAMES = ["▹▹▹▹▹", "▸▹▹▹▹", "▹▸▹▹▹", "▹▹▸▹▹", "▹▹▹▸▹", "▹▹▹▹▸", "▹▹▹▹▹"]

_SPINNER_FRAMES_BOUNCE = [
    "⣀⣤⣤⣤⣤",
    "⣤⣀⣤⣤⣤",
    "⣤⣤⣀⣤⣤",
    "⣤⣤⣤⣀⣤",
    "⣤⣤⣤⣤⣀",
]


def _spinner_frame(step: int, frames: list[str] | None = None) -> str:
    f = frames or _SPINNER_FRAMES
    return f[step % len(f)]


# ── styled text helpers ───────────────────────────────────────────


def _strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*[a-zA-Z]", "", text)


def _truncate_visible(s: str, max_vis: int) -> str:
    """Truncate `s` at `max_vis` visible characters, preserving ANSI codes."""
    if max_vis <= 0:
        return ""
    out: list[str] = []
    vis = 0
    i = 0
    while i < len(s) and vis < max_vis:
        if s[i] == "\033":
            out.append(s[i])
            i += 1
            if i < len(s) and s[i] == "[":
                out.append(s[i])
                i += 1
                while i < len(s) and (s[i].isnumeric() or s[i] == ";"):
                    out.append(s[i])
                    i += 1
                if i < len(s):
                    out.append(s[i])
                    i += 1
        else:
            out.append(s[i])
            i += 1
            vis += 1
    return "".join(out)


# ── keyboard input (non-blocking, Unix termios) ─────────────────


def _read_key(timeout: float = 0.01) -> str | None:
    """
    Read one keypress non-blocking. Returns None if no key available.

    Arrow keys → 'arrow_up', 'arrow_down', 'arrow_left', 'arrow_right'
    Number/digit keys → the digit character
    Escape → 'esc'
    'q' → 'q'
    """
    fd = sys.stdin.fileno()
    try:
        import termios, tty
    except ImportError:
        return None
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        if not select.select([sys.stdin], [], [], timeout)[0]:
            return None
        ch = sys.stdin.read(1)
        if ch == "\033":
            time.sleep(0.005)
            if select.select([sys.stdin], [], [], 0)[0]:
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    mapping = {"A": "up", "B": "down", "C": "right", "D": "left"}
                    direction = mapping.get(ch3, ch3)
                    return f"arrow_{direction}"
                return f"esc_{ch2}"
            return "esc"
        return ch
    except Exception:
        return None
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


# ── word-wrap ANSI-safe ──────────────────────────────────────────


def _word_wrap_lines(
    colourised_lines: list[str], max_vis: int
) -> list[str]:
    """
    Word-wrap a list of colourised log lines so each visual line fits *max_vis* chars.

    ANSI colour codes are detected and re-applied to each wrapped fragment.
    """
    out: list[str] = []
    for raw in colourised_lines:
        stripped = _strip_ansi(raw)
        visible_len = len(stripped)
        if visible_len <= max_vis:
            out.append(raw)
            continue

        # Extract leading ANSI prefix (one or more SGR sequences)
        prefix_match = re.match(r"^(\033\[[0-9;]*m)+", raw)
        prefix = prefix_match.group(0) if prefix_match else ""

        words = stripped.split(" ")
        current = ""
        for word in words:
            candidate = (current + " " + word).strip()
            if len(candidate) <= max_vis:
                current = candidate
            else:
                if current:
                    out.append(prefix + current + _RESET)
                current = word
        if current:
            out.append(prefix + current + _RESET)
    return out


def styled(text: str, fg: int = FG.WHITE, bold: bool = False, dim: bool = False,
           italic: bool = False, underline: bool = False, reverse: bool = False,
           bg: int | None = None) -> str:
    codes = []
    if bold:
        codes.append(1)
    if dim:
        codes.append(2)
    if italic:
        codes.append(3)
    if underline:
        codes.append(4)
    if reverse:
        codes.append(7)
    codes.extend([38, 5, fg])
    if bg is not None:
        codes.extend([48, 5, bg])
    return f"\033[{';'.join(str(c) for c in codes)}m{text}{_RESET}"


def fg_text(text: str, color: int) -> str:
    return f"{_fg(color)}{text}{_RESET}"


def _gradient(text: str, start: int, end: int) -> str:
    n = max(len(text), 1)
    out = []
    for i, ch in enumerate(text):
        t = i / (n - 1) if n > 1 else 0.5
        c = int(start + (end - start) * t)
        out.append(f"{_fg(c)}{ch}")
    out.append(_RESET)
    return "".join(out)


def _status_tag(state: str) -> str:
    return {
        "starting": f"{_fg(FG.WARNING)}{Box.DOT} STARTING{_RESET}",
        "ready": f"{_fg(FG.SUCCESS)}{Box.DOT} READY{_RESET}",
        "error": f"{_fg(FG.ERROR)}{Box.DOT} ERROR{_RESET}",
    }.get(state, f"{_fg(FG.DIM)}{Box.RING} ?{_RESET}")


# ── Panel ─────────────────────────────────────────────────────────


def render_panel(
    title: str,
    content: str,
    *,
    width: int = 0,
    border_fg: int = FG.BORDER,
    title_fg: int = FG.PRIMARY,
    subtitle: str = "",
    subtitle_fg: int = FG.DIM,
    padding_y: int = 1,
    padding_x: int = 2,
    box_chars: tuple[str, str, str, str, str, str] | None = None,
    fill: bool = False,
) -> str:
    """
    Render a bordered panel with title and optional subtitle.

    box_chars: (tl, tr, bl, br, h, v) — defaults to light rounded.
    fill: if True, fill remaining terminal height with empty lines.
    """
    w = width or _COLS
    bc = box_chars or (Box.TL, Box.TR, Box.BL, Box.BR, Box.H, Box.V)
    tl, tr, bl, br, h, v = bc

    inner_w = w - 2
    pad_l = " " * padding_x
    lines: list[str] = []

    # ── top border with title ──────────────────────────────────
    top = f"{_fg(border_fg)}{tl}{h}"
    ttl = f"  {title}  "
    remaining = inner_w - len(_strip_ansi(ttl)) - 1
    if remaining >= 0:
        top += f"{_fg(title_fg)}{ttl}{_RESET}{_fg(border_fg)}{h * remaining}{tr}{_RESET}"
    else:
        top += f"{h * (inner_w - 1)}{tr}{_RESET}"
    lines.append(top)

    # ── padding top ────────────────────────────────────────────
    for _ in range(padding_y):
        lines.append(f"{_fg(border_fg)}{v}{_RESET}{' ' * inner_w}{_fg(border_fg)}{v}{_RESET}")

    # ── content ────────────────────────────────────────────────
    max_content_vis = inner_w - padding_x
    if content:
        for line in content.split("\n"):
            visible_len = len(_strip_ansi(line))
            if visible_len > max_content_vis:
                stripped = _truncate_visible(line, max_content_vis)
            else:
                stripped = line
            pad = inner_w - len(_strip_ansi(stripped))
            filler = max(0, pad - padding_x)
            lines.append(
                f"{_fg(border_fg)}{v}{_RESET}{pad_l}{stripped}{' ' * filler}{_fg(border_fg)}{v}{_RESET}"
            )

    # ── subtitle ───────────────────────────────────────────────
    if subtitle:
        sub = f"  {subtitle}  "
        sub_visible = len(_strip_ansi(sub))
        sub_pad = max(0, inner_w - sub_visible)
        lines.append(
            f"{_fg(border_fg)}{v}{_RESET}{_fg(subtitle_fg)}{sub}{_RESET}{' ' * sub_pad}{_fg(border_fg)}{v}{_RESET}"
        )

    # ── padding bottom ─────────────────────────────────────────
    for _ in range(padding_y):
        lines.append(f"{_fg(border_fg)}{v}{_RESET}{' ' * inner_w}{_fg(border_fg)}{v}{_RESET}")

    # ── bottom border ──────────────────────────────────────────
    bottom = f"{_fg(border_fg)}{bl}{h * inner_w}{br}{_RESET}"
    lines.append(bottom)

    # ── fill remaining terminal height ─────────────────────────
    if fill:
        total_rows = _ROWS
        current = len(lines)
        remaining_rows = total_rows - current
        if remaining_rows > 0:
            for _ in range(remaining_rows):
                lines.append(f"{' ' * w}")

    return "\n".join(lines)


# ── Elegant gradient header ───────────────────────────────────────


def render_gradient_header(
    title: str,
    subtitle: str = "",
    info: dict | None = None,
    width: int = 0,
    step: int = 0,
) -> str:
    """
    A beautiful header with a gradient title bar, info grid, and clean borders.
    """
    w = width or _COLS
    inner = w - 2
    lines: list[str] = []

    # ── top border bar (solid colour) ──────────────────────────
    left_cap = f"{_fg(FG.GRADIENT_B)}{Box.TLB}{_RESET}"
    right_cap = f"{_fg(FG.GRADIENT_B)}{Box.TRB}{_RESET}"
    bar = f"{_fg(FG.GRADIENT_B)}{Box.H2 * (inner)}{_RESET}"
    lines.append(f"{left_cap}{bar}{right_cap}")

    # ── title line (gradient text, heavy sides) ────────────────
    styled_title = _gradient(f"  {title}  ", FG.GRADIENT_A, FG.GRADIENT_B)
    left_v = f"{_fg(FG.GRADIENT_B)}{Box.V2}{_RESET}"
    right_v = f"{_fg(FG.GRADIENT_B)}{Box.V2}{_RESET}"
    remaining = inner - len(_strip_ansi(styled_title))
    if remaining >= 0:
        lines.append(
            f"{left_v}{styled_title}{' ' * remaining}{right_v}"
        )
    else:
        lines.append(f"{left_v}{' ' * inner}{right_v}")

    # ── divider ────────────────────────────────────────────────
    div = f"{_fg(FG.BORDER)}{Box.V_DASH}{Box.H * (inner - 2)}{Box.V_DASH}{_RESET}"
    lines.append(f"{left_v}{div}{right_v}")

    # ── info grid (2 columns) ──────────────────────────────────
    if info:
        items = list(info.items())
        pairs = []
        for i in range(0, len(items), 2):
            pair = items[i]
            if i + 1 < len(items):
                pairs.append((pair, items[i + 1]))
            else:
                pairs.append((pair, ("", "")))
        for (k1, v1), (k2, v2) in pairs:
            col_w = inner // 2
            col1 = f"{_fg(FG.DIM)}{k1}:{_RESET}  {_fg(FG.WHITE)}{v1}{_RESET}"
            col2 = f"{_fg(FG.DIM)}{k2}:{_RESET}  {_fg(FG.WHITE)}{v2}{_RESET}" if k2 else ""
            row = f"{left_v}  {col1}{' ' * max(1, col_w - len(_strip_ansi(col1)))}  {col2}{' ' * max(1, inner - col_w - len(_strip_ansi(col2)) - 4)}{right_v}"
            lines.append(row)

    # ── bottom border ──────────────────────────────────────────
    lines.append(
        f"{_fg(FG.GRADIENT_B)}{Box.BLB}{_RESET}"
        f"{_fg(FG.GRADIENT_B)}{Box.H2 * inner}{_RESET}"
        f"{_fg(FG.GRADIENT_B)}{Box.BRB}{_RESET}"
    )

    return "\n".join(lines)


# ── Metrics row ──────────────────────────────────────────────────


def _round_int(v: float) -> int:
    return int(v + 0.5)


def render_metrics_row(
    cpu: float,
    memory: float,
    disk: float,
    width: int = 0,
) -> str:
    """
    Render a resource usage bar row between header and tab bar.

    Shows CPU / Memory / Disk as horizontal bar meters with block chars.
    """
    w = width or _COLS
    bar_w = 10
    gutter = 2  # left/right gutter in spaces

    def _meter(val: float) -> str:
        filled = _round_int(val / 100 * bar_w)
        filled = max(0, min(bar_w, filled))
        empty = bar_w - filled
        fg_c = FG.SUCCESS if val < 60 else (FG.WARNING if val < 85 else FG.ERROR)
        return (
            f"{_fg(fg_c)}{Box.BLOCK * filled}{_RESET}"
            f"{_fg(FG.DIM)}{Box.BLOCK4 * empty}{_RESET}"
            f" {_fg(fg_c)}{_round_int(val):>3d}%{_RESET}"
        )

    cpu_m = _meter(cpu)
    mem_m = _meter(memory)
    disk_m = _meter(disk)

    label_cpu = f"{_fg(FG.MUTED)}CPU{_RESET}"
    label_mem = f"{_fg(FG.MUTED)}MEM{_RESET}"
    label_disk = f"{_fg(FG.MUTED)}DSK{_RESET}"

    row = f"{' ' * gutter}{label_cpu}  {cpu_m}   {label_mem}  {mem_m}   {label_disk}  {disk_m}  "
    row_pad = max(0, w - len(_strip_ansi(row)))
    row_full = f"{row}{' ' * row_pad}"

    div_w = max(0, w - gutter * 2 - 2)
    div = f"{' ' * gutter}{_fg(FG.BORDER)}{Box.V_DASH}{Box.H * div_w}{Box.V_DASH}{_RESET}"
    div_pad = max(0, w - len(_strip_ansi(div)))
    div_full = f"{div}{' ' * div_pad}"

    return f"{div_full}\n{row_full}\n{div_full}"


def render_tab_bar(
    tabs: list[tuple[str, str, str]],
    active_id: str,
    width: int = 0,
    step: int = 0,
    badges: dict[str, tuple[int, int]] | None = None,
    new_counts: dict[str, int] | None = None,
) -> str:
    """
    Render a tab bar with bounce-animated loading spinners.

    Uses FG-colour stacking (never mid-string RESET) so active-tab
    background highlighting is not broken by the dot colour.

    Args:
        tabs: (id, label, state) where state is starting/ready/error
        active_id: currently selected tab id
        width: total width
        step: animation frame counter
        badges: optional per-tab (errors, warnings) counts
        new_counts: optional per-tab unread new log count
    """
    w = width or _COLS
    inner = w - 2
    parts: list[str] = []
    badges = badges or {}
    new_counts = new_counts or {}

    for tid, label, state in tabs:
        is_active = tid == active_id

        if state == "starting":
            dot_glyph = _spinner_frame(step, _SPINNER_FRAMES_BOUNCE)
            dot_fg_code = FG.WARNING
        elif state == "ready":
            dot_glyph = Box.DOT
            dot_fg_code = FG.SUCCESS
        elif state == "error":
            dot_glyph = Box.DOT
            dot_fg_code = FG.ERROR
        else:
            dot_glyph = Box.RING
            dot_fg_code = FG.DIM

        # optional count badges (errors / warnings)
        badge_str = ""
        err_cnt, warn_cnt = badges.get(tid, (0, 0))
        if err_cnt > 0:
            badge_str += f" {_fg(FG.ERROR)}{err_cnt}{Box.DOT}{_RESET}"
        if warn_cnt > 0:
            badge_str += f" {_fg(FG.WARNING)}{warn_cnt}{Box.DOT_SM}{_RESET}"

        # unread new-log indicator
        n = new_counts.get(tid, 0)
        if n > 0 and not is_active:
            badge_str += f" {_fg(FG.INFO)}+{n}{_RESET}"

        if is_active:
            # FG-stacking: set bg once, override FG for dot, restore FG for label
            entry = (
                f"{_bg(FG.PRIMARY)}{_fg(FG.BLACK)}"
                f"  {_fg(dot_fg_code)}{dot_glyph}{_fg(FG.BLACK)}  {label}  "
                f"{badge_str}"
                f"{_RESET}"
            )
        else:
            entry = (
                f"  {_fg(dot_fg_code)}{dot_glyph}{_RESET}  {label}  "
                f"{badge_str}"
            )
        parts.append(entry)

    bar = f"  {_fg(FG.DIM)}{Box.V_DASH}{_RESET}  ".join(parts)
    if len(_strip_ansi(bar)) > inner:
        bar = _truncate_visible(bar, inner - 3) + "..."

    return (
        f"{_fg(FG.BORDER)}{Box.H_DASH * w}{_RESET}\n"
        f"{bar}{' ' * max(0, w - len(_strip_ansi(bar)))}\n"
        f"{_fg(FG.BORDER)}{Box.H_DASH * w}{_RESET}"
    )


# ── Footer ────────────────────────────────────────────────────────


def render_footer(
    text: str,
    width: int = 0,
    fg: int = FG.MUTED,
    elapsed: str = "",
) -> str:
    """Render an elegant footer with optional elapsed time."""
    w = width or _COLS
    inner = w - 2

    elapsed_tag = f"  {_fg(FG.DIM)}{elapsed}{_RESET}" if elapsed else ""
    bar = f"  {_fg(fg)}{text}{_RESET}{elapsed_tag}"
    if len(_strip_ansi(bar)) > inner:
        bar = _truncate_visible(bar, inner - 3) + "..."

    pad = max(0, w - len(_strip_ansi(bar)) - 1)
    return (
        f"{_fg(FG.BORDER)}{Box.H_DASH * w}{_RESET}\n"
        f" {bar}{' ' * pad}\n"
        f"{_fg(FG.BORDER)}{Box.H_DASH * w}{_RESET}"
    )


# ── Live Display ──────────────────────────────────────────────────


class LiveDisplay:
    """
    Full-screen live-updating display using ANSI cursor control.

    Usage::

        with LiveDisplay() as display:
            while running:
                display.update(render_some_content())
                time.sleep(0.05)
    """

    def __init__(self, refresh_rate: float = 10):
        self._rate = refresh_rate
        self._prev = ""
        self._prev_line_count = 0
        self._running = False

    def __enter__(self) -> LiveDisplay:
        _refresh_size()
        sys.stdout.write(_HIDE_CURSOR)
        sys.stdout.write(_CLEAR)
        sys.stdout.write(_MOVE_HOME)
        sys.stdout.flush()
        self._running = True
        return self

    def __exit__(self, *args):
        self._running = False
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.write(_MOVE_HOME)
        sys.stdout.write(_CLEAR)
        sys.stdout.flush()

    def update(self, renderable: str):
        """Replace screen content without causing terminal scroll."""
        _refresh_size()
        max_rows = max(1, _ROWS - 1)
        lines = renderable.rstrip("\n").split("\n")
        visible = lines[:max_rows]
        buf = []
        for i, line in enumerate(visible, 1):
            buf.append(f"\033[{i};1H{line}\033[K")
        for i in range(len(visible) + 1, min(self._prev_line_count, max_rows) + 1):
            buf.append(f"\033[{i};1H\033[K")
        sys.stdout.write("".join(buf))
        self._prev = renderable
        self._prev_line_count = len(visible)
        sys.stdout.flush()


# ── colourise log lines ───────────────────────────────────────────


def _colourise(line: str) -> str:
    """Apply ANSI colour to a log line based on content keywords."""
    if "ERROR" in line or "CRITICAL" in line or "Traceback" in line:
        return f"{_fg(FG.ERROR)}{line}{_RESET}"
    if "WARNING" in line or "WARN" in line:
        return f"{_fg(FG.WARNING)}{line}{_RESET}"
    if "200" in line or "3xx" in line.lower() or "success" in line.lower():
        return f"{_fg(FG.SUCCESS)}{line}{_RESET}"
    if any(kw in line.lower() for kw in ("ready", "started", "listening", "running on", "complete", "compiled")):
        return f"{_fg(FG.SUCCESS)}{line}{_RESET}"
    if "INFO" in line:
        return f"{_fg(FG.INFO)}{line}{_RESET}"
    return f"{_fg(FG.WHITE)}{line}{_RESET}"


def _highlight_search(lines: list[str], term: str) -> list[str]:
    """
    Filter and highlight log lines by search term (case-insensitive).

    Returns only matching lines with matched substrings wrapped in
    reverse-video ANSI. Lines already have ``_colourise()`` wrapping
    (colour prefix + text + RESET suffix).
    """
    if not term:
        return lines
    term_lower = term.lower()
    out: list[str] = []
    for coloured in lines:
        stripped = _strip_ansi(coloured)
        if term_lower not in stripped.lower():
            continue
        # Find all match positions in visible text
        matches: list[tuple[int, int]] = []
        start = 0
        stripped_lower = stripped.lower()
        while True:
            idx = stripped_lower.find(term_lower, start)
            if idx < 0:
                break
            matches.append((idx, idx + len(term)))
            start = idx + len(term)
        # Extract ANSI prefix (everything before visible text)
        prefix = coloured[:len(coloured) - len(stripped) - len(_RESET)]
        suffix = _RESET
        result = prefix
        pos = 0
        for m_start, m_end in matches:
            if m_start > pos:
                result += stripped[pos:m_start]
            result += f"{_REVERSE}{stripped[m_start:m_end]}{_RESET}"
            if m_end < len(stripped):
                result += prefix
            pos = m_end
        if pos < len(stripped):
            result += stripped[pos:]
        result += suffix
        out.append(result)
    return out


# ── TabConfig dataclass ───────────────────────────────────────────


@dataclass
class TabConfig:
    """Configuration for a single dev-server log tab."""

    id: str
    title: str
    lines: deque
    port: int = 0
    url_path: str = ""


# ── DevDashboard ──────────────────────────────────────────────────


class DevDashboard:
    """
    Tabbed live-updating dev server dashboard.

    Pure-ANSI rendering — no external dependencies. Beautiful
    gradient header, animated spinners, colourised logs.

    Usage::

        dashboard = DevDashboard(
            title="My App",
            tabs=[TabConfig("api", "API", log_deque, port=8000)],
            info={"Repo": "/path"},
        )
        dashboard.serve(stop_check=lambda: flag)
    """

    def __init__(
        self,
        title: str = "SloughGPT Dev Server",
        tabs: list[TabConfig] | None = None,
        info: dict | None = None,
        on_restart: Optional[Callable[[], bool]] = None,
    ):
        self._title = title
        self._tabs: list[TabConfig] = tabs or []
        self._tab_map: dict[str, TabConfig] = {t.id: t for t in self._tabs}
        self._states: dict[str, str] = {t.id: "starting" for t in self._tabs}
        self._active_tab: str = self._tabs[0].id if self._tabs else ""
        self._info: dict = info or {}
        if "Theme" not in self._info:
            self._info["Theme"] = "Default"
        if self._restarting:
            self._info["Status"] = "RESTARTING"
        elif "Status" in self._info:
            del self._info["Status"]
        self._shutdown = False
        self._start_time = time.monotonic()
        self._frame = 0
        self._startup_phase = True
        self._on_restart = on_restart
        self._restarting = False

        # scroll support: scroll offset per tab (0 = latest)
        self._scroll_offsets: dict[str, int] = {t.id: 0 for t in self._tabs}
        self._scroll_follow: dict[str, bool] = {t.id: True for t in self._tabs}

        # last activity per tab (low-res timestamp)
        self._last_activity: dict[str, float] = {t.id: 0.0 for t in self._tabs}

        # resource metrics (collected every ~2s in serve loop)
        self._metrics: dict[str, float] = {"cpu": 0.0, "memory": 0.0, "disk": 0.0}
        self._last_metrics_collect: float = 0.0
        self._linux_cpu_sample: tuple[float, float] | None = None  # (idle, total)

        # search/filter mode
        self._search_mode: bool = False
        self._search_query: str = ""

        # help overlay
        self._show_help: bool = False

        # tracked line count per tab (for "N new" indicator)
        self._seen_line_count: dict[str, int] = {t.id: len(t.lines) for t in self._tabs}

    def set_status(self, tab_id: str, state: str):
        if tab_id in self._states:
            was = self._states[tab_id]
            self._states[tab_id] = state
            if state == "ready" and was != "ready":
                self._last_activity[tab_id] = time.monotonic()
                self._startup_phase = False

    def set_info(self, key: str, value: str):
        self._info[key] = value

    def set_metrics(self, cpu: float = 0, memory: float = 0, disk: float = 0):
        """Push external resource metrics. Overrides auto-collection."""
        self._metrics["cpu"] = max(0, min(100, cpu))
        self._metrics["memory"] = max(0, min(100, memory))
        self._metrics["disk"] = max(0, min(100, disk))
        self._last_metrics_collect = time.monotonic()

    def _collect_metrics(self):
        """Collect system metrics (CPU / memory / disk) using subprocess calls."""
        now = time.monotonic()
        if now - self._last_metrics_collect < 2.0:
            return
        self._last_metrics_collect = now

        m: dict[str, float] = {"cpu": 0.0, "memory": 0.0, "disk": 0.0}

        import platform
        is_linux = platform.system() == "Linux"

        if is_linux:
            # CPU via /proc/stat (non-blocking: store sample, compute delta next call)
            try:
                with open("/proc/stat") as f:
                    line = f.readline()
                parts = line.split()
                # user, nice, system, idle, iowait, irq, softirq, steal
                vals = [int(x) for x in parts[1:9]]
                idle = vals[3] + vals[4]
                total = sum(vals)
                if self._linux_cpu_sample is not None:
                    prev_idle, prev_total = self._linux_cpu_sample
                    d_idle = idle - prev_idle
                    d_total = total - prev_total
                    if d_total > 0:
                        m["cpu"] = min(100, max(0, (1 - d_idle / d_total) * 100))
                self._linux_cpu_sample = (idle, total)
            except Exception:
                pass

            # Memory via /proc/meminfo
            try:
                info = {}
                with open("/proc/meminfo") as f:
                    for line in f:
                        if ":" in line:
                            key, val = line.split(":", 1)
                            # values are in kB, strip " kB"
                            info[key.strip()] = int(val.split()[0])
                total = info.get("MemTotal", 0)
                available = info.get("MemAvailable", info.get("MemFree", 0))
                if total > 0:
                    used = total - available
                    m["memory"] = min(100, used / total * 100)
            except Exception:
                pass
        else:
            # macOS — CPU via top
            try:
                out = subprocess.check_output(
                    ["top", "-l", "1", "-n", "0"],
                    timeout=3, stderr=subprocess.DEVNULL, text=True,
                )
                user = sys_v = 0.0
                for line in out.split("\n"):
                    if "CPU usage" in line:
                        parts = line.replace(",", "").split()
                        for i, p in enumerate(parts):
                            if p == "user" and i > 0:
                                user = float(parts[i - 1].rstrip("%"))
                            elif p == "sys" and i > 0:
                                sys_v = float(parts[i - 1].rstrip("%"))
                        m["cpu"] = min(100, user + sys_v)
                        break
            except Exception:
                pass

            # Memory via vm_stat
            try:
                out = subprocess.check_output(
                    ["vm_stat"],
                    timeout=3, stderr=subprocess.DEVNULL, text=True,
                )
                pages = {}
                for line in out.split("\n"):
                    if ":" in line:
                        key, val = line.split(":", 1)
                        val = val.strip().rstrip(".")
                        try:
                            pages[key.strip()] = int(val)
                        except ValueError:
                            pass
                active = pages.get("Pages active", 0)
                wired = pages.get("Pages wired down", 0)
                compressed = pages.get("Pages stored in compressor", 0)
                free = pages.get("Pages free", 0)
                total = active + wired + compressed + free
                if total > 0:
                    used = active + wired + compressed
                    m["memory"] = min(100, used / total * 100)
            except Exception:
                pass

        # Disk via df (works on both Linux and macOS)
        try:
            out = subprocess.check_output(
                ["df", "-k", "."],
                timeout=3, stderr=subprocess.DEVNULL, text=True,
            )
            lines = out.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 5:
                    used = int(parts[2])
                    total = int(parts[1])
                    if total > 0:
                        m["disk"] = min(100, used / total * 100)
        except Exception:
            pass

        self._metrics = m

    def _mark_active_seen(self):
        """Record the current line count of the active tab as 'seen'."""
        tab = self._tab_map.get(self._active_tab)
        if tab:
            self._seen_line_count[self._active_tab] = len(tab.lines)

    def handle_arrow_key(self, key: str) -> bool:
        """Handle a keyboard key. Returns True if display should refresh immediately."""
        if key == "arrow_left":
            self._cycle_tab(-1)
            return True
        if key == "arrow_right":
            self._cycle_tab(1)
            return True
        if key == "arrow_up":
            self._scroll(-1)
            return True
        if key == "arrow_down":
            self._scroll(1)
            return True
        if key in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            idx = int(key) - 1
            if idx < len(self._tabs):
                self._active_tab = self._tabs[idx].id
                self._mark_active_seen()
            return True
        if key == "t":
            new_theme = cycle_theme()
            self._info["Theme"] = new_theme.title()
            return True
        if key == " ":
            tid = self._active_tab
            if tid:
                self._scroll_follow[tid] = not self._scroll_follow.get(tid, True)
                if self._scroll_follow[tid]:
                    self._scroll_offsets[tid] = 0
            return True
        return False

    def _cycle_tab(self, direction: int):
        if not self._tabs:
            return
        idx = next(i for i, t in enumerate(self._tabs) if t.id == self._active_tab)
        idx = (idx + direction) % len(self._tabs)
        self._active_tab = self._tabs[idx].id
        self._mark_active_seen()

    def _scroll(self, direction: int):
        """Adjust scroll offset for active tab. Direction: -1=up, +1=down."""
        tab = self._tab_map.get(self._active_tab)
        if not tab or not tab.lines:
            return
        total = len(tab.lines)
        max_vis = max(0, total - 50)
        current = self._scroll_offsets.get(self._active_tab, 0)
        # up arrow (-1) → larger offset (skip more recent lines)
        # down arrow (+1) → smaller offset (show more recent lines)
        new = current - direction
        if new < 0:
            new = 0
        if new > max_vis:
            new = max_vis
        self._scroll_offsets[self._active_tab] = new
        if current == new:
            return  # no movement — don't toggle follow
        if direction < 0:
            self._scroll_follow[self._active_tab] = False
        if new == 0:
            self._scroll_follow[self._active_tab] = True

    def serve(self, stop_check: Optional[Callable[[], bool]] = None):
        """Render the dashboard live until stop_check or Ctrl+C.

        Keyboard controls:
          ←/→      switch tabs
          ↑/↓      scroll log history
          1-9      jump to tab by number
          /        search/filter logs
          ?        help overlay
          C        clear log buffer (shift+c)
          t        cycle colour theme
          Space    toggle auto-scroll
          q        quit
          Ctrl+C   quit
        """
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, "_shutdown", True))
        signal.signal(signal.SIGTERM, lambda s, f: setattr(self, "_shutdown", True))

        try:
            with LiveDisplay() as display:
                while not self._shutdown:
                    if stop_check and stop_check():
                        self._shutdown = True
                        break

                    # startup animation: rapid pulsing frames for first second
                    if self._startup_phase and self._frame < 15:
                        rendered = self._render()
                        display.update(rendered)
                        self._frame += 1
                        time.sleep(0.07)
                        continue

                    # keyboard input (non-blocking)
                    key = _read_key(0.04)
                    if key == "q" and not self._search_mode:
                        self._shutdown = True
                        break
                    if key:

                        # ── search mode handling ─────────────────
                        if self._search_mode:
                            if key == "esc" or key == "\r" or key == "\n":
                                self._search_mode = False
                                self._search_query = ""
                                rendered = self._render()
                                display.update(rendered)
                                self._frame += 1
                                continue
                            if key in ("\x7f", "\b"):  # backspace
                                self._search_query = self._search_query[:-1]
                                rendered = self._render()
                                display.update(rendered)
                                self._frame += 1
                                continue
                            if len(key) == 1 and key.isprintable():
                                self._search_query += key
                                rendered = self._render()
                                display.update(rendered)
                                self._frame += 1
                                continue
                            # fall through to normal handling for arrow keys etc

                        # ── enter search mode ────────────────────
                        if key == "/" and not self._search_mode:
                            self._search_mode = True
                            self._search_query = ""
                            rendered = self._render()
                            display.update(rendered)
                            self._frame += 1
                            continue

                        # ── help overlay toggle ───────────────────
                        if key == "?":
                            self._show_help = not self._show_help
                            rendered = self._render()
                            display.update(rendered)
                            self._frame += 1
                            continue
                        if self._show_help and key in ("esc", "q"):
                            self._show_help = False
                            rendered = self._render()
                            display.update(rendered)
                            self._frame += 1
                            continue

                        # ── clear logs ────────────────────────────
                        if key == "C":
                            active_tab = self._tab_map.get(self._active_tab)
                            if active_tab:
                                active_tab.lines.clear()
                                self._scroll_offsets[self._active_tab] = 0
                            rendered = self._render()
                            display.update(rendered)
                            self._frame += 1
                            continue

                        # ── restart servers ──────────────────────
                        if key == "r" and self._on_restart and not self._restarting:
                            self._restarting = True
                            rendered = self._render()
                            display.update(rendered)
                            # Run restart in a thread so the UI stays responsive
                            def _do_restart():
                                try:
                                    success = self._on_restart()
                                except Exception:
                                    success = False
                                self._restarting = False
                                if success:
                                    # Reset states to starting
                                    for tid in self._states:
                                        self._states[tid] = "starting"
                                    self._startup_phase = True
                                    self._start_time = time.monotonic()
                            threading.Thread(target=_do_restart, daemon=True).start()
                            self._frame += 1
                            continue

                        self.handle_arrow_key(key)
                        rendered = self._render()
                        display.update(rendered)
                        self._frame += 1
                        continue

                    rendered = self._render()
                    self._frame += 1

                    if all(s == "error" for s in self._states.values()):
                        display.update(rendered)
                        time.sleep(3)
                        break

                    display.update(rendered)
                    time.sleep(0.08)
        except Exception:
            pass

    def _render_help_overlay(self, w: int) -> str:
        """Render keyboard help as a centered panel replacing the log panel."""
        help_lines = [
            ("Navigation", [
                ("← / →", "Switch tabs"),
                ("1-9", "Jump to tab"),
                ("↑ / ↓", "Scroll log history"),
                ("Space", "Toggle auto-scroll"),
            ]),
            ("Search & Filter", [
                ("/", "Search/filter logs"),
                ("Esc / Enter", "Exit search"),
                ("C", "Clear log buffer"),
            ]),
            ("Display", [
                ("t", "Cycle colour theme"),
                ("?", "Toggle this help"),
            ]),
            ("General", [
                ("r", "Restart servers"),
                ("q / Ctrl+C", "Quit dashboard"),
            ]),
        ]
        inner = w - 4
        lines: list[str] = []

        # top border
        lines.append(f"{_fg(FG.PRIMARY)}{Box.TL}{Box.H * (inner)}{Box.TR}{_RESET}")

        # title
        title = " Keyboard Shortcuts "
        pad = inner - len(title)
        left_pad = pad // 2
        right_pad = pad - left_pad
        lines.append(
            f"{_fg(FG.PRIMARY)}{Box.V}{_RESET}"
            f"{' ' * left_pad}{_fg(FG.WHITE)}{_BOLD}{title}{_RESET}{' ' * right_pad}"
            f"{_fg(FG.PRIMARY)}{Box.V}{_RESET}"
        )

        # divider
        lines.append(
            f"{_fg(FG.PRIMARY)}{Box.V}{_RESET}"
            f"{_fg(FG.BORDER)}{Box.H * inner}{_RESET}"
            f"{_fg(FG.PRIMARY)}{Box.V}{_RESET}"
        )

        for section_name, shortcuts in help_lines:
            # section header
            section_line = f"  {_fg(FG.INFO)}{section_name}{_RESET}"
            lines.append(
                f"{_fg(FG.PRIMARY)}{Box.V}{_RESET}"
                f"{section_line}{' ' * max(0, inner - len(_strip_ansi(section_line)))}"
                f"{_fg(FG.PRIMARY)}{Box.V}{_RESET}"
            )

            for key_desc, action in shortcuts:
                entry = f"    {_fg(FG.WHITE)}{key_desc}{_RESET}  {_fg(FG.DIM)}{action}{_RESET}"
                lines.append(
                    f"{_fg(FG.PRIMARY)}{Box.V}{_RESET}"
                    f"{entry}{' ' * max(0, inner - len(_strip_ansi(entry)))}"
                    f"{_fg(FG.PRIMARY)}{Box.V}{_RESET}"
                )

        # bottom border
        lines.append(f"{_fg(FG.PRIMARY)}{Box.BL}{Box.H * (inner)}{Box.BR}{_RESET}")

        return render_panel(
            "Help",
            "\n".join(lines),
            width=w,
            border_fg=FG.PRIMARY,
            subtitle=f"  {_fg(FG.DIM)}Press ? or Esc to close{_RESET}",
        )

    def _render(self) -> str:
        _refresh_size()
        w = _COLS
        parts: list[str] = []

        # ── startup pulsing overlay ──────────────────────────
        if self._startup_phase:
            pulse_idx = self._frame % 4
            pulse_chars = [Box.BLOCK4, Box.BLOCK3, Box.BLOCK2, Box.BLOCK]
            pulse = pulse_chars[pulse_idx]
            bar = pulse * (w - 2)
            parts.append(
                f"{_fg(FG.GRADIENT_B)}{Box.TLB}{_RESET}"
                f"{_fg(FG.WARNING)}{bar}{_RESET}"
                f"{_fg(FG.GRADIENT_B)}{Box.TRB}{_RESET}"
            )
            msg = f"  {Box.DOT}  Booting servers  {Box.DOT_SM}  {_spinner_frame(self._frame // 2)}  "
            pad_w = max(0, w - len(msg) - 6)
            parts.append(
                f"{_fg(FG.GRADIENT_B)}{Box.V2}{_RESET}  "
                f"{_fg(FG.WARNING)}{msg}{_RESET}{' ' * pad_w}"
                f"  {_fg(FG.GRADIENT_B)}{Box.V2}{_RESET}"
            )
            parts.append(
                f"{_fg(FG.GRADIENT_B)}{Box.BLB}{_RESET}"
                f"{_fg(FG.WARNING)}{Box.H2 * (w - 2)}{_RESET}"
                f"{_fg(FG.GRADIENT_B)}{Box.BRB}{_RESET}"
            )

            # Add spacer rows
            for _ in range(3):
                parts.append("")

        # ── gradient header ───────────────────────────────────
        parts.append(render_gradient_header(
            self._title,
            info=self._info,
            width=w,
            step=self._frame,
        ))

        # ── resource metrics bar ──────────────────────────────
        self._collect_metrics()
        parts.append(render_metrics_row(
            self._metrics["cpu"],
            self._metrics["memory"],
            self._metrics["disk"],
            width=w,
        ))

        # ── tab bar with animated spinners + error badges ──────
        tab_data = [(t.id, t.title, self._states.get(t.id, "starting")) for t in self._tabs]
        badge_counts: dict[str, tuple[int, int]] = {}
        for t in self._tabs:
            errs = sum(1 for line in t.lines if "ERROR" in line)
            warns = sum(1 for line in t.lines if "WARNING" in line or "WARN" in line)
            badge_counts[t.id] = (errs, warns)
        parts.append(render_tab_bar(
            tab_data, self._active_tab, width=w, step=self._frame,
            badges=badge_counts,
        ))

        # ── log content panel (or help overlay) ───────────────
        if self._show_help:
            parts.append(self._render_help_overlay(w))
            # skip rest of rendering after the panel
            parts.append("")
            elapsed_sec = time.monotonic() - self._start_time
            elapsed_str = f"uptime {int(elapsed_sec // 60)}m {int(elapsed_sec % 60)}s"
            parts.append(render_footer(
                f"  ? / Esc to close  {Box.DOT}  q to quit",
                width=w,
                elapsed=elapsed_str,
            ))
            return "\n".join(parts)

        active_tab = self._tab_map.get(self._active_tab)
        if active_tab:
            state = self._states.get(active_tab.id, "starting")
            border_fg = {
                "starting": FG.WARNING,
                "ready": FG.SUCCESS,
                "error": FG.ERROR,
            }.get(state, FG.WARNING)

            # word-wrap logs
            content_lines = list(active_tab.lines)
            # determine slice based on scroll offset
            scroll_off = self._scroll_offsets.get(active_tab.id, 0)
            max_vis_logs = 50
            start_idx = scroll_off
            end_idx = start_idx + max_vis_logs
            visible_lines = content_lines[start_idx:end_idx]

            # search mode: filter and highlight
            search_query = self._search_query if self._search_mode else ""
            if search_query:
                visible_lines = [l for l in content_lines if search_query.lower() in l.lower()]

            # search bar always shown when in search mode
            search_bar = ""
            if self._search_mode:
                if search_query:
                    search_bar = (
                        f"{_fg(FG.INFO)}  / {search_query}{_RESET}"
                        f"  {_fg(FG.DIM)}(searching...){_RESET}"
                    )
                else:
                    search_bar = f"{_fg(FG.INFO)}  / {_fg(FG.DIM)}(type to search, Esc to cancel){_RESET}"
                search_bar = search_bar.ljust(max(0, w - 4))

            if visible_lines:
                colourised = [_colourise(l) for l in visible_lines]
                if search_query:
                    colourised = _highlight_search(colourised, search_query)
                max_w = w - 2 - 2  # inner width minus padding_x*2
                wrapped = _word_wrap_lines(colourised, max_w)
                content = "\n".join(wrapped)
                if search_bar and search_query:
                    match_cnt = len(visible_lines)
                    search_bar = (
                        f"{_fg(FG.INFO)}  / {search_query}{_RESET}"
                        f"  {_fg(FG.DIM)}({match_cnt} lines){_RESET}"
                    ).ljust(max(0, w - 4))
                    content = f"{search_bar}\n{content}"
                elif search_bar:
                    content = f"{search_bar}\n{content}"
            else:
                if search_bar:
                    content = search_bar
                elif search_query:
                    content = f"{_fg(FG.DIM)}{_ITALIC}    (no matches for '{search_query}'){_RESET}"
                elif active_tab.lines and scroll_off > 0:
                    content = f"{_fg(FG.DIM)}{_ITALIC}    (scrolled past top){_RESET}"
                else:
                    content = f"{_fg(FG.DIM)}{_ITALIC}  Waiting for process output...{_RESET}"

            port_str = f":{active_tab.port}{active_tab.url_path}" if active_tab.port else ""

            # connection health indicator
            last_act = self._last_activity.get(active_tab.id, 0.0)
            if state == "ready" and last_act > 0:
                age = time.monotonic() - last_act
                if age < 10:
                    health_tag = f"{_fg(FG.SUCCESS)}{Box.DOT} active{_RESET}"
                elif age < 60:
                    health_tag = f"{_fg(FG.MUTED)}{Box.DOT_SM} idle{_RESET}"
                else:
                    health_tag = f"{_fg(FG.DIM)}{Box.RING} stale{_RESET}"
            elif state == "error":
                health_tag = f"{_fg(FG.ERROR)}{Box.DOT} down{_RESET}"
            else:
                health_tag = f"{_fg(FG.WARNING)}{Box.DOT} booting{_RESET}"

            total_logs = len(active_tab.lines)
            scroll_info = ""
            if scroll_off > 0 or total_logs > max_vis_logs:
                follow = self._scroll_follow.get(active_tab.id, True)
                follow_indicator = f"{Box.DOT_SM} live" if follow else f"{Box.RING} paused"
                follow_fg = FG.SUCCESS if follow else FG.MUTED
                scroll_info = (
                    f"  {_fg(follow_fg)}{follow_indicator}{_RESET}"
                    f"  {_fg(FG.DIM)}[{scroll_off}+{min(max_vis_logs, total_logs - scroll_off)}]{_RESET}"
                )

            subtitle = (
                f"{_status_tag(state)}"
                f"  {_fg(FG.DIM)}{total_logs} lines{_RESET}"
                f"{scroll_info}"
                f"  {health_tag}"
                f"  {_fg(FG.MUTED)}{port_str}{_RESET}"
                f"  {_fg(FG.DIM)}Tab:{_RESET} {active_tab.title}"
            )

            parts.append(
                render_panel(
                    active_tab.title,
                    content,
                    width=w,
                    border_fg=border_fg,
                    subtitle=subtitle,
                    fill=True,
                )
            )

        # ── footer with elapsed time ──────────────────────────
        elapsed_sec = time.monotonic() - self._start_time
        elapsed_str = f"uptime {int(elapsed_sec // 60)}m {int(elapsed_sec % 60)}s"
        ports = sorted(set(t.port for t in self._tabs if t.port))
        url_strs = [f"localhost:{p}" for p in ports]
        sep = "  \u2022  "
        footer_text = f"  Ctrl+C  {sep}{sep.join(url_strs)}"
        if 8000 in ports:
            footer_text += f"  {sep}Docs :8000/docs"
        footer_text += (
            f"  {sep}\u2190\u2192 tabs  \u2191\u2193 scroll  "
            f"space pause  / search  ? help  C clear  r restart  t theme  q quit"
        )
        parts.append(render_footer(footer_text, width=w, elapsed=elapsed_str))

        return "\n".join(parts)
