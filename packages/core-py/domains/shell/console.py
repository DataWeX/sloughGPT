"""
Console — structured output for the shell REPL.

Provides formatted output primitives (tables, boxes, separators, status
indicators) and a safe-print mechanism that saves and restores the
readline input buffer so background output never corrupts the user's
in-progress input.

All public methods:
  write, print, rule, separator, section, panel, box,
  status, table, table_from_dicts, kvlist, progress, spinner,
  confirm, ask, json, paginate,
  error, success, info, warn, note,
  badge, columns, tree, log, markdown,
  summary, header, live, capture, indent,
  select, clear, hide_cursor, show_cursor,
  styled, download_bar

Usage:
  from .console import Console
  c = Console(io, has_readline=True)
  c.table(data, header=["A", "B"])
  c.box("Hello")
  c.separator()
  c.status("ok", "Task complete")
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any


_COLOR_ENABLED = not os.environ.get("NO_COLOR")
if _COLOR_ENABLED:
    _C_CYAN = "\033[36m"
    _C_GREEN = "\033[32m"
    _C_YELLOW = "\033[33m"
    _C_RED = "\033[31m"
    _C_DIM = "\033[2m"
    _C_BOLD = "\033[1m"
    _C_ITALIC = "\033[3m"
    _C_UNDERLINE = "\033[4m"
    _C_REVERSE = "\033[7m"
    _C_RESET = "\033[0m"
else:
    _C_CYAN = _C_GREEN = _C_YELLOW = _C_RED = _C_DIM = _C_BOLD = ""
    _C_ITALIC = _C_UNDERLINE = _C_REVERSE = _C_RESET = ""


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_C_RESET}" if _COLOR_ENABLED and code else text


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# ── Inline markdown helpers ───────────────────────────────────────────────────


_RE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_RE_CODE = re.compile(r"`(.+?)`")
_RE_LINK = re.compile(r"\[(.+?)\]\((.+?)\)")


def _render_inline(text: str) -> str:
    text = _RE_BOLD.sub(lambda m: _color(m.group(1), _C_BOLD), text)
    text = _RE_ITALIC.sub(lambda m: _color(m.group(1), _C_ITALIC), text)
    text = _RE_CODE.sub(lambda m: _color(m.group(1), _C_CYAN), text)
    text = _RE_LINK.sub(lambda m: f"{m.group(1)} ({_color(m.group(2), _C_DIM)})", text)
    return text


@dataclass
class Block:
    """A structured output block that an LLM can read and render.

    Each Console method emits one ``Block`` with a ``type`` tag and a
    ``data`` dict carrying all information needed to render the output
    without loss.
    """
    type: str
    data: dict
    meta: dict = field(default_factory=dict)


class Console:
    """Structured output with input-buffer preservation.

    Args:
        io: A ShellIO-compatible object (ConsoleIO, MemoryIO, etc.).
        has_readline: If True, attempts to save/restore the readline buffer
            around output to avoid corrupting the user's in-progress input.
    """

    def __init__(self, io: Any, has_readline: bool = False) -> None:
        self._io = io
        self._has_readline = has_readline
        self._blocks: list[Block] = []
        self._tui_repl = None
        if has_readline:
            try:
                import readline  # noqa: F401
            except ImportError:
                self._has_readline = False

    # ── Structured block recording ───────────────────────────────────

    def _emit(self, type: str, data: dict, **meta) -> None:
        """Record a structured output block."""
        self._blocks.append(Block(type=type, data=data, meta=meta))

    def get_blocks(self) -> list[dict]:
        """Return all recorded blocks as plain dicts (JSON-safe)."""
        return [asdict(b) for b in self._blocks]

    def get_json(self, indent: int = 2) -> str:
        """Return all recorded blocks as JSON string."""
        return json.dumps(self.get_blocks(), indent=indent, default=str)

    def clear_blocks(self) -> None:
        """Clear all recorded blocks."""
        self._blocks.clear()

    def last_block(self) -> dict | None:
        """Return the most recent block dict, or None."""
        if self._blocks:
            return asdict(self._blocks[-1])
        return None

    # ── Input-safe output ────────────────────────────────────────────

    def write(self, text: str, end: str = "\n") -> None:
        """Write text via the underlying IO, preserving the input line."""
        self._emit("write", {"text": text, "end": end})
        if not self._has_readline or not text.strip():
            self._io.write(text, end=end)
            return

        try:
            import readline
            buf = readline.get_line_buffer()
        except (ImportError, RuntimeError):
            self._io.write(text, end=end)
            return

        if not buf:
            self._io.write(text, end=end)
            return

        lines_out = text.count("\n") + (1 if end == "\n" else 0)

        self._io.write("\033[s", end="")
        self._io.write(f"\033[{lines_out}B", end="")
        self._io.write(text, end=end)
        self._io.write("\033[u", end="")
        self._io.write(readline.get_line_buffer(), end="")

    # ── Print raw text ───────────────────────────────────────────────

    def print(self, *args, **kwargs) -> None:
        end = kwargs.get("end", "\n")
        text = " ".join(str(a) for a in args)
        self._emit("print", {"text": text, "end": end})
        self.write(text, end=end)

    # ── Rule (horizontal line with optional label) ────────────────────

    def rule(self, label: str = "", char: str = "─", width: int | None = None) -> None:
        """Print a horizontal rule, optionally with a centered label."""
        self._emit("rule", {"label": label, "char": char, "width": width})
        cols = width or shutil.get_terminal_size().columns
        if not label:
            self._io.write(_color(char * cols, _C_DIM))
            return
        inner = f" {label} "
        side_len = max(0, (cols - len(inner)) // 2)
        side = char * side_len
        line = _color(f"{side}{inner}{side}", _C_DIM)
        self._io.write(line)

    # ── Separator ────────────────────────────────────────────────────

    def separator(self, char: str = "─", color: str = _C_DIM) -> str:
        """Print a full-width horizontal separator. Returns the line."""
        self._emit("separator", {"char": char})
        cols = shutil.get_terminal_size().columns
        line = char * cols
        rendered = _color(line, color) if color else line
        self._io.write(rendered)
        return rendered

    # ── Section header ───────────────────────────────────────────────

    def section(self, title: str, char: str = "─", width: int | None = None) -> None:
        """Print a section header in the form '── title ──'. Deprecated: use rule()."""
        self.rule(title, char, width)

    # ── Panel (box with title) ────────────────────────────────────────

    def panel(self, text: str, title: str = "", width: int | None = None,
              title_align: str = "left") -> None:
        """Draw a box with an optional title line."""
        self._emit("panel", {"text": text, "title": title, "width": width, "title_align": title_align})
        cols = width or shutil.get_terminal_size().columns
        inner_w = cols - 4
        if title:
            max_t = inner_w - 4
            t = title[:max_t]
            if title_align == "center":
                t = t.center(inner_w)
            elif title_align == "right":
                t = t.rjust(inner_w)
            else:
                t = " " + t + " "
            top = _color(f"┌─{t}─┐", _C_DIM)
        else:
            top = _color("┌" + "─" * inner_w + "┐", _C_DIM)
        self._io.write("  " + top)
        for line in text.split("\n"):
            padded = line.ljust(inner_w)
            self._io.write(f"  {_C_DIM}│{_C_RESET} {padded} {_C_DIM}│{_C_RESET}")
        self._io.write("  " + _color("└" + "─" * inner_w + "┘", _C_DIM))

    # ── Box (legacy) ─────────────────────────────────────────────────

    def box(self, text: str, width: int | None = None) -> None:
        """Draw a simple box around text. Shorthand for ``panel(text)``."""
        self.panel(text, width=width)

    # ── Status indicators ────────────────────────────────────────────

    def status(self, kind: str, message: str, detail: str = "") -> None:
        """Print a colored status line."""
        self._emit("status", {"kind": kind, "message": message, "detail": detail})
        symbols = {
            "ok": _color("✓", _C_GREEN),
            "warn": _color("⚠", _C_YELLOW),
            "error": _color("✗", _C_RED),
            "info": _color("ℹ", _C_CYAN),
            "step": _color("→", _C_CYAN),
        }
        prefix = symbols.get(kind, kind)
        parts = [f"  {prefix} {message}"]
        if detail:
            parts.append(_color(f"({detail})", _C_DIM))
        self._io.write(" ".join(parts))

    # ── Table ────────────────────────────────────────────────────────

    def table(self, rows: list[list[str]], header: list[str] | None = None,
              separator_after_header: bool = True) -> None:
        """Print a formatted table with aligned columns."""
        self._emit("table", {"header": header, "rows": rows, "separator_after_header": separator_after_header})
        if not rows:
            self._io.write("  (empty)")
            return
        cols = max(len(r) for r in rows)
        if header:
            cols = max(cols, len(header))
        widths = [0] * cols
        for row in rows:
            for i, cell in enumerate(row):
                if i < cols:
                    widths[i] = max(widths[i], len(str(cell)))
        if header:
            for i, cell in enumerate(header):
                if i < cols:
                    widths[i] = max(widths[i], len(str(cell)))
        fmt = "  ".join("{{:<{}}}".format(w) for w in widths)
        if header:
            self._io.write(f"  {fmt.format(*header)}")
            if separator_after_header:
                self._io.write(f"  {'  '.join('─' * w for w in widths)}")
        for row in rows:
            padded = list(row) + [""] * (cols - len(row))
            self._io.write(f"  {fmt.format(*padded)}")

    # ── Key-Value list ───────────────────────────────────────────────

    def kvlist(self, items: list[tuple[str, str]], indent: int = 2) -> None:
        """Print a key-value list with aligned values."""
        self._emit("kvlist", {"items": items, "indent": indent})
        if not items:
            return
        max_k = max(len(k) for k, _ in items)
        pad = " " * indent
        for k, v in items:
            self._io.write(f"{pad}{k:<{max_k}}  {v}")

    # ── Progress bar ─────────────────────────────────────────────────

    def progress(self, label: str, current: int, total: int,
                 bar_width: int = 20) -> None:
        """Print an overwritable progress bar line."""
        self._emit("progress", {"label": label, "current": current, "total": total, "bar_width": bar_width})
        frac = current / max(total, 1)
        filled = int(frac * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = f"{frac * 100:5.1f}%"
        self._io.write(f"\r  {label}: [{bar}] {pct} ({current}/{total})", end="")
        if current >= total:
            self._io.write("")

    # ── Spinner context manager ───────────────────────────────────────

    def spinner(self, message: str = "", rate: float = 0.1) -> _Spinner:
        """Return a context manager that shows a spinner while a task runs.

        Usage::

            with c.spinner("Loading") as s:
                do_work()
            s.ok("done")          # overwrites spinner with ✓ done
            s.fail("error msg")   # overwrites spinner with ✗ error msg
        """
        return _Spinner(self, message, rate)

    # ── Confirm ──────────────────────────────────────────────────────

    def confirm(self, message: str, default: bool = False) -> bool:
        """Prompt yes/no and return True/False."""
        self._emit("confirm", {"message": message, "default": default})
        if self._tui_repl is not None and hasattr(self._tui_repl, "prompt_confirm"):
            return self._tui_repl.prompt_confirm(message, default)
        hint = "Y/n" if default else "y/N"
        self._io.write(f"  {message} [{hint}] ", end="")
        raw = self._io.read("").strip().lower()
        result = default if not raw else raw in ("y", "yes", "ye", "true", "1")
        return result

    # ── Ask — prompt for free-form input ──────────────────────────────

    def ask(self, message: str, default: str = "") -> str:
        """Prompt for free-form input with an optional default."""
        self._emit("ask", {"message": message, "default": default})
        if self._tui_repl is not None and hasattr(self._tui_repl, "prompt_ask"):
            return self._tui_repl.prompt_ask(message, default)
        suffix = f" [{default}]" if default else ""
        self._io.write(f"  {message}{suffix} ", end="")
        raw = self._io.read("").strip()
        result = raw if raw else default
        return result

    # ── JSON output ──────────────────────────────────────────────────

    def json(self, data: Any, indent: int = 2) -> None:
        """Print pretty-printed JSON with syntax highlighting."""
        self._emit("json", {"data": data, "indent": indent})
        text = json.dumps(data, indent=indent, default=str)
        for line in text.split("\n"):
            self._io.write(f"  {line}")

    # ── Pager (paginate long output) ──────────────────────────────────

    def paginate(self, lines: list[str], page_size: int | None = None) -> None:
        """Print lines page-by-page, prompting for Enter to continue."""
        self._emit("paginate", {"line_count": len(lines), "page_size": page_size})
        if page_size is None:
            _, page_size = shutil.get_terminal_size()
            page_size = max(5, page_size - 2)
        total = len(lines)
        pos = 0
        while pos < total:
            chunk = lines[pos:pos + page_size]
            for line in chunk:
                self._io.write(line)
            pos += len(chunk)
            if pos < total:
                pct = pos * 100 // total
                self._io.write(_color(f"  --More--({pct}%)--", _C_DIM), end="")
                self._io.read("")
                self._io.write("\033[1A\033[K", end="")

    # ── Error with hint ──────────────────────────────────────────────

    def error(self, message: str, hint: str = "") -> None:
        """Print a red error, optionally with a dim hint below."""
        self._emit("error", {"message": message, "hint": hint})
        self._io.write(f"  {_color('Error:', _C_RED)} {message}")
        if hint:
            self._io.write(f"    {_color(hint, _C_DIM)}")

    # ── Success ──────────────────────────────────────────────────────

    def success(self, message: str) -> None:
        """Print a green checkmark and message."""
        self._emit("success", {"message": message})
        self._io.write(f"  {_color('✓', _C_GREEN)} {message}")

    # ── Info ─────────────────────────────────────────────────────────

    def info(self, message: str) -> None:
        """Print a cyan info prefix and message."""
        self._emit("info", {"message": message})
        self._io.write(f"  {_color('ℹ', _C_CYAN)} {message}")

    # ── Warn ─────────────────────────────────────────────────────────

    def warn(self, message: str) -> None:
        """Print a yellow warning message."""
        self._emit("warn", {"message": message})
        self._io.write(f"  {_color('⚠', _C_YELLOW)} {message}")

    # ── Table from dicts ─────────────────────────────────────────────

    def table_from_dicts(self, data: list[dict], **kwargs) -> None:
        """Print a table from a list of dicts (auto-headers from keys)."""
        if not data:
            self._io.write("  (empty)")
            return
        header = list(data[0].keys())
        rows = [[str(r.get(k, "")) for k in header] for r in data]
        self.table(rows, header=header, **kwargs)

    # ── Capture context manager ──────────────────────────────────────

    def capture(self) -> _Capture:
        """Return a context manager that captures all writes to a string.

        Usage::

            with c.capture() as cap:
                c.print("hello")
            text = cap.get()  # "hello"
        """
        return _Capture(self._io)

    # ── Indent context manager ───────────────────────────────────────

    def indent(self, level: int = 2, char: str = " ") -> _Indent:
        """Return a context manager that indents all output.

        Usage::

            with c.indent(4):
                c.print("indented text")
        """
        return _Indent(self, level, char)

    # ── Columns ──────────────────────────────────────────────────────

    def columns(self, items: list[str], col_count: int | None = None,
                spacing: int = 2) -> None:
        """Render items in aligned columns, auto-fitting to terminal width.

        Args:
            items: Strings to arrange.
            col_count: Number of columns (default: auto-fit to width).
            spacing: Spaces between columns.
        """
        self._emit("columns", {"items": items, "col_count": col_count, "spacing": spacing})
        if not items:
            return
        cols = shutil.get_terminal_size().columns
        max_w = max(len(i) for i in items)
        if col_count is None:
            col_count = max(1, (cols - spacing) // (max_w + spacing))
        rows_n = (len(items) + col_count - 1) // col_count
        fmt = (" " * spacing).join("{{:<{}}}".format(max_w) for _ in range(col_count))
        for r in range(rows_n):
            row_items = []
            for c in range(col_count):
                idx = r + c * rows_n
                if idx < len(items):
                    row_items.append(items[idx])
                else:
                    row_items.append("")
            self._io.write(f"  {fmt.format(*row_items)}")

    # ── Tree ─────────────────────────────────────────────────────────

    def tree(self, data: dict[str, list[str] | dict],
             prefix: str = "") -> None:
        """Render a nested structure as a Unicode tree."""
        self._emit("tree", {"data": data, "prefix": prefix})
        items = list(data.items())
        for i, (label, children) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            self._io.write(f"  {prefix}{connector}{label}")
            if isinstance(children, dict):
                ext = "    " if is_last else "│   "
                self.tree(children, prefix + ext)
            elif isinstance(children, list):
                ext = "    " if is_last else "│   "
                for j, child in enumerate(children):
                    c = "└── " if j == len(children) - 1 else "├── "
                    self._io.write(f"  {prefix}{ext}{c}{child}")

    # ── Log ──────────────────────────────────────────────────────────

    def log(self, message: str, level: str = "info") -> None:
        """Print a timestamped log line with a level tag."""
        self._emit("log", {"message": message, "level": level})
        ts = time.strftime("%H:%M:%S")
        tags = {
            "info": _color(" INFO ", _C_CYAN),
            "warn": _color(" WARN ", _C_YELLOW),
            "error": _color("ERROR ", _C_RED),
            "debug": _color("DEBUG ", _C_DIM),
        }
        tag = tags.get(level, f" {level.upper():<5}")
        self._io.write(f"  {_color(ts, _C_DIM)} {tag} {message}")

    # ── Markdown ─────────────────────────────────────────────────────

    def markdown(self, text: str) -> None:
        """Render basic Markdown with ANSI formatting."""
        self._emit("markdown", {"text": text})
        blocks = re.split(r"\n\s*\n", text)
        in_code = False
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                continue
            if stripped.startswith("```"):
                if not in_code:
                    in_code = True
                    lang = stripped[3:].strip()
                    if lang:
                        self._io.write(f"  {_color(lang, _C_DIM)}")
                    continue
                else:
                    in_code = False
                    continue
            if in_code:
                self._io.write(f"  {_color(stripped, _C_CYAN)}")
                continue
            if re.match(r"^-{3,}\s*$", stripped):
                self.rule(width=shutil.get_terminal_size().columns - 4)
                continue
            if stripped.startswith(">"):
                for line in stripped.split("\n"):
                    content = line.lstrip(">").strip()
                    self._io.write(f"  {_C_DIM}│{_C_RESET} {_render_inline(content)}")
                continue
            h_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if h_match:
                level = len(h_match.group(1))
                content = _render_inline(h_match.group(2))
                if level == 1:
                    self._io.write(f"  {_color(content, _C_BOLD)}")
                    self._io.write(f"  {_color('─' * (len(stripped) - level - 1), _C_DIM)}")
                else:
                    self._io.write(f"  {_color(content, _C_BOLD)}")
                continue
            lines = stripped.split("\n")
            list_type = None
            for line in lines:
                b_match = re.match(r"^(\s*)[-*+]\s+(.+)$", line)
                n_match = re.match(r"^(\s*)\d+[.)]\s+(.+)$", line)
                if b_match:
                    indent = len(b_match.group(1))
                    content = _render_inline(b_match.group(2))
                    self._io.write(f"  {'  ' * indent}• {content}")
                elif n_match:
                    indent = len(n_match.group(1))
                    content = _render_inline(n_match.group(2))
                    self._io.write(f"  {'  ' * indent}{content}")
                else:
                    self._io.write(f"  {_render_inline(line)}")

    # ── Badge ────────────────────────────────────────────────────────

    def badge(self, label: str, color: str = "info") -> None:
        """Print an inline colored badge/pill."""
        self._emit("badge", {"label": label, "color": color})
        palette = {
            "info": (_C_CYAN, _C_RESET),
            "ok": (_C_GREEN, _C_RESET),
            "warn": (_C_YELLOW, _C_RESET),
            "error": (_C_RED, _C_RESET),
        }
        fg, _ = palette.get(color, (_C_DIM, _C_RESET))
        self._io.write(f"  {fg}{_C_BOLD}{label}{_C_RESET}")

    # ── Live display context manager ─────────────────────────────────

    def live(self) -> _Live:
        """Return a context manager for a live-updating display region."""
        self._emit("live", {})
        return _Live(self)

    # ── Summary card ─────────────────────────────────────────────────

    def summary(self, title: str, items: list[tuple[str, str]],
                width: int | None = None) -> None:
        """Print a compact summary card with a title and key-value rows."""
        self._emit("summary", {"title": title, "items": items, "width": width})
        cols = width or shutil.get_terminal_size().columns
        inner_w = cols - 4
        self._io.write("  " + _color("┌" + "─" * inner_w + "┐", _C_DIM))
        t = f" {title} ".center(inner_w)
        self._io.write(f"  {_C_DIM}│{_C_RESET} {_color(t, _C_BOLD)} {_C_DIM}│{_C_RESET}")
        self._io.write("  " + _color("│" + " " * inner_w + "│", _C_DIM))
        for k, v in items:
            line = f"  {_C_DIM}│{_C_RESET}  {k:<20} {v}{_C_DIM} │{_C_RESET}"
            self._io.write(line)
        self._io.write("  " + _color("└" + "─" * inner_w + "┘", _C_DIM))

    # ── Page header ─────────────────────────────────────────────────

    def header(self, title: str, subtitle: str = "") -> None:
        """Print a page header block with optional subtitle."""
        self._emit("header", {"title": title, "subtitle": subtitle})
        self._io.write(f"  {_color(title, _C_BOLD)}")
        if subtitle:
            self._io.write(f"  {_color(subtitle, _C_DIM)}")
        self.rule()

    # ── Note ─────────────────────────────────────────────────────────

    def note(self, message: str) -> None:
        """Print a dimmed annotation."""
        self._emit("note", {"message": message})
        self._io.write(f"  {_color(message, _C_DIM)}")

    # ── Select (interactive numbered choice) ─────────────────────────

    def select(self, title: str, options: list[str]) -> str:
        """Show a numbered menu and return the chosen option string."""
        self._emit("select", {"title": title, "options": options})
        self._io.write(f"  {title}")
        for i, opt in enumerate(options, 1):
            self._io.write(f"    {i}. {opt}")
        while True:
            self._io.write("  Enter number: ", end="")
            try:
                raw = self._io.read("").strip()
            except (EOFError, KeyboardInterrupt):
                return options[0] if options else ""
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx]

    # ── Cursor control ───────────────────────────────────────────────

    def hide_cursor(self) -> None:
        """Hide the terminal cursor."""
        self._emit("hide_cursor", {})
        self._io.write("\033[?25l", end="")
        self._io.flush()

    def show_cursor(self) -> None:
        """Show the terminal cursor."""
        self._emit("show_cursor", {})
        self._io.write("\033[?25h", end="")
        self._io.flush()

    # ── Clear screen ─────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear the terminal screen and reset cursor position."""
        self._emit("clear", {})
        self._io.write("\033[2J\033[H", end="")

    # ── Style utility ────────────────────────────────────────────────

    def styled(self, text: str, style: str = "bold") -> str:
        """Return *text* wrapped in an ANSI style (no I/O)."""
        codes = {
            "bold": _C_BOLD,
            "dim": _C_DIM,
            "italic": _C_ITALIC,
            "underline": _C_UNDERLINE,
            "cyan": _C_CYAN,
            "green": _C_GREEN,
            "yellow": _C_YELLOW,
            "red": _C_RED,
        }
        code = codes.get(style, "")
        return _color(text, code)

    # ── Download progress bar ────────────────────────────────────────

    def download_bar(self, label: str, current: int, total: int,
                     bytes_done: int = 0, bytes_total: int = 0,
                     speed: float = 0.0) -> None:
        """Print an overwritable download progress line with ETA."""
        self._emit("download_bar", {"label": label, "current": current, "total": total})
        frac = current / max(total, 1)
        bar_w = 15
        filled = int(frac * bar_w)
        bar = "█" * filled + "░" * (bar_w - filled)
        pct = f"{frac * 100:5.1f}%"
        size_str = f"{_human_size(bytes_done)}/{_human_size(bytes_total)}"
        speed_str = f"{_human_size(speed)}/s" if speed else ""
        eta = ""
        if speed > 0 and bytes_total > bytes_done:
            eta_sec = (bytes_total - bytes_done) / speed
            if eta_sec < 60:
                eta = f"{eta_sec:.0f}s"
            elif eta_sec < 3600:
                eta = f"{eta_sec / 60:.0f}m {eta_sec % 60:.0f}s"
        parts = [f"\r  {label[:20]:<20} [{bar}] {pct}"]
        if size_str:
            parts.append(size_str)
        if speed_str:
            parts.append(speed_str)
        if eta:
            parts.append(_color(eta, _C_DIM))
        self._io.write(" ".join(parts), end="")
        if current >= total:
            self._io.write("")


# ── Spinner helper ────────────────────────────────────────────────────────────


class _Spinner:
    """Context manager that drives a spinner animation.

    Returned by ``Console.spinner()``.  Use ``.ok(msg)`` / ``.fail(msg)``
    to replace the spinner with a final status line.
    """

    def __init__(self, console: Console, message: str, rate: float) -> None:
        self._console = console
        self._message = message
        self._rate = rate
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _Spinner:
        self._stop.clear()
        io = self._console._io
        msg = self._message
        frames = SPINNER_FRAMES

        def _spin():
            i = 0
            while not self._stop.is_set():
                f = frames[i % len(frames)]
                io.write(f"\r  {f} {msg}\033[K", end="")
                io.flush()
                i += 1
                self._stop.wait(self._rate)

        self._thread = threading.Thread(target=_spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._console._io.write("\r\033[K", end="")

    def ok(self, message: str = "") -> None:
        """Replace spinner with a success (green checkmark) line."""
        self.__exit__()
        self._console.success(message or self._message)

    def fail(self, message: str = "") -> None:
        """Replace spinner with a failure (red cross) line."""
        self.__exit__()
        self._console.error(message or self._message)


# ── Live helper ───────────────────────────────────────────────────────────────


class _Live:
    """Context manager for a live-updating display region.

    Returned by ``Console.live()``.  Each call to ``.update(text)``
    replaces the previously rendered content.
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._line_count = 0

    def __enter__(self) -> _Live:
        return self

    def __exit__(self, *exc) -> None:
        pass

    def update(self, text: str) -> None:
        """Replace the live region with *text*."""
        lines = text.split("\n")
        n = len(lines)
        if self._line_count > 0:
            self._console._io.write(f"\033[{self._line_count}A", end="")
        for i, line in enumerate(lines):
            self._console._io.write(f"\r\033[K{line}", end="\n" if i < n - 1 else "")
        self._line_count = n
        self._console._io.flush()


# ── Capture helper ────────────────────────────────────────────────────────────


class _Capture:
    """Context manager that captures all writes to an internal buffer.

    Returned by ``Console.capture()``.  Use ``.get()`` after the
    ``with`` block to retrieve captured text.
    """

    def __init__(self, io: Any) -> None:
        self._io = io
        self._buf: list[str] = []
        self._orig_write = io.write

    def __enter__(self) -> _Capture:
        self._buf.clear()
        self._io.write = self._capture_write  # type: ignore
        return self

    def __exit__(self, *exc) -> None:
        self._io.write = self._orig_write  # type: ignore

    def _capture_write(self, text: str, end: str = "\n") -> None:
        self._buf.append(text + end if text else end)

    def get(self) -> str:
        return "".join(self._buf)


# ── Indent helper ─────────────────────────────────────────────────────────────


class _Indent:
    """Context manager that indents writes by a given level.

    Returned by ``Console.indent()``.  Prepends *level* spaces to each
    output line while the context is active.
    """

    def __init__(self, console: Console, level: int, char: str) -> None:
        self._console = console
        self._prefix = char * level
        self._orig_write = console._io.write

    def __enter__(self) -> _Indent:
        io = self._console._io
        prefix = self._prefix
        orig = self._orig_write

        def _indent_write(text: str, end: str = "\n") -> None:
            lines = text.split("\n")
            indented = "\n".join(prefix + l for l in lines)
            orig(indented, end=end)

        io.write = _indent_write  # type: ignore
        return self

    def __exit__(self, *exc) -> None:
        self._console._io.write = self._orig_write  # type: ignore
