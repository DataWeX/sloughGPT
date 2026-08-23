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

from .interactive import InteractivePrompt


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
        self._interactive = InteractivePrompt(io)
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
        """Prompt yes/no with arrow-key toggle and return True/False."""
        if self._tui_repl is not None and hasattr(self._tui_repl, "prompt_confirm"):
            result = self._tui_repl.prompt_confirm(message, default)
        else:
            result = self._interactive.confirm(message, default)
        self._emit("confirm", {"message": message, "default": default, "result": result})
        return result

    # ── Ask — prompt for free-form input ──────────────────────────────

    def ask(self, message: str, default: str = "") -> str:
        """Prompt for free-form input with cursor movement and an optional default."""
        self._emit("ask", {"message": message, "default": default})
        if self._tui_repl is not None and hasattr(self._tui_repl, "prompt_ask"):
            return self._tui_repl.prompt_ask(message, default)
        return self._interactive.ask(message, default)

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

    def tree_multi(self, data: dict[str, list[str] | dict],
                   title: str = "Select items") -> list[str]:
        """Show a tree and let user select multiple leaf items.

        Returns list of selected leaf labels.
        """
        self._emit("tree_multi", {"data": data, "title": title})
        leaves: list[str] = []

        def _collect(d: dict, prefix: str = "") -> None:
            for i, (label, children) in enumerate(d.items()):
                is_last = i == len(d) - 1
                connector = "\u251c\u2500\u2500 " if not is_last else "\u2514\u2500\u2500 "
                if isinstance(children, dict):
                    self._io.write(f"  {prefix}{connector}{label}")
                    ext = "\u2502   " if not is_last else "    "
                    _collect(children, prefix + ext)
                elif isinstance(children, list):
                    ext = "\u2502   " if not is_last else "    "
                    for j, child in enumerate(children):
                        c = "\u251c\u2500\u2500 " if j < len(children) - 1 else "\u2514\u2500\u2500 "
                        self._io.write(f"  {prefix}{ext}{c}{child}")
                        leaves.append(child)

        _collect(data)
        return self._interactive.select_multi(title, leaves)

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
        """Show an interactive arrow-key menu and return the chosen option.

        Uses arrow keys and type-to-filter when the terminal supports it,
        falling back to a numbered menu for non-TTY environments.
        """
        self._emit("select", {"title": title, "options": options})
        if self._tui_repl is not None and hasattr(self._tui_repl, "prompt_select"):
            return self._tui_repl.prompt_select(title, options)
        return self._interactive.select(title, options)

    def select_multi(self, title: str, options: list[str]) -> list[str]:
        """Show an interactive multi-select menu with checkboxes.

        Use Space to toggle items, arrows to move, Enter to confirm.
        Falls back to comma-separated numbered input for non-TTY.
        """
        self._emit("select_multi", {"title": title, "options": options})
        return self._interactive.select_multi(title, options)

    def select_with_details(self, title: str, options: list[str],
                            details: list[str]) -> str:
        """Show an interactive selector with a detail pane below the list.

        Uses arrow keys and type-to-filter when the terminal supports it,
        falling back to a numbered menu for non-TTY environments.
        """
        self._emit("select_with_details", {"title": title, "options": options, "details": details})
        return self._interactive.select_with_details(title, options, details)

    def confirm_multi(self, title: str, items: list[str],
                      default: bool = True) -> list[str]:
        """Show a multi-confirm prompt: list items and ask y/N for each.

        Returns list of items that were confirmed (y).
        """
        self._emit("confirm_multi", {"title": title, "items": items, "default": default})
        return self._interactive.confirm_multi(title, items, default)

    def select_with_preview(self, title: str, options: list[str],
                            preview_fn: "Callable[[str], str]") -> str:
        """Show an interactive selector with a live preview panel."""
        self._emit("select_with_preview", {"title": title, "options": options})
        return self._interactive.select_with_preview(title, options, preview_fn)

    def edit(self, message: str, default: str = "",
             validator: "Callable[[str], str | None] | None" = None) -> str:
        """Interactive text input with inline validation."""
        self._emit("edit", {"message": message, "default": default})
        return self._interactive.edit(message, default, validator)

    def pager(self, content: str, title: str = "Output") -> None:
        """Display long content in a scrollable pager view."""
        self._emit("pager", {"title": title})
        self._interactive.pager(content, title)

    def diff(self, left_label: str, left_lines: list[str],
             right_label: str, right_lines: list[str],
             title: str = "") -> None:
        """Show a side-by-side diff with colored additions/removals."""
        self._emit("diff", {"title": title})
        self._interactive.diff(left_label, left_lines, right_label, right_lines, title)

    def password(self, message: str) -> str:
        """Interactive password input with masked characters."""
        self._emit("password", {"message": message})
        return self._interactive.password(message)

    def confirm_action(self, action: str, details: str = "",
                       danger: bool = False) -> bool:
        """Confirm an action with a descriptive prompt."""
        self._emit("confirm_action", {"action": action, "danger": danger})
        return self._interactive.confirm_action(action, details, danger)

    def countdown(self, seconds: int, message: str = "Starting in") -> bool:
        """Show a visual countdown timer."""
        self._emit("countdown", {"seconds": seconds, "message": message})
        return self._interactive.countdown(seconds, message)

    def banner(self, text: str, style: str = "double") -> None:
        """Display a styled banner with box-drawing characters."""
        self._emit("banner", {"text": text, "style": style})
        self._interactive.banner(text, style)

    def slider(self, message: str, min_val: int = 0, max_val: int = 100,
               default: int = 50, step: int = 1) -> int:
        """Interactive numeric slider."""
        self._emit("slider", {"message": message, "min": min_val, "max": max_val})
        return self._interactive.slider(message, min_val, max_val, default, step)

    def toggle(self, message: str, default: bool = False) -> bool:
        """Interactive on/off toggle switch."""
        self._emit("toggle", {"message": message, "default": default})
        return self._interactive.toggle(message, default)

    def tag_input(self, message: str, defaults: list[str] | None = None,
                  placeholder: str = "Add tag...") -> list[str]:
        """Interactive tag input."""
        self._emit("tag_input", {"message": message})
        return self._interactive.tag_input(message, defaults, placeholder)

    def select_tree(self, title: str, tree: dict[str, list[str] | dict],
                    expanded: set[str] | None = None) -> str | None:
        """Interactive tree selector with expand/collapse."""
        self._emit("select_tree", {"title": title})
        return self._interactive.select_tree(title, tree, expanded)

    def spin_wait(self, message: str, check_fn: "Callable[[], bool]",
                  interval: float = 0.1, timeout: float = 0) -> bool:
        """Wait for a condition with a spinner."""
        self._emit("spin_wait", {"message": message})
        return self._interactive.spin_wait(message, check_fn, interval, timeout)

    def confirm_dangerous(self, action: str, phrase: str = "yes, I am sure") -> bool:
        """Confirm a dangerous action by typing a phrase."""
        self._emit("confirm_dangerous", {"action": action})
        return self._interactive.confirm_dangerous(action, phrase)

    def file_browser(self, title: str, start_dir: str = ".",
                     pattern: str = "*") -> str | None:
        """Interactive file browser with directory navigation."""
        self._emit("file_browser", {"title": title})
        return self._interactive.file_browser(title, start_dir, pattern)

    def history_search(self, history: list[str], message: str = "History:") -> str | None:
        """Interactive history search with type-to-filter."""
        self._emit("history_search", {"message": message})
        return self._interactive.history_search(history, message)

    def process_manager(self, processes: list[dict[str, str]],
                        message: str = "Processes:") -> dict[str, str] | None:
        """Interactive process manager with live status."""
        self._emit("process_manager", {"message": message})
        return self._interactive.process_manager(processes, message)

    def log_viewer(self, logs: list[str], message: str = "Logs:") -> str | None:
        """Interactive log viewer with scroll and filter."""
        self._emit("log_viewer", {"message": message})
        return self._interactive.log_viewer(logs, message)

    def progress_step(self, steps: list[str], current: int, done: bool = False) -> None:
        """Display a step-by-step progress indicator."""
        self._emit("progress_step", {"current": current, "done": done})
        self._interactive.progress_step(steps, current, done)

    def multi_choice(self, title: str, options: list[str],
                     defaults: list[int] | None = None) -> list[str]:
        """Select multiple options with numbered keys."""
        self._emit("multi_choice", {"title": title, "options": options})
        return self._interactive.multi_choice(title, options, defaults)

    def date_picker(self, message: str, default: str = "") -> str:
        """Interactive date picker with year/month/day navigation."""
        self._emit("date_picker", {"message": message})
        return self._interactive.date_picker(message, default)

    def color_picker_rgb(self, message: str, default: str = "#ffffff") -> str:
        """Interactive hex color picker with RGB sliders."""
        self._emit("color_picker_rgb", {"message": message})
        return self._interactive.color_picker_rgb(message, default)

    def confirm_timeout(self, message: str, timeout: float = 5.0,
                        default: bool = True) -> bool:
        """Confirm with an auto-timeout."""
        self._emit("confirm_timeout", {"message": message, "timeout": timeout})
        return self._interactive.confirm_timeout(message, timeout, default)

    def spin_until(self, message: str, async_fn: "Callable[[], Any]",
                   check: "Callable[[Any], bool]",
                   interval: float = 0.1, timeout: float = 0) -> "Any":
        """Wait for an async function's result to satisfy a condition."""
        self._emit("spin_until", {"message": message})
        return self._interactive.spin_until(message, async_fn, check, interval, timeout)

    def progress_multi(self, items: list[tuple[str, int, int]]) -> None:
        """Display multiple progress bars stacked."""
        self._emit("progress_multi", {})
        self._interactive.progress_multi(items)

    def time_picker(self, message: str, default: str = "") -> str:
        """Interactive time picker with hour/minute/AM-PM navigation."""
        self._emit("time_picker", {"message": message})
        return self._interactive.time_picker(message, default)

    def progress_eta(self, label: str, current: int, total: int,
                     elapsed: float = 0) -> None:
        """Display a progress bar with estimated time remaining."""
        self._emit("progress_eta", {"label": label, "current": current, "total": total})
        self._interactive.progress_eta(label, current, total, elapsed)

    def select_with_search(self, title: str, options: list[str]) -> str:
        """Select with prominent search bar and live filtering."""
        self._emit("select_with_search", {"title": title})
        return self._interactive.select_with_search(title, options)

    def table_select(self, headers: list[str], rows: list[list[str]],
                     title: str = "Select row") -> int | None:
        """Interactive table with row selection."""
        self._emit("table_select", {"title": title})
        return self._interactive.table_select(headers, rows, title)

    def year_picker(self, message: str, default: int = 0,
                    min_year: int = 1900, max_year: int = 2100) -> int:
        """Interactive year picker with arrow keys."""
        self._emit("year_picker", {"message": message})
        return self._interactive.year_picker(message, default, min_year, max_year)

    def month_picker(self, message: str, default: int = 0) -> int:
        """Interactive month picker with names."""
        self._emit("month_picker", {"message": message})
        return self._interactive.month_picker(message, default)

    def confirm_list(self, title: str, items: list[str],
                     default: bool = True) -> list[str]:
        """Confirm each item in a list with y/N."""
        self._emit("confirm_list", {"title": title, "items": items})
        return self._interactive.confirm_list(title, items, default)

    def table_edit(self, headers: list[str], rows: list[list[str]],
                   title: str = "Edit table") -> list[list[str]]:
        """Interactive table with cell editing."""
        self._emit("table_edit", {"title": title})
        return self._interactive.table_edit(headers, rows, title)

    def duration_picker(self, message: str, default: int = 0) -> int:
        """Interactive duration picker in seconds."""
        self._emit("duration_picker", {"message": message})
        return self._interactive.duration_picker(message, default)

    def confirm_text(self, message: str, target: str, hint: str = "") -> bool:
        """Confirm by typing exact text."""
        self._emit("confirm_text", {"message": message, "target": target})
        return self._interactive.confirm_text(message, target, hint)

    def table_sort(self, headers: list[str], rows: list[list[str]],
                   title: str = "Sort table") -> list[list[str]]:
        """Interactive table with column sorting."""
        self._emit("table_sort", {"title": title})
        return self._interactive.table_sort(headers, rows, title)

    def notify(self, title: str, message: str = "", level: str = "info") -> None:
        """Display a styled notification banner."""
        self._emit("notify", {"title": title, "level": level})
        self._interactive.notify(title, message, level)

    def week_picker(self, message: str, default: int = 0) -> int:
        """Interactive week-of-year picker."""
        self._emit("week_picker", {"message": message})
        return self._interactive.week_picker(message, default)

    def quarter_picker(self, message: str, default: int = 0) -> int:
        """Interactive quarter picker."""
        self._emit("quarter_picker", {"message": message})
        return self._interactive.quarter_picker(message, default)

    def confirm_delete(self, item: str, count: int = 1) -> bool:
        """Confirm deletion with type-to-confirm."""
        self._emit("confirm_delete", {"item": item})
        return self._interactive.confirm_delete(item, count)

    def confirm_overwrite(self, path: str) -> bool:
        """Confirm overwriting an existing file."""
        self._emit("confirm_overwrite", {"path": path})
        return self._interactive.confirm_overwrite(path)

    def progress_ring(self, label: str, current: int, total: int) -> None:
        """Display a circular progress indicator."""
        self._emit("progress_ring", {"label": label})
        self._interactive.progress_ring(label, current, total)

    def timezone_picker(self, message: str, default: str = "UTC") -> str:
        """Interactive timezone picker."""
        self._emit("timezone_picker", {"message": message})
        return self._interactive.timezone_picker(message, default)

    def currency_picker(self, message: str, default: str = "USD") -> str:
        """Interactive currency picker."""
        self._emit("currency_picker", {"message": message})
        return self._interactive.currency_picker(message, default)

    def language_picker(self, message: str, default: str = "en") -> str:
        """Interactive language picker."""
        self._emit("language_picker", {"message": message})
        return self._interactive.language_picker(message, default)

    def confirm_with_preview(self, message: str, preview: str,
                             default: bool = False) -> bool:
        """Confirm with a preview of what will happen."""
        self._emit("confirm_with_preview", {"message": message})
        return self._interactive.confirm_with_preview(message, preview, default)

    def select_with_preview(self, message: str, options: list[str],
                            preview_fn: Callable[[str], str],
                            default: str = "") -> str:
        """Select from options with a live preview panel."""
        self._emit("select_with_preview", {"message": message})
        return self._interactive.select_with_preview(message, options, preview_fn)

    def progress_bar(self, label: str, current: int, total: int,
                     width: int = 30) -> None:
        """Display a progress bar with percentage."""
        self._emit("progress_bar", {"label": label})
        self._interactive.progress_bar(label, current, total, width)

    def date_range_picker(self, message: str,
                          default_start: str = "", default_end: str = "") -> tuple[str, str]:
        """Pick a date range."""
        self._emit("date_range_picker", {"message": message})
        return self._interactive.date_range_picker(message, default_start, default_end)

    def color_picker(self, message: str, default: str = "#ffffff") -> str:
        """Interactive color picker."""
        self._emit("color_picker", {"message": message})
        return self._interactive.color_picker(message, default)

    def time_range_picker(self, message: str,
                          default_start: str = "", default_end: str = "") -> tuple[str, str]:
        """Pick a time range."""
        self._emit("time_range_picker", {"message": message})
        return self._interactive.time_range_picker(message, default_start, default_end)

    def number_range_picker(self, message: str, min_val: int = 0,
                            max_val: int = 100, default: int = 0,
                            step: int = 1) -> int:
        """Pick a number from a range."""
        self._emit("number_range_picker", {"message": message})
        return self._interactive.number_range_picker(message, min_val, max_val, default, step)

    def confirm_with_details(self, message: str, details: dict[str, str],
                             default: bool = False) -> bool:
        """Confirm with key-value details displayed."""
        self._emit("confirm_with_details", {"message": message})
        return self._interactive.confirm_with_details(message, details, default)

    def spinner_with_status(self, message: str, status: str) -> None:
        """Display a spinner with a status message."""
        self._emit("spinner_with_status", {"message": message})
        self._interactive.spinner_with_status(message, status)

    def select_with_filter(self, message: str, options: list[str],
                           default: str = "") -> str:
        """Select from options with type-to-filter."""
        self._emit("select_with_filter", {"message": message})
        return self._interactive.select_with_filter(message, options, default)

    def confirm_with_preview_and_edit(self, message: str, preview: str,
                                      edit_prompt: str = "Edit: ",
                                      default: bool = False) -> tuple[bool, str]:
        """Confirm with preview, with option to edit."""
        self._emit("confirm_with_preview_and_edit", {"message": message})
        return self._interactive.confirm_with_preview_and_edit(message, preview, edit_prompt, default)

    def progress_bar_colored(self, label: str, current: int, total: int,
                             width: int = 30) -> None:
        """Display a colored progress bar."""
        self._emit("progress_bar_colored", {"label": label})
        self._interactive.progress_bar_colored(label, current, total, width)

    def spinner_with_progress(self, message: str, current: int, total: int) -> None:
        """Display a spinner with progress percentage."""
        self._emit("spinner_with_progress", {"message": message})
        self._interactive.spinner_with_progress(message, current, total)

    def select_with_icons(self, message: str, options: list[tuple[str, str]],
                          default: str = "") -> str:
        """Select from options with icons."""
        self._emit("select_with_icons", {"message": message})
        return self._interactive.select_with_icons(message, options, default)

    def confirm_with_warning(self, message: str, warning: str,
                             default: bool = False) -> bool:
        """Confirm with a warning message displayed."""
        self._emit("confirm_with_warning", {"message": message})
        return self._interactive.confirm_with_warning(message, warning, default)

    def progress_bar_eta(self, label: str, current: int, total: int,
                         elapsed: float, width: int = 30) -> None:
        """Display a progress bar with percentage and ETA."""
        self._emit("progress_bar_eta", {"label": label})
        self._interactive.progress_bar_eta(label, current, total, elapsed, width)

    def spinner_with_dots(self, message: str) -> None:
        """Display a spinner with dots animation."""
        self._emit("spinner_with_dots", {"message": message})
        self._interactive.spinner_with_dots(message)

    def select_with_pagination(self, message: str, options: list[str],
                               page_size: int = 10, default: str = "") -> str:
        """Select from options with page navigation."""
        self._emit("select_with_pagination", {"message": message})
        return self._interactive.select_with_pagination(message, options, page_size, default)

    def select_with_search_and_preview(self, message: str, options: list[str],
                                       preview_fn: Callable[[str], str]) -> str:
        """Select from options with type-to-filter and live preview."""
        self._emit("select_with_search_and_preview", {"message": message})
        return self._interactive.select_with_search_and_preview(message, options, preview_fn)

    def progress_bar_with_status(self, label: str, current: int, total: int,
                                 status: str = "", width: int = 30) -> None:
        """Display a progress bar with percentage and status message."""
        self._emit("progress_bar_with_status", {"label": label})
        self._interactive.progress_bar_with_status(label, current, total, status, width)

    def spinner_with_eta(self, message: str, elapsed: float, progress: float = 0) -> None:
        """Display a spinner with elapsed time and optional ETA."""
        self._emit("spinner_with_eta", {"message": message})
        self._interactive.spinner_with_eta(message, elapsed, progress)

    def select_with_grouping(self, message: str,
                             groups: dict[str, list[str]],
                             default: str = "") -> str:
        """Select from categorized options with group headers."""
        self._emit("select_with_grouping", {"message": message})
        return self._interactive.select_with_grouping(message, groups, default)

    def multi_select_with_preview(self, message: str, options: list[str],
                                  preview_fn: Callable[[str], str]) -> list[str]:
        """Multi-select from options with live preview."""
        self._emit("multi_select_with_preview", {"message": message})
        return self._interactive.multi_select_with_preview(message, options, preview_fn)

    def progress_bar_indeterminate(self, label: str, status: str = "") -> None:
        """Display an indeterminate progress bar."""
        self._emit("progress_bar_indeterminate", {"label": label})
        self._interactive.progress_bar_indeterminate(label, status)

    def table_with_search(self, headers: list[str], rows: list[list[str]],
                          title: str = "") -> list[list[str]]:
        """Display a searchable table."""
        self._emit("table_with_search", {"title": title})
        return self._interactive.table_with_search(headers, rows, title)

    def select_with_countdown(self, message: str, options: list[str],
                              timeout: int = 10, default: int = 0) -> str:
        """Select from options with auto-select countdown."""
        self._emit("select_with_countdown", {"message": message})
        return self._interactive.select_with_countdown(message, options, timeout, default)

    def confirm_with_countdown(self, message: str, timeout: int = 10,
                               default: bool = False) -> bool:
        """Confirm with auto-confirm countdown."""
        self._emit("confirm_with_countdown", {"message": message})
        return self._interactive.confirm_with_countdown(message, timeout, default)

    def progress_bar_stripe(self, label: str, current: int, total: int,
                            width: int = 30) -> None:
        """Display a striped progress bar."""
        self._emit("progress_bar_stripe", {"label": label})
        self._interactive.progress_bar_stripe(label, current, total, width)

    def spinner_with_dots_eta(self, message: str, elapsed: float,
                              progress: float = 0) -> None:
        """Display a spinner with dots and ETA."""
        self._emit("spinner_with_dots_eta", {"message": message})
        self._interactive.spinner_with_dots_eta(message, elapsed, progress)

    def confirm_with_phrase(self, message: str, phrase: str = "yes") -> bool:
        """Confirm by typing a specific phrase."""
        self._emit("confirm_with_phrase", {"message": message})
        return self._interactive.confirm_with_phrase(message, phrase)

    def progress_bar_gradient(self, label: str, current: int, total: int,
                              width: int = 30) -> None:
        """Display a gradient-colored progress bar."""
        self._emit("progress_bar_gradient", {"label": label})
        self._interactive.progress_bar_gradient(label, current, total, width)

    def spinner_pulse(self, message: str, duration: float = 2.0) -> None:
        """Display a pulsing dot spinner."""
        self._emit("spinner_pulse", {"message": message})
        self._interactive.spinner_pulse(message, duration)

    def select_with_preview_and_icons(self, message: str,
                                      options: list[tuple[str, str]],
                                      preview_fn: Callable[[str], str]) -> str:
        """Select from icon+label options with live preview."""
        self._emit("select_with_preview_and_icons", {"message": message})
        return self._interactive.select_with_preview_and_icons(message, options, preview_fn)

    def multi_confirm(self, message: str, items: list[str],
                      default: bool = True) -> dict[str, bool]:
        """Confirm multiple items with toggle."""
        self._emit("multi_confirm", {"message": message})
        return self._interactive.multi_confirm(message, items, default)

    def progress_bar_segmented(self, label: str, segments: list[tuple[str, int]],
                               width: int = 30) -> None:
        """Display a segmented progress bar."""
        self._emit("progress_bar_segmented", {"label": label})
        self._interactive.progress_bar_segmented(label, segments, width)

    def spinner_wave(self, message: str, duration: float = 2.0) -> None:
        """Display a wave animation spinner."""
        self._emit("spinner_wave", {"message": message})
        self._interactive.spinner_wave(message, duration)

    def select_with_tags(self, message: str, options: list[str],
                         tags: dict[str, list[str]]) -> str:
        """Select from options filtered by tags."""
        self._emit("select_with_tags", {"message": message})
        return self._interactive.select_with_tags(message, options, tags)

    def select_with_preview_and_grouping(self, message: str,
                                         groups: dict[str, list[str]],
                                         preview_fn: Callable[[str], str]) -> str:
        """Select from categorized options with group headers and live preview."""
        self._emit("select_with_preview_and_grouping", {"message": message})
        return self._interactive.select_with_preview_and_grouping(message, groups, preview_fn)

    def confirm_list_with_preview(self, message: str, items: list[str],
                                  preview_fn: Callable[[str], str],
                                  default: bool = True) -> list[str]:
        """Confirm a list of items with preview."""
        self._emit("confirm_list_with_preview", {"message": message})
        return self._interactive.confirm_list_with_preview(message, items, preview_fn, default)

    def progress_bar_multi_segment(self, label: str,
                                   segments: list[tuple[str, int, str]],
                                   width: int = 30) -> None:
        """Display a multi-segment progress bar."""
        self._emit("progress_bar_multi_segment", {"label": label})
        self._interactive.progress_bar_multi_segment(label, segments, width)

    def spinner_bounce(self, message: str, duration: float = 2.0) -> None:
        """Display a bouncing animation spinner."""
        self._emit("spinner_bounce", {"message": message})
        self._interactive.spinner_bounce(message, duration)

    def select_with_confirm(self, message: str, options: list[str],
                            default: str = "") -> str:
        """Select an option and confirm."""
        self._emit("select_with_confirm", {"message": message})
        return self._interactive.select_with_confirm(message, options, default)

    def confirm_with_preview_and_timeout(self, message: str, preview: str,
                                         timeout: int = 10,
                                         default: bool = False) -> bool:
        """Confirm with preview and auto-confirm countdown."""
        self._emit("confirm_with_preview_and_timeout", {"message": message})
        return self._interactive.confirm_with_preview_and_timeout(message, preview, timeout, default)

    def progress_bar_animated(self, label: str, current: int, total: int,
                              width: int = 30) -> None:
        """Display an animated shimmer progress bar."""
        self._emit("progress_bar_animated", {"label": label})
        self._interactive.progress_bar_animated(label, current, total, width)

    def spinner_clock(self, message: str, duration: float = 2.0) -> None:
        """Display a clock animation spinner."""
        self._emit("spinner_clock", {"message": message})
        self._interactive.spinner_clock(message, duration)

    def select_with_preview_and_confirm(self, message: str, options: list[str],
                                        preview_fn: Callable[[str], str],
                                        default: str = "") -> str:
        """Select with preview and confirm."""
        self._emit("select_with_preview_and_confirm", {"message": message})
        return self._interactive.select_with_preview_and_confirm(message, options, preview_fn, default)

    def confirm_with_preview_and_countdown(self, message: str, preview: str,
                                           timeout: int = 10,
                                           default: bool = True) -> bool:
        """Confirm with preview and countdown."""
        self._emit("confirm_with_preview_and_countdown", {"message": message})
        return self._interactive.confirm_with_preview_and_countdown(message, preview, timeout, default)

    def progress_bar_with_status_and_eta(self, label: str, current: int,
                                         total: int, status: str,
                                         width: int = 30,
                                         elapsed: float = 0.0) -> None:
        """Display a progress bar with status and ETA."""
        self._emit("progress_bar_with_status_and_eta", {"label": label})
        self._interactive.progress_bar_with_status_and_eta(label, current, total, status, width, elapsed)

    def spinner_with_messages(self, message: str, messages: list[str],
                              duration: float = 3.0) -> None:
        """Display a spinner cycling through messages."""
        self._emit("spinner_with_messages", {"message": message})
        self._interactive.spinner_with_messages(message, messages, duration)

    def multi_select_with_filter(self, message: str, options: list[str],
                                 default: list[str] | None = None) -> list[str]:
        """Multi-select with type-to-filter."""
        self._emit("multi_select_with_filter", {"message": message})
        return self._interactive.multi_select_with_filter(message, options, default)

    def confirm_with_countdown_and_preview(self, message: str, preview: str,
                                           timeout: int = 10,
                                           default: bool = True) -> bool:
        """Confirm with preview and countdown."""
        self._emit("confirm_with_countdown_and_preview", {"message": message})
        return self._interactive.confirm_with_countdown_and_preview(message, preview, timeout, default)

    def progress_bar_with_steps(self, label: str, steps: list[str],
                                current_step: int, width: int = 30) -> None:
        """Display a multi-step progress bar."""
        self._emit("progress_bar_with_steps", {"label": label})
        self._interactive.progress_bar_with_steps(label, steps, current_step, width)

    def spinner_with_eta_message(self, message: str, total: int,
                                 duration: float = 3.0) -> None:
        """Display a spinner with ETA message."""
        self._emit("spinner_with_eta_message", {"message": message})
        self._interactive.spinner_with_eta_message(message, total, duration)

    def table_with_search_and_preview(self, headers: list[str],
                                      rows: list[list[str]],
                                      preview_fn: Callable[[list[str]], str]) -> list[str]:
        """Searchable table with live preview."""
        self._emit("table_with_search_and_preview", {})
        return self._interactive.table_with_search_and_preview(headers, rows, preview_fn)

    def select_with_filter_and_confirm(self, message: str,
                                       options: list[str],
                                       default: str = "") -> str:
        """Select with type-to-filter and confirm."""
        self._emit("select_with_filter_and_confirm", {"message": message})
        return self._interactive.select_with_filter_and_confirm(message, options, default)

    def select_with_filter_and_preview(self, message: str,
                                       options: list[str],
                                       preview_fn: Callable[[str], str]) -> str:
        """Select with type-to-filter and live preview."""
        self._emit("select_with_filter_and_preview", {"message": message})
        return self._interactive.select_with_filter_and_preview(message, options, preview_fn)

    def select_table_with_preview(self, headers: list[str],
                                  rows: list[list[str]],
                                  preview_fn: Callable[[list[str]], str]) -> list[str]:
        """Select a row from a table with live preview."""
        self._emit("select_table_with_preview", {})
        return self._interactive.select_table_with_preview(headers, rows, preview_fn)

    def confirm_with_preview_and_edit_with_timeout(self, message: str,
                                                   preview: str,
                                                   edit_prompt: str = "Edit:",
                                                   timeout: int = 10,
                                                   default: bool = True) -> tuple[bool, str]:
        """Confirm with preview, optional edit, and timeout."""
        self._emit("confirm_with_preview_and_edit_with_timeout", {"message": message})
        return self._interactive.confirm_with_preview_and_edit_with_timeout(
            message, preview, edit_prompt, timeout, default)

    def progress_bar_with_eta_and_status(self, label: str, current: int,
                                         total: int, status: str,
                                         elapsed: float = 0.0,
                                         width: int = 30) -> None:
        """Display a progress bar with ETA and status."""
        self._emit("progress_bar_with_eta_and_status", {"label": label})
        self._interactive.progress_bar_with_eta_and_status(
            label, current, total, status, elapsed, width)

    def spinner_with_dots_and_status(self, message: str, status: str,
                                     duration: float = 2.0) -> None:
        """Display a spinner with dots and status."""
        self._emit("spinner_with_dots_and_status", {"message": message})
        self._interactive.spinner_with_dots_and_status(message, status, duration)

    def select_with_preview_and_countdown(self, message: str,
                                          options: list[str],
                                          preview_fn: Callable[[str], str],
                                          timeout: int = 10,
                                          default: str = "") -> str:
        """Select with preview and countdown."""
        self._emit("select_with_preview_and_countdown", {"message": message})
        return self._interactive.select_with_preview_and_countdown(
            message, options, preview_fn, timeout, default)

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
