"""
CLILogger — Rich-powered output for ``sloughgpt`` CLI commands.

Inherits Logger and routes records through ``rich.console.Console`` for
formatted terminal output with tables, panels, and syntax highlighting.

Usage::

    from domains.logging import CLILogger, LogLevel

    log = CLILogger("slo.cli")
    log.info("model loaded", model="gpt2", params="124M")
    log.success("training complete", loss="0.42")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Logger, LogLevel, LogRecord


# ── Rich imports (lazy — only loaded when CLILogger is instantiated) ────

_console = None
_Table = None
_Panel = None
_Syntax = None
_box = None

# While the shell TUI is active the terminal belongs to curses, so the rich
# console must not write to it.  Handlers (e.g. the LogBuffer console pane)
# keep receiving records regardless.
_TERMINAL_ENABLED = True


def set_cli_terminal(enabled: bool) -> None:
    """Enable or disable rich terminal output (used by the shell TUI)."""
    global _TERMINAL_ENABLED
    _TERMINAL_ENABLED = enabled


def _cli_print(renderable) -> None:
    """Print via the rich console unless terminal output is disabled."""
    if _TERMINAL_ENABLED:
        _console.print(renderable)


def _ensure_rich():
    global _console, _Table, _Panel, _Syntax, _box
    if _console is not None:
        return
    from rich.console import Console
    from rich.table import Table as RichTable
    from rich.panel import Panel as RichPanel
    from rich.syntax import Syntax as RichSyntax
    from rich import box as rich_box

    _console = Console(highlight=False)
    _Table = RichTable
    _Panel = RichPanel
    _Syntax = RichSyntax
    _box = rich_box


# ── Level → Rich style mapping ────────────────────────────────────────

_LEVEL_STYLE = {
    LogLevel.DEBUG:    ("dim cyan",    "debug"),
    LogLevel.INFO:     ("green",       "info"),
    LogLevel.WARNING:  ("bold yellow", "warning"),
    LogLevel.ERROR:    ("bold red",    "error"),
    LogLevel.CRITICAL: ("bold white on red", "critical"),
}

# Semantic prefix icons (inherited from CLI Printer convention)
_ICON = {
    LogLevel.DEBUG:    ("dim", "·"),
    LogLevel.INFO:     ("blue", "ℹ"),
    LogLevel.WARNING:  ("yellow", "!"),
    LogLevel.ERROR:    ("red", "✗"),
    LogLevel.CRITICAL: ("bold red", "✗"),
}


class CLILogger(Logger):
    """Rich-based CLI logger — inherits from Logger, outputs via Rich Console.

    Supports all base Logger methods plus CLI-specific helpers:
    ``success()``, ``step()``, ``header()``, ``section()``, ``table()``.

    Parameters:
        name:    Logger name (e.g. ``"slo.cli"``).
        level:   Minimum severity to emit.
        context: Default context attached to every record.
    """

    def __init__(
        self,
        name: str = "slo.cli",
        level: LogLevel = LogLevel.INFO,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, level=level, context=context)
        _ensure_rich()

    # ── Core emit ───────────────────────────────────────────────────────

    def emit(self, record: LogRecord) -> None:
        """Format and print the record via Rich Console (thread-safe)."""
        style, _ = _LEVEL_STYLE.get(record.level, ("white", record.level.value))
        icon_style, icon = _ICON.get(record.level, ("white", "·"))

        # Build Rich Text parts
        from rich.text import Text

        parts = Text()

        # Icon
        parts.append(f"  {icon} ", style=icon_style)

        # Level tag
        parts.append(f"[{record.level.value}] ", style=style)

        # Logger name
        parts.append(f"{record.logger} ", style="dim")

        # Context
        if record.context:
            ctx_str = " ".join(f"{k}={v}" for k, v in record.context.items())
            parts.append(f"{ctx_str} ", style="dim")

        # Message
        parts.append(record.message)

        # Exception
        if record.exception:
            parts.append(f" — {record.exception}", style="red")

        with self._lock:
            _cli_print(parts)

    # ── CLI-specific helpers (not on base Logger) ───────────────────────

    def success(self, msg: str, **ctx: Any) -> None:
        """Log a success (green checkmark)."""
        from rich.text import Text
        parts = Text()
        parts.append("  ✓ ", style="green")
        parts.append(msg)
        if ctx:
            ctx_str = " ".join(f"{k}={v}" for k, v in ctx.items())
            parts.append(f" {ctx_str}", style="dim")
        with self._lock:
            _cli_print(parts)

    def step(self, msg: str, **ctx: Any) -> None:
        """Log a step/action (cyan arrow)."""
        from rich.text import Text
        parts = Text()
        parts.append("  → ", style="cyan")
        parts.append(msg)
        if ctx:
            ctx_str = " ".join(f"{k}={v}" for k, v in ctx.items())
            parts.append(f" {ctx_str}", style="dim")
        with self._lock:
            _cli_print(parts)

    def header(self, title: str, char: str = "=") -> None:
        """Print a bold header with a separator line."""
        width = _console.width
        with self._lock:
            _cli_print(f"[bold]{title}[/]")
            _cli_print(f"[dim]{char * width}[/]")

    def section(self, title: str) -> None:
        """Print a section divider."""
        width = _console.width
        with self._lock:
            _cli_print("")
            _cli_print(f"[bold]{title}[/]")
            _cli_print(f"[dim]{'-' * width}[/]")

    def table(
        self,
        headers: List[str],
        rows: List[List[str]],
        align: Optional[List[str]] = None,
    ) -> None:
        """Print a Rich table."""
        if not rows:
            return
        t = _Table(box=_box.SIMPLE, show_header=True, header_style="bold")
        for i, h in enumerate(headers):
            justification = {"l": "left", "r": "right", "c": "center"}.get(
                align[i] if align and i < len(align) else "l", "left"
            )
            t.add_column(h, justify=justification)
        for row in rows:
            t.add_row(*row)
        with self._lock:
            _cli_print(t)

    def json(self, data: Any, indent: int = 2) -> None:
        """Pretty-print JSON with syntax highlighting."""
        import json as _json
        text = _json.dumps(data, indent=indent, default=str)
        syntax = _Syntax(text, "json", theme="monokai", line_numbers=False)
        with self._lock:
            _cli_print(syntax)

    def status(self, label: str, value: str, status: str = "ok") -> None:
        """Print a key-value status line with a colored indicator."""
        colors = {"ok": "green", "warn": "yellow", "error": "red", "info": "blue"}
        icons = {"ok": "✓", "warn": "!", "error": "✗", "info": "ℹ"}
        color = colors.get(status, "white")
        icon = icons.get(status, "•")
        with self._lock:
            _cli_print(f"  [[{color}]{icon}[/]] {label}: {value}")

    def divider(self, char: str = "-") -> None:
        """Print a separator line."""
        with self._lock:
            _cli_print(f"[dim]{char * _console.width}[/]")

    def key_value(self, key: str, value: str, indent: int = 2):
        """Print a dim key: value pair."""
        padding = " " * indent
        if key:
            with self._lock:
                _cli_print(f"{padding}[dim]{key}:[/] {value}")
        else:
            with self._lock:
                _cli_print(f"{padding}{value}")

    def blank(self, count: int = 1):
        """Print blank lines."""
        with self._lock:
            for _ in range(count):
                _cli_print("")

    def command(self, cmd: str, description: str = ""):
        """Print a command with optional description."""
        from rich.text import Text
        parts = Text()
        parts.append(f"  {cmd:<30}", style="cyan")
        if description:
            parts.append(f" {description}", style="dim")
        with self._lock:
            _cli_print(parts)
