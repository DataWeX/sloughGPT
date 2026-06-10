"""
Printer — Consistent CLI output with Rich.

Same API as before (success, error, header, table, etc.)
but powered by rich under the hood.

TUI sub-module provides live-updating dashboards and panels
for long-running processes. Use ``printer.tui`` or import
``core.tui`` directly.
"""

import os
import json as _json
from typing import Optional, List
from rich.console import Console
from rich.table import Table as RichTable
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich import box


_console = Console(highlight=False)


class Printer:
    """Consistent CLI output formatting powered by Rich."""

    def __init__(self, width: Optional[int] = None, color: bool = True):
        self.width = width or _console.width
        self._color_enabled = color

    @property
    def color_enabled(self) -> bool:
        return self._color_enabled

    @color_enabled.setter
    def color_enabled(self, value: bool):
        self._color_enabled = value

    def success(self, message: str):
        _console.print(f"  [green]✓[/] {message}")

    def error(self, message: str):
        _console.print(f"  [red]✗[/] {message}")

    def warning(self, message: str):
        _console.print(f"  [yellow]![/] {message}")

    def info(self, message: str):
        _console.print(f"  [blue]ℹ[/] {message}")

    def step(self, message: str):
        _console.print(f"  [cyan]→[/] {message}")

    def header(self, title: str, char: str = "="):
        line = char * self.width
        _console.print(f"[bold]{title}[/]")
        _console.print(f"[dim]{line}[/]")

    def section(self, title: str):
        _console.print()
        _console.print(f"[bold]{title}[/]")
        _console.print(f"[dim]{'-' * self.width}[/]")

    def key_value(self, key: str, value: str, indent: int = 2):
        padding = " " * indent
        _console.print(f"{padding}[dim]{key}:[/] {value}")

    def table(
        self,
        headers: List[str],
        rows: List[List[str]],
        align: Optional[List[str]] = None,
    ):
        if not rows:
            return
        t = RichTable(box=box.SIMPLE, show_header=True, header_style="bold")
        for i, h in enumerate(headers):
            justification = {"l": "left", "r": "right", "c": "center"}.get(
                align[i] if align and i < len(align) else "l", "left"
            )
            t.add_column(h, justify=justification)
        for row in rows:
            t.add_row(*row)
        _console.print(t)

    def divider(self, char: str = "-"):
        _console.print(f"[dim]{char * self.width}[/]")

    def print_json(self, data: dict, indent: int = 2):
        text = _json.dumps(data, indent=indent, default=str)
        syntax = Syntax(text, "json", theme="monokai", line_numbers=False)
        _console.print(syntax)

    def blank(self, count: int = 1):
        for _ in range(count):
            _console.print()

    def status(self, label: str, value: str, status: str = "ok"):
        colors = {"ok": "green", "warn": "yellow", "error": "red", "info": "blue"}
        color = colors.get(status, "white")
        indicator = {"ok": "✓", "warn": "!", "error": "✗", "info": "ℹ"}.get(status, "•")
        _console.print(f"  [[{color}]{indicator}[/]] {label}: {value}")

    def command(self, cmd: str, description: str = ""):
        if description:
            _console.print(f"  [cyan]{cmd:<30}[/] [dim]{description}[/]")
        else:
            _console.print(f"  [cyan]{cmd}[/]")


printer = Printer()

# TUI sub-module — pure-ANSI components
from core.tui import DevDashboard, TabConfig
