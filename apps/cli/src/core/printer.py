"""
Printer — Consistent CLI output with Rich, backed by the OOP logging hierarchy.

Inherits from ``CLILogger`` (domains.logging) so that all output goes through
the structured Logger system.  Keeps the existing API (success, error, header,
table, etc.) for backward compatibility with all CLI commands.

Usage::

    from core.printer import printer

    printer.success("model loaded")
    printer.error("connection failed")
    printer.table(["Name", "Size"], [["gpt2", "500MB"]])
"""

import os
import sys
import json as _json
from typing import Optional, List

# Lazy import of CLILogger to avoid circular deps in CLI context.
# The CLI runs from apps/cli/ which adds core-py to sys.path at startup.
from domains.logging import CLILogger, LogLevel


class Printer(CLILogger):
    """CLI output formatter — inherits from CLILogger, adds Printer-specific helpers.

    Inherits: debug, info, warning, error, critical, exception, emit,
              success, step, header, section, table, json, status, divider.

    Adds: key_value, print_json, blank, command.
    """

    def __init__(self, name: str = "slo.cli", level: LogLevel = LogLevel.DEBUG):
        super().__init__(name=name, level=level)
        # Printer exposes width as a public attribute (used by commands)
        try:
            from rich.console import Console
            self.width = Console(highlight=False).width
        except Exception:
            self.width = 80

    # ── Printer-specific helpers (not on CLILogger) ─────────────────────

    def key_value(self, key: str, value: str, indent: int = 2):
        """Print a dim key: value pair."""
        from rich.console import Console
        c = Console(highlight=False)
        padding = " " * indent
        c.print(f"{padding}[dim]{key}:[/] {value}")

    def print_json(self, data: dict, indent: int = 2):
        """Pretty-print JSON with syntax highlighting."""
        # Delegate to CLILogger.json()
        self.json(data, indent=indent)

    def blank(self, count: int = 1):
        """Print blank lines."""
        from rich.console import Console
        c = Console(highlight=False)
        for _ in range(count):
            c.print()

    def command(self, cmd: str, description: str = ""):
        """Print a command with optional description."""
        from rich.console import Console
        c = Console(highlight=False)
        if description:
            c.print(f"  [cyan]{cmd:<30}[/] [dim]{description}[/]")
        else:
            c.print(f"  [cyan]{cmd}[/]")

    # ── Backward compat properties ──────────────────────────────────────

    @property
    def color_enabled(self) -> bool:
        return True

    @color_enabled.setter
    def color_enabled(self, value: bool):
        pass  # Rich handles color detection internally


printer = Printer()
