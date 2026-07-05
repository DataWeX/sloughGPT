"""OpenCode-inspired Rich components for SloughGPT TUI.

Adopts opencode's clean, minimal terminal UI style:
- 256-color palette with specific color codes
- Clean borders (NO double borders, minimal panels)
- Modular components
- Vim-like shortcuts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from rich.console import Console, ConsoleOptions, RenderResult
from rich.style import Style
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.box import Box, ROUNDED, HEAVY, SIMPLE
from rich.panel import Panel as RichPanel
from rich.table import Table


CONSOLE = Console()


@dataclass
class Color:
    PRIMARY = "cyan"
    SECONDARY = "magenta"
    SUCCESS = "green"
    WARNING = "yellow"
    ERROR = "red"
    MUTED = "color(241)"
    HIGHLIGHT = "color(69)"
    DIM = "color(241)"
    WHITE = "white"
    BORDER = "color(240)"


class Style:
    def __init__(
        self,
        fg: str = "",
        bg: str = "",
        bold: bool = False,
        italic: bool = False,
        dim: bool = False,
    ):
        self.fg = fg
        self.bg = bg
        self.bold = bold
        self.italic = italic
        self.dim = dim

    def render(self, text: str) -> str:
        parts = []
        if self.fg:
            parts.append(f"({self.fg})")
        if self.bold:
            parts.append("bold")
        if self.dim:
            parts.append("dim")
        return f"[{' '.join(parts)}]{text}[/]"


@dataclass
class Border:
    NONE = ""
    LIGHT = "─│┌┐└┘├┤┬┴┼"
    ROUNDED = "─│╭╮╯╰├┤╭╮╯╰"
    HEAVY = "━┃┏┓┗┛┣┫┳┻╋"


class Header:
    """Clean header bar like opencode."""

    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle

    def render(self) -> None:
        title = Text(f" {self.title} ", style=f"bold {Color.PRIMARY}")
        CONSOLE.print()
        CONSOLE.print(title)
        if self.subtitle:
            CONSOLE.print(Text(f"  {self.subtitle}", style=Color.MUTED))


class Section:
    """Minimal section with clean border."""

    def __init__(
        self,
        title: str,
        content: str = "",
        border_style: str = Color.BORDER,
    ):
        self.title = title
        self.content = content
        self.border_style = border_style

    def render(self) -> None:
        if self.content:
            CONSOLE.print(Panel(
                self.content,
                title=f" {self.title} ",
                border_style=self.border_style,
                box=ROUNDED,
                padding=(0, 1),
            ))
        else:
            CONSOLE.print(Text(f"\n{self.title}", style=f"bold {Color.PRIMARY}"))


class StatusLine:
    """Single line status display."""

    def __init__(self, items: Dict[str, str]):
        self.items = items

    def render(self) -> None:
        parts = []
        for key, value in self.items.items():
            parts.append(f"[bold {Color.PRIMARY}]{key}[/]: [bold white]{value}[/]")
        CONSOLE.print("  " + "  |  ".join(parts))


class StatusTable:
    """Table for displaying status/key-value pairs."""

    def __init__(self, title: str = ""):
        self.title = title
        self.rows: List[Tuple[str, str]] = []

    def add(self, key: str, value: str) -> "StatusTable":
        self.rows.append((key, value))
        return self

    def render(self) -> None:
        table = Table(
            show_header=False,
            box=SIMPLE,
            border_style=Color.BORDER,
            padding=(0, 2),
            pad_edge=False,
        )
        table.add_column(style=Color.MUTED, width=15)
        table.add_column(style=Color.WHITE)

        for key, value in self.rows:
            table.add_row(key, value)

        if self.title:
            CONSOLE.print(Panel(
                table,
                title=f" {self.title} ",
                border_style=Color.BORDER,
                box=ROUNDED,
                padding=(0, 1),
            ))
        else:
            CONSOLE.print(table)


class ListView:
    """Simple selectable list like opencode's navigation."""

    def __init__(self, title: str = ""):
        self.title = title
        self.items: List[Tuple[str, str]] = []
        self.selected = 0

    def add(self, label: str, description: str = "") -> "ListView":
        self.items.append((label, description))
        return self

    def render(self) -> None:
        if self.title:
            CONSOLE.print(Text(f"\n{self.title}", style=f"bold {Color.PRIMARY}"))

        for i, (label, desc) in enumerate(self.items):
            prefix = ">"
            marker = f"[{Color.HIGHLIGHT}]{prefix}[/] "
            if i == self.selected:
                line = Text(marker + label, style=f"bold {Color.WHITE}")
            else:
                line = Text(f"  {label}", style=Color.DIM)

            if desc:
                line += Text(f"  {desc}", style=Color.MUTED)

            CONSOLE.print(line)


class ProgressBar:
    """Clean progress bar."""

    def __init__(self, description: str = "Working"):
        self.description = description
        self.total = 100
        self.current = 0

    def update(self, current: int, total: Optional[int] = None) -> None:
        self.current = current
        if total:
            self.total = total

    def render(self) -> None:
        pct = self.current / max(self.total, 1) * 100
        filled = int(pct / 100 * 20)
        bar = "█" * filled + "░" * (20 - filled)
        CONSOLE.print(f"  {bar} {pct:3.0f}%  {self.description}")


class LiveProgress:
    """Progress with live updates."""

    def __init__(self, description: str = "Working"):
        self.description = description
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=CONSOLE,
        )

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, *args):
        self.progress.__exit__(*args)

    def add_task(self, description: str, total: Optional[int] = None):
        return self.progress.add_task(description, total=total)


class ChoiceMenu:
    """OpenCode-style choice menu with keyboard hints."""

    def __init__(self, title: str = "Options"):
        self.title = title
        self.choices: List[Tuple[str, str, str]] = []

    def add(self, label: str, description: str, key: str = "") -> "ChoiceMenu":
        self.choices.append((label, description, key))
        return self

    def render(self) -> None:
        CONSOLE.print()
        CONSOLE.print(Text(f"  {self.title}", style=f"bold {Color.PRIMARY}"))
        CONSOLE.print()

        for label, desc, key in self.choices:
            key_hint = f" [{Color.HIGHLIGHT}]{key}[/]" if key else ""
            CONSOLE.print(f"    {label}{key_hint}")
            if desc:
                CONSOLE.print(f"      {desc}", style=Color.MUTED)


class HealthIndicator:
    """Health status with colored dot."""

    HEALTHY = Color.SUCCESS
    WARNING = Color.WARNING
    ERROR = Color.ERROR
    UNKNOWN = Color.MUTED

    @classmethod
    def indicator(cls, status: str) -> str:
        if status == "healthy":
            return f"[{cls.HEALTHY}]●[/]"
        elif status == "degraded":
            return f"[{cls.WARNING}]◐[/]"
        return f"[{cls.ERROR}]○[/]"

    @classmethod
    def status(cls, name: str, code: int, detail: str = "") -> None:
        if 200 <= code < 300:
            icon = f"[{cls.HEALTHY}]●[/]"
            label = "healthy"
        elif 400 <= code < 500:
            icon = f"[{cls.WARNING}]◐[/]"
            label = "degraded"
        elif code >= 500:
            icon = f"[{cls.ERROR}]○[/]"
            label = "unhealthy"
        else:
            icon = f"[{cls.UNKNOWN}]○[/]"
            label = "unknown"

        CONSOLE.print(f"  {icon} {name}  {label}  {detail}", style=Color.MUTED if detail else "")


class InfoBox:
    """Minimal info box."""

    def __init__(self, text: str, style: str = Color.PRIMARY):
        self.text = text
        self.style = style

    def render(self) -> None:
        CONSOLE.print()
        CONSOLE.print(Text(f"  {self.text}  ", style=f"bold {self.style}"))


class Divider:
    """Clean divider line."""

    def __init__(self, char: str = "─", style: str = Color.BORDER):
        self.char = char
        self.style = style

    def render(self) -> None:
        CONSOLE.print(Text(self.char * 60, style=self.style))


class LiveTable:
    """Table that updates in place."""

    def __init__(self, columns: List[str]):
        self.columns = columns
        self.rows: List[List[str]] = []

    def add_row(self, *values: str):
        self.rows.append(list(values))

    def render(self) -> Panel:
        table = Table(
            show_header=True,
            header_style=f"bold {Color.PRIMARY}",
            box=ROUNDED,
            border_style=Color.BORDER,
            padding=(0, 1),
        )
        for col in self.columns:
            table.add_column(col)

        for row in self.rows:
            table.add_row(*[str(v) for v in row])

        return Panel(table, border_style=Color.BORDER, padding=(0, 1))


def header(title: str, subtitle: str = ""):
    Header(title, subtitle).render()


def section(title: str, content: str = ""):
    Section(title, content).render()


def status_table(title: str = "") -> StatusTable:
    return StatusTable(title)


def choice_menu(title: str = "Options") -> ChoiceMenu:
    return ChoiceMenu(title)


def health_indicator(name: str, code: int, detail: str = ""):
    HealthIndicator.status(name, code, detail)


def info(text: str):
    InfoBox(text).render()


def divider():
    Divider().render()


def live_progress(description: str = "Working") -> LiveProgress:
    return LiveProgress(description)


def print_divider(char: str = "─"):
    CONSOLE.print(Text(char * 60, style=Color.BORDER))
