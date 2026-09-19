"""PromptRenderer — single owner for shell prompt expansion.

Previously ``ShellREPL._render_prompt`` expanded PS1 escapes inline
while ``TuiRepl._render_input`` hardcoded its default prompt. Both now build a
:class:`PromptContext` and call :meth:`PromptRenderer.render`, so PS1,
exit-code prefix, and log badge behave identically in line mode and TUI
(the TUI adds its own horizontal scroll view on top).

Readline safety (wrapping ANSI in ``\\x01..\\x02``) stays in
``domain.shell._internal.io.readline_safe_prompt`` and is applied at the
``ConsoleIO.read`` boundary, not here, so this renderer stays pure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptContext:
    """Precomputed values for prompt expansion (no I/O inside render)."""

    ps1: str = "\u0343"
    cwd: str = "~"
    user: str = "user"
    host: str = ""
    time_str: str = ""
    cmd_count: int = 1
    exit_code: int = 0
    model: str = ""
    soul: str = ""
    badge: str = ""
    exit_prefix: str = ""
    suffix: str = ""


class PromptRenderer:
    """Pure PS1 expander: ``\\h \\w \\t \\u \\s \\# \\n \\m \\S``."""

    @staticmethod
    def render(ctx: PromptContext) -> str:
        s = ctx.ps1
        s = s.replace("\\h", ctx.host)
        s = s.replace("\\w", ctx.cwd)
        s = s.replace("\\t", ctx.time_str)
        s = s.replace("\\u", ctx.user)
        s = s.replace("\\s", "sloughgpt")
        s = s.replace("\\#", str(ctx.cmd_count))
        s = s.replace("\\n", "\n")
        s = s.replace("\\m", ctx.model)
        s = s.replace("\\S", ctx.soul)
        if ctx.exit_code != 0 and ctx.exit_prefix:
            s = f"{ctx.exit_prefix}{s}"
        if ctx.badge:
            s = f"{s}{ctx.badge}"
        if ctx.suffix:
            s = f"{s}{ctx.suffix}"
        return s
