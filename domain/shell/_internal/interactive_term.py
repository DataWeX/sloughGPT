"""
InteractivePrompt — arrow-key navigation and type-to-filter for line-mode.

Provides gh-auth-login-style interactive selection in the default
(readline-based) shell.  Falls back to a numbered menu when the terminal
does not support raw input (e.g. piped stdin, tests with MemoryIO).

Usage:
    from domain.shell._internal.interactive import InteractivePrompt
    prompt = InteractivePrompt(io)
    choice = prompt.select("Pick an account", ["GitHub.com", "GitHub Enterprise"])
"""

from __future__ import annotations

import os
import sys

from .tui_session import (
    KEY_ALT,
    KEY_BACKSPACE,
    KEY_CHAR,
    KEY_CTRL,
    KEY_CTRL_A,
    KEY_CTRL_C,
    KEY_CTRL_E,
    KEY_DELETE,
    KEY_DOWN,
    KEY_END,
    KEY_ENTER,
    KEY_ESC,
    KEY_HOME,
    KEY_LEFT,
    KEY_PAGE_DOWN,
    KEY_PAGE_UP,
    KEY_RIGHT,
    KEY_SPACE,
    KEY_TAB,
    KEY_UP,
    Key,
    _RawMode,
    decode_key,
    terminal_bytes,
)

# ── Raw key reading ──────────────────────────────────────────────────────────

# The prompt layer has always spoken `_KEY_*`, so those names stay. What
# changes is that each is now *bound* to the constant it stood for rather than
# restating its string: eighteen literals were copied here from `tui_session`,
# byte-identical with nothing keeping the two in sync, which is how a vocabulary
# ends up with two answers to "what is `left`". An assignment cannot drift from
# its constant, and because it is not an import nothing mistakes a re-export for
# a dead name and deletes it.
_KEY_UP = KEY_UP
_KEY_DOWN = KEY_DOWN
_KEY_LEFT = KEY_LEFT
_KEY_RIGHT = KEY_RIGHT
_KEY_ENTER = KEY_ENTER
_KEY_ESC = KEY_ESC
_KEY_BACKSPACE = KEY_BACKSPACE
_KEY_DELETE = KEY_DELETE
_KEY_HOME = KEY_HOME
_KEY_END = KEY_END
_KEY_CTRL_C = KEY_CTRL_C
_KEY_CTRL_A = KEY_CTRL_A
_KEY_CTRL_E = KEY_CTRL_E
_KEY_PAGE_UP = KEY_PAGE_UP
_KEY_PAGE_DOWN = KEY_PAGE_DOWN
_KEY_SPACE = KEY_SPACE
_KEY_TAB = KEY_TAB
_KEY_CHAR = KEY_CHAR

# The two chord kinds, added when Ctrl+W and Alt+D became bindable in the
# prompt fields. Unlike the twenty above there was no literal here to replace
# — nothing in the prompt layer had ever named a chord — so these are bound
# rather than re-pointed, and `matches(_KEY_CTRL, char="w")` is how a caller
# says "Ctrl+W" without the vocabulary growing a `KEY_CTRL_W`.
_KEY_CTRL = KEY_CTRL
_KEY_ALT = KEY_ALT


def _terminal_width() -> int:
    """Get terminal width in columns."""
    try:
        return os.get_terminal_size().columns
    except (AttributeError, ValueError, OSError):
        return 80


def _terminal_height() -> int:
    """Get terminal height in rows."""
    try:
        return os.get_terminal_size().lines
    except (AttributeError, ValueError, OSError):
        return 24


def _truncate(text: str, width: int) -> str:
    """Truncate *text* to *width* columns, adding an ellipsis when cut."""
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def _dash_row(widths: list[int]) -> str:
    """Render a box-drawing dash separator row matching *widths*."""
    return "  ".join("\u2500" * w for w in widths)


# `_RawKey` was a second class with `_RawKey.__init__` and a `RawKey(...)` repr
# — structurally identical to `Key`, and built nowhere in production:
# `_read_raw_key` was annotated *and* documented as returning one, then
# returned `decode_key(...)`, a `Key`. Two types for one value is the
# duplication the shared vocabulary exists to remove, so the class is now an
# alias: the name `interactive.py` re-exports and two tests import survives,
# the second implementation does not, and the annotation on `_read_raw_key`
# below is finally true rather than aspirational.
_RawKey = Key


def _read_raw_key(fd: int) -> _RawKey:
    """Read a single key press from a raw-mode file descriptor.

    Kept as a named wrapper because the interactive facade re-exports it and
    tests import it by name.  The escape-sequence parsing lives in
    `tui_session.decode_key`, which `TtySession.read_key` also calls, so both
    backends decode the wire format through one implementation.  That is the
    point: a fix to arrow-key or delete handling cannot land in one copy and
    miss the other, which is the failure that put a second, subtly different
    decoder in the TUI to begin with.

    Args:
        fd: file descriptor for the terminal (e.g. sys.stdin.fileno())

    Returns:
        _RawKey with kind and optional char.
    """
    # The byte source comes from the same module for the same reason the
    # decoder does: it owns the "how long do I wait for the rest of a sequence"
    # decision, so a bound added there reaches this path too rather than leaving
    # one entry point hanging on a lone Esc and the other not.
    return decode_key(terminal_bytes(fd))


# ── Terminal helpers ──────────────────────────────────────────────────────────


class _RawTerminal(_RawMode):
    """Context manager that puts the terminal in raw mode and restores it.

    The behaviour is `_RawMode`, the same scope `TtySession.raw()` hands out, so
    the two backends enter and leave raw mode identically.  This subclass exists
    only to keep the name the interactive facade re-exports; it adds nothing.
    """


# ── ANSI helpers ──────────────────────────────────────────────────────────────

_ERASE_LINE = "\033[2K"
_CURSOR_UP = "\033[1A"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_REVERSE = "\033[7m"
_RESET_REVERSE = "\033[27m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _rewind(rows: int) -> None:
    """Return to the top of a frame `rows` rows tall that was just painted.

    A prompt that repaints on every keystroke has to undo the frame it just
    wrote before writing the next one, or the two stack: the erase inside the
    paint clears a single row while the paint writes N, so a handful of keys
    scrolls a dozen stale copies past the user.

    The order is the other half of it. The cursor sits on the frame's *last*
    row, so the erasure has to start there and only then walk upward — the
    shape most of these clears had was `up` then `erase`, which erases the row
    *above* the frame and leaves the frame's own last row behind. On a
    terminal parked at row 0 the upward move clamps and the mistake hides;
    anywhere else it shows.

    `rows` is measured off the frame actually written rather than guessed from
    the prompt's data, because the frame also carries the query line, the help
    line and any inline error, none of which the data counts.

    This lives here because every prompt module repaints, and a second copy of
    it is how the two shapes came to differ.
    """
    if not rows:
        return  # first paint: there is no frame to undo
    sys.stdout.write(f"\r{_ERASE_LINE}")
    for _ in range(rows - 1):
        sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")


# ── InteractivePrompt ─────────────────────────────────────────────────────────
