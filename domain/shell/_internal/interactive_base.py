"""Shared substrate for the InteractivePrompt families.

The state every family needs (`_io`, `_is_tty`), and the four private helpers more than one family calls. Every family mixin inherits from this class, which is why it lives in its own module: the families import it here, and the assembly module imports the families.
"""

from __future__ import annotations

import os
import sys

from .interactive_term import (
    _KEY_ALT,
    _KEY_BACKSPACE,
    _KEY_CHAR,
    _KEY_CTRL,
    _KEY_DELETE,
    _KEY_END,
    _KEY_HOME,
    _KEY_LEFT,
    _KEY_RIGHT,
    _RESET_REVERSE,
    _REVERSE,
)
from .textedit import delete_back, delete_forward, word_backward, word_forward
from .tui_session import TtySession


class InteractiveBase:
    """TTY detection and fd access, shared by every prompt family.

    Every `_raw` path starts by asking whether the terminal supports raw input and, if so, which descriptor to read.
    """

    def __init__(self, io: object) -> None:
        self._io = io
        self._session = TtySession(io)
        self._is_tty = self._detect_tty()

    def _detect_tty(self) -> bool:
        """Whether raw terminal input is available. Declared, not probed.

        The IO layer says whether it holds the only handle on the terminal, and
        that is the whole question. The old `os.isatty(sys.stdin.fileno())`
        fallback could not be trusted: a curses session is a tty, so it answered
        True for `TuiIo` even though the event loop in front of it owned the
        descriptor. The prompt layer then entered raw mode behind curses' back
        and read from the same fd the loop was polling. No probe distinguishes
        the two cases — a declaration can, and `TuiIo.owns_stdin` is it.

        The `_is_tty` and `os.isatty` fallbacks remain for duck-typed IO objects
        that predate the declaration, including the stand-ins tests assign to
        `_interactive._io`. New ShellIO implementations declare instead.
        """
        declared = getattr(self._io, "owns_stdin", None)
        if declared is not None:
            return bool(declared)
        if hasattr(self._io, "_is_tty"):
            return self._io._is_tty
        try:
            fd = sys.stdin.fileno()
            return os.isatty(fd)
        except (AttributeError, ValueError, OSError):
            return False

    def _tree_leaves(self, tree: dict) -> list[str]:
        leaves: list[str] = []
        for _k, v in tree.items():
            if isinstance(v, dict):
                leaves.extend(self._tree_leaves(v))
            elif isinstance(v, list):
                leaves.extend(v)
        return leaves

    def _hex_to_rgb(self, hex_str: str) -> tuple[int, int, int]:
        h = hex_str.lstrip("#")
        if len(h) == 3:
            h = h[0] * 2 + h[1] * 2 + h[2] * 2
        try:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        except (ValueError, IndexError):
            return (255, 255, 255)

    # ── type-to-filter boxes ─────────────────────────────────────────────────

    @staticmethod
    def _filter_key(query: str, caret: int, key) -> tuple[str, int, bool]:
        """Apply one keystroke to a type-to-filter box.

        Returns `(query, caret, edited)`.

        A filter box compares and renders a `str`, so it keeps one — but the
        motion and the deletes run over the list-backed buffer `textedit` was
        written for, which is what makes a filter box and the shell's own
        prompt agree on where a word starts. One scan, one definition.

        `edited` reports whether the *text* changed, not whether the caret
        merely moved: repositioning the caret does not narrow or widen what is
        being filtered, so it must not reset the caller's selection. Keys the
        box does not own (Up, Down, Enter, function keys) fall through
        untouched and come back with `edited` False, so a caller can hand its
        own navigation keys in first and treat whatever returns as its own.

        The caret is expected to start at `len(query)` — at the end — so a box
        nobody has ever moved behaves exactly as it did before it had a caret.
        """
        buf = list(query)
        caret = max(0, min(caret, len(buf)))

        if key.matches(_KEY_LEFT, ctrl=True):
            caret = word_backward(buf, caret)
        elif key.matches(_KEY_RIGHT, ctrl=True):
            caret = word_forward(buf, caret)
        elif key.matches(_KEY_CTRL, char="w"):
            caret, _ = delete_back(buf, caret)
        elif key.matches(_KEY_ALT, char="d"):
            caret, _ = delete_forward(buf, caret)
        elif key.matches(_KEY_LEFT):
            caret = max(0, caret - 1)
        elif key.matches(_KEY_RIGHT):
            caret = min(len(buf), caret + 1)
        elif key.kind == _KEY_HOME:
            caret = 0
        elif key.kind == _KEY_END:
            caret = len(buf)
        elif key.kind == _KEY_BACKSPACE or (key.kind == _KEY_CHAR and key.char == "\x7f"):
            # Rubout arrives as a kind on some backends and as a plain
            # character on others; both mean "take the one before the caret".
            if caret > 0:
                del buf[caret - 1]
                caret -= 1
        elif key.kind == _KEY_DELETE:
            if caret < len(buf):
                del buf[caret]
        elif key.kind == _KEY_CHAR:
            buf.insert(caret, key.char)
            caret += 1

        new_query = "".join(buf)
        return new_query, caret, new_query != query

    @staticmethod
    def _filter_view(query: str, caret: int, style: str | None = None) -> str:
        """*query* with the caret drawn as the shell's reverse-video bar.

        The same bar every other caret-bearing field draws (`interactive_prompt`
        renders `before` + reverse `|` + `after`), so a filter box reads as the
        same control as the prompt it sits beside. `_RESET_REVERSE` is
        `\\033[27m` — reverse off only — so the surrounding `_DIM` survives
        across the marker instead of popping back to normal for `after`.

        `style` is the ANSI prefix the bar takes. The search boxes that already
        painted a bar after their query pass their own — `_CYAN` in most, `_RESET`
        in one — so they do not restyle at rest: with the caret at the end, which
        is where it starts, what comes out is byte-identical to what they painted
        before they had a caret at all.

        The bar is composed here rather than at the call site on purpose. An
        f-string nested in another f-string's expression — what an inline marker
        amounts to — parses only on Python 3.12+, and this project declares 3.9.
        """
        if style is None:
            bar = f"{_REVERSE}|{_RESET_REVERSE}"
        else:
            bar = f"{style}│"
        return f"{query[:caret]}{bar}{query[caret:]}"
