"""OptionsScreen prompt family. — _OptionsPrompts mixin.

The driver for `options_screen`: a RetroBIOS-style options form where
every row is one option of a command line, named by its flag on the
right.  All state lives in `options_screen.OptionsScreen`; this module
only maps raw keys onto model moves and model style tags onto ANSI.

Modal key map (the one invariant: Esc unwinds exactly one level):

  form    Up/Down rows, Left/Right set, Enter open/edit/run,
          Esc submit, Ctrl+C abort, ? help
  picker  Up/Down (also Left/Right) choose, Enter accept, Esc back
  input   type/edit via _filter_key, Enter commit, Esc back

Fits the family shape every other interactive prompt follows: a public
method with a non-TTY fallback and a `_raw` loop over `self._session`.
"""

from __future__ import annotations

import sys
import termios

from domain.shell._internal.interactive_base import InteractiveBase
from domain.shell._internal.interactive_term import (
    _BOLD,
    _CYAN,
    _DIM,
    _ERASE_LINE,
    _HIDE_CURSOR,
    _KEY_CHAR,
    _KEY_CTRL_C,
    _KEY_DOWN,
    _KEY_ENTER,
    _KEY_ESC,
    _KEY_LEFT,
    _KEY_RIGHT,
    _KEY_UP,
    _RESET,
    _RESET_REVERSE,
    _REVERSE,
    _SHOW_CURSOR,
    _YELLOW,
    _rewind,
)
from domain.shell._internal.options_screen import (
    MODE_INPUT,
    MODE_PICKER,
    OptionsScreen,
)

# Model style tag -> ANSI prefix.  The model stays free of escapes; this
# table is the whole translation, so restyling is a one-place change.
_STYLES = {
    "title": _BOLD,
    "sub": _DIM,
    "row": "",
    "rowsel": _REVERSE + _CYAN,
    "pick": _DIM,
    "picksel": _CYAN,
    "input": _REVERSE,
    "status": _YELLOW,
    "foot": _DIM,
    "help": _DIM,
    "empty": "",
}


class _OptionsPrompts(InteractiveBase):
    """The options-screen form-over-flags family."""

    def options_screen(
        self, rows: list[dict], message: str = "Options"
    ) -> dict[str, str]:
        """Interactive options screen; returns {key: value}.

        Each row is one option of a command line, named by its flag on
        the right of the screen.  See `options_screen.OptionsScreen` for
        the row spec.  Non-TTY returns each row's default value.
        """
        if not self._is_tty:
            screen = OptionsScreen(rows, message)
            return screen.values()
        return self._options_screen_raw(OptionsScreen(rows, message))

    def _options_screen_raw(self, screen: OptionsScreen) -> dict[str, str]:
        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            out = []
            for text, style in screen.render():
                prefix = _STYLES.get(style, "")
                # reverse must be explicitly released or it bleeds into
                # the next repaint's plain rows.
                close = _RESET_REVERSE if style in ("rowsel", "input") else _RESET
                out.append(f"{_ERASE_LINE}\r  {prefix}{text}{close}")
            return "\n".join(out) + f"\n{_HIDE_CURSOR}"

        def _clear() -> None:
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while not screen.done:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    self._dispatch(screen, key)
        except (termios.error, OSError):
            return screen.values()
        _clear()
        return screen.values()

    def _dispatch(self, screen: OptionsScreen, key) -> None:
        """One key -> one model move.  Split by mode so no branch has to
        remember which keys belong to which level."""
        if key.kind == _KEY_CTRL_C:
            screen.abort()
            return
        if screen.mode == MODE_INPUT:
            if key.kind == _KEY_ENTER:
                screen.input_commit()
            elif key.kind == _KEY_ESC:
                screen.back()
            else:
                # Motion, rubout, word jumps — the same editing every
                # other field in the family gets, unchanged.
                buf, caret, _ = self._filter_key(screen.input_buf, screen.input_caret, key)
                screen.set_input(buf, caret)
            return
        if screen.mode == MODE_PICKER:
            if key.kind == _KEY_ENTER:
                screen.picker_select()
            elif key.kind == _KEY_ESC:
                screen.back()
            elif key.matches(_KEY_UP, _KEY_LEFT):
                screen.picker_move(-1)
            elif key.matches(_KEY_DOWN, _KEY_RIGHT):
                screen.picker_move(1)
            return
        # MODE_FORM
        if key.kind == _KEY_ESC:
            screen.back()
        elif key.kind == _KEY_ENTER:
            screen.open()
        elif key.matches(_KEY_UP):
            screen.move(-1)
        elif key.matches(_KEY_DOWN):
            screen.move(1)
        elif key.matches(_KEY_LEFT):
            screen.cycle(-1)
        elif key.matches(_KEY_RIGHT):
            screen.cycle(1)
        elif key.matches(_KEY_CHAR, char="?"):
            screen.toggle_help()
