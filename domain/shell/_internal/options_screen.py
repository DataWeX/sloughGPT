"""OptionsScreen — the pure model behind the options-screen prompt.

A RetroBIOS-style "Options" screen: every row is one option of a command
line, named by its flag on the right of the screen.  Arrow keys navigate
rows, Left/Right cycle a choice's value, Enter opens an inline picker
(choosing) or an edit field (typing), and action rows run a callback.
Esc unwinds exactly one level: picker -> form, input -> form, form ->
submit.  "?" toggles the help overlay.

Zero terminal imports.  The model owns all state and produces styled
text lines; the driver (`interactive_options`) only maps keys to model
moves and styles to ANSI.

Row spec (a dict, like the rest of the family):
    label      left-hand text, e.g. "Preferred regions"
    flag       right-hand flag name, e.g. "--region" (None for action rows)
    type       "choice" | "text" | "bool" | "action"     (default "text")
    options    list[str] — required for "choice"
    value      current value (str for choice/text, bool for bool)
    key        values-dict key; defaults to flag-without-dashes, else label
    omit       values (str-compared) that suppress the flag in the command
               line — the "Copy from: none" case, where none means no --from
    invert     bool rows only: emit the flag when the value is False
    on_label / off_label   bool display strings (default "on"/"off")
    action     "action" rows only: Callable[[OptionsScreen], None]

Command-line assembly: choice/text emit `[flag, value]` unless the value
is empty or listed in `omit`; bool emits `[flag]` when the effective
truth differs from `invert`; actions never emit.  "Show the command
line" is the built-in action factory `show_command_line_action`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

_CARET = "\u258c"  # left half block — the caret glyph, pure text

MODE_FORM = "form"
MODE_PICKER = "picker"
MODE_INPUT = "input"

_KINDS = ("choice", "text", "bool", "action")

_FOOTER = "Enter Open  Esc Back  Left/Right Set  ? Help"
_SUBTITLE = "Each row is an option of the command line, named on the right."


@dataclass
class OptionRow:
    """One normalized row.  Built from the spec dict; see module docstring."""

    label: str
    flag: str | None = None
    kind: str = "text"
    options: list[str] = field(default_factory=list)
    value: str | bool = ""
    key: str = ""
    omit: tuple[str, ...] = ()
    invert: bool = False
    on_label: str = "on"
    off_label: str = "off"
    action: Callable[[OptionsScreen], None] | None = None

    @classmethod
    def from_spec(cls, spec: dict) -> OptionRow:
        kind = str(spec.get("type", "text"))
        if kind not in _KINDS:
            raise ValueError(f"unknown row type {kind!r} (expected one of {_KINDS})")
        flag = spec.get("flag")
        label = str(spec.get("label", ""))
        key = str(spec.get("key") or (str(flag).lstrip("-") if flag else label))
        omit = spec.get("omit") or ()
        if isinstance(omit, str):
            omit = (omit,)
        return cls(
            label=label,
            flag=str(flag) if flag else None,
            kind=kind,
            options=[str(o) for o in spec.get("options") or []],
            value=spec.get("value", ""),
            key=key,
            omit=tuple(str(o) for o in omit),
            invert=bool(spec.get("invert", False)),
            on_label=str(spec.get("on_label", "on")),
            off_label=str(spec.get("off_label", "off")),
            action=spec.get("action"),
        )

    @property
    def is_action(self) -> bool:
        return self.kind == "action"

    def display_value(self) -> str:
        """The value as the screen shows it."""
        if self.kind == "bool":
            return self.on_label if bool(self.value) else self.off_label
        return str(self.value)

    def argv(self) -> list[str]:
        """This row's contribution to the command line."""
        if self.is_action or not self.flag:
            return []
        if self.kind == "bool":
            return [self.flag] if bool(self.value) != self.invert else []
        text = str(self.value)
        if text == "" or text in self.omit:
            return []
        return [self.flag, text]


class OptionsScreen:
    """State machine over rows: form <-> picker, form <-> input.

    `done` flips exactly once, when the form level is left for good:
    Esc submits, Ctrl+C aborts.  Every other transition is reversible,
    which is the invariant the whole screen rests on.
    """

    def __init__(self, rows: list[dict] | list[OptionRow], message: str = "Options") -> None:
        self.message = message
        self.rows: list[OptionRow] = [
            r if isinstance(r, OptionRow) else OptionRow.from_spec(r) for r in rows
        ]
        self.cursor = 0
        self.mode = MODE_FORM
        self.picker_index = 0
        self.input_buf = ""
        self.input_caret = 0
        self.help = False
        self.status = ""
        self.done = False
        self.aborted = False

    # ── form-level moves ────────────────────────────────────────────

    def move(self, delta: int) -> None:
        """Move the row cursor, wrapping.  No-op on an empty screen."""
        if not self.rows:
            return
        self.cursor = (self.cursor + delta) % len(self.rows)

    def current(self) -> OptionRow | None:
        return self.rows[self.cursor] if self.rows else None

    def cycle(self, delta: int) -> None:
        """Left/Right on the form: set a choice's value, flip a bool.

        Text and action rows ignore it — there is nothing to step
        through, and silently editing a free-text value from an arrow
        key would eat characters the user meant for the edit field.
        """
        row = self.current()
        if row is None:
            return
        if row.kind == "choice" and row.options:
            idx = row.options.index(str(row.value)) if str(row.value) in row.options else 0
            row.value = row.options[(idx + delta) % len(row.options)]
        elif row.kind == "bool":
            row.value = not bool(row.value)

    def open(self) -> None:
        """Enter: a picker for choices, an edit field for text, flip for
        bools, run for actions."""
        row = self.current()
        if row is None:
            return
        if row.kind == "choice" and row.options:
            self.picker_index = (
                row.options.index(str(row.value)) if str(row.value) in row.options else 0
            )
            self.mode = MODE_PICKER
        elif row.kind == "text":
            self.input_buf = str(row.value)
            self.input_caret = len(self.input_buf)
            self.mode = MODE_INPUT
        elif row.kind == "bool":
            row.value = not bool(row.value)
        elif row.is_action and row.action is not None:
            row.action(self)

    def back(self) -> None:
        """Esc: one level, never two.  At the form it submits."""
        if self.mode == MODE_INPUT:
            self.input_buf, self.input_caret, self.mode = "", 0, MODE_FORM
        elif self.mode == MODE_PICKER:
            self.mode = MODE_FORM
        else:
            self.done = True

    def abort(self) -> None:
        self.aborted = True
        self.done = True

    def toggle_help(self) -> None:
        self.help = not self.help

    # ── picker moves ────────────────────────────────────────────────

    def picker_move(self, delta: int) -> None:
        row = self.current()
        if self.mode != MODE_PICKER or row is None or not row.options:
            return
        self.picker_index = (self.picker_index + delta) % len(row.options)

    def picker_select(self) -> None:
        row = self.current()
        if self.mode != MODE_PICKER or row is None or not row.options:
            return
        row.value = row.options[self.picker_index]
        self.mode = MODE_FORM

    # ── input moves (editing itself is the driver's `_filter_key`) ──

    def set_input(self, buf: str, caret: int) -> None:
        self.input_buf = buf
        self.input_caret = max(0, min(caret, len(buf)))

    def input_commit(self) -> None:
        row = self.current()
        if self.mode != MODE_INPUT or row is None:
            return
        row.value = self.input_buf
        self.input_buf, self.input_caret, self.mode = "", 0, MODE_FORM

    # ── results ─────────────────────────────────────────────────────

    def command_line(self) -> list[str]:
        parts: list[str] = []
        for row in self.rows:
            parts.extend(row.argv())
        return parts

    def command_line_str(self) -> str:
        import re
        import shlex

        # shlex.quote escapes `~`, which would break the one thing this
        # line is for: paste-back into a shell with the tilde still
        # expanding.  Tokens that are shell-safe — including a leading
        # `~/` — stay bare; everything else gets real quoting.
        safe = re.compile(r"^[A-Za-z0-9_@%+=:,./~^-]+$")
        return " ".join(p if safe.match(p) else shlex.quote(p) for p in self.command_line())

    def values(self) -> dict[str, str]:
        """Keyed by the row's `key` (flag name by default)."""
        return {row.key: row.display_value() for row in self.rows if not row.is_action}

    # ── render (pure text + style tags; the driver owns the ANSI) ───

    def render(self, cols: int = 80) -> list[tuple[str, str]]:
        """One (text, style) per screen line.  Styles: title sub row
        rowsel pick picksel input status foot help empty."""
        lines: list[tuple[str, str]] = [(self.message, "title"), (_SUBTITLE, "sub"), ("", "empty")]
        if not self.rows:
            lines.append(("(no options)", "sub"))
        for i, row in enumerate(self.rows):
            selected = i == self.cursor
            left = f"{'>' if selected else ' '} {row.label}"
            if not row.is_action:
                left += f": {row.display_value()}"
            line = self._place_flag(left, row.flag, cols)
            lines.append((line, "rowsel" if selected else "row"))
            # The picker opens under its own row, so "where am I" never
            # leaves the screen while a submenu is up.
            if selected and self.mode == MODE_PICKER and row.options:
                for j, opt in enumerate(row.options):
                    mark = ">" if j == self.picker_index else " "
                    lines.append(
                        (f"    {mark} {opt}", "picksel" if j == self.picker_index else "pick")
                    )
            if selected and self.mode == MODE_INPUT:
                buf, caret = self.input_buf, self.input_caret
                lines.append((f"    {buf[:caret]}{_CARET}{buf[caret:]}", "input"))
        if self.status:
            lines.append((self.status, "status"))
        lines.append(("", "empty"))
        if self.help:
            lines.append(("Up/Down rows   Left/Right set   Enter open/edit/run", "help"))
            lines.append(("Esc back/submit   ? close help", "help"))
        else:
            lines.append((_FOOTER, "foot"))
        return lines

    @staticmethod
    def _place_flag(left: str, flag: str | None, cols: int) -> str:
        """Flag right-aligned to a common edge, like the reference screen."""
        if not flag:
            return left
        room = cols - len(flag) - 1
        if len(left) >= room:
            return f"{left} {flag}"  # narrow terminal: never drop the flag
        return f"{left.ljust(room)}{flag}"


def show_command_line_action(screen: OptionsScreen) -> None:
    """Built-in action for a "Show the command line" row."""
    screen.status = screen.command_line_str() or "(no flags set)"
