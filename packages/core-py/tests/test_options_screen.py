"""Tests for the options-screen family — form-over-flags (RetroBIOS-style).

The model (`options_screen.OptionsScreen`) is pure state: rows, cursor,
modal substate, command-line assembly.  It is tested directly, no
terminal involved.  The driver (inlined in `interactive.InteractivePrompt`
on main) is tested through its scripted-key seam: `_options_screen_raw`
accepts a `read_key` callable and an `fd`, so a fixed key sequence
replays against the real loop while stdout is captured and the fd is
/dev/null (which `_RawTerminal` tolerates as a non-tty).
"""

from __future__ import annotations

import contextlib as _contextlib
import io as _io_mod
import os as _os
import re as _re

import pytest

from domain.shell._internal.interactive import (
    _KEY_BACKSPACE,
    _KEY_CHAR,
    _KEY_DOWN,
    _KEY_ENTER,
    _KEY_ESC,
    _KEY_LEFT,
    _KEY_RIGHT,
    InteractivePrompt,
    _RawKey,
)
from domain.shell._internal.io import MemoryIO
from domain.shell._internal.options_screen import (
    MODE_FORM,
    MODE_INPUT,
    MODE_PICKER,
    OptionRow,
    OptionsScreen,
    show_command_line_action,
)

# ── spec fixture: the reference screen from the design screenshot ───


def _specs() -> list[dict]:
    return [
        {
            "label": "Preferred regions",
            "flag": "--region",
            "type": "choice",
            "options": ["any", "us", "eu"],
            "value": "any",
        },
        {
            "label": "Game data",
            "flag": "--no-game-data",
            "type": "bool",
            "value": False,
            "on_label": "removed",
            "off_label": "kept",
        },
        {
            "label": "Cores",
            "flag": "--core",
            "type": "choice",
            "options": ["all", "snes", "nes"],
            "value": "all",
        },
        {
            "label": "Parallel downloads",
            "flag": "--jobs",
            "type": "choice",
            "options": ["1", "4", "8"],
            "value": "8",
        },
        {"label": "BIOS folder", "flag": "--dest", "type": "text", "value": "~/retrodeck"},
        {
            "label": "Copy from",
            "flag": "--from",
            "type": "choice",
            "options": ["none", "local"],
            "value": "none",
            "omit": ["none"],
        },
        {
            "label": "Preview the file list",
            "type": "action",
            "action": lambda s: setattr(s, "status", "preview: 3 files"),
        },
        {"label": "Show the command line", "type": "action", "action": show_command_line_action},
    ]


# ── OptionRow / spec normalization ─────────────────────────────────


class TestRowSpec:
    def test_key_defaults_to_flag_without_dashes(self):
        row = OptionRow.from_spec({"label": "X", "flag": "--jobs", "value": "1"})
        assert row.key == "jobs"

    def test_key_falls_back_to_label_when_no_flag(self):
        row = OptionRow.from_spec({"label": "Go", "type": "action"})
        assert row.key == "Go"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unknown row type"):
            OptionRow.from_spec({"label": "X", "type": "wat"})

    def test_omit_accepts_bare_string(self):
        row = OptionRow.from_spec({"label": "X", "flag": "--from", "value": "none", "omit": "none"})
        assert row.omit == ("none",)

    def test_bool_display_uses_labels(self):
        row = OptionRow.from_spec(
            {
                "label": "G",
                "value": True,
                "on_label": "removed",
                "off_label": "kept",
                "type": "bool",
            }
        )
        assert row.display_value() == "removed"
        row.value = False
        assert row.display_value() == "kept"


# ── command-line assembly ──────────────────────────────────────────


class TestCommandLine:
    def test_choice_emits_flag_and_value(self):
        s = OptionsScreen(
            [
                {
                    "label": "J",
                    "flag": "--jobs",
                    "type": "choice",
                    "options": ["1", "8"],
                    "value": "8",
                }
            ]
        )
        assert s.command_line() == ["--jobs", "8"]

    def test_omit_value_suppresses_flag(self):
        s = OptionsScreen(_specs())
        assert "--from" not in s.command_line()

    def test_bool_emits_only_when_active(self):
        s = OptionsScreen(
            [{"label": "G", "flag": "--no-game-data", "type": "bool", "value": False}]
        )
        assert s.command_line() == []
        s.rows[0].value = True
        assert s.command_line() == ["--no-game-data"]

    def test_bool_invert_flips_emission(self):
        s = OptionsScreen(
            [{"label": "G", "flag": "--color", "type": "bool", "value": True, "invert": True}]
        )
        assert s.command_line() == []
        s.rows[0].value = False
        assert s.command_line() == ["--color"]

    def test_actions_never_emit(self):
        s = OptionsScreen(_specs())
        assert all(p != "preview" for p in s.command_line())

    def test_full_reference_screen_command_line(self):
        s = OptionsScreen(_specs())
        assert s.command_line() == [
            "--region",
            "any",
            "--core",
            "all",
            "--jobs",
            "8",
            "--dest",
            "~/retrodeck",
        ]

    def test_show_command_line_action_sets_status(self):
        s = OptionsScreen(_specs())
        show_command_line_action(s)
        assert s.status == "--region any --core all --jobs 8 --dest ~/retrodeck"

    def test_empty_status_reads_as_no_flags(self):
        s = OptionsScreen([{"label": "A", "type": "action", "action": show_command_line_action}])
        show_command_line_action(s)
        assert s.status == "(no flags set)"

    def test_values_dict_keys_and_actions_excluded(self):
        s = OptionsScreen(_specs())
        vals = s.values()
        assert "region" in vals and "dest" in vals
        assert "Preview the file list" not in vals and "Show the command line" not in vals
        assert vals["region"] == "any" and vals["no-game-data"] == "kept"


# ── navigation and modality ────────────────────────────────────────


class TestFormNavigation:
    def test_move_wraps_both_directions(self):
        s = OptionsScreen(_specs())
        s.move(1)
        assert s.cursor == 1
        s.move(-1)
        assert s.cursor == 0
        s.move(-1)
        assert s.cursor == len(s.rows) - 1

    def test_move_on_empty_screen_is_noop(self):
        s = OptionsScreen([])
        s.move(1)  # must not raise
        assert s.current() is None

    def test_cycle_choice_wraps(self):
        s = OptionsScreen(_specs())
        s.cycle(1)
        assert s.rows[0].value == "us"
        s.cycle(-1)
        assert s.rows[0].value == "any"
        s.cycle(-1)
        assert s.rows[0].value == "eu"

    def test_cycle_bool_flips(self):
        s = OptionsScreen(_specs())
        s.move(1)  # -> Game data
        s.cycle(1)
        assert s.rows[1].value is True
        assert s.values()["no-game-data"] == "removed"

    def test_cycle_text_row_is_noop(self):
        s = OptionsScreen(_specs())
        before = s.rows[4].value
        s.move(4)
        s.cycle(1)
        assert s.rows[4].value == before

    def test_cycle_out_of_options_value_starts_at_zero(self):
        s = OptionsScreen(
            [
                {
                    "label": "J",
                    "flag": "--jobs",
                    "type": "choice",
                    "options": ["1", "4", "8"],
                    "value": "999",
                }
            ]
        )
        s.cycle(1)
        assert s.rows[0].value == "4"


class TestPicker:
    def test_open_choice_enters_picker_at_current_value(self):
        s = OptionsScreen(_specs())
        s.cycle(1)  # region -> us
        s.open()
        assert s.mode == MODE_PICKER
        assert s.picker_index == 1

    def test_picker_select_sets_value_and_returns_to_form(self):
        s = OptionsScreen(_specs())
        s.open()
        s.picker_move(1)
        s.picker_select()
        assert s.mode == MODE_FORM
        assert s.rows[0].value == "us"

    def test_picker_move_wraps(self):
        s = OptionsScreen(_specs())
        s.open()
        s.picker_move(-1)
        assert s.picker_index == len(s.rows[0].options) - 1

    def test_picker_esc_back_unwinds_one_level_only(self):
        s = OptionsScreen(_specs())
        s.open()
        s.back()
        assert s.mode == MODE_FORM and not s.done
        assert s.rows[0].value == "any"  # selection discarded

    def test_form_esc_submits(self):
        s = OptionsScreen(_specs())
        s.back()
        assert s.done and not s.aborted

    def test_abort_sets_both_flags(self):
        s = OptionsScreen(_specs())
        s.abort()
        assert s.done and s.aborted


class TestInput:
    def test_open_text_enters_input_prefilled_at_end(self):
        s = OptionsScreen(_specs())
        s.move(4)
        s.open()
        assert s.mode == MODE_INPUT
        assert s.input_buf == "~/retrodeck"
        assert s.input_caret == len("~/retrodeck")

    def test_commit_sets_value_and_returns_to_form(self):
        s = OptionsScreen(_specs())
        s.move(4)
        s.open()
        s.set_input("/opt/bios", 9)
        s.input_commit()
        assert s.mode == MODE_FORM
        assert s.rows[4].value == "/opt/bios"

    def test_esc_back_discards_edit(self):
        s = OptionsScreen(_specs())
        s.move(4)
        s.open()
        s.set_input("garbage", 7)
        s.back()
        assert s.mode == MODE_FORM and not s.done
        assert s.rows[4].value == "~/retrodeck"

    def test_set_input_clamps_caret(self):
        s = OptionsScreen(_specs())
        s.move(4)
        s.open()
        s.set_input("ab", 99)
        assert s.input_caret == 2
        s.set_input("ab", -5)
        assert s.input_caret == 0


class TestActions:
    def test_open_action_runs_callback(self):
        s = OptionsScreen(_specs())
        s.move(6)
        s.open()
        assert s.status == "preview: 3 files"
        assert s.mode == MODE_FORM and not s.done

    def test_action_without_callback_is_noop(self):
        s = OptionsScreen([{"label": "A", "type": "action"}])
        s.open()  # must not raise
        assert not s.done

    def test_open_bool_toggles(self):
        s = OptionsScreen(_specs())
        s.move(1)
        s.open()
        assert s.rows[1].value is True


# ── render ─────────────────────────────────────────────────────────


class TestRender:
    def _texts(self, s: OptionsScreen) -> list[str]:
        return [t for t, _ in s.render()]

    def test_rows_named_by_flag_on_the_right(self):
        text = "\n".join(self._texts(OptionsScreen(_specs())))
        assert "Preferred regions: any" in text
        # flag lands right of the label, on the same line
        for line in text.splitlines():
            if "Preferred regions" in line:
                assert line.rstrip().endswith("--region")
                break
        else:
            pytest.fail("no regions row rendered")

    def test_cursor_marker_on_selected_row(self):
        # render(): title, subtitle, blank, then the rows.
        lines = self._texts(OptionsScreen(_specs()))
        assert lines[3].startswith("> ")
        assert lines[4].startswith("  ")

    def test_action_row_has_no_value(self):
        text = "\n".join(self._texts(OptionsScreen(_specs())))
        assert "Show the command line\n" in text or text.endswith("Show the command line")

    def test_picker_expands_under_selected_row(self):
        s = OptionsScreen(_specs())
        s.open()
        lines = self._texts(s)
        region_idx = next(i for i, t in enumerate(lines) if "Preferred regions" in t)
        picker_block = lines[region_idx + 1 : region_idx + 1 + len(s.rows[0].options)]
        assert [t.strip().lstrip(">").strip() for t in picker_block] == s.rows[0].options
        assert picker_block[0].startswith("    >")

    def test_input_line_carries_caret_glyph(self):
        s = OptionsScreen(_specs())
        s.move(4)
        s.open()
        s.set_input("ab", 1)
        text = "\n".join(self._texts(s))
        assert "a\u258cb" in text

    def test_help_overlay_replaces_footer(self):
        s = OptionsScreen(_specs())
        footer = self._texts(s)[-1]
        s.toggle_help()
        help_lines = self._texts(s)[-2:]
        assert footer == "Enter Open  Esc Back  Left/Right Set  ? Help"
        assert help_lines[0].startswith("Up/Down rows")
        assert "Esc" in help_lines[1]
        assert footer not in help_lines

    def test_empty_rows_render_placeholder(self):
        text = "\n".join(self._texts(OptionsScreen([])))
        assert "(no options)" in text

    def test_status_line_appears_above_footer(self):
        s = OptionsScreen(_specs())
        s.status = "hello"
        lines = self._texts(s)
        assert "hello" in lines[-3]

    def test_narrow_terminal_keeps_flag(self):
        s = OptionsScreen(_specs())
        line = next(t for t, _ in s.render(cols=30) if "Preferred regions" in t)
        assert "--region" in line


# ── driver: scripted keys over the real prompt object ──────────────


def _test_fd() -> int:
    """A real fd that is not a tty: `_RawTerminal` swallows the resulting
    termios.error, so the loop runs headless without patching internals."""
    return _os.open(_os.devnull, _os.O_RDONLY)


def _chars(text: str) -> list[_RawKey]:
    return [_RawKey(_KEY_CHAR, c) for c in text]


_Ansi = _re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _plain(raw: str) -> str:
    """Escape-stripped capture — assertions read text, not markup."""
    return _Ansi.sub("", raw)


def _drive(rows: list[dict], keys: list, message: str = "Options") -> tuple[dict, str]:
    """Run the raw driver over *keys*, returning (values, stdout)."""
    pending = list(keys)

    def read_key() -> _RawKey:
        if not pending:
            raise AssertionError("key script exhausted before the prompt returned")
        return pending.pop(0)

    p = InteractivePrompt(MemoryIO())
    fd = _test_fd()
    buf = _io_mod.StringIO()
    try:
        with _contextlib.redirect_stdout(buf):
            values = p._options_screen_raw(OptionsScreen(rows, message), read_key=read_key, fd=fd)
    finally:
        _os.close(fd)
    return values, buf.getvalue()


class TestDriverFallback:
    def test_non_tty_returns_defaults(self):
        io = MemoryIO()
        p = InteractivePrompt(io)
        p._is_tty = False
        values = p.options_screen(_specs())
        assert values["region"] == "any"
        assert values["dest"] == "~/retrodeck"
        assert values["no-game-data"] == "kept"


class TestDriverLoop:
    def test_cycle_with_right_arrow_reaches_result(self):
        values, _ = _drive(_specs(), [_RawKey(_KEY_RIGHT), _RawKey(_KEY_ESC)])
        assert values["region"] == "us"

    def test_picker_roundtrip_down_enter(self):
        values, _ = _drive(
            _specs(),
            [_RawKey(_KEY_ENTER), _RawKey(_KEY_DOWN), _RawKey(_KEY_ENTER), _RawKey(_KEY_ESC)],
        )
        assert values["region"] == "us"

    def test_picker_esc_discards(self):
        values, _ = _drive(
            _specs(),
            [_RawKey(_KEY_ENTER), _RawKey(_KEY_DOWN), _RawKey(_KEY_ESC), _RawKey(_KEY_ESC)],
        )
        assert values["region"] == "any"

    def test_text_edit_commit_via_driver(self):
        keys = [
            *[_RawKey(_KEY_DOWN)] * 4,  # -> BIOS folder (row 5)
            _RawKey(_KEY_ENTER),  # open input
            *_chars("X"),  # appended at end
            _RawKey(_KEY_ENTER),  # commit
            _RawKey(_KEY_ESC),  # submit form
        ]
        values, _ = _drive(_specs(), keys)
        assert values["dest"] == "~/retrodeckX"

    def test_text_edit_esc_discards_via_driver(self):
        keys = [
            *[_RawKey(_KEY_DOWN)] * 4,
            _RawKey(_KEY_ENTER),
            *_chars("junk"),
            _RawKey(_KEY_ESC),  # cancel edit
            _RawKey(_KEY_ESC),  # submit form
        ]
        values, _ = _drive(_specs(), keys)
        assert values["dest"] == "~/retrodeck"

    def test_backspace_edits_inside_input(self):
        keys = [
            *[_RawKey(_KEY_DOWN)] * 4,
            _RawKey(_KEY_ENTER),
            _RawKey(_KEY_BACKSPACE),  # ~/retrodec
            _RawKey(_KEY_ENTER),
            _RawKey(_KEY_ESC),
        ]
        values, _ = _drive(_specs(), keys)
        assert values["dest"] == "~/retrodec"

    def test_action_row_runs_and_stays_open(self):
        keys = [
            *[_RawKey(_KEY_DOWN)] * 6,  # -> Preview the file list
            _RawKey(_KEY_ENTER),  # run preview
            *[_RawKey(_KEY_DOWN)],  # -> Show the command line
            _RawKey(_KEY_ENTER),  # run it
            _RawKey(_KEY_ESC),  # submit
        ]
        values, out = _drive(_specs(), keys)
        # both actions fired; the screen never left for a submenu
        assert "--region any" in out
        assert "preview: 3 files" in out
        assert values["region"] == "any"

    def test_help_toggle_renders_overlay(self):
        _, out = _drive(_specs(), [_RawKey(_KEY_CHAR, "?"), _RawKey(_KEY_ESC)])
        assert "Up/Down rows" in out

    def test_left_arrow_cycles_backward(self):
        values, _ = _drive(_specs(), [_RawKey(_KEY_LEFT), _RawKey(_KEY_ESC)])
        assert values["region"] == "eu"

    def test_render_frame_names_flags_on_right(self):
        _, out = _drive(_specs(), [_RawKey(_KEY_ESC)])
        for line in _plain(out).splitlines():
            if "Preferred regions" in line:
                assert line.rstrip().endswith("--region")
                return
        pytest.fail("regions row never rendered")
