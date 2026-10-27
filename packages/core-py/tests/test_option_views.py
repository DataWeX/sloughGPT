"""Card D — the three option views, pinned against drift.

One descriptor per row (`{label, value, flag, setter}`) drives the screen
row, the command-line equivalent and the help text.  These tests pin the
three views *and* the invariant the card exists for: a registry row's
flag can never disagree with the command it claims to describe — the
docstring and the parser are checked independently, so drift in either
direction fails the build.

Views are pure data (`option_views.py`); the driver that renders them
(TTY loop or plain lines) is `console.run_options_screen`.
"""

from __future__ import annotations

import inspect
import os as _os
import re as _re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from domain.shell._internal.config import get_api_base
from domain.shell._internal.history import HistoryStore
from domain.shell._internal.io import MemoryIO, capture_cmd
from domain.shell._internal.option_views import (
    boot_rows,
    command_rows,
    option_commands,
    settings_rows,
)
from domain.shell._internal.options_screen import OptionsScreen


@pytest.fixture
def repl():
    from domain.shell._internal.init import reset_init_system
    from domain.shell._internal.repl import ShellREPL
    from domain.shell._internal.runtime import DaitRuntime
    from domain.shell._internal.state import reset_shell_state_db, set_shell_state_db

    reset_init_system()
    with tempfile.TemporaryDirectory() as tmp:
        st = Path(tmp) / "sloughgpt"
        st.mkdir(parents=True, exist_ok=True)
        set_shell_state_db(str(st / "shell_state_mogdb"))
        with (
            patch(
                "domain.shell._internal.runtime._probe_api",
                return_value={"available": False, "error": "mock"},
            ),
            patch(
                "domain.shell._internal.repl.ShellREPL._get_current_model", return_value=""
            ),
            patch(
                "domain.shell._internal.repl.ShellREPL._get_current_soul", return_value=""
            ),
            patch.object(ShellREPL, "_setup_readline"),
            patch(
                "domain.shell._internal.runtime.APIServerProcess.start",
                return_value={"ok": True, "message": "mocked"},
            ),
        ):
            rt = DaitRuntime()
            r = ShellREPL(rt)
            r._perms._granted.update(["tee", "xargs", "cp", "mv", "touch", "chmod"])
            r.io._tty = None
            r.io._uses_stdin = False
            r.console._interactive._is_tty = False
            yield r
            reset_init_system()
            reset_shell_state_db()


def _flags(rows: list[dict]) -> list[str]:
    return [str(r["flag"]) for r in rows]


def _run_tty(repl, fake_raw, fn, *args) -> str:
    """Run *fn* with the interactive side forced to TTY and the raw driver faked."""
    mem = MemoryIO()
    ip = repl.console._interactive
    old = (repl.io, repl.console._io, ip._io, ip._is_tty, ip._options_screen_raw)
    repl.io = mem
    repl.console._io = mem
    ip._io = mem
    ip._is_tty = True
    ip._options_screen_raw = fake_raw
    try:
        fn(*args)
    finally:
        (
            repl.io,
            repl.console._io,
            ip._io,
            ip._is_tty,
            ip._options_screen_raw,
        ) = old
    return mem.get_output()


# ── the anti-drift core: registry vs docstring vs parser ────────────


def _parser_source(fn) -> str:
    """The method's code, docstring removed — what actually runs."""
    return inspect.getsource(fn).split('"""', 2)[-1]


class TestCommandRegistry:
    def test_registry_lists_the_flag_complete_commands(self):
        assert option_commands() == ["ai", "logs"]

    @pytest.mark.parametrize("name", ["logs", "ai"])
    def test_registry_flags_match_the_command_docstring(self, repl, name):
        """The docstring names the flags; the registry must name exactly those."""
        fn = getattr(repl, f"_cmd_{name}")
        documented = set(_re.findall(r"--[a-z-]+", fn.__doc__ or ""))
        assert set(_flags(command_rows(name, repl))) == documented

    @pytest.mark.parametrize("name", ["logs", "ai"])
    def test_registry_flags_exist_in_the_parser(self, repl, name):
        """Every row's flag is really parsed — not just documented."""
        code = _parser_source(getattr(repl, f"_cmd_{name}"))
        for flag in _flags(command_rows(name, repl)):
            assert f'"{flag}"' in code or f"'{flag}'" in code, f"{flag} not parsed"

    @pytest.mark.parametrize("name", ["logs", "ai"])
    def test_docstring_and_parser_agree(self, repl, name):
        """The drift the card exists for: docstring vs parser, checked both ways."""
        fn = getattr(repl, f"_cmd_{name}")
        documented = set(_re.findall(r"--[a-z-]+", fn.__doc__ or ""))
        code = _parser_source(fn)
        missing = {f for f in documented if f'"{f}"' not in code and f"'{f}'" not in code}
        assert not missing, f"documented but never parsed: {missing}"

    def test_unknown_command_is_a_clear_error(self, repl):
        out = capture_cmd(repl, repl._cmd_options, "train")
        assert repl._last_exit_code == 1
        assert "train" in out and ("unknown" in out.lower() or "Unknown" in out)
        # and it tells you what it does know
        assert "logs" in out

    def test_missing_argument_prints_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_options, "")
        assert "Usage" in out
        assert "logs" in out

    def test_values_persist_between_opens(self, repl):
        """A row's setter remembers your choice — the projection has state."""
        rows = command_rows("logs", repl)
        rows[0]["value"] = "ERROR"
        OptionsScreen(rows, "logs").apply()
        assert command_rows("logs", repl)[0]["value"] == "ERROR"

    def test_logs_view_prints_plain_rows_and_equivalent_line(self, repl):
        out = capture_cmd(repl, repl._cmd_options, "logs")
        assert "Each row is an option of the command line" in out
        assert "--level" in out and "--follow" in out
        assert "Enter Open" in out and "? Help" in out
        # the CLI half, derived from the same rows — never hand-typed
        assert "Equivalent: logs" in out
        assert "--level" in out

    def test_equivalent_line_is_the_real_command(self, repl):
        rows = command_rows("logs", repl)
        rows[0]["value"] = "INFO"
        rows[3]["value"] = True  # --follow
        screen = OptionsScreen(rows, "logs")
        assert screen.command_line_str() == "--level INFO --lines 30 --follow"


# ── settings ─────────────────────────────────────────────────────────


class TestSettingsView:
    def test_rows_carry_the_real_keys(self, repl):
        rows = settings_rows(repl)
        assert _flags(rows) == ["--api-base", "--history", "--api"]
        assert [r["label"] for r in rows] == [
            "API base",
            "History size",
            "Auto-start API",
        ]
        by_flag = {r["flag"]: r for r in rows}
        assert by_flag["--api-base"]["value"] == get_api_base()
        assert by_flag["--history"]["value"] == str(repl._history_store.max_entries)
        assert by_flag["--api"]["value"] is True  # auto-start defaults on

    def test_apply_writes_real_state(self, repl, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://before:1")
        screen = OptionsScreen(settings_rows(repl), "Settings")
        screen.rows[0].value = "http://after:9999"
        screen.rows[1].value = "42"
        screen.rows[2].value = False
        applied = screen.apply()
        assert applied == ["api-base", "history", "api"]
        assert get_api_base() == "http://after:9999"
        assert repl._history_store.max_entries == 42

    def test_settings_prints_plain_rows_without_a_tty(self, repl):
        out = capture_cmd(repl, repl._cmd_settings, "")
        assert "Each row is an option of the command line" in out
        assert "API base" in out and "--api-base" in out
        assert "History size" in out and "--history" in out
        assert "Auto-start API" in out and "--api" in out
        assert "Enter Open" in out and "- ? Help" in out.replace("  ", " ")
        # nothing was edited, so nothing was written
        assert "applied" not in out

    def test_tty_submit_applies_the_screen(self, repl, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://before:1")

        def submit(screen, **kw):
            screen.rows[0].value = "http://tty:7"
            screen.done = True
            return screen.values()

        out = _run_tty(repl, submit, repl._cmd_settings, "")
        assert "applied" in out
        assert get_api_base() == "http://tty:7"

    def test_tty_abort_writes_nothing(self, repl, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://before:1")

        def abort(screen, **kw):
            screen.aborted = True
            screen.done = True
            return screen.values()

        out = _run_tty(repl, abort, repl._cmd_settings, "")
        assert "cancel" in out.lower()
        assert get_api_base() == "http://before:1"

    def test_settings_and_boot_agree_on_auto_start_api(self, repl):
        """The same pref backs both views — they cannot disagree."""
        s = {r["flag"]: r for r in settings_rows(repl)}["--api"]
        b = {r["flag"]: r for r in boot_rows(repl)}["--api"]
        assert s["value"] == b["value"]
        screen = OptionsScreen(settings_rows(repl), "Settings")
        screen.rows[2].value = False
        screen.apply()
        assert {r["flag"]: r for r in boot_rows(repl)}["--api"]["value"] is False

    def test_help_covers_the_new_commands(self, repl):
        assert "settings" in capture_cmd(repl, repl._cmd_help, "settings")
        assert "options" in capture_cmd(repl, repl._cmd_help, "options")


# ── boot ─────────────────────────────────────────────────────────────


class TestBootView:
    def test_boot_rows(self, repl):
        rows = boot_rows(repl)
        assert _flags(rows) == ["--api", "--model"]
        by_flag = {r["flag"]: r for r in rows}
        assert by_flag["--api"]["value"] is True
        assert by_flag["--model"]["value"] == _os.environ.get(
            "SLO_AUTOLOAD_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        )

    @pytest.fixture
    def _boot_api(self, repl):
        api = MagicMock()
        api.is_running = False
        api.start.return_value = {"ok": True, "message": "started"}
        with (
            patch.object(
                type(repl.os), "api", new_callable=PropertyMock, return_value=api
            ),
            patch.object(
                type(repl.os),
                "api_status",
                new_callable=PropertyMock,
                return_value={"available": False},
            ),
            patch.object(repl.os, "boot", return_value=("log", {"available": False})),
        ):
            yield api

    def test_bare_non_tty_output_is_unchanged(self, repl, _boot_api):
        """Plain CLI: no view, same auto-start line as before card D."""
        out = capture_cmd(repl, repl._cmd_boot, "")
        assert "Each row is an option" not in out
        assert "Auto-starting API server" in out
        assert "auto-start failed" not in out
        assert repl._last_exit_code == 0

    def test_no_api_skips_autostart(self, repl, _boot_api):
        out = capture_cmd(repl, repl._cmd_boot, "--no-api")
        assert _boot_api.start.call_count == 0
        assert "Each row is an option" not in out
        assert "Skipping API auto-start" in out

    def test_api_forces_autostart_despite_pref_off(self, repl, _boot_api):
        settings = settings_rows(repl)
        settings[2]["value"] = False
        OptionsScreen(settings, "Settings").apply()
        capture_cmd(repl, repl._cmd_boot, "--api")
        assert _boot_api.start.call_count == 1

    def test_bare_tty_opens_the_view_then_boots(self, repl, _boot_api):
        seen: dict = {}

        def submit(screen, **kw):
            seen["message"] = screen.message
            screen.done = True
            return screen.values()

        out = _run_tty(repl, submit, repl._cmd_boot, "")
        assert seen["message"] == "Boot"
        assert "applied" in out
        assert repl.os.boot.call_count == 1
        assert repl._running is True

    def test_bare_tty_abort_cancels_the_boot(self, repl, _boot_api):
        def abort(screen, **kw):
            screen.aborted = True
            screen.done = True
            return screen.values()

        out = _run_tty(repl, abort, repl._cmd_boot, "")
        assert "cancel" in out.lower()
        assert repl.os.boot.call_count == 0
        assert repl._running is False

    def test_flags_suppress_the_view_even_on_a_tty(self, repl, _boot_api):
        def must_not_run(screen, **kw):  # pragma: no cover - failure path
            raise AssertionError("the view must not open when flags are given")

        out = _run_tty(repl, must_not_run, repl._cmd_boot, "--no-api")
        assert "Each row is an option" not in out
        assert _boot_api.start.call_count == 0


# ── history cap (the settings row that makes dead config real) ───────


class TestHistoryCap:
    def test_store_trims_to_max_entries(self):
        store = HistoryStore(max_entries=3)
        for i in range(5):
            store.append(f"cmd{i}")
        assert store.entries == ["cmd2", "cmd3", "cmd4"]
        assert store.max_entries == 3

    def test_lowering_the_cap_trims_immediately(self):
        store = HistoryStore(max_entries=10)
        for i in range(10):
            store.append(f"cmd{i}")
        store.max_entries = 4
        assert len(store.entries) == 4

    def test_settings_history_row_takes_effect(self, repl):
        store = repl._history_store
        store.entries.clear()
        rows = settings_rows(repl)
        rows[1]["value"] = "2"
        OptionsScreen(rows, "Settings").apply()
        for i in range(5):
            store.append(f"x{i}")
        assert store.entries == ["x3", "x4"]
