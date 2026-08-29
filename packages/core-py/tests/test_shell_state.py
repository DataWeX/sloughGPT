"""Tests for ShellState — persistence, history, aliases, env vars.

Covers:
  - add_history with dedup
  - set_alias / unset_alias
  - set_env
  - to_dict summary
  - save/load round-trip via temp MogDB
  - history max length enforcement
"""

import json
import pytest
from pathlib import Path
from domains.shell.state import ShellState, _MAX_HISTORY, set_shell_state_db, reset_shell_state_db


@pytest.fixture(autouse=True)
def _temp_mogdb(tmp_path, monkeypatch):
    """Point the shell state module at a temporary MogDB for every test."""
    db_path = str(tmp_path / "test_shell_state")
    set_shell_state_db(db_path)
    yield
    reset_shell_state_db()


class TestShellState:
    def test_init_defaults(self):
        s = ShellState()
        assert s.history == []
        assert s.aliases == {}
        assert s.env == {}
        assert s.first_run is True

    def test_add_history(self):
        s = ShellState()
        s.add_history("ls -la")
        assert s.history == ["ls -la"]

    def test_add_history_dedup(self):
        s = ShellState()
        s.add_history("ls")
        s.add_history("ls")
        assert s.history == ["ls"]

    def test_add_history_different(self):
        s = ShellState()
        s.add_history("ls")
        s.add_history("pwd")
        assert s.history == ["ls", "pwd"]

    def test_add_history_empty(self):
        s = ShellState()
        s.add_history("")
        s.add_history(None)
        assert s.history == []

    def test_set_alias(self):
        s = ShellState()
        s.set_alias("ll", "ls -la")
        assert s.aliases["ll"] == "ls -la"

    def test_unset_alias(self):
        s = ShellState()
        s.set_alias("ll", "ls -la")
        assert s.unset_alias("ll") is True
        assert "ll" not in s.aliases

    def test_unset_alias_missing(self):
        s = ShellState()
        assert s.unset_alias("nonexistent") is False

    def test_set_env(self):
        s = ShellState()
        s.set_env("EDITOR", "vim")
        assert s.env["EDITOR"] == "vim"

    def test_to_dict(self):
        s = ShellState()
        s.add_history("test")
        s.set_alias("q", "exit")
        d = s.to_dict()
        assert d["history_count"] == 1
        assert d["aliases"] == {"q": "exit"}
        assert d["first_run"] is True

    def test_save_load_roundtrip(self):
        s = ShellState()
        s.add_history("cmd1")
        s.set_alias("a", "b")
        s.set_env("X", "1")
        s.first_run = False
        s.save()

        s2 = ShellState()
        assert s2.history == ["cmd1"]
        assert s2.aliases == {"a": "b"}
        assert s2.env == {"X": "1"}
        assert s2.first_run is False

    def test_load_nonexistent(self):
        s = ShellState()
        assert s.history == []

    def test_history_max_length(self):
        s = ShellState()
        for i in range(_MAX_HISTORY + 50):
            s.add_history(f"cmd{i}")
        s.save()
        s2 = ShellState()
        assert len(s2.history) == _MAX_HISTORY
        assert s2.history[-1] == f"cmd{_MAX_HISTORY + 49}"

    def test_save_overwrites_existing(self):
        s = ShellState()
        s.add_history("first")
        s.add_history("second")
        s.save()
        s2 = ShellState()
        assert s2.history == ["first", "second"]
