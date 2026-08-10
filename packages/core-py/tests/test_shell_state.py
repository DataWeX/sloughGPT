"""Tests for ShellState — persistence, history, aliases, env vars.

Covers:
  - add_history with dedup
  - set_alias / unset_alias
  - set_env
  - to_dict summary
  - save/load round-trip via tmp_path
  - history max length enforcement
"""

import json
import pytest
from pathlib import Path
from domains.shell.state import ShellState, _MAX_HISTORY


class TestShellState:
    def test_init_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.shell.state._STATE_FILE", tmp_path / "state.json")
        s = ShellState()
        assert s.history == []
        assert s.aliases == {}
        assert s.env == {}
        assert s.first_run is True

    def test_add_history(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.shell.state._STATE_FILE", tmp_path / "state.json")
        s = ShellState()
        s.add_history("ls -la")
        assert s.history == ["ls -la"]

    def test_add_history_dedup(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.shell.state._STATE_FILE", tmp_path / "state.json")
        s = ShellState()
        s.add_history("ls")
        s.add_history("ls")
        assert s.history == ["ls"]

    def test_add_history_different(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.shell.state._STATE_FILE", tmp_path / "state.json")
        s = ShellState()
        s.add_history("ls")
        s.add_history("pwd")
        assert s.history == ["ls", "pwd"]

    def test_add_history_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.shell.state._STATE_FILE", tmp_path / "state.json")
        s = ShellState()
        s.add_history("")
        s.add_history(None)
        assert s.history == []

    def test_set_alias(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.shell.state._STATE_FILE", tmp_path / "state.json")
        s = ShellState()
        s.set_alias("ll", "ls -la")
        assert s.aliases["ll"] == "ls -la"

    def test_unset_alias(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.shell.state._STATE_FILE", tmp_path / "state.json")
        s = ShellState()
        s.set_alias("ll", "ls -la")
        assert s.unset_alias("ll") is True
        assert "ll" not in s.aliases

    def test_unset_alias_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.shell.state._STATE_FILE", tmp_path / "state.json")
        s = ShellState()
        assert s.unset_alias("nonexistent") is False

    def test_set_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.shell.state._STATE_FILE", tmp_path / "state.json")
        s = ShellState()
        s.set_env("EDITOR", "vim")
        assert s.env["EDITOR"] == "vim"

    def test_to_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.shell.state._STATE_FILE", tmp_path / "state.json")
        s = ShellState()
        s.add_history("test")
        s.set_alias("q", "exit")
        d = s.to_dict()
        assert d["history_count"] == 1
        assert d["aliases"] == {"q": "exit"}
        assert d["first_run"] is True

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        state_file = tmp_path / "state.json"
        monkeypatch.setattr("domains.shell.state._STATE_FILE", state_file)
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

    def test_load_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.shell.state._STATE_FILE", tmp_path / "nonexistent.json")
        s = ShellState()
        assert s.history == []

    def test_load_corrupt(self, tmp_path, monkeypatch):
        state_file = tmp_path / "state.json"
        state_file.write_text("not json {{{")
        monkeypatch.setattr("domains.shell.state._STATE_FILE", state_file)
        s = ShellState()
        assert s.history == []
