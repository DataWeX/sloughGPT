"""Tests for domains/shell/state.py — file-backed shell state persistence."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch
from domains.shell.state import ShellState, _MAX_HISTORY


class TestShellState:
    def _make_state(self, tmp_path, state_data=None):
        state_file = tmp_path / "shell_state.json"
        if state_data:
            state_file.write_text(json.dumps(state_data))
        with patch("domains.shell.state._STATE_FILE", state_file), \
             patch("domains.shell.state._CONFIG_DIR", tmp_path):
            return ShellState()

    def test_fresh_state(self, tmp_path):
        state = self._make_state(tmp_path)
        assert state.history == []
        assert state.aliases == {}
        assert state.env == {}
        assert state.first_run is True

    def test_loads_existing_state(self, tmp_path):
        data = {
            "history": ["ls", "pwd"],
            "aliases": {"q": "exit"},
            "env": {"EDITOR": "vim"},
            "first_run": False,
        }
        state = self._make_state(tmp_path, data)
        assert state.history == ["ls", "pwd"]
        assert state.aliases == {"q": "exit"}
        assert state.env == {"EDITOR": "vim"}
        assert state.first_run is False

    def test_add_history(self, tmp_path):
        state = self._make_state(tmp_path)
        state.add_history("ls")
        state.add_history("pwd")
        assert state.history == ["ls", "pwd"]

    def test_add_history_dedup_consecutive(self, tmp_path):
        state = self._make_state(tmp_path)
        state.add_history("ls")
        state.add_history("ls")
        state.add_history("ls")
        assert state.history == ["ls"]

    def test_add_history_allows_non_consecutive(self, tmp_path):
        state = self._make_state(tmp_path)
        state.add_history("ls")
        state.add_history("pwd")
        state.add_history("ls")
        assert state.history == ["ls", "pwd", "ls"]

    def test_add_history_ignores_empty(self, tmp_path):
        state = self._make_state(tmp_path)
        state.add_history("")
        state.add_history(None)
        assert state.history == []

    def test_set_alias(self, tmp_path):
        state = self._make_state(tmp_path)
        state.set_alias("q", "exit")
        assert state.aliases["q"] == "exit"

    def test_unset_alias(self, tmp_path):
        state = self._make_state(tmp_path)
        state.set_alias("q", "exit")
        assert state.unset_alias("q") is True
        assert "q" not in state.aliases

    def test_unset_alias_nonexistent(self, tmp_path):
        state = self._make_state(tmp_path)
        assert state.unset_alias("nonexistent") is False

    def test_set_env(self, tmp_path):
        state = self._make_state(tmp_path)
        state.set_env("EDITOR", "vim")
        assert state.env["EDITOR"] == "vim"

    def test_to_dict(self, tmp_path):
        state = self._make_state(tmp_path)
        state.add_history("ls")
        state.set_alias("h", "history")
        state.set_env("FOO", "bar")
        d = state.to_dict()
        assert d["history_count"] == 1
        assert d["aliases"] == {"h": "history"}
        assert d["env_vars"] == 1
        assert d["first_run"] is True

    def test_save_persists(self, tmp_path):
        state_file = tmp_path / "shell_state.json"
        with patch("domains.shell.state._STATE_FILE", state_file), \
             patch("domains.shell.state._CONFIG_DIR", tmp_path):
            state = ShellState()
            state.add_history("test_cmd")
            state.set_alias("x", "y")
            state.save()

        saved = json.loads(state_file.read_text())
        assert saved["history"] == ["test_cmd"]
        assert saved["aliases"] == {"x": "y"}
        assert saved["first_run"] is True
        assert "last_session" in saved

    def test_save_truncates_history(self, tmp_path):
        state_file = tmp_path / "shell_state.json"
        with patch("domains.shell.state._STATE_FILE", state_file), \
             patch("domains.shell.state._CONFIG_DIR", tmp_path):
            state = ShellState()
            for i in range(_MAX_HISTORY + 10):
                state.add_history(f"cmd{i}")
            state.save()

        saved = json.loads(state_file.read_text())
        assert len(saved["history"]) == _MAX_HISTORY

    def test_load_corrupt_file(self, tmp_path):
        state_file = tmp_path / "shell_state.json"
        state_file.write_text("NOT JSON!!!")
        with patch("domains.shell.state._STATE_FILE", state_file), \
             patch("domains.shell.state._CONFIG_DIR", tmp_path):
            state = ShellState()
        assert state.history == []
        assert state.aliases == {}

    def test_save_failure_is_silent(self, tmp_path):
        state_file = tmp_path / "shell_state.json"
        with patch("domains.shell.state._STATE_FILE", state_file), \
             patch("domains.shell.state._CONFIG_DIR", tmp_path), \
             patch.object(Path, "write_text", side_effect=OSError("disk full")):
            state = ShellState()
            state.add_history("cmd")
            state.save()  # should not raise
        assert state.history == ["cmd"]
