"""Exit-code contract of apps.cli.src.cli.main — SystemExit must propagate.

main() historically swallowed SystemExit (``except SystemExit: pass``),
so every failure path — usage errors, the framework's KeyboardInterrupt
handler, command ``sys.exit(1)`` — left the process at exit code 0.
These tests pin the contract: nonzero codes survive main(), 0/None return
normally, string codes become 1 (printed to stderr), bare KI is 130.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def cli_mod():
    from apps.cli.src import cli as mod

    return mod


class TestMainExitCodes:
    def test_nonzero_exit_code_propagates(self, cli_mod, monkeypatch):
        def boom(**_kwargs):
            raise SystemExit(2)

        monkeypatch.setattr(cli_mod, "cli", boom)
        with pytest.raises(SystemExit) as ei:
            cli_mod.main()
        assert ei.value.code == 2

    def test_zero_exit_code_returns_normally(self, cli_mod, monkeypatch):
        monkeypatch.setattr(cli_mod, "cli", MagicMock(side_effect=SystemExit(0)))
        assert cli_mod.main() is None

    def test_none_exit_code_returns_normally(self, cli_mod, monkeypatch):
        monkeypatch.setattr(cli_mod, "cli", MagicMock(side_effect=SystemExit(None)))
        assert cli_mod.main() is None

    def test_string_exit_code_becomes_one_on_stderr(self, cli_mod, monkeypatch, capsys):
        monkeypatch.setattr(cli_mod, "cli", MagicMock(side_effect=SystemExit("bye")))
        with pytest.raises(SystemExit) as ei:
            cli_mod.main()
        assert ei.value.code == 1
        assert "bye" in capsys.readouterr().err

    def test_keyboard_interrupt_exits_130(self, cli_mod, monkeypatch):
        monkeypatch.setattr(cli_mod, "cli", MagicMock(side_effect=KeyboardInterrupt))
        with pytest.raises(SystemExit) as ei:
            cli_mod.main()
        assert ei.value.code == 130

    def test_unknown_command_exits_1_through_main(self, cli_mod, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sloughgpt", "definitely-not-a-command-xyz"])
        with pytest.raises(SystemExit) as ei:
            cli_mod.main()
        assert ei.value.code == 1

    def test_help_exits_zero_and_returns_normally(self, cli_mod, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sloughgpt", "--help"])
        assert cli_mod.main() is None
