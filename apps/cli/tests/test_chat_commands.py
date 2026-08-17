"""Tests for apps/cli/src/commands/chat.py — chat and generate commands."""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(autouse=True)
def mock_log(monkeypatch):
    fake_log = MagicMock()
    fake_log.header = MagicMock()
    fake_log.section = MagicMock()
    fake_log.info = MagicMock()
    fake_log.warning = MagicMock()
    fake_log.error = MagicMock()
    fake_log.success = MagicMock()
    fake_log.step = MagicMock()
    fake_log.key_value = MagicMock()
    fake_log.blank = MagicMock()
    fake_log.table = MagicMock()
    fake_log.status = MagicMock()
    fake_log.command = MagicMock()
    import commands.chat as mod
    monkeypatch.setattr(mod, "log", fake_log)
    return fake_log


class TestCmdGenerate:
    def test_generate_calls_engine(self, monkeypatch):
        from commands.chat import cmd_generate
        args = MagicMock()
        args.prompt = "Hello"
        args.max_tokens = 50
        args.temperature = 0.8

        mock_engine = MagicMock()
        mock_engine.generate.return_value = "Generated text"
        mock_engine.load_soul.return_value = MagicMock(name="test_soul")

        import domains.core
        monkeypatch.setattr(domains.core, "SloEngine", lambda **kw: mock_engine)
        import utils.helpers
        monkeypatch.setattr(utils.helpers, "local_soul_candidate_paths", lambda x: [])

        cmd_generate(args)
        mock_engine.generate.assert_called_once()

    def test_generate_no_model_demo_mode(self, monkeypatch):
        from commands.chat import cmd_generate
        args = MagicMock()
        args.prompt = "Hello"
        args.max_tokens = 50
        args.temperature = 0.8

        mock_engine = MagicMock()
        mock_engine.generate.return_value = "Demo output"

        import domains.core
        monkeypatch.setattr(domains.core, "SloEngine", lambda **kw: mock_engine)
        import utils.helpers
        monkeypatch.setattr(utils.helpers, "local_soul_candidate_paths", lambda x: [])

        cmd_generate(args)


class TestCmdChat:
    def test_chat_no_serve_flag(self, monkeypatch):
        from commands.chat import cmd_chat
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        args.no_serve = True
        args.auto_model = None
        args.model = None
        args.max_tokens = 100
        args.temperature = 0.8

        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))

        cmd_chat(args)

    def test_chat_quit_exits(self, monkeypatch):
        from commands.chat import cmd_chat
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        args.no_serve = True
        args.auto_model = None
        args.model = None
        args.max_tokens = 100
        args.temperature = 0.8

        import requests
        monkeypatch.setattr(requests, "get", MagicMock(return_value=MagicMock(status_code=200)))
        monkeypatch.setattr("builtins.input", lambda _: "quit")

        cmd_chat(args)
