"""Tests for apps/cli/src/commands/chat.py — chat and generate commands."""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(autouse=True)
def mock_log(monkeypatch):
    fake_log = MagicMock()
    import commands.chat as mod
    monkeypatch.setattr(mod, "log", fake_log)
    return fake_log


@pytest.fixture
def mock_soul_paths(monkeypatch):
    """Mock local_soul_candidate_paths."""
    import utils.helpers
    paths = MagicMock(return_value=[])
    monkeypatch.setattr(utils.helpers, "local_soul_candidate_paths", paths)
    return paths


class TestCmdGenerate:
    def test_creates_engine_and_generates(self, monkeypatch, mock_soul_paths):
        from commands.chat import cmd_generate
        args = MagicMock()
        args.prompt = "Hello"
        args.max_tokens = 50
        args.temperature = 0.8

        mock_engine = MagicMock()
        mock_engine.generate.return_value = "Generated text"

        import domains.core
        monkeypatch.setattr(domains.core, "SloEngine", lambda **kw: mock_engine)

        cmd_generate(args)

        mock_engine.generate.assert_called_once_with(
            "Hello", max_new_tokens=50, temperature=0.8
        )

    def test_header_shows_prompt_info(self, mock_log, monkeypatch, mock_soul_paths):
        from commands.chat import cmd_generate
        args = MagicMock()
        args.prompt = "Test prompt"
        args.max_tokens = 100
        args.temperature = 0.5

        mock_engine = MagicMock()
        mock_engine.generate.return_value = "ok"

        import domains.core
        monkeypatch.setattr(domains.core, "SloEngine", lambda **kw: mock_engine)

        cmd_generate(args)

        mock_log.header.assert_called_with("Text Generation")
        mock_log.key_value.assert_any_call("Prompt", "Test prompt")
        mock_log.key_value.assert_any_call("Max Tokens", "100")
        mock_log.key_value.assert_any_call("Temperature", "0.5")

    def test_no_soul_files_warns_demo_mode(self, mock_log, monkeypatch, mock_soul_paths):
        from commands.chat import cmd_generate
        args = MagicMock()
        args.prompt = "Hi"
        args.max_tokens = 10
        args.temperature = 0.8

        mock_soul_paths.return_value = []
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "Demo"

        import domains.core
        monkeypatch.setattr(domains.core, "SloEngine", lambda **kw: mock_engine)

        cmd_generate(args)

        mock_log.warning.assert_called_with("No model found, using demo mode")

    def test_soul_file_loaded(self, mock_log, monkeypatch):
        from commands.chat import cmd_generate
        from pathlib import Path
        args = MagicMock()
        args.prompt = "Hi"
        args.max_tokens = 10
        args.temperature = 0.8

        mock_engine = MagicMock()
        mock_soul = MagicMock()
        mock_soul.name = "test-soul"
        mock_engine.load_soul.return_value = mock_soul
        mock_engine.generate.return_value = "ok"

        import domains.core
        monkeypatch.setattr(domains.core, "SloEngine", lambda **kw: mock_engine)
        import utils.helpers
        monkeypatch.setattr(
            utils.helpers, "local_soul_candidate_paths",
            lambda x: [Path("/fake/model.soul")]
        )

        cmd_generate(args)

        mock_engine.load_soul.assert_called_once_with("/fake/model.soul")
        mock_log.success.assert_called()
        assert "test-soul" in str(mock_log.success.call_args)


class TestCmdChat:
    def test_no_serve_flag_returns_when_api_down(self, monkeypatch, mock_log):
        from commands.chat import cmd_chat
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        args.no_serve = True
        args.auto_model = None
        args.model = None

        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))

        cmd_chat(args)

        mock_log.error.assert_called_with("API not reachable and --no-serve set")

    def test_quit_exits_loop(self, monkeypatch, mock_log):
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

        mock_log.header.assert_called_with("SloughGPT Chat")

    def test_auto_model_triggers_load(self, monkeypatch, mock_log):
        from commands.chat import cmd_chat
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        args.no_serve = True
        args.auto_model = "gpt2"
        args.model = None
        args.max_tokens = 100
        args.temperature = 0.8

        import requests
        mock_get = MagicMock(return_value=MagicMock(status_code=200))
        mock_post = MagicMock(return_value=MagicMock(status_code=200, json=MagicMock(return_value={})))
        monkeypatch.setattr(requests, "get", mock_get)
        monkeypatch.setattr(requests, "post", mock_post)
        monkeypatch.setattr("builtins.input", lambda _: "quit")

        cmd_chat(args)

        mock_post.assert_called()
        load_call = [c for c in mock_post.call_args_list if "/models/load" in str(c)]
        assert len(load_call) > 0

    def test_user_input_sent_to_api(self, monkeypatch, mock_log):
        from commands.chat import cmd_chat
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        args.no_serve = True
        args.auto_model = None
        args.model = None
        args.max_tokens = 50
        args.temperature = 0.8

        call_count = [0]
        def mock_input(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello AI"
            return "quit"

        import requests
        mock_get = MagicMock(return_value=MagicMock(status_code=200))
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": "Response"}
        mock_post = MagicMock(return_value=mock_resp)
        monkeypatch.setattr(requests, "get", mock_get)
        monkeypatch.setattr(requests, "post", mock_post)
        monkeypatch.setattr("builtins.input", mock_input)

        cmd_chat(args)

        generate_calls = [c for c in mock_post.call_args_list if "/generate" in str(c)]
        assert len(generate_calls) > 0
        assert generate_calls[0][1]["json"]["prompt"] == "Hello AI"
