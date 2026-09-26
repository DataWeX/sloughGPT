"""Tests for apps/cli/src/commands/dev.py — dev server and health commands."""

import json
import os
import sys
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(autouse=True)
def mock_log(monkeypatch):
    fake_log = MagicMock()
    import commands.dev as mod

    monkeypatch.setattr(mod, "log", fake_log)
    return fake_log


class TestHelpers:
    def test_repo_root_returns_path(self):
        from commands.dev import _repo_root

        root = _repo_root()
        assert root.exists()
        assert (root / "apps").is_dir()

    def test_check_port_closed(self):
        from commands.dev import _check_port

        assert _check_port(1) is False

    def test_check_api_ready_closed(self):
        from commands.dev import _check_api_ready

        assert _check_api_ready(1) is False

    def test_get_startup_progress_none(self):
        from commands.dev import _get_startup_progress

        assert _get_startup_progress(1) is None


class TestCmdHealth:
    def test_server_down_logs_error(self, mock_log, monkeypatch):
        import requests
        from commands.dev import cmd_health

        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        cmd_health(args)
        mock_log.header.assert_called_with("API Health Check")
        mock_log.error.assert_called_with("API not reachable")

    def test_server_up_logs_success(self, mock_log, monkeypatch):
        import requests
        from commands.dev import cmd_health

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "model_loaded": True}
        monkeypatch.setattr(requests, "get", MagicMock(return_value=mock_resp))
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        cmd_health(args)
        mock_log.success.assert_called()
        assert "Healthy" in mock_log.success.call_args[0][0]

    def test_server_up_displays_keys(self, mock_log, monkeypatch):
        import requests
        from commands.dev import cmd_health

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "model": "gpt2"}
        monkeypatch.setattr(requests, "get", MagicMock(return_value=mock_resp))
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        cmd_health(args)
        kv_keys = [c[0][0] for c in mock_log.key_value.call_args_list]
        assert "Endpoint" in kv_keys


class TestCmdApiStatus:
    def test_server_down_logs_not_reachable(self, mock_log, monkeypatch):
        import requests
        from commands.dev import cmd_api_status

        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        cmd_api_status(args)
        mock_log.header.assert_called_with("SloughGPT API Status")
        mock_log.status.assert_called()

    def test_endpoints_checked(self, mock_log, monkeypatch):
        import requests
        from commands.dev import cmd_api_status

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        monkeypatch.setattr(requests, "get", MagicMock(return_value=mock_resp))
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        cmd_api_status(args)
        assert mock_log.status.call_count >= 4


class TestCmdApiTest:
    def test_server_down_logs_error(self, mock_log, monkeypatch):
        import requests
        from commands.dev import cmd_api_test

        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        monkeypatch.setattr(requests, "post", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        args.endpoint = "/health"
        cmd_api_test(args)
        mock_log.header.assert_called_with("API Endpoint Tests")


class TestStatusBlock:
    """Tests for the StatusBlock in-place update class."""

    def test_update_writes_to_tty(self):

        from commands.dev import StatusBlock

        logger = MagicMock()
        logger._lock = __import__("threading").Lock()
        logger._colors = False
        logger.cursor_up = MagicMock()
        logger.clear_line = MagicMock()

        stream = MagicMock()
        stream.isatty.return_value = True
        logger._stream = stream

        block = StatusBlock(logger)
        assert block._is_tty is True

        block.update("  SloughGPT", "  API: starting")

        assert logger.info.call_count == 2
        calls = [c[0][0] for c in logger.info.call_args_list]
        assert "  SloughGPT" in calls
        assert "  API: starting" in calls

    def test_update_clears_previous_on_tty(self):
        from commands.dev import StatusBlock

        logger = MagicMock()
        logger._lock = __import__("threading").Lock()
        logger._colors = False
        logger.cursor_up = MagicMock()
        logger.clear_line = MagicMock()

        stream = MagicMock()
        stream.isatty.return_value = True
        logger._stream = stream

        block = StatusBlock(logger)
        block.update("  Line 1", "  Line 2")
        assert len(block._lines) == 2

        block.update("  New Line 1")

        # Should have called cursor_up and clear_line to clear previous lines
        assert logger.cursor_up.called
        assert logger.clear_line.called
        assert len(block._lines) == 1

    def test_first_update_no_clear(self):
        from commands.dev import StatusBlock

        logger = MagicMock()
        logger._lock = __import__("threading").Lock()
        logger._colors = False
        logger.cursor_up = MagicMock()
        logger.clear_line = MagicMock()

        stream = MagicMock()
        stream.isatty.return_value = True
        logger._stream = stream

        block = StatusBlock(logger)
        block.update("  Only line")

        # First update should not call cursor_up or clear_line
        assert not logger.cursor_up.called
        assert not logger.clear_line.called
        assert len(block._lines) == 1

    def test_non_tty_uses_info(self):
        import threading

        from commands.dev import StatusBlock

        logger = MagicMock()
        logger._stream = MagicMock()
        logger._stream.isatty.return_value = False
        logger._lock = threading.Lock()
        logger._colors = False

        block = StatusBlock(logger)
        assert block._is_tty is False

        block.update("  Line 1", "  Line 2")
        assert logger.info.call_count == 2

    def test_non_tty_prints_only_once(self):
        from commands.dev import StatusBlock

        logger = MagicMock()
        logger._stream = MagicMock()
        logger._stream.isatty.return_value = False
        logger._lock = __import__("threading").Lock()
        logger._colors = False

        block = StatusBlock(logger)
        block.update("  Line 1")
        assert logger.info.call_count == 1

        # Second update should NOT print again
        block.update("  Line 2")
        assert logger.info.call_count == 1

    def test_line_count_tracking(self):
        import threading

        from commands.dev import StatusBlock

        logger = MagicMock()
        stream = MagicMock()
        stream.isatty.return_value = True
        logger._stream = stream
        logger._lock = threading.Lock()
        logger._colors = False
        logger.cursor_up = MagicMock()
        logger.clear_line = MagicMock()

        block = StatusBlock(logger)
        block.update("a", "b", "c")
        assert len(block._lines) == 3
        block.update("x")
        assert len(block._lines) == 1


class TestPortHelpers:
    """Tests for port utility functions."""

    def test_is_port_bound_free(self):
        from commands.dev import _is_port_bound

        assert _is_port_bound(49999) is False

    def test_find_free_port_same_if_free(self):
        from commands.dev import _find_free_port

        assert _find_free_port(49999) == 49999

    def test_find_free_port_skips_bound(self):
        import socket

        from commands.dev import _find_free_port, _is_port_bound

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("localhost", 49998))
            sock.listen(1)
            assert _is_port_bound(49998) is True
            result = _find_free_port(49998)
            assert result != 49998
            assert result >= 49998
        finally:
            sock.close()

    def test_check_web_ready_nonexistent(self):
        from commands.dev import _check_web_ready

        assert _check_web_ready(49997) is False


class TestCleanup:
    """Tests for _cleanup function handling None processes."""

    def test_cleanup_with_none_processes(self):
        """_cleanup should not crash when api_proc or web_proc is None."""
        from commands.dev import _cleanup

        with patch("commands.dev._kill_port"):
            _cleanup(None, None, 8000, 3000)

    def test_cleanup_with_api_proc_none(self):
        """_cleanup should handle api_proc=None and a valid web_proc."""
        from commands.dev import _cleanup

        mock_web_proc = MagicMock()
        mock_web_proc.poll.return_value = None
        with patch("commands.dev._kill_port"):
            _cleanup(None, mock_web_proc, 8000, 3000)
        mock_web_proc.terminate.assert_called_once()

    def test_cleanup_with_web_proc_none(self):
        """_cleanup should handle web_proc=None and a valid api_proc."""
        from commands.dev import _cleanup

        mock_api_proc = MagicMock()
        mock_api_proc.poll.return_value = None
        with patch("commands.dev._kill_port"):
            _cleanup(mock_api_proc, None, 8000, 3000)
        mock_api_proc.terminate.assert_called_once()

    def test_cleanup_skips_already_terminated(self):
        """_cleanup should not terminate processes that already exited."""
        from commands.dev import _cleanup

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # already exited
        with patch("commands.dev._kill_port"):
            _cleanup(mock_proc, mock_proc, 8000, 3000)
        mock_proc.terminate.assert_not_called()


class TestCmdApiAndWebMonitorLoop:
    """Tests for _cmd_api_and_web monitor loop handling web_proc=None."""

    def test_monitor_loop_web_proc_none_no_crash(self, mock_log, monkeypatch):
        """Monitor loop should not crash when web_proc is None (web reused)."""
        from commands.dev import _cmd_api_and_web

        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        args.web_port = 3000

        with (
            patch("commands.dev._check_api_ready", return_value=True),
            patch("commands.dev._check_web_ready", return_value=True),
            patch("commands.dev._check_port", return_value=True),
            patch("commands.dev._repo_root") as mock_root,
            patch("commands.dev._cleanup"),
            patch("commands.dev.signal"),
            patch("commands.dev.webbrowser"),
            patch("commands.dev.time") as mock_time,
        ):
            mock_root.return_value = MagicMock()
            mock_time.sleep.side_effect = [None, None, KeyboardInterrupt]

            # Simulate web_reused=True by making _check_web_ready return True
            # This means web_proc stays None
            _cmd_api_and_web(args)

            # Should not raise AttributeError: 'NoneType' object has no attribute 'poll'
            # The function should complete without crashing


class TestStartupBudget:
    """The CLI's API-wait budget must cover the server's real startup cost."""

    def test_budget_covers_stage_ready_worst_case(self):
        from commands.dev import API_STARTUP_TIMEOUT

        # Stage READY's model_ready hook polls 120s (startup.py) under a 130s
        # hook timeout, and Stage CRITICAL spends ~40s before READY begins.
        # A budget below that kills a healthy server mid-model-load and
        # reports it as "error".
        assert API_STARTUP_TIMEOUT >= 160

    def test_no_hardcoded_90s_api_waits_remain(self):
        import commands.dev as mod

        src = open(mod.__file__).read()
        assert "range(90)" not in src, "reintroduced a hardcoded 90s API wait"
        assert "timeout=90" not in src, "reintroduced a hardcoded 90s API wait"
        # every readiness loop must be driven by the shared budget
        assert src.count("API_STARTUP_TIMEOUT") >= 5


class TestLatestStartupPhase:
    """The wait loops surface the API's own startup markers while polling."""

    def test_returns_latest_phase_marker(self):
        from commands.dev import _latest_startup_phase

        lines = [
            "06:46:15 INF [START] startup Phase: all routers registered (61 routes)",
            "06:46:30 INF [START] startup Phase 4: loading model Qwen (background)",
            "06:47:00 INF [SYS] startup Startup complete — server ready for requests",
        ]
        assert _latest_startup_phase(lines) == "Startup complete — server ready for requests"

    def test_prefers_stage_markers_over_older_phases(self):
        from commands.dev import _latest_startup_phase

        lines = [
            "06:46:30 INF [START] startup Phase 4: loading model (background)",
            "06:46:30 INF [START] startup Stage READY: running 2 hooks",
        ]
        assert _latest_startup_phase(lines) == "Stage READY: running 2 hooks"

    def test_empty_when_no_marker(self):
        from commands.dev import _latest_startup_phase

        assert _latest_startup_phase([]) == ""
        assert _latest_startup_phase(["06:46:15 INF [INFRA] quantization Quarantine"]) == ""

    def test_ignores_non_startup_lines_and_truncates(self):
        from commands.dev import _latest_startup_phase

        lines = ["noise", "no marker here"]
        assert _latest_startup_phase(lines) == ""
        long_marker = "Phase 4: loading model " + "x" * 200
        out = _latest_startup_phase([f"INF startup {long_marker}"])
        assert len(out) <= 72


class TestWebDevEnv:
    """The web dev server must receive the requested port via PORT."""

    def test_sets_port_and_keeps_other_env(self, monkeypatch, tmp_path):
        from commands.dev import _web_dev_env

        monkeypatch.setenv("NVM_DIR", str(tmp_path))  # no versions/node → PATH untouched
        out = _web_dev_env({"PATH": "/base/bin", "FORCE_COLOR": "1"}, 3999)
        assert out["PORT"] == "3999"
        assert out["FORCE_COLOR"] == "1"
        assert out["PATH"] == "/base/bin"

    def test_prepends_nvm_node_to_path(self, monkeypatch, tmp_path):
        from commands.dev import _web_dev_env

        nvm = tmp_path / "nvm"
        bin_dir = nvm / "versions" / "node" / "v22.0.0" / "bin"
        bin_dir.mkdir(parents=True)
        monkeypatch.setenv("NVM_DIR", str(nvm))

        out = _web_dev_env({"PATH": "/base/bin"}, 3000)
        assert out["PATH"].split(os.pathsep) == [str(bin_dir), "/base/bin"]

    def test_node_env_does_not_mutate_input(self, monkeypatch, tmp_path):
        from commands.dev import _node_env

        nvm = tmp_path / "nvm"
        (nvm / "versions" / "node" / "v22.0.0" / "bin").mkdir(parents=True)
        monkeypatch.setenv("NVM_DIR", str(nvm))

        base = {"PATH": "/base/bin"}
        out = _node_env(base)
        assert base == {"PATH": "/base/bin"}
        assert out["PATH"].split(os.pathsep)[0].startswith(str(nvm))


class TestCmdDevWebPort:
    """`slo dev --web-port N` must make the dev server listen on N.

    Next.js reads PORT from env; Vite ignores PORT and needs --port argv.
    The web subprocess previously got neither, so Next always listened on
    its default 3000 while the readiness check polled the requested port.
    """

    @pytest.mark.parametrize("watch_web", [False, True])
    @pytest.mark.parametrize(
        ("dev_script", "expected_tail"),
        [
            ("next dev", ["npm", "run", "dev"]),
            ("vite", ["npm", "run", "dev", "--", "--port", "3999", "--host", "0.0.0.0"]),
        ],
    )
    def test_web_subprocess_receives_web_port(
        self, mock_log, monkeypatch, tmp_path, watch_web, dev_script, expected_tail
    ):
        import commands.dev as mod

        web_root = tmp_path / "apps" / "web"
        web_root.mkdir(parents=True)
        (web_root / "package.json").write_text(
            json.dumps({"scripts": {"dev": dev_script}}), encoding="utf-8"
        )

        popen_calls = []

        def fake_popen(cmd, **kwargs):
            popen_calls.append((list(cmd), kwargs))
            proc = MagicMock()
            proc.poll.return_value = None
            return proc

        class FakeDashboard:
            def __init__(self, *a, **k):
                pass

            def serve(self, stop_check=None):
                return None

        with (
            patch("commands.dev.subprocess.Popen", side_effect=fake_popen),
            patch("commands.dev._repo_root", return_value=tmp_path),
            patch("commands.dev._read_stream"),
            patch("commands.dev._preflight_model_check"),
            patch("commands.dev._check_api_ready", return_value=False),
            patch("commands.dev._check_port", return_value=False),
            patch("commands.dev._kill_port"),
            patch("commands.dev._cleanup"),
            patch("commands.dev.find_server_python", return_value="/usr/bin/python3"),
            patch("commands.dev.signal"),
            patch("commands.dev.time"),
            patch("core.tui.DevDashboard", FakeDashboard),
        ):
            args = MagicMock()
            args.model = None
            args.port = 8000
            args.web_port = 3999
            args.watch_web = watch_web
            args.host = "localhost"

            mod.cmd_dev(args)

        web_calls = [c for c in popen_calls if c[0] and c[0][0] in ("npm", "npx")]
        assert web_calls, f"web subprocess never spawned: {popen_calls}"
        cmd, kwargs = web_calls[0]
        assert cmd[-len(expected_tail) :] == expected_tail
        assert kwargs["env"]["PORT"] == "3999"


class TestWebDevCmd:
    """package.json's dev script decides how the port reaches the server."""

    @pytest.fixture
    def web_root(self, tmp_path, monkeypatch):
        root = tmp_path / "apps" / "web"
        root.mkdir(parents=True)
        monkeypatch.setattr("commands.dev._repo_root", lambda: tmp_path)
        return root

    def test_next_script_uses_plain_npm_run_dev(self, web_root):
        from commands.dev import _web_dev_cmd

        (web_root / "package.json").write_text('{"scripts": {"dev": "next dev"}}')
        assert _web_dev_cmd(3999) == ["npm", "run", "dev"]

    def test_vite_script_appends_port_and_host(self, web_root):
        from commands.dev import _web_dev_cmd

        (web_root / "package.json").write_text('{"scripts": {"dev": "vite"}}')
        assert _web_dev_cmd(3999) == [
            "npm",
            "run",
            "dev",
            "--",
            "--port",
            "3999",
            "--host",
            "0.0.0.0",
        ]

    def test_missing_package_json_defaults_to_next_form(self, web_root):
        from commands.dev import _web_dev_cmd

        assert _web_dev_cmd(3000) == ["npm", "run", "dev"]

    def test_malformed_package_json_defaults_to_next_form(self, web_root):
        from commands.dev import _web_dev_cmd

        (web_root / "package.json").write_text("{not json")
        assert _web_dev_cmd(3000) == ["npm", "run", "dev"]

    def test_package_json_without_scripts_defaults(self, web_root):
        from commands.dev import _web_dev_cmd

        (web_root / "package.json").write_text('{"name": "web"}')
        assert _web_dev_cmd(3000) == ["npm", "run", "dev"]


class TestCmdDevFailureReporting:
    """A dead or timing-out service must end the dev session with a visible error.

    Previously only web death was noticed, `status["api"]` could never become
    "error", and `_stop_check` required both services to be in error — so a
    crashed API left the TUI showing "starting" forever with no message.
    """

    @staticmethod
    def _scaffold(monkeypatch, tmp_path, stopped, api_dead=False, web_dead=False):
        import commands.dev as mod

        def fake_popen(cmd, **kwargs):
            proc = MagicMock()
            is_npm = bool(cmd) and cmd[0] in ("npm", "npx")
            dead = web_dead if is_npm else api_dead
            proc.poll.return_value = 1 if dead else None
            proc.returncode = 1 if dead else None
            return proc

        class FakeDashboard:
            def __init__(self, *a, **k):
                pass

            def set_status(self, *a, **k):
                pass

            def serve(self, stop_check=None):
                import time as _t

                for _ in range(200):  # up to 4s for the poll thread to react
                    if stop_check is None or stop_check():
                        stopped.append(True)
                        return
                    _t.sleep(0.02)
                stopped.append(False)

        ctx = (
            patch("commands.dev.subprocess.Popen", side_effect=fake_popen),
            patch("commands.dev._repo_root", return_value=tmp_path),
            patch("commands.dev._read_stream"),
            patch("commands.dev._preflight_model_check"),
            patch("commands.dev._check_api_ready", return_value=False),
            patch("commands.dev._check_port", return_value=False),
            patch("commands.dev._kill_port"),
            patch("commands.dev._cleanup"),
            patch("commands.dev.find_server_python", return_value="/usr/bin/python3"),
            patch("commands.dev.signal"),
            patch("commands.dev.time"),
            patch("core.tui.DevDashboard", FakeDashboard),
        )
        return mod, ctx

    @staticmethod
    def _args(web_port=4590):
        args = MagicMock()
        args.model = None
        args.port = 8177
        args.web_port = web_port
        args.watch_web = False
        args.host = "localhost"
        return args

    @staticmethod
    def _error_messages(mock_log):
        return [str(c[0][0]) for c in mock_log.error.call_args_list]

    def test_api_death_reports_and_stops(self, mock_log, monkeypatch, tmp_path):
        stopped = []
        mod, ctx = self._scaffold(monkeypatch, tmp_path, stopped, api_dead=True)
        with ExitStack() as stack:
            for c in ctx:
                stack.enter_context(c)
            mod.cmd_dev(self._args())

        assert stopped == [True], "session must stop when the API dies"
        assert any("API server exited" in m for m in self._error_messages(mock_log)), (
            f"no API death message: {self._error_messages(mock_log)}"
        )

    def test_web_death_reports_and_stops(self, mock_log, monkeypatch, tmp_path):
        stopped = []
        mod, ctx = self._scaffold(monkeypatch, tmp_path, stopped, web_dead=True)
        with ExitStack() as stack:
            for c in ctx:
                stack.enter_context(c)
            mod.cmd_dev(self._args())

        assert stopped == [True], "session must stop when the web server dies"
        assert any("Web server exited" in m for m in self._error_messages(mock_log)), (
            f"no web death message: {self._error_messages(mock_log)}"
        )

    def test_api_timeout_reports_error_instead_of_waiting_forever(
        self, mock_log, monkeypatch, tmp_path
    ):
        stopped = []
        mod, ctx = self._scaffold(monkeypatch, tmp_path, stopped)  # both alive, never ready
        with ExitStack() as stack:
            for c in ctx:
                stack.enter_context(c)
            mod.cmd_dev(self._args())

        assert stopped == [True], "session must stop when the startup budget runs out"
        assert any("did not become ready within" in m for m in self._error_messages(mock_log)), (
            f"no timeout message: {self._error_messages(mock_log)}"
        )
