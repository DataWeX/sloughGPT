"""Pre-configured mock factories for common test patterns.

Usage in conftest.py:
    from tests.fixtures.mocks import MockAPI, MockCLI, MockLogger

    @pytest.fixture
    def mock_api():
        return MockAPI()

Or in tests:
    def test_something(mock_api):
        mock_api.set_response(200, {"models": [{"name": "gpt2"}]})
        result = mock_api.get("/models")
        assert result["models"][0]["name"] == "gpt2"
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch


class MockAPI:
    """Pre-configured mock for HTTP API calls."""

    def __init__(self):
        self._responses: dict[str, tuple[int, Any]] = {}
        self._calls: list[dict] = []
        self._mock_get = MagicMock(side_effect=self._do_request)
        self._mock_post = MagicMock(side_effect=self._do_request)

    def _do_request(self, url: str = "", **kwargs) -> MagicMock:
        self._calls.append({"url": url, **kwargs})
        status, data = self._responses.get(url, (200, {"status": "ok"}))
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = data
        resp.ok = 200 <= status < 300
        resp.text = json.dumps(data)
        resp.headers = {"content-type": "application/json"}
        return resp

    def set_response(self, status: int, data: Any, url: str = "") -> None:
        self._responses[url] = (status, data)

    def get_calls(self) -> list[dict]:
        return list(self._calls)

    def clear(self) -> None:
        self._calls.clear()
        self._responses.clear()

    @contextmanager
    def intercept(self):
        """Context manager that patches requests.get and requests.post."""
        import requests as req
        with patch.object(req, "get", self._mock_get):
            with patch.object(req, "post", self._mock_post):
                yield self


class MockCLI:
    """Pre-configured mock for CLI command testing."""

    def __init__(self):
        self.captured_output: list[str] = []
        self.captured_errors: list[str] = []
        self._mock_echo = MagicMock(side_effect=lambda msg: self.captured_output.append(str(msg)))
        self._mock_log = MagicMock()

    def echo(self, msg: str) -> None:
        self.captured_output.append(str(msg))

    def get_output(self) -> str:
        return "\n".join(self.captured_output)

    def get_last_output(self) -> str:
        return self.captured_output[-1] if self.captured_output else ""

    def assert_output_contains(self, text: str) -> None:
        output = self.get_output()
        assert text in output, f"Expected '{text}' in output:\n{output}"

    def assert_output_not_contains(self, text: str) -> None:
        output = self.get_output()
        assert text not in output, f"Unexpected '{text}' in output:\n{output}"

    def clear(self) -> None:
        self.captured_output.clear()
        self.captured_errors.clear()

    @contextmanager
    def capture(self):
        """Context manager that captures echo() calls."""
        try:
            import apps.cli.src.cli as cli_mod
            original_echo = cli_mod.echo
            cli_mod.echo = self._mock_echo
            yield self
        finally:
            cli_mod.echo = original_echo


class MockLogger:
    """Pre-configured mock for logger testing."""

    def __init__(self):
        self.headers: list[str] = []
        self.sections: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.successes: list[str] = []
        self.steps: list[str] = []
        self.kvs: list[tuple[str, str]] = []

        self._mock = MagicMock()
        self._mock.header = lambda msg: self.headers.append(msg)
        self._mock.section = lambda msg: self.sections.append(msg)
        self._mock.info = lambda msg: self.infos.append(msg)
        self._mock.warning = lambda msg: self.warnings.append(msg)
        self._mock.error = lambda msg: self.errors.append(msg)
        self._mock.success = lambda msg: self.successes.append(msg)
        self._mock.step = lambda msg: self.steps.append(msg)
        self._mock.key_value = lambda k, v: self.kvs.append((k, v))
        self._mock.blank = lambda: None
        self._mock.table = lambda h, r: None
        self._mock.status = lambda n, v, s: None
        self._mock.command = lambda msg: None

    @property
    def mock(self) -> MagicMock:
        return self._mock

    def assert_warning(self, text: str) -> None:
        assert any(text in w for w in self.warnings), \
            f"Expected warning containing '{text}', got: {self.warnings}"

    def assert_error(self, text: str) -> None:
        assert any(text in e for e in self.errors), \
            f"Expected error containing '{text}', got: {self.errors}"

    def assert_no_errors(self) -> None:
        assert len(self.errors) == 0, f"Expected no errors, got: {self.errors}"

    def clear(self) -> None:
        self.headers.clear()
        self.sections.clear()
        self.infos.clear()
        self.warnings.clear()
        self.errors.clear()
        self.successes.clear()
        self.steps.clear()
        self.kvs.clear()


class MockProcess:
    """Pre-configured mock for subprocess calls."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs) -> MagicMock:
        self._calls.append(cmd)
        proc = MagicMock()
        proc.returncode = self._returncode
        proc.stdout = self._stdout
        proc.stderr = self._stderr
        proc.communicate.return_value = (self._stdout, self._stderr)
        return proc

    def get_calls(self) -> list[list[str]]:
        return list(self._calls)


class MockFilesystem:
    """In-memory filesystem mock for file-dependent tests."""

    def __init__(self):
        self._files: dict[str, str] = {}
        self._dirs: set[str] = set()

    def add_file(self, path: str, content: str = "") -> None:
        self._files[path] = content
        # Ensure parent dirs exist
        parts = path.split("/")
        for i in range(1, len(parts)):
            self._dirs.add("/".join(parts[:i]))

    def add_dir(self, path: str) -> None:
        self._dirs.add(path)

    def read_file(self, path: str) -> str:
        if path not in self._files:
            raise FileNotFoundError(path)
        return self._files[path]

    def file_exists(self, path: str) -> bool:
        return path in self._files

    def dir_exists(self, path: str) -> bool:
        return path in self._dirs

    def list_files(self, dir_path: str = "") -> list[str]:
        prefix = dir_path.rstrip("/") + "/" if dir_path else ""
        return [
            p for p in self._files
            if p.startswith(prefix) and "/" not in p[len(prefix):]
        ]

    @contextmanager
    def patch(self):
        """Context manager that patches Path.exists, Path.read_text, etc."""
        from pathlib import Path
        original_exists = Path.exists
        original_read = Path.read_text

        def mock_exists(self_path):
            path_str = str(self_path)
            if path_str in self._files or path_str in self._dirs:
                return True
            return original_exists(self_path)

        def mock_read(self_path, *args, **kwargs):
            path_str = str(self_path)
            if path_str in self._files:
                return self._files[path_str]
            return original_read(self_path, *args, **kwargs)

        with patch.object(Path, "exists", mock_exists):
            with patch.object(Path, "read_text", mock_read):
                yield self
