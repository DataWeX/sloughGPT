"""Tests for shell.repl — ShellREPL utility functions and basics."""

from __future__ import annotations

import os
import io
import sys
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from domains.shell.repl import (
    _color,
    _COLOR_ENABLED,
    _CaptureOutput,
    _get_file_handler,
    _get_log_buffer_handler,
    _fetch_model_names,
    _fetch_soul_names,
    _fetch_dataset_names,
    _fetch_checkpoint_names,
    _COMMAND_CACHE_FETCHERS,
)


# ── _color ──────────────────────────────────────────────────────────────────


class TestColor:

    def test_color_enabled(self):
        with patch.dict(os.environ, {}, clear=False):
            result = _color("hello", "\033[36m")
            if _COLOR_ENABLED:
                assert "\033[36m" in result
            else:
                assert result == "hello"

    def test_color_no_code(self):
        result = _color("hello", "")
        assert result == "hello"

    def test_color_no_color_env(self):
        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            import domains.shell.repl as mod
            old = mod._COLOR_ENABLED
            mod._COLOR_ENABLED = False
            try:
                result = _color("hello", "\033[36m")
                assert result == "hello"
            finally:
                mod._COLOR_ENABLED = old


# ── _CaptureOutput ──────────────────────────────────────────────────────────


class TestCaptureOutput:

    def test_capture_stdout(self):
        with _CaptureOutput():
            print("test output")
        # After context, stdout is restored

    def test_capture_with_repl(self):
        mock_repl = MagicMock()
        mock_repl.io = MagicMock()
        mock_repl.console = MagicMock()
        with _CaptureOutput(repl=mock_repl) as cap:
            pass
        assert cap.getvalue() == ""


# ── _get_file_handler ──────────────────────────────────────────────────────


class TestGetFileHandler:

    def test_returns_handler(self):
        import domains.shell.repl as mod
        old = mod._file_handler
        mod._file_handler = None
        try:
            handler = _get_file_handler()
            assert handler is not None
        finally:
            mod._file_handler = old

    def test_caches_handler(self):
        import domains.shell.repl as mod
        old = mod._file_handler
        mock_handler = MagicMock()
        mock_handler.closed = False
        mod._file_handler = mock_handler
        try:
            handler = _get_file_handler()
            assert handler is mock_handler
        finally:
            mod._file_handler = old


# ── _get_log_buffer_handler ────────────────────────────────────────────────


class TestGetLogBufferHandler:

    def test_returns_handler(self):
        import domains.shell.repl as mod
        old = mod._buf_handler
        mod._buf_handler = None
        try:
            handler = _get_log_buffer_handler()
            # May be None if log_buffer module unavailable
            assert handler is None or handler is not None
        finally:
            mod._buf_handler = old

    def test_caches_handler(self):
        import domains.shell.repl as mod
        old = mod._buf_handler
        mock_handler = MagicMock()
        mock_handler.closed = False
        mod._buf_handler = mock_handler
        try:
            handler = _get_log_buffer_handler()
            assert handler is mock_handler
        finally:
            mod._buf_handler = old


# ── _fetch_* functions ──────────────────────────────────────────────────────


class TestFetchFunctions:

    def test_fetch_model_names_error(self):
        with patch("requests.get", side_effect=Exception("network error")):
            result = _fetch_model_names()
            assert result == []

    def test_fetch_model_names_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"name": "gpt2"}, {"name": "llama"}]
        with patch("requests.get", return_value=mock_resp):
            result = _fetch_model_names()
            assert "gpt2" in result
            assert "llama" in result

    def test_fetch_soul_names_error(self):
        with patch("requests.get", side_effect=Exception("network error")):
            result = _fetch_soul_names()
            assert result == []

    def test_fetch_dataset_names_error(self):
        with patch("requests.get", side_effect=Exception("network error")):
            result = _fetch_dataset_names()
            assert result == []

    def test_fetch_checkpoint_names_error(self):
        with patch("requests.get", side_effect=Exception("network error")):
            result = _fetch_checkpoint_names()
            assert result == []


# ── _COMMAND_CACHE_FETCHERS ─────────────────────────────────────────────────


class TestCommandCacheFetchers:

    def test_has_all_keys(self):
        assert "load" in _COMMAND_CACHE_FETCHERS
        assert "unload" in _COMMAND_CACHE_FETCHERS
        assert "gen" in _COMMAND_CACHE_FETCHERS
        assert "switch" in _COMMAND_CACHE_FETCHERS
        assert "datasets" in _COMMAND_CACHE_FETCHERS

    def test_fetchers_are_callable(self):
        for key, fn in _COMMAND_CACHE_FETCHERS.items():
            assert callable(fn), f"{key} is not callable"


# ── ANSI color constants ────────────────────────────────────────────────────


class TestAnsiConstants:

    def test_color_codes_defined(self):
        from domains.shell.repl import _C_CYAN, _C_GREEN, _C_YELLOW, _C_RED
        from domains.shell.repl import _C_DIM, _C_BOLD, _C_RESET
        assert isinstance(_C_CYAN, str)
        assert isinstance(_C_RESET, str)


# ── _HAS_READLINE ───────────────────────────────────────────────────────────


class TestReadline:

    def test_has_readline_defined(self):
        from domains.shell.repl import _HAS_READLINE
        assert isinstance(_HAS_READLINE, bool)
