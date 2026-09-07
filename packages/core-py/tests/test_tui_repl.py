"""Tests for shell.tui_repl — TuiIo, _complete_path, constants."""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock

from domains.shell.tui_repl import TuiIo, _complete_path, _STYLE_PAIRS, _P_LOG_INFO
from domains.shell.surface import TextSurface


# ── _complete_path ──────────────────────────────────────────────────────────


class TestCompletePath:

    def test_empty(self):
        result = _complete_path("")
        assert isinstance(result, list)

    def test_home_dir(self):
        result = _complete_path("~/")
        assert isinstance(result, list)

    def test_nonexistent(self):
        result = _complete_path("/nonexistent/path/xyz")
        assert result == []

    def test_dot(self):
        result = _complete_path(".")
        assert isinstance(result, list)
        assert len(result) > 0


# ── _STYLE_PAIRS ───────────────────────────────────────────────────────────


class TestStylePairs:

    def test_has_all_styles(self):
        assert _P_LOG_INFO in _STYLE_PAIRS.values()


# ── TuiIo ──────────────────────────────────────────────────────────────────


class TestTuiIo:

    def test_init(self):
        surface = TextSurface()
        io = TuiIo(surface)
        assert io._surface is surface

    def test_write(self):
        surface = TextSurface()
        io = TuiIo(surface)
        io.write("hello")
        rendered = surface.render(10, 0)
        assert len(rendered) > 0

    def test_write_with_end(self):
        surface = TextSurface()
        io = TuiIo(surface)
        io.write("hello", end="")
        rendered = surface.render(10, 0)
        assert len(rendered) > 0

    def test_flush(self):
        surface = TextSurface()
        io = TuiIo(surface)
        io.flush()  # Should not raise

    def test_read_raises(self):
        surface = TextSurface()
        io = TuiIo(surface)
        with pytest.raises(NotImplementedError):
            io.read("prompt: ")


# ── TuiRepl constants ──────────────────────────────────────────────────────


class TestTuiReplConstants:

    def test_console_ratio(self):
        from domains.shell.tui_repl import TuiRepl
        assert TuiRepl.CONSOLE_RATIO == 0.3

    def test_console_min(self):
        from domains.shell.tui_repl import TuiRepl
        assert TuiRepl.CONSOLE_MIN == 4

    def test_output_min(self):
        from domains.shell.tui_repl import TuiRepl
        assert TuiRepl.OUTPUT_MIN == 6


# ── _read_escape_remainder ──────────────────────────────────────────────────


class TestReadEscapeRemainder:

    def test_no_more_bytes(self):
        from domains.shell.tui_repl import _read_escape_remainder
        mock_stdscr = MagicMock()
        mock_stdscr.getch.return_value = -1
        result = _read_escape_remainder(mock_stdscr, {})
        assert result is None

    def test_ctrl_sequence(self):
        from domains.shell.tui_repl import _read_escape_remainder
        mock_stdscr = MagicMock()
        mock_stdscr.getch.side_effect = [ord("["), ord("5"), ord("C")]
        result = _read_escape_remainder(mock_stdscr, {})
        assert result == "seq:ctrl-right"

    def test_ctrl_left(self):
        from domains.shell.tui_repl import _read_escape_remainder
        mock_stdscr = MagicMock()
        mock_stdscr.getch.side_effect = [ord("["), ord("5"), ord("D")]
        result = _read_escape_remainder(mock_stdscr, {})
        assert result == "seq:ctrl-left"

    def test_alt_key(self):
        from domains.shell.tui_repl import _read_escape_remainder
        mock_stdscr = MagicMock()
        mock_stdscr.getch.side_effect = [ord("f")]
        result = _read_escape_remainder(mock_stdscr, {"f": "find"})
        assert result == "alt:find"

    def test_alt_key_no_mapping(self):
        from domains.shell.tui_repl import _read_escape_remainder
        mock_stdscr = MagicMock()
        mock_stdscr.getch.side_effect = [ord("x")]
        result = _read_escape_remainder(mock_stdscr, {})
        assert result is None
