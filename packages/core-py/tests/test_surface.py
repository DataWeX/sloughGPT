"""Tests for shell.surface — TextSurface, LogSurface, strip_ansi, clip."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from domains.shell.surface import (
    TextSurface,
    LogSurface,
    RenderLine,
    strip_ansi,
    clip,
    STYLE_INFO,
    STYLE_WARN,
    STYLE_ERROR,
)


# ── strip_ansi ────────────────────────────────────────────────────────────


class TestStripAnsi:

    def test_strips_csi(self):
        assert strip_ansi("\033[31mred\033[0m") == "red"

    def test_strips_complex(self):
        assert strip_ansi("\033[1;32mbold green\033[0m") == "bold green"

    def test_no_ansi(self):
        assert strip_ansi("plain text") == "plain text"

    def test_empty(self):
        assert strip_ansi("") == ""

    def test_only_ansi(self):
        assert strip_ansi("\033[2J\033[H") == ""


# ── clip ──────────────────────────────────────────────────────────────────


class TestClip:

    def test_no_clip_needed(self):
        assert clip("hello", 10) == "hello"

    def test_exact_width(self):
        assert clip("hello", 5) == "hello"

    def test_truncates_with_ellipsis(self):
        result = clip("hello world", 8)
        assert len(result) <= 8
        assert result.endswith("\u2026")

    def test_width_zero(self):
        assert clip("hello", 0) == ""

    def test_width_negative(self):
        assert clip("hello", -1) == ""

    def test_short_width(self):
        result = clip("hello", 2)
        assert len(result) <= 2
        assert not result.endswith("\u2026")


# ── TextSurface ───────────────────────────────────────────────────────────


class TestTextSurface:

    def test_write_single_line(self):
        s = TextSurface()
        s.write("hello\n")
        # After newline, _partial becomes ""
        assert "hello" in s.capture

    def test_write_multiple_lines(self):
        s = TextSurface()
        s.write("line1\nline2\n")
        capture = s.capture
        assert "line1" in capture
        assert "line2" in capture

    def test_write_partial_line(self):
        s = TextSurface()
        s.write("hello", end="")
        s.write(" world\n")
        capture = s.capture
        assert "hello world" in capture

    def test_write_strips_ansi(self):
        s = TextSurface()
        s.write("\033[31mred\033[0m\n")
        assert "red" in s.capture

    def test_clear(self):
        s = TextSurface()
        s.write("hello\n")
        s.clear()
        assert s.capture == []

    def test_render_returns_last_n_lines(self):
        s = TextSurface()
        for i in range(10):
            s.write(f"line{i}\n")
        lines = s.render(3)
        # Render returns last 3 lines including empty partial
        assert len(lines) >= 3
        # The last rendered line may be empty partial; check that real lines are there
        texts = [l.text for l in lines if l.text]
        assert "line9" in texts

    def test_render_with_offset(self):
        s = TextSurface()
        for i in range(10):
            s.write(f"line{i}\n")
        lines = s.render(3, offset=2)
        assert len(lines) >= 3
        texts = [l.text for l in lines if l.text]
        assert any("line" in t for t in texts)

    def test_render_clips_to_width(self):
        s = TextSurface()
        s.set_width(5)
        s.write("hello world\n")
        lines = s.render(1)
        assert len(lines[0].text) <= 5

    def test_render_empty(self):
        s = TextSurface()
        assert s.render(5) == []

    def test_render_zero_rows(self):
        s = TextSurface()
        s.write("hello\n")
        assert s.render(0) == []

    def test_set_width(self):
        s = TextSurface()
        s.set_width(40)
        assert s._width == 40

    def test_set_width_minimum(self):
        s = TextSurface()
        s.set_width(0)
        assert s._width == 1

    def test_maxlen_enforced(self):
        s = TextSurface()
        for i in range(3000):
            s.write(f"line{i}\n")
        assert len(s.capture) <= 2000


# ── LogSurface ────────────────────────────────────────────────────────────


class TestLogSurface:

    def _make_buffer(self, entries):
        buf = MagicMock()
        buf.get.return_value = entries
        return buf

    def test_render_empty(self):
        buf = self._make_buffer([])
        surface = LogSurface(buf)
        assert surface.render(10) == []

    def test_render_zero_rows(self):
        buf = self._make_buffer([])
        surface = LogSurface(buf)
        assert surface.render(0) == []

    def test_render_entries(self):
        entry = MagicMock()
        entry.timestamp = time.time()
        entry.level = "INFO"
        entry.source = "test"
        entry.message = "hello"
        buf = self._make_buffer([entry])
        surface = LogSurface(buf)
        lines = surface.render(10)
        assert len(lines) == 1
        assert "hello" in lines[0].text
        assert lines[0].style == STYLE_INFO

    def test_render_error_style(self):
        entry = MagicMock()
        entry.timestamp = time.time()
        entry.level = "ERROR"
        entry.source = "test"
        entry.message = "fail"
        buf = self._make_buffer([entry])
        surface = LogSurface(buf)
        lines = surface.render(10)
        assert lines[0].style == STYLE_ERROR

    def test_render_warning_style(self):
        entry = MagicMock()
        entry.timestamp = time.time()
        entry.level = "WARNING"
        entry.source = "test"
        entry.message = "warn"
        buf = self._make_buffer([entry])
        surface = LogSurface(buf)
        lines = surface.render(10)
        assert lines[0].style == STYLE_WARN

    def test_render_with_offset(self):
        entries = []
        for i in range(10):
            e = MagicMock()
            e.timestamp = time.time()
            e.level = "INFO"
            e.source = "test"
            e.message = f"msg{i}"
            entries.append(e)
        buf = self._make_buffer(entries)
        surface = LogSurface(buf)
        lines = surface.render(3, offset=2)
        assert len(lines) == 3

    def test_render_clips_to_width(self):
        entry = MagicMock()
        entry.timestamp = time.time()
        entry.level = "INFO"
        entry.source = "test"
        entry.message = "x" * 100
        buf = self._make_buffer([entry])
        surface = LogSurface(buf)
        surface.set_width(20)
        lines = surface.render(10)
        assert len(lines[0].text) <= 20

    def test_set_width(self):
        buf = self._make_buffer([])
        surface = LogSurface(buf)
        surface.set_width(120)
        assert surface._width == 120
