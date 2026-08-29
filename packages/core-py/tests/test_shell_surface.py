"""Tests for domains.shell.surface — surface content rendering."""

import time

import pytest
from domains.shell.surface import (
    strip_ansi, clip, RenderLine, TextSurface, LogSurface,
    STYLE_INFO, STYLE_WARN, STYLE_ERROR, STYLE_DEBUG, STYLE_CRITICAL,
)
from domains.shell.log_buffer import LogBuffer, LogEntry


class TestStripAnsi:
    def test_plain_text_unchanged(self):
        assert strip_ansi("hello") == "hello"

    def test_strips_color_codes(self):
        text = "\x1b[31mred\x1b[0m"
        assert strip_ansi(text) == "red"

    def test_strips_multiple_codes(self):
        text = "\x1b[1m\x1b[32mbold green\x1b[0m"
        assert strip_ansi(text) == "bold green"

    def test_empty_string(self):
        assert strip_ansi("") == ""


class TestClip:
    def test_short_text_unchanged(self):
        assert clip("hello", 10) == "hello"

    def test_exact_width(self):
        assert clip("hello", 5) == "hello"

    def test_truncates_with_ellipsis(self):
        assert clip("hello world", 8) == "hello w\u2026"

    def test_width_zero(self):
        assert clip("hello", 0) == ""

    def test_width_negative(self):
        assert clip("hello", -5) == ""

    def test_very_narrow(self):
        assert clip("hello", 2) == "he"


class TestRenderLine:
    def test_default_style(self):
        rl = RenderLine(text="hello")
        assert rl.text == "hello"
        assert rl.style is None

    def test_with_style(self):
        rl = RenderLine(text="error", style=STYLE_ERROR)
        assert rl.style == STYLE_ERROR


class TestTextSurface:
    def test_empty_render(self):
        s = TextSurface()
        assert s.render(10) == []

    def test_write_and_render(self):
        s = TextSurface()
        s.write("hello")  # default end="\n"
        lines = s.render(10)
        assert len(lines) == 1
        assert lines[0].text == "hello"

    def test_multiple_writes(self):
        s = TextSurface()
        s.write("line1")
        s.write("line2")
        s.write("line3")
        lines = s.render(10)
        assert [l.text for l in lines] == ["line1", "line2", "line3"]

    def test_render_limit(self):
        s = TextSurface()
        for i in range(10):
            s.write(f"line{i}")
        lines = s.render(3)
        assert len(lines) == 3
        assert lines[0].text == "line7"
        assert lines[2].text == "line9"

    def test_render_with_offset(self):
        s = TextSurface()
        for i in range(5):
            s.write(f"line{i}")
        lines = s.render(2, offset=0)
        assert [l.text for l in lines] == ["line3", "line4"]
        lines = s.render(2, offset=2)
        assert [l.text for l in lines] == ["line1", "line2"]

    def test_write_without_newline_end(self):
        s = TextSurface()
        s.write("partial", end="")
        lines = s.render(10)
        assert len(lines) == 1
        assert lines[0].text == "partial"

    def test_clear(self):
        s = TextSurface()
        s.write("hello")
        s.clear()
        assert s.render(10) == []

    def test_capture(self):
        s = TextSurface()
        s.write("a")
        s.write("b")
        assert s.capture == ["a", "b"]

    def test_capture_partial(self):
        s = TextSurface()
        s.write("a")
        s.write("partial", end="")
        assert s.capture == ["a", "partial"]

    def test_strips_ansi(self):
        s = TextSurface()
        s.write("\x1b[31mred\x1b[0m")
        lines = s.render(10)
        assert lines[0].text == "red"

    def test_set_width(self):
        s = TextSurface()
        s.set_width(5)
        s.write("hello world")
        lines = s.render(10)
        assert len(lines[0].text) <= 5


class TestLogSurface:
    def _make_entry(self, level="INFO", message="hello"):
        return LogEntry(timestamp=time.time(), level=level, source="test", message=message)

    def test_empty_render(self):
        buf = LogBuffer(max_size=100)
        s = LogSurface(buf)
        assert s.render(10) == []

    def test_render_with_entries(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(self._make_entry("INFO", "hello"))
        buf._entries.append(self._make_entry("ERROR", "fail"))
        s = LogSurface(buf)
        lines = s.render(10)
        assert len(lines) == 2
        assert "INFO" in lines[0].text
        assert "ERROR" in lines[1].text

    def test_level_styles(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(self._make_entry("WARNING", "warn"))
        s = LogSurface(buf)
        lines = s.render(10)
        assert lines[0].style == STYLE_WARN

    def test_render_limit(self):
        buf = LogBuffer(max_size=100)
        for i in range(5):
            buf._entries.append(self._make_entry("INFO", f"msg{i}"))
        s = LogSurface(buf)
        lines = s.render(2)
        assert len(lines) == 2

    def test_set_width(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(self._make_entry("INFO", "a" * 200))
        s = LogSurface(buf)
        s.set_width(20)
        lines = s.render(10)
        assert len(lines[0].text) <= 20

    def test_render_offset(self):
        buf = LogBuffer(max_size=100)
        for i in range(5):
            buf._entries.append(self._make_entry("INFO", f"msg{i}"))
        s = LogSurface(buf)
        lines = s.render(2, offset=0)
        assert len(lines) == 2
        lines = s.render(2, offset=2)
        assert len(lines) == 2
