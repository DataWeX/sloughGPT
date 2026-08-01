"""Tests for domains/shell/surface.py — content surfaces."""

from domains.shell.log_buffer import LogBuffer, LogEntry
from domains.shell.surface import (
    LogSurface,
    RenderLine,
    STYLE_ERROR,
    STYLE_INFO,
    STYLE_WARN,
    TextSurface,
    clip,
    strip_ansi,
)


class TestClip:
    def test_short_text_unchanged(self):
        assert clip("hello", 10) == "hello"

    def test_long_text_truncated(self):
        assert clip("hello world", 5) == "hello"

    def test_zero_width(self):
        assert clip("hello", 0) == ""


class TestStripAnsi:
    def test_plain_text_unchanged(self):
        assert strip_ansi("hello") == "hello"

    def test_sgr_colors_stripped(self):
        assert strip_ansi("\x1b[36mDevelopment:\x1b[0m") == "Development:"

    def test_multiple_sequences(self):
        assert strip_ansi("\x1b[1mBold\x1b[0m and \x1b[32mgreen\x1b[0m") == "Bold and green"

    def test_parameterised_sgr(self):
        assert strip_ansi("\x1b[38;5;208morange\x1b[0m") == "orange"

    def test_cursor_position_sequence(self):
        assert strip_ansi("x\x1b[94Gy") == "xy"

    def test_charset_selection(self):
        assert strip_ansi("\x1b(Bascii\x1b(B") == "ascii"

    def test_save_restore_cursor(self):
        assert strip_ansi("\x1b[s\x1b[1B\x1b[u") == ""

    def test_mixed_content(self):
        assert strip_ansi("\x1b[32m  \u2139 \x1b[0m\x1b[2m[info] \x1b[0mmsg") == "  \u2139 [info] msg"


class TestTextSurface:
    def test_write_appends_lines(self):
        s = TextSurface()
        s.write("line one")
        assert s.capture == ["line one"]

    def test_write_with_end(self):
        s = TextSurface()
        s.write("a", end="")
        s.write("b", end="")
        assert s.capture == ["ab"]

    def test_empty_write_appends_blank(self):
        s = TextSurface()
        s.write("")
        assert s.capture == [""]

    def test_clear_empties(self):
        s = TextSurface()
        s.write("data")
        s.clear()
        assert s.capture == []

    def test_render_clips_to_width(self):
        s = TextSurface()
        s.set_width(5)
        s.write("hello world")
        lines = s.render(10)
        assert lines[0].text == "hello"

    def test_render_tails_only(self):
        s = TextSurface()
        for i in range(10):
            s.write(f"line {i}")
        lines = s.render(3)
        assert len(lines) == 3
        assert lines[-1].text == "line 9"

    def test_write_after_clear_starts_fresh(self):
        s = TextSurface()
        s.write("stale")
        s.clear()
        s.write("fresh")
        assert s.capture == ["fresh"]

    def test_render_zero_rows(self):
        s = TextSurface()
        s.write("x")
        assert s.render(0) == []

    def test_render_offset_scrolls_back(self):
        s = TextSurface()
        for i in range(10):
            s.write(f"line {i}")
        lines = s.render(3, offset=4)
        assert len(lines) == 3
        assert lines[-1].text == "line 5"
        assert lines[0].text == "line 3"

    def test_render_offset_past_start(self):
        s = TextSurface()
        for i in range(4):
            s.write(f"line {i}")
        lines = s.render(3, offset=50)
        assert len(lines) == 3
        assert lines[0].text == "line 0"

    def test_render_offset_zero_tails(self):
        s = TextSurface()
        for i in range(10):
            s.write(f"line {i}")
        lines = s.render(3, offset=0)
        assert lines[0].text == "line 7"

    def test_render_default_style_none(self):
        s = TextSurface()
        s.write("plain")
        assert s.render(1)[0].style is None

    def test_write_strips_ansi(self):
        s = TextSurface()
        s.write("\x1b[36mDevelopment:\x1b[0m")
        assert s.capture == ["Development:"]

    def test_write_strips_ansi_across_partial_lines(self):
        s = TextSurface()
        s.write("\x1b[36mDev", end="")
        s.write("elopment\x1b[0m", end="")
        s.write(":", end="")
        assert s.capture == ["Development:"]


class TestLogSurface:
    def _buffer(self):
        b = LogBuffer()
        b.append(LogEntry(timestamp=1.0, level="INFO", source="kernel", message="boot ok"))
        b.append(LogEntry(timestamp=2.0, level="WARNING", source="runtime", message="orphan killed"))
        b.append(LogEntry(timestamp=3.0, level="ERROR", source="server", message="boom"))
        return b

    def test_renders_formatted_entries(self):
        s = LogSurface(self._buffer())
        s.set_width(80)
        lines = s.render(10)
        assert len(lines) == 3
        assert "INFO" in lines[0].text
        assert "kernel" in lines[0].text
        assert "boot ok" in lines[0].text

    def test_level_styles(self):
        s = LogSurface(self._buffer())
        s.set_width(80)
        lines = s.render(10)
        assert lines[0].style == STYLE_INFO
        assert lines[1].style == STYLE_WARN
        assert lines[2].style == STYLE_ERROR

    def test_render_tails(self):
        s = LogSurface(self._buffer())
        s.set_width(80)
        lines = s.render(2)
        assert len(lines) == 2
        assert "boom" in lines[-1].text

    def test_render_zero_rows(self):
        s = LogSurface(self._buffer())
        assert s.render(0) == []

    def test_render_offset_scrolls_back(self):
        s = LogSurface(self._buffer())
        s.set_width(80)
        lines = s.render(1, offset=1)
        assert len(lines) == 1
        assert "orphan" in lines[0].text

    def test_render_offset_zero_tails(self):
        s = LogSurface(self._buffer())
        s.set_width(80)
        lines = s.render(1, offset=0)
        assert "boom" in lines[0].text

    def test_empty_buffer(self):
        s = LogSurface(LogBuffer())
        s.set_width(80)
        assert s.render(10) == []

    def test_clips_long_lines(self):
        b = LogBuffer()
        b.append(LogEntry(timestamp=1.0, level="INFO", source="s", message="x" * 100))
        s = LogSurface(b)
        s.set_width(10)
        lines = s.render(1)
        assert len(lines[0].text) <= 10
