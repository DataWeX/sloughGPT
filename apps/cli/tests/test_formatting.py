"""Tests for apps/cli/src/utils/formatting.py — pure formatting functions."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestFormatSize:
    def test_zero_bytes(self):
        from utils.formatting import format_size
        assert format_size(0) == "0 B"

    def test_bytes(self):
        from utils.formatting import format_size
        assert format_size(512) == "512 B"

    def test_kilobytes_integer(self):
        from utils.formatting import format_size
        assert format_size(1024) == "1 KB"

    def test_kilobytes_fractional(self):
        from utils.formatting import format_size
        assert format_size(1536) == "1.5 KB"

    def test_megabytes(self):
        from utils.formatting import format_size
        assert format_size(1048576) == "1 MB"

    def test_megabytes_fractional(self):
        from utils.formatting import format_size
        assert format_size(1572864) == "1.5 MB"

    def test_gigabytes(self):
        from utils.formatting import format_size
        assert format_size(1073741824) == "1 GB"

    def test_negative_returns_zero(self):
        from utils.formatting import format_size
        assert format_size(-100) == "0 B"

    def test_large_terabytes(self):
        from utils.formatting import format_size
        result = format_size(1099511627776)
        assert "TB" in result

    def test_petabytes(self):
        from utils.formatting import format_size
        result = format_size(1125899906842624)
        assert "PB" in result


class TestFormatTime:
    def test_zero(self):
        from utils.formatting import format_time
        assert format_time(0) == "0ms"

    def test_milliseconds(self):
        from utils.formatting import format_time
        assert format_time(0.5) == "500ms"

    def test_seconds(self):
        from utils.formatting import format_time
        assert format_time(5.3) == "5.3s"

    def test_minutes(self):
        from utils.formatting import format_time
        assert format_time(65) == "1m 5s"

    def test_exact_minute(self):
        from utils.formatting import format_time
        assert format_time(60) == "1m"

    def test_hours(self):
        from utils.formatting import format_time
        assert format_time(3661) == "1h 1m"

    def test_negative_returns_zero(self):
        from utils.formatting import format_time
        assert format_time(-10) == "0s"


class TestFormatNumber:
    def test_zero(self):
        from utils.formatting import format_number
        assert format_number(0) == "0"

    def test_thousands(self):
        from utils.formatting import format_number
        assert format_number(1234) == "1,234"

    def test_millions(self):
        from utils.formatting import format_number
        assert format_number(1234567) == "1,234,567"

    def test_negative(self):
        from utils.formatting import format_number
        assert format_number(-1234) == "-1,234"


class TestTruncate:
    def test_no_truncation_needed(self):
        from utils.formatting import truncate
        assert truncate("hello", 10) == "hello"

    def test_exact_length(self):
        from utils.formatting import truncate
        assert truncate("hello", 5) == "hello"

    def test_truncation_with_suffix(self):
        from utils.formatting import truncate
        result = truncate("hello world", 8)
        assert result == "hello..."

    def test_custom_suffix(self):
        from utils.formatting import truncate
        result = truncate("hello world", 8, suffix="…")
        assert result == "hello w…"

    def test_very_short_max(self):
        from utils.formatting import truncate
        result = truncate("hello", 3)
        assert len(result) == 3
        assert result.endswith("...")


class TestWrapText:
    def test_empty_string(self):
        from utils.formatting import wrap_text
        assert wrap_text("") == []

    def test_single_word(self):
        from utils.formatting import wrap_text
        assert wrap_text("hello") == ["hello"]

    def test_wraps_at_width(self):
        from utils.formatting import wrap_text
        result = wrap_text("hello world foo bar", width=12)
        assert all(len(line) <= 12 for line in result)

    def test_preserves_all_words(self):
        from utils.formatting import wrap_text
        text = "the quick brown fox jumps over the lazy dog"
        result = wrap_text(text, width=20)
        combined = " ".join(result)
        assert combined == text


class TestIndent:
    def test_single_line(self):
        from utils.formatting import indent
        assert indent("hello", 2) == "  hello"

    def test_multiple_lines(self):
        from utils.formatting import indent
        result = indent("line1\nline2", 4)
        assert result == "    line1\n    line2"

    def test_zero_indent(self):
        from utils.formatting import indent
        assert indent("hello", 0) == "hello"


class TestPad:
    def test_left_pad(self):
        from utils.formatting import pad
        assert pad("hi", 5) == "hi   "

    def test_right_pad(self):
        from utils.formatting import pad
        assert pad("hi", 5, alignment="right") == "   hi"

    def test_center_pad(self):
        from utils.formatting import pad
        assert pad("hi", 6, alignment="center") == "  hi  "
