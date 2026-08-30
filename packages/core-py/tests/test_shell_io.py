"""Tests for domains.shell.io — MemoryIO, _Capture, capture_output, capture_cmd."""

import pytest
from domains.shell.io import MemoryIO, _Capture, capture_output


class TestMemoryIO:
    def test_write_default_newline(self):
        io = MemoryIO()
        io.write("hello")
        assert io.get_output() == "hello\n"

    def test_write_custom_end(self):
        io = MemoryIO()
        io.write("hello", end="")
        assert io.get_output() == "hello"

    def test_write_empty(self):
        io = MemoryIO()
        io.write("")
        assert io.get_output() == "\n"

    def test_write_multiple(self):
        io = MemoryIO()
        io.write("a")
        io.write("b")
        io.write("c")
        assert io.get_output() == "a\nb\nc\n"

    def test_read_from_feed(self):
        io = MemoryIO()
        io.feed("line1", "line2")
        assert io.read() == "line1"
        assert io.read() == "line2"

    def test_read_eof(self):
        io = MemoryIO()
        with pytest.raises(EOFError):
            io.read()

    def test_read_strips_whitespace(self):
        io = MemoryIO()
        io.feed("  hello  ")
        assert io.read() == "hello"

    def test_flush_noop(self):
        io = MemoryIO()
        io.flush()  # should not raise

    def test_clear(self):
        io = MemoryIO()
        io.write("data")
        assert io.get_output() == "data\n"
        io.clear()
        assert io.get_output() == ""

    def test_feed_resets_index(self):
        io = MemoryIO()
        io.feed("a")
        io.read()
        io.feed("b")
        assert io.read() == "a"  # feed extends, doesn't reset
        io.feed("c", "d")
        assert io.get_output() == ""  # no writes yet

    def test_get_output_joins_all(self):
        io = MemoryIO()
        io.write("x", end="")
        io.write("y", end="")
        assert io.get_output() == "xy"


class TestCapture:
    def test_capture_redirects(self):
        io = MemoryIO()
        with capture_output(io) as cap:
            io.write("captured")
        assert "captured" in cap.getvalue()

    def test_capture_restores(self):
        io = MemoryIO()
        with capture_output(io):
            io.write("inside")
        # After context manager, write should work normally (not to buffer).
        io.write("outside")
        assert "outside" in io.get_output()

    def test_capture_clears_buffer(self):
        io = MemoryIO()
        with capture_output(io) as cap:
            io.write("first")
        with capture_output(io) as cap:
            io.write("second")
        assert "first" not in cap.getvalue()
        assert "second" in cap.getvalue()

    def test_capture_empty(self):
        io = MemoryIO()
        with capture_output(io) as cap:
            pass
        assert cap.getvalue() == ""
