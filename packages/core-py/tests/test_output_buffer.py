"""
Tests for the shared OutputBuffer (moved from shell/stdio.py to infrastructure).
"""
import logging
import pytest
from domains.infrastructure.output_buffer import (
    OutputLine,
    OutputBuffer,
    _Subscriber,
    BufferLogHandler,
    get_server_buffer,
    install_log_bridge,
)


class TestOutputLine:
    def test_render_plain(self):
        line = OutputLine("hello")
        assert line.render() == "hello"

    def test_render_with_style(self):
        line = OutputLine("hello", style="\033[31m")
        result = line.render(color=True)
        assert "\033[31m" in result
        assert "hello" in result

    def test_render_no_color(self):
        line = OutputLine("hello", style="\033[31m")
        assert line.render(color=False) == "hello"

    def test_to_dict(self):
        line = OutputLine("hello", level="error", source="test")
        d = line.to_dict()
        assert d["text"] == "hello"
        assert d["level"] == "error"
        assert d["source"] == "test"
        assert "ts" in d

    def test_to_sse(self):
        line = OutputLine("hello")
        s = line.to_sse()
        assert '"text": "hello"' in s


class TestOutputBuffer:
    def test_append_and_count(self):
        buf = OutputBuffer(max_lines=100)
        buf.append_text("a")
        buf.append_text("b")
        assert buf.count == 2

    def test_capacity_eviction(self):
        buf = OutputBuffer(max_lines=5)
        for i in range(10):
            buf.append_text(f"line{i}")
        assert buf.count == 5
        assert buf.tail(1)[0].text == "line9"

    def test_clear(self):
        buf = OutputBuffer()
        buf.append_text("a")
        buf.clear()
        assert buf.count == 0
        assert buf.seq == 0

    def test_seq_increments(self):
        buf = OutputBuffer()
        buf.append_text("a")
        assert buf.seq == 1
        buf.append_text("b")
        assert buf.seq == 2

    def test_tail(self):
        buf = OutputBuffer()
        for i in range(10):
            buf.append_text(f"line{i}")
        tail = buf.tail(3)
        assert [l.text for l in tail] == ["line7", "line8", "line9"]

    def test_tail_dicts(self):
        buf = OutputBuffer()
        buf.append_text("hello")
        dicts = buf.tail_dicts(1)
        assert dicts[0]["text"] == "hello"

    def test_subscribe_receives_lines(self):
        buf = OutputBuffer()
        sub = buf.subscribe("test")
        buf.append_text("before")
        buf.append_text("after")
        lines = sub.read(timeout=0.1)
        assert len(lines) == 2
        assert lines[0].text == "before"
        buf.unsubscribe("test")

    def test_unsubscribe(self):
        buf = OutputBuffer()
        sub = buf.subscribe("test")
        buf.unsubscribe("test")
        assert "test" not in buf._subscribers

    def test_viewport_scroll(self):
        buf = OutputBuffer()
        for i in range(20):
            buf.append_text(f"line{i}")
        buf.set_viewport(5)
        assert len(buf.visible_lines) == 5
        buf.scroll(3)
        assert buf.visible_lines[0].text == "line3"


class TestBufferLogHandler:
    def test_routes_logging_to_buffer(self):
        buf = OutputBuffer()
        handler = BufferLogHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.getLogger("test_buffer_handler")
        logger.addHandler(handler)
        logger.warning("test message")
        logger.removeHandler(handler)
        lines = buf.tail(5)
        assert any("test message" in l.text for l in lines)


class TestSingletons:
    def test_get_server_buffer_returns_same_instance(self):
        b1 = get_server_buffer()
        b2 = get_server_buffer()
        assert b1 is b2

    def test_install_log_bridge(self):
        buf = OutputBuffer()
        handler = install_log_bridge(buffer=buf)
        assert isinstance(handler, BufferLogHandler)
        # Verify it's added to root logger
        assert handler in logging.root.handlers
        logging.root.handlers.remove(handler)
