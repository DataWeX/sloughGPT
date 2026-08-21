"""Tests for output_buffer — structured log capture, ring buffer, subscribers."""

import asyncio
import json
import logging
import sys
import threading

import pytest

from domains.infrastructure.output_buffer import (
    BufferLogHandler,
    OutputBuffer,
    OutputLine,
    _TeeWriter,
    _norm_level,
    get_server_buffer,
    install_log_bridge,
    install_stdio_bridge,
)


class TestOutputLine:
    def test_defaults_autoset_timestamp(self):
        line = OutputLine(text="hi")
        assert line.level == "info"
        assert line.source == ""
        assert line.tag == ""
        assert line.context == {}
        assert line.timestamp > 0

    def test_render_plain(self):
        line = OutputLine(text="hi")
        assert line.render() == "hi"

    def test_render_indent(self):
        line = OutputLine(text="hi", indent=3)
        assert line.render() == "   hi"

    def test_render_style_colored(self):
        line = OutputLine(text="hi", style="\033[31m")
        assert line.render() == "\033[31mhi\033[0m"

    def test_render_style_no_color(self):
        line = OutputLine(text="hi", style="\033[31m")
        assert line.render(color=False) == "hi"

    def test_to_dict_minimal(self):
        d = OutputLine(text="hi").to_dict()
        assert set(d) == {"text", "level", "source", "ts"}

    def test_to_dict_with_tag_and_context(self):
        d = OutputLine(text="hi", tag="REQ", context={"method": "GET"}).to_dict()
        assert d["tag"] == "REQ"
        assert d["context"] == {"method": "GET"}

    def test_to_sse_roundtrip(self):
        line = OutputLine(text="hi", tag="REQ", context={"method": "GET"})
        assert json.loads(line.to_sse()) == line.to_dict()

    def test_repr_includes_tag(self):
        assert "[REQ]" in repr(OutputLine(text="x", tag="REQ"))
        assert "[REQ]" not in repr(OutputLine(text="x"))


class TestOutputBuffer:
    def test_append_and_count(self):
        buf = OutputBuffer()
        buf.append(OutputLine(text="a"))
        buf.append(OutputLine(text="b"))
        assert buf.count == 2
        assert buf.seq == 2
        assert [l.text for l in buf.lines] == ["a", "b"]

    def test_append_text_returns_line(self):
        buf = OutputBuffer()
        line = buf.append_text("hello", tag="START")
        assert line.text == "hello"
        assert line.tag == "START"
        assert buf.count == 1

    def test_append_log_fields(self):
        buf = OutputBuffer()
        line = buf.append_log("boom", level="error", source="slo.x", tag="ERR",
                             context={"code": 1})
        assert line.level == "error"
        assert line.source == "slo.x"
        assert line.tag == "ERR"
        assert line.context == {"code": 1}

    def test_tail_returns_last_n(self):
        buf = OutputBuffer()
        for i in range(10):
            buf.append_text(f"line-{i}")
        assert [l.text for l in buf.tail(3)] == ["line-7", "line-8", "line-9"]
        assert len(buf.tail(100)) == 10

    def test_tail_dicts(self):
        buf = OutputBuffer()
        buf.append_text("x", level="warn")
        d = buf.tail_dicts(1)[0]
        assert d["text"] == "x"
        assert d["level"] == "warn"

    def test_clear_resets(self):
        buf = OutputBuffer()
        buf.append_text("a")
        buf.clear()
        assert buf.count == 0
        assert buf.seq == 0

    def test_ring_capacity_trimmed(self):
        buf = OutputBuffer(max_lines=5)
        for i in range(10):
            buf.append_text(f"line-{i}")
        assert buf.count == 5
        assert [l.text for l in buf.lines] == ["line-5", "line-6", "line-7", "line-8", "line-9"]

    def test_lines_returns_underlying_list(self):
        buf = OutputBuffer()
        buf.append_text("a")
        assert buf.lines is buf._lines


class TestViewport:
    def test_set_viewport_clamps_positive(self):
        buf = OutputBuffer()
        buf.set_viewport(0)
        assert buf._view_height == 1

    def test_visible_lines_with_viewport(self):
        buf = OutputBuffer()
        for i in range(10):
            buf.append_text(f"line-{i}")
        buf.set_viewport(3)
        buf.scroll_to_bottom()
        assert [l.text for l in buf.visible_lines] == ["line-7", "line-8", "line-9"]

    def test_scroll_up(self):
        buf = OutputBuffer()
        for i in range(10):
            buf.append_text(f"line-{i}")
        buf.set_viewport(3)
        buf.scroll_to_bottom()
        buf.scroll(-1)
        assert [l.text for l in buf.visible_lines] == ["line-6", "line-7", "line-8"]

    def test_scroll_clamped_at_bottom(self):
        buf = OutputBuffer()
        for i in range(10):
            buf.append_text(f"line-{i}")
        buf.set_viewport(3)
        buf.scroll(-100)
        buf.scroll(1000)
        assert [l.text for l in buf.visible_lines] == ["line-7", "line-8", "line-9"]

    def test_scroll_to_bottom(self):
        buf = OutputBuffer()
        for i in range(10):
            buf.append_text(f"line-{i}")
        buf.set_viewport(3)
        buf.scroll(-100)
        buf.scroll_to_bottom()
        assert [l.text for l in buf.visible_lines] == ["line-7", "line-8", "line-9"]


class TestSubscribers:
    def test_subscribe_read_receives_lines(self):
        buf = OutputBuffer()
        sub = buf.subscribe("s1")
        buf.append_text("a")
        buf.append_text("b")
        got = sub.read_all()
        assert [l.text for l in got] == ["a", "b"]

    def test_subscribe_read_clears_pending(self):
        buf = OutputBuffer()
        sub = buf.subscribe("s1")
        buf.append_text("a")
        sub.read_all()
        assert sub.read_all() == []

    def test_subscribe_read_with_timeout_returns_pending(self):
        buf = OutputBuffer()
        sub = buf.subscribe("s1")
        buf.append_text("a")
        got = sub.read(timeout=0.2)
        assert [l.text for l in got] == ["a"]

    def test_unsubscribe_stops_delivery(self):
        buf = OutputBuffer()
        sub = buf.subscribe("s1")
        buf.unsubscribe("s1")
        buf.append_text("a")
        assert sub.read_all() == []

    def test_default_subscriber_name_unique(self):
        buf = OutputBuffer()
        s1 = buf.subscribe()
        s2 = buf.subscribe()
        assert s1.name != s2.name

    def test_async_read_receives_lines(self):
        buf = OutputBuffer()
        sub = buf.subscribe("s1")
        buf.append_text("a")
        buf.append_text("b")

        async def read():
            return await sub.async_read(timeout=0.2)

        got = asyncio.run(read())
        assert [l.text for l in got] == ["a", "b"]

    def test_async_read_timeout_returns_empty(self):
        buf = OutputBuffer()
        sub = buf.subscribe("s1")

        async def read():
            return await sub.async_read(timeout=0.05)

        got = asyncio.run(read())
        assert got == []

    def test_async_read_wakes_on_new_line(self):
        buf = OutputBuffer()
        sub = buf.subscribe("s1")

        async def scenario():
            async def writer():
                await asyncio.sleep(0.05)
                buf.append_text("delayed")

            asyncio.create_task(writer())
            return await sub.async_read(timeout=2.0)

        got = asyncio.run(scenario())
        assert len(got) == 1
        assert got[0].text == "delayed"

    def test_async_read_thread_safe(self):
        buf = OutputBuffer()
        sub = buf.subscribe("s1")

        async def scenario():
            def bg():
                import time
                time.sleep(0.05)
                buf.append_text("from-thread")

            t = threading.Thread(target=bg)
            t.start()
            got = await sub.async_read(timeout=2.0)
            t.join()
            return got

        got = asyncio.run(scenario())
        assert len(got) == 1
        assert got[0].text == "from-thread"

    def test_async_read_clears_pending(self):
        buf = OutputBuffer()
        sub = buf.subscribe("s1")
        buf.append_text("a")

        async def read():
            await sub.async_read(timeout=0.2)
            return await sub.async_read(timeout=0.05)

        got = asyncio.run(read())
        assert got == []

    def test_sync_read_still_works(self):
        buf = OutputBuffer()
        sub = buf.subscribe("s1")
        buf.append_text("a")
        got = sub.read(timeout=0.2)
        assert [l.text for l in got] == ["a"]


class TestBufferLogHandler:
    def test_captures_record(self):
        buf = OutputBuffer()
        handler = BufferLogHandler(buf)
        record = logging.LogRecord("slo.test", logging.WARNING, "x.py", 1,
                                   "warned %s", ("now",), None)
        handler.emit(record)
        line = buf.lines[0]
        assert line.level == "warning"
        assert line.source == "slo.test"
        assert line.text == "warned now"

    def test_captures_extra_fields(self):
        buf = OutputBuffer()
        handler = BufferLogHandler(buf)
        record = logging.LogRecord("slo.test", logging.ERROR, "x.py", 1,
                                   "failed", (), None)
        record.tag = "ERR"
        record.context = {"code": 42}
        record.error_code = "E42"
        handler.emit(record)
        line = buf.lines[0]
        assert line.tag == "ERR"
        assert line.context["code"] == 42
        assert line.context["error_code"] == "E42"

    def test_emit_swallows_errors(self, monkeypatch):
        buf = OutputBuffer()
        handler = BufferLogHandler(buf)
        record = logging.LogRecord("slo.test", logging.INFO, "x.py", 1,
                                   "ok", (), None)
        monkeypatch.setattr(record, "getMessage", lambda: (_ for _ in ()).throw(ValueError()))
        handler.emit(record)
        assert buf.count == 0

    def test_stray_extra_fields_captured_automatically(self):
        buf = OutputBuffer()
        handler = BufferLogHandler(buf)
        record = logging.LogRecord("slo.test", logging.INFO, "x.py", 1,
                                   "trained", (), None)
        record.vocab_size = 512
        record.corpus_size = 128000
        handler.emit(record)
        assert buf.lines[0].context == {"vocab_size": 512, "corpus_size": 128000}

    def test_explicit_context_wins_over_stray(self):
        buf = OutputBuffer()
        handler = BufferLogHandler(buf)
        record = logging.LogRecord("slo.test", logging.INFO, "x.py", 1,
                                   "msg", (), None)
        record.context = {"mode": "explicit"}
        record.mode = "stray"
        handler.emit(record)
        assert buf.lines[0].context == {"mode": "explicit"}

    def test_mixed_tag_context_and_stray_all_surface(self):
        # Production shape from e.g. routers/inference.py request logs.
        buf = OutputBuffer()
        handler = BufferLogHandler(buf)
        record = logging.LogRecord("slo.routers.inference", logging.INFO, "x.py", 1,
                                   "generate", (), None)
        record.tag = "INFO"
        record.context = {"provider": "hf-default"}
        record.elapsed_ms = 511
        record.result = "ok"
        handler.emit(record)
        line = buf.lines[0]
        assert line.tag == "INFO"
        assert line.context["provider"] == "hf-default"
        assert line.context["elapsed_ms"] == 511
        assert line.context["result"] == "ok"


class TestTeeWriter:
    def test_writes_through_to_original(self):
        original = _Capture()
        buf = OutputBuffer()
        w = _TeeWriter(original, buf)
        w.write("hello\n")
        assert original.data == ["hello\n"]

    def test_plain_line_parsed(self):
        buf = OutputBuffer()
        w = _TeeWriter(_Capture(), buf)
        w.write("just some text\n")
        assert buf.lines[0].text == "just some text"
        assert buf.lines[0].source == "stdout"

    def test_structured_line_parsed(self):
        buf = OutputBuffer()
        w = _TeeWriter(_Capture(), buf)
        w.write("12:34:56 ERR [BOOT] slo.startup something failed\n")
        line = buf.lines[0]
        assert line.text == "something failed"
        assert line.level == "error"
        assert line.source == "slo.startup"
        assert line.tag == "BOOT"

    def test_uvicorn_line_parsed(self):
        buf = OutputBuffer()
        w = _TeeWriter(_Capture(), buf)
        w.write("INFO:     Application startup complete.\n")
        line = buf.lines[0]
        assert line.text == "Application startup complete."
        assert line.level == "info"
        assert line.source == "uvicorn"

    def test_system_warning_parsed(self):
        buf = OutputBuffer()
        w = _TeeWriter(_Capture(), buf)
        w.write("(python:1234): GLib-WARNING **: 01:02:03.456: cannot connect\n")
        line = buf.lines[0]
        assert line.text == "cannot connect"
        assert line.level == "warning"
        assert line.source == "system"

    def test_ansi_stripped(self):
        buf = OutputBuffer()
        w = _TeeWriter(_Capture(), buf)
        w.write("\033[31mred text\033[0m\n")
        assert buf.lines[0].text == "red text"

    def test_blank_lines_skipped(self):
        buf = OutputBuffer()
        w = _TeeWriter(_Capture(), buf)
        w.write("   \nnot blank\n")
        assert buf.count == 1

    def test_partial_lines_buffered(self):
        buf = OutputBuffer()
        w = _TeeWriter(_Capture(), buf)
        w.write("part1\npart2")
        assert buf.count == 1
        w.write("\n")
        assert buf.count == 2

    def test_flush_delegates(self):
        class Fake:
            def __init__(self):
                self.flushed = False
            def flush(self):
                self.flushed = True
        f = Fake()
        w = _TeeWriter(f, OutputBuffer())
        w.flush()
        assert f.flushed is True

    def test_attr_delegation(self):
        class Fake:
            encoding = "utf-8"
        w = _TeeWriter(Fake(), OutputBuffer())
        assert w.encoding == "utf-8"


class _Capture:
    def __init__(self):
        self.data = []

    def write(self, data):
        self.data.append(data)

    def flush(self):
        pass


class TestNormLevel:
    @pytest.mark.parametrize("raw,expected", [
        ("INFO", "info"), ("inf", "info"), ("WARNING", "warning"),
        ("wrn", "warning"), ("ERROR", "error"), ("err", "error"),
        ("DEBUG", "debug"), ("dbg", "debug"), ("CRITICAL", "critical"),
        ("weird", "weird"),
    ])
    def test_norm_level(self, raw, expected):
        assert _norm_level(raw) == expected


class TestSingletons:
    def test_get_server_buffer_singleton(self, monkeypatch):
        monkeypatch.setattr("domains.infrastructure.output_buffer._server_buffer", None)
        b1 = get_server_buffer()
        b2 = get_server_buffer()
        assert b1 is b2
        assert b1._max == 10_000

    def test_install_log_bridge_adds_handler(self):
        buf = OutputBuffer()
        handler = install_log_bridge(buf)
        try:
            assert handler in logging.root.handlers
        finally:
            logging.root.removeHandler(handler)

    def test_install_stdio_bridge_tees(self, monkeypatch):
        buf = OutputBuffer()
        real_stdout = sys.stdout
        monkeypatch.setattr(sys, "stdout", real_stdout, raising=False)
        install_stdio_bridge(buf)
        sys.stdout.write("hello world\n")
        assert buf.count == 1
        assert buf.lines[0].text == "hello world"


class TestSubscriberReadTimeoutZero:
    def test_read_timeout_zero_returns_pending(self):
        """read(timeout=0) should return immediately with any pending lines."""
        buf = OutputBuffer()
        sub = buf.subscribe("s1")
        buf.append_text("a")
        got = sub.read(timeout=0)
        assert [l.text for l in got] == ["a"]

    def test_read_timeout_zero_no_pending(self):
        """read(timeout=0) with no pending lines returns empty immediately."""
        buf = OutputBuffer()
        sub = buf.subscribe("s1")
        got = sub.read(timeout=0)
        assert got == []


class TestBufferLogHandlerLevelFilter:
    def test_handler_respects_level_through_logger(self):
        """BufferLogHandler should only emit records at or above its level when used via Logger."""
        buf = OutputBuffer()
        handler = BufferLogHandler(buf, level=logging.WARNING)
        logger = logging.getLogger("slo.test.level_filter")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            logger.info("info msg")
            logger.warning("warn msg")
            # Only WARNING should be captured (INFO is below handler threshold)
            assert buf.count == 1
            assert buf.lines[0].text == "warn msg"
        finally:
            logger.removeHandler(handler)
