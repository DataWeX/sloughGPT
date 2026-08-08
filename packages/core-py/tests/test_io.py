"""Tests for ShellIO — ConsoleIO, MemoryIO, capture helpers."""

from __future__ import annotations

import io as _io
import sys
from types import SimpleNamespace

import pytest

from domains.shell.io import (
    ConsoleIO, MemoryIO, capture_cmd, capture_output,
)


class _FakeTty:
    def __init__(self):
        self._w = []
        self._r = []
        self._closed = False

    def write(self, text):
        self._w.append(text)

    def flush(self):
        pass

    def readline(self):
        return self._r.pop(0)

    def close(self):
        self._closed = True


class TestConsoleIOTtyPaths:
    def test_open_dev_tty_success(self, monkeypatch):
        fake = _FakeTty()
        monkeypatch.setattr("builtins.open", lambda *a, **k: fake)
        c = ConsoleIO()
        assert c._tty is fake

    def test_write_to_tty(self, monkeypatch):
        fake = _FakeTty()
        c = ConsoleIO()
        c._tty = fake
        c.write("hi", end="!")
        assert "".join(fake._w) == "hi!"
        assert c._is_tty is True

    def test_read_from_tty(self, monkeypatch):
        fake = _FakeTty()
        fake._r = ["answer\n"]
        c = ConsoleIO()
        c._tty = fake
        assert c.read("> ") == "answer"
        assert "".join(fake._w) == "> "

    def test_flush_tty(self, monkeypatch):
        fake = _FakeTty()
        c = ConsoleIO()
        c._tty = fake
        c.flush()

    def test_close_tty(self, monkeypatch):
        fake = _FakeTty()
        c = ConsoleIO()
        c._tty = fake
        c.close()
        assert fake._closed
        assert c._tty is None

    def test_close_tty_raises(self, monkeypatch):
        class _BadTty:
            def close(self):
                raise OSError("boom")

        c = ConsoleIO()
        c._tty = _BadTty()
        c.close()
        assert c._tty is None


class TestConsoleIOFallback:
    def test_write_stdout_fallback(self, monkeypatch):
        c = ConsoleIO()
        c._tty = None
        buf = _io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        c.write("out")
        assert buf.getvalue() == "out\n"

    def test_read_input_fallback(self, monkeypatch):
        c = ConsoleIO()
        c._tty = None
        monkeypatch.setattr("builtins.input", lambda prompt: "typed")
        assert c.read("> ") == "typed"

    def test_flush_stdout_fallback(self, monkeypatch):
        c = ConsoleIO()
        c._tty = None
        buf = _io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        c.flush()
        assert c._is_tty is False

    def test_close_without_tty(self):
        c = ConsoleIO()
        c._tty = None
        c.close()

    def test_readline_import_failure(self, monkeypatch):
        import builtins
        _orig = builtins.__import__

        def _no_readline(name, *a, **k):
            if name == "readline":
                raise ImportError("no readline")
            return _orig(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _no_readline)
        monkeypatch.setattr(sys, "modules", {k: v for k, v in sys.modules.items() if k != "readline"})
        c = ConsoleIO()
        assert c._has_readline is False


class TestConsoleIOCompletionHistory:
    def test_setup_completion_disabled(self):
        c = ConsoleIO()
        c._has_readline = False
        c.setup_completion(lambda text, state: None)

    def test_setup_completion_enabled(self, monkeypatch):
        import readline
        calls = []
        monkeypatch.setattr(readline, "set_completer", lambda f: calls.append("set"))
        monkeypatch.setattr(readline, "parse_and_bind", lambda s: calls.append(s))
        c = ConsoleIO()
        c._has_readline = True
        c.setup_completion(lambda text, state: None)
        assert calls[0] == "set"

    def test_setup_completion_exception(self, monkeypatch):
        import readline
        monkeypatch.setattr(readline, "parse_and_bind", lambda s: (_ for _ in ()).throw(RuntimeError()))
        c = ConsoleIO()
        c._has_readline = True
        c.setup_completion(lambda text, state: None)

    def test_save_history_disabled(self):
        c = ConsoleIO()
        c._has_readline = False
        c.save_history("/tmp/nope")
        c.load_history("/tmp/nope")

    def test_save_history_enabled(self, monkeypatch):
        import readline
        calls = []
        monkeypatch.setattr(readline, "write_history_file", lambda p: calls.append(p))
        c = ConsoleIO()
        c._has_readline = True
        c.save_history("/tmp/hist")
        assert calls == ["/tmp/hist"]

    def test_save_history_exception(self, monkeypatch):
        import readline
        monkeypatch.setattr(readline, "write_history_file", lambda p: (_ for _ in ()).throw(RuntimeError()))
        c = ConsoleIO()
        c._has_readline = True
        c.save_history("/tmp/hist")

    def test_load_history_enabled(self, monkeypatch):
        import readline
        calls = []
        monkeypatch.setattr(readline, "read_history_file", lambda p: calls.append(p))
        monkeypatch.setattr(readline, "set_history_length", lambda n: calls.append(n))
        c = ConsoleIO()
        c._has_readline = True
        c.load_history("/tmp/hist")
        assert calls == ["/tmp/hist", 500]

    def test_load_history_missing_file(self, monkeypatch):
        import readline
        monkeypatch.setattr(readline, "read_history_file", lambda p: (_ for _ in ()).throw(FileNotFoundError()))
        c = ConsoleIO()
        c._has_readline = True
        c.load_history("/tmp/absent")


class TestMemoryIO:
    def test_read_raises_eof_when_empty(self):
        io = MemoryIO()
        with pytest.raises(EOFError):
            io.read()

    def test_feed_and_read_strips(self):
        io = MemoryIO()
        io.feed("  one  ", "two")
        assert io.read() == "one"
        assert io.read() == "two"

    def test_clear_empties_output(self):
        io = MemoryIO()
        io.write("x")
        io.clear()
        assert io.get_output() == ""

    def test_write_empty_text(self):
        io = MemoryIO()
        io.write("")
        assert io.get_output() == "\n"

    def test_flush_is_noop(self):
        MemoryIO().flush()


class TestCapture:
    def test_capture_output_roundtrip(self):
        io = MemoryIO()
        with capture_output(io) as cap:
            io.write("captured")
        assert cap.getvalue() == "captured\n"
        io.write("after")
        assert io.get_output() == "after\n"

    def test_capture_cmd(self):
        calls = []

        class _Repl:
            def __init__(self):
                self.io = MemoryIO()
                self.console = SimpleNamespace(_io=MemoryIO())

        repl = _Repl()
        old_io = repl.io

        def _cmd(x):
            calls.append(x)
            repl.io.write("out")

        result = capture_cmd(repl, _cmd, 42)
        assert calls == [42]
        assert result == "out\n"
        assert repl.io is old_io
        assert repl.console._io is not old_io
