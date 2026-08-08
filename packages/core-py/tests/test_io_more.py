"""Coverage tests for ShellIO (domains.shell.io)."""

import builtins
import types

import pytest

from domains.shell.io import ConsoleIO, MemoryIO, capture_cmd, capture_output


class _FakeTty:
    def __init__(self, line="  hello  \n", fail_close=False):
        self.writes = []
        self._line = line
        self._fail_close = fail_close

    def write(self, s):
        self.writes.append(s)

    def flush(self):
        pass

    def readline(self):
        return self._line

    def close(self):
        if self._fail_close:
            raise OSError("boom")
        self.writes = None


@pytest.fixture
def _no_readline(monkeypatch):
    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "readline":
            raise ImportError("no readline")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)


class TestConsoleIOInit:
    def test_init_readline_present(self):
        c = ConsoleIO()
        assert c._has_readline is True

    def test_init_without_readline(self, _no_readline):
        c = ConsoleIO()
        assert c._has_readline is False


class TestConsoleIOTty:
    def _console_with_tty(self, **kw):
        c = ConsoleIO()
        c._tty = _FakeTty(**kw)
        return c

    def test_is_tty(self):
        c = ConsoleIO()
        assert c._is_tty is (c._tty is not None)
        c._tty = _FakeTty()
        assert c._is_tty is True

    def test_write_through_tty(self):
        c = self._console_with_tty()
        c.write("hi", end="\n")
        assert c._tty.writes[-1] == "hi\n"

    def test_read_through_tty(self):
        c = self._console_with_tty()
        assert c.read(">> ") == "hello"
        assert c._tty.writes[-1] == ">> "

    def test_flush_through_tty(self):
        c = self._console_with_tty()
        c.flush()

    def test_write_fallback_stdout(self, capsys):
        c = ConsoleIO()
        c._tty = None
        c.write("fallback")
        assert capsys.readouterr().out == "fallback\n"

    def test_read_fallback_input(self, monkeypatch):
        c = ConsoleIO()
        c._tty = None
        monkeypatch.setattr(builtins, "input", lambda prompt="": " typed \n")
        assert c.read(">> ") == "typed"

    def test_flush_fallback_stdout(self):
        c = ConsoleIO()
        c._tty = None
        c.flush()

    def test_close_tty(self):
        c = self._console_with_tty()
        c.close()
        assert c._tty is None

    def test_close_tty_that_fails(self):
        c = self._console_with_tty(fail_close=True)
        c.close()
        assert c._tty is None

    def test_close_no_tty(self):
        c = ConsoleIO()
        c._tty = None
        c.close()


class TestConsoleIOReadline:
    def test_setup_completion_present(self):
        c = ConsoleIO()
        c.setup_completion(lambda text, state: None)

    def test_setup_completion_missing(self, _no_readline):
        c = ConsoleIO()
        c.setup_completion(lambda text, state: None)

    def test_setup_completion_readline_error(self, monkeypatch):
        c = ConsoleIO()
        monkeypatch.setattr("readline.set_completer", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
        c.setup_completion(lambda text, state: None)

    def test_save_history_present(self, tmp_path):
        c = ConsoleIO()
        path = str(tmp_path / "hist")
        c.save_history(path)
        assert (tmp_path / "hist").exists()

    def test_save_history_missing(self, _no_readline, tmp_path):
        c = ConsoleIO()
        c.save_history(str(tmp_path / "hist"))

    def test_save_history_error(self, monkeypatch, tmp_path):
        c = ConsoleIO()
        monkeypatch.setattr("readline.write_history_file", lambda p: (_ for _ in ()).throw(OSError("x")))
        c.save_history(str(tmp_path / "hist"))

    def test_load_history_present(self, tmp_path):
        c = ConsoleIO()
        path = str(tmp_path / "hist")
        with open(path, "w") as f:
            f.write("echo hi\n")
        c.load_history(path)

    def test_load_history_missing_file(self, tmp_path):
        c = ConsoleIO()
        c.load_history(str(tmp_path / "does-not-exist"))

    def test_load_history_missing_readline(self, _no_readline):
        c = ConsoleIO()
        c.load_history("/nope")


class TestMemoryIO:
    def test_write_empty_text_uses_just_end(self):
        m = MemoryIO()
        m.write("", end="X")
        assert m.get_output() == "X"

    def test_feed_and_read_sequence(self):
        m = MemoryIO()
        m.feed("one", "two")
        assert m.read() == "one"
        assert m.read() == "two"

    def test_feed_resets_index(self):
        m = MemoryIO()
        m.feed("a", "b")
        m.read()
        m.feed("c")
        assert m.read() == "a"

    def test_read_exhausted_raises(self):
        m = MemoryIO()
        with pytest.raises(EOFError):
            m.read()

    def test_clear(self):
        m = MemoryIO()
        m.write("x")
        m.clear()
        assert m.get_output() == ""

    def test_flush_is_noop(self):
        MemoryIO().flush()


class TestCapture:
    def test_capture_redirects_and_restores(self):
        m = MemoryIO()
        with capture_output(m) as cap:
            m.write("one")
            m.write("two", end=" ")
            assert cap.getvalue() == "one\ntwo "
        m.write("after")
        assert m.get_output() == "after\n"
        assert cap.getvalue() == "one\ntwo "

    def test_capture_empty(self):
        m = MemoryIO()
        with capture_output(m) as cap:
            pass
        assert cap.getvalue() == ""

    def test_capture_restores_on_exception(self):
        m = MemoryIO()
        with pytest.raises(RuntimeError):
            with capture_output(m):
                m.write("boom")
                raise RuntimeError("x")
        m.write("after")
        assert m.get_output() == "after\n"

    def test_capture_empty_text_uses_end(self):
        m = MemoryIO()
        with capture_output(m) as cap:
            m.write("", end="|")
        assert cap.getvalue() == "|"


class TestCaptureCmd:
    def test_capture_cmd_returns_output_and_restores(self):
        repl = types.SimpleNamespace(
            io=MemoryIO(),
            console=types.SimpleNamespace(_io=MemoryIO()),
        )
        old_io = repl.io
        old_console_io = repl.console._io

        def _greet(name):
            repl.io.write("hi " + name)

        out = capture_cmd(repl, _greet, "bob")
        assert out == "hi bob\n"
        assert repl.io is old_io
        assert repl.console._io is old_console_io

    def test_capture_cmd_restores_on_error(self):
        repl = types.SimpleNamespace(
            io=MemoryIO(),
            console=types.SimpleNamespace(_io=MemoryIO()),
        )
        old_io = repl.io

        def _explode():
            repl.io.write("partial")
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            capture_cmd(repl, _explode)
        assert repl.io is old_io
