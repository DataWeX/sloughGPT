"""Tests for shell.pretty — highlighting and traceback formatting."""

from __future__ import annotations

from domain.shell._internal import pretty


class TestColorSupported:
    def test_returns_bool(self):
        assert isinstance(pretty.color_supported(), bool)

    def test_no_color_disables(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert pretty.color_supported() is False


class TestHighlight:
    def test_plain_fallback_without_color(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert pretty.highlight("x = 1", "python") == "x = 1"

    def test_unknown_lexer_falls_back_to_text(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        out = pretty.highlight("hello", "no-such-lexer-xyz")
        assert "hello" in out

    def test_python_highlight_contains_source(self):
        import re

        out = pretty.highlight("x = 1", "python")
        stripped = re.sub(r"\x1b\[[0-9;]*m", "", out)
        assert "x = 1" in stripped


class TestFormatTraceback:
    def test_includes_exception(self):
        try:
            raise ValueError("boom-test-123")
        except ValueError:
            out = pretty.format_traceback()
        assert "ValueError" in out
        assert "boom-test-123" in out

    def test_plain_without_color(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        try:
            raise RuntimeError("plain-test")
        except RuntimeError:
            out = pretty.format_traceback()
        assert "RuntimeError" in out
        assert "\x1b[" not in out


class TestFormatErrorBrief:
    def test_type_and_message(self):
        assert pretty.format_error_brief(ValueError("bad")) == "ValueError: bad"

    def test_bare_type_without_message(self):
        assert pretty.format_error_brief(KeyboardInterrupt()) == "KeyboardInterrupt"


class TestTerminalPageSize:
    def test_sane_minimum(self):
        assert pretty.terminal_page_size() >= 5


class TestLiveCapable:
    def test_dumb_term_not_capable(self, monkeypatch):
        monkeypatch.setenv("TERM", "dumb")
        assert pretty.live_capable() is False


class TestLivePager:
    def _pager(self, keys, height=4):
        emitted: list[str] = []
        key_iter = iter(keys)
        return pretty.LivePager(
            emit=emitted.append,
            read_key=lambda: next(key_iter),
            height_fn=lambda: height,
        ), emitted

    def test_passthrough_until_full(self):
        pager, emitted = self._pager([], height=4)
        assert pager.process("a", "\n") == ("a", "\n")
        assert pager.process("b", "\n") == ("b", "\n")
        assert emitted == []
        assert not pager.dropped

    def test_quit_drops_rest(self):
        pager, emitted = self._pager(["q"], height=4)
        for i in range(1, 5):
            assert pager.process(f"l{i}", "\n") == (f"l{i}", "\n")
        assert pager.process("l5", "\n") is None
        assert pager.dropped
        assert pager.process("l6", "\n") is None
        assert any("1049" in chunk for chunk in emitted)

    def test_scroll_past_end_resumes_streaming(self):
        pager, emitted = self._pager([" "], height=4)
        for i in range(1, 5):
            pager.process(f"l{i}", "\n")
        # Space at the bottom exits the viewer; chunk already shown.
        assert pager.process("l5", "\n") is None
        assert not pager.dropped
        # Next overflow re-opens the viewer.
        pager2_keys = iter(["q"])
        pager.read_key = lambda: next(pager2_keys)
        assert pager.process("l6", "\n") is None
        assert pager.dropped

    def test_scroll_up_and_down(self):
        pager, emitted = self._pager(["k", "k", "j", "q"], height=4)
        for i in range(1, 5):
            pager.process(f"l{i}", "\n")
        assert pager.process("l5", "\n") is None
        assert pager.dropped

    def test_reset_clears(self):
        pager, emitted = self._pager(["q"], height=4)
        for i in range(1, 5):
            pager.process(f"l{i}", "\n")
        pager.process("l5", "\n")
        assert pager.dropped
        pager.reset()
        assert not pager.dropped
        assert pager.process("l6", "\n") == ("l6", "\n")

    def test_status_line(self):
        pager, _ = self._pager([], height=10)
        pager.process("a\nb\nc", "\n")
        assert "1-3/3" in pager._status()


class TestPager:
    def test_passes_through_without_prompt_fn(self):
        pager = pretty.Pager(page_size=2)
        assert pager.process("a\nb\nc\nd") == ("a\nb\nc\nd", "\n")
        assert not pager.dropped

    def test_pauses_when_page_fills(self):
        answers = iter(["", "q"])
        pager = pretty.Pager(page_size=2, prompt_fn=lambda: next(answers))
        assert pager.process("l1", "\n") == ("l1", "\n")
        assert pager.process("l2", "\n") == ("l2", "\n")
        # Third line overflows the page → prompt → Enter continues.
        assert pager.process("l3", "\n") == ("l3", "\n")
        assert not pager.dropped

    def test_quit_drops_rest_until_reset(self):
        answers = iter(["q"])
        pager = pretty.Pager(page_size=1, prompt_fn=lambda: next(answers))
        assert pager.process("l1", "\n") == ("l1", "\n")
        assert pager.process("l2", "\n") is None
        assert pager.dropped
        assert pager.process("l3", "\n") is None
        pager.reset()
        assert not pager.dropped
        assert pager.process("l4", "\n") == ("l4", "\n")

    def test_eof_quits(self):
        def _raise():
            raise EOFError

        pager = pretty.Pager(page_size=0, prompt_fn=_raise)
        assert pager.process("l1", "\n") is None
        assert pager.dropped
