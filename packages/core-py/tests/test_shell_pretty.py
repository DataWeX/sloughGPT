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
