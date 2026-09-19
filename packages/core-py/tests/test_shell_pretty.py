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
