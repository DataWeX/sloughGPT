"""Tests for domains.errors — domain exceptions and require_non_empty_prompt.

Covers: exception hierarchy, http_status codes, code attributes, validation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.errors import (
    SloughGPTDomainError,
    InvalidGenerationInputError,
    EmptyPromptError,
    require_non_empty_prompt,
)


class TestExceptionHierarchy:
    def test_base_exception(self):
        e = SloughGPTDomainError("test")
        assert isinstance(e, Exception)
        assert e.http_status == 500
        assert e.code == "domain_error"

    def test_invalid_generation_input(self):
        e = InvalidGenerationInputError("bad input")
        assert isinstance(e, SloughGPTDomainError)
        assert e.http_status == 422
        assert e.code == "invalid_generation_input"

    def test_empty_prompt(self):
        e = EmptyPromptError()
        assert isinstance(e, InvalidGenerationInputError)
        assert e.code == "empty_prompt"
        assert "empty" in str(e).lower()

    def test_empty_prompt_custom_message(self):
        e = EmptyPromptError("custom")
        assert "custom" in str(e)


class TestRequireNonEmptyPrompt:
    def test_valid_prompt(self):
        assert require_non_empty_prompt("hello") == "hello"

    def test_strips_whitespace(self):
        assert require_non_empty_prompt("  hello  ") == "hello"

    def test_empty_string(self):
        with pytest.raises(EmptyPromptError):
            require_non_empty_prompt("")

    def test_whitespace_only(self):
        with pytest.raises(EmptyPromptError):
            require_non_empty_prompt("   ")

    def test_not_a_string(self):
        with pytest.raises(InvalidGenerationInputError):
            require_non_empty_prompt(123)

    def test_none(self):
        with pytest.raises(InvalidGenerationInputError):
            require_non_empty_prompt(None)

    def test_custom_field_name(self):
        with pytest.raises(EmptyPromptError) as exc_info:
            require_non_empty_prompt("", field_name="message")
        assert "message" in str(exc_info.value)

    def test_returns_stripped(self):
        assert require_non_empty_prompt("  test  ") == "test"
