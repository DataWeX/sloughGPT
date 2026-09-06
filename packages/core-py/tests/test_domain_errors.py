"""Tests for domain exceptions and require_non_empty_prompt."""
from __future__ import annotations

import pytest

from domains.errors import (
    EmptyPromptError,
    InvalidGenerationInputError,
    SloughGPTDomainError,
    require_non_empty_prompt,
)


class TestDomainExceptions:
    def test_base_exception(self):
        e = SloughGPTDomainError("fail")
        assert str(e) == "fail"
        assert e.http_status == 400
        assert e.code == "E_DOMAIN"

    def test_invalid_input(self):
        e = InvalidGenerationInputError("bad")
        assert e.http_status == 422
        assert e.code == "E_VAL_FIELD"

    def test_empty_prompt(self):
        e = EmptyPromptError()
        assert "must not be empty" in str(e)
        assert e.code == "E_VAL_FIELD"

    def test_hierarchy(self):
        from domains.infrastructure.errors import ValidationError, AppError
        assert issubclass(InvalidGenerationInputError, ValidationError)
        assert issubclass(InvalidGenerationInputError, AppError)
        assert issubclass(EmptyPromptError, InvalidGenerationInputError)


class TestRequireNonEmptyPrompt:
    def test_valid(self):
        assert require_non_empty_prompt("hello") == "hello"

    def test_strips_whitespace(self):
        assert require_non_empty_prompt("  hello  ") == "hello"

    def test_empty_raises(self):
        with pytest.raises(EmptyPromptError):
            require_non_empty_prompt("")

    def test_whitespace_only_raises(self):
        with pytest.raises(EmptyPromptError):
            require_non_empty_prompt("   ")

    def test_non_string_raises(self):
        with pytest.raises(InvalidGenerationInputError):
            require_non_empty_prompt(123)

    def test_custom_field_name(self):
        with pytest.raises(InvalidGenerationInputError, match="query"):
            require_non_empty_prompt(123, field_name="query")
