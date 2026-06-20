"""Tests for domains/errors.py — domain-layer exceptions."""

import pytest
from domains.errors import (
    SloughGPTDomainError,
    InvalidGenerationInputError,
    EmptyPromptError,
    require_non_empty_prompt,
)


class TestExceptionHierarchy:
    def test_base_error(self):
        e = SloughGPTDomainError("msg")
        assert str(e) == "msg"
        assert e.http_status == 500
        assert e.code == "domain_error"

    def test_invalid_input_inherits(self):
        e = InvalidGenerationInputError("bad")
        assert isinstance(e, SloughGPTDomainError)
        assert e.http_status == 422
        assert e.code == "invalid_generation_input"

    def test_empty_prompt_inherits(self):
        e = EmptyPromptError()
        assert isinstance(e, InvalidGenerationInputError)
        assert e.code == "empty_prompt"
        assert "must not be empty" in str(e)

    def test_empty_prompt_custom_message(self):
        e = EmptyPromptError("custom msg")
        assert str(e) == "custom msg"


class TestRequireNonEmptyPrompt:
    def test_valid_string(self):
        result = require_non_empty_prompt("hello world")
        assert result == "hello world"

    def test_strips_whitespace(self):
        result = require_non_empty_prompt("  hello  ")
        assert result == "hello"

    def test_empty_string_raises(self):
        with pytest.raises(EmptyPromptError):
            require_non_empty_prompt("")

    def test_whitespace_only_raises(self):
        with pytest.raises(EmptyPromptError):
            require_non_empty_prompt("   ")

    def test_non_string_raises(self):
        with pytest.raises(InvalidGenerationInputError):
            require_non_empty_prompt(123)

    def test_none_raises(self):
        with pytest.raises(InvalidGenerationInputError):
            require_non_empty_prompt(None)

    def test_custom_field_name(self):
        with pytest.raises(EmptyPromptError, match="my_field must not be empty"):
            require_non_empty_prompt("   ", field_name="my_field")

    def test_custom_field_name_non_string(self):
        with pytest.raises(InvalidGenerationInputError, match="query must be a string"):
            require_non_empty_prompt(42, field_name="query")
