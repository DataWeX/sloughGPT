"""Tests for domain.errors — exception hierarchy and require_non_empty_prompt."""

import pytest

from domain.errors._internal.errors import (
    EmptyPromptError,
    InvalidGenerationInputError,
    SloughGPTDomainError,
    require_non_empty_prompt,
)
from domain.infrastructure._internal.errors import AppError, ValidationError


class TestExceptionHierarchy:
    def test_sloughgpt_domain_error_is_app_error(self):
        assert issubclass(SloughGPTDomainError, AppError)

    def test_invalid_generation_input_is_validation_error(self):
        assert issubclass(InvalidGenerationInputError, ValidationError)

    def test_empty_prompt_is_invalid_generation_input(self):
        assert issubclass(EmptyPromptError, InvalidGenerationInputError)

    def test_default_http_status(self):
        exc = SloughGPTDomainError("test")
        assert exc.http_status == 400
        assert exc.code == "E_DOMAIN"

    def test_invalid_generation_input_http_status(self):
        exc = InvalidGenerationInputError("bad input")
        assert exc.http_status == 422
        assert exc.code == "E_VAL_FIELD"

    def test_empty_prompt_code(self):
        exc = EmptyPromptError()
        assert exc.code == "E_VAL_FIELD"

    def test_empty_prompt_default_message(self):
        exc = EmptyPromptError()
        assert str(exc) == "prompt must not be empty"

    def test_empty_prompt_custom_message(self):
        exc = EmptyPromptError("custom msg")
        assert str(exc) == "custom msg"

    def test_inheritance_chain_catch(self):
        with pytest.raises(AppError):
            raise EmptyPromptError()

    def test_inheritance_chain_catch_intermediate(self):
        with pytest.raises(InvalidGenerationInputError):
            raise EmptyPromptError()

    def test_domain_error_message_preserved(self):
        exc = SloughGPTDomainError("something broke")
        assert str(exc) == "something broke"

    def test_invalid_generation_message_preserved(self):
        exc = InvalidGenerationInputError("bad shape")
        assert str(exc) == "bad shape"


class TestRequireNonEmptyPrompt:
    def test_valid_string(self):
        assert require_non_empty_prompt("hello world") == "hello world"

    def test_strips_whitespace(self):
        assert require_non_empty_prompt("  hello  ") == "hello"

    def test_strips_tabs_newlines(self):
        assert require_non_empty_prompt("\t\nhello\n\t") == "hello"

    def test_empty_string_raises(self):
        with pytest.raises(EmptyPromptError):
            require_non_empty_prompt("")

    def test_whitespace_only_raises(self):
        with pytest.raises(EmptyPromptError):
            require_non_empty_prompt("   ")

    def test_tabs_only_raises(self):
        with pytest.raises(EmptyPromptError):
            require_non_empty_prompt("\t\t\t")

    def test_non_string_raises(self):
        with pytest.raises(InvalidGenerationInputError) as exc_info:
            require_non_empty_prompt(123)
        assert "must be a string" in str(exc_info.value)

    def test_none_raises(self):
        with pytest.raises(InvalidGenerationInputError):
            require_non_empty_prompt(None)

    def test_list_raises(self):
        with pytest.raises(InvalidGenerationInputError):
            require_non_empty_prompt(["hello"])

    def test_custom_field_name(self):
        with pytest.raises(InvalidGenerationInputError) as exc_info:
            require_non_empty_prompt(123, field_name="query")
        assert "query" in str(exc_info.value)

    def test_custom_field_name_empty(self):
        with pytest.raises(EmptyPromptError) as exc_info:
            require_non_empty_prompt("", field_name="query")
        assert "query" in str(exc_info.value)

    def test_single_char_valid(self):
        assert require_non_empty_prompt("a") == "a"

    def test_long_string_valid(self):
        long = "x" * 10000
        assert require_non_empty_prompt(long) == long
