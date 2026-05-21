"""``domains.errors`` — domain exceptions and prompt guards (no HTTP)."""

from __future__ import annotations

import pytest

from domains.errors import (
    EmptyPromptError,
    InvalidGenerationInputError,
    SloughGPTDomainError,
    require_non_empty_prompt,
)


def test_require_non_empty_prompt_ok() -> None:
    assert require_non_empty_prompt("  hello  ") == "hello"


def test_require_non_empty_prompt_empty_raises() -> None:
    with pytest.raises(EmptyPromptError) as ei:
        require_non_empty_prompt("")
    assert ei.value.http_status == 422
    assert ei.value.code == "empty_prompt"


def test_require_non_empty_prompt_whitespace_raises() -> None:
    with pytest.raises(EmptyPromptError):
        require_non_empty_prompt("   \t\n")


def test_require_non_empty_prompt_wrong_type() -> None:
    with pytest.raises(InvalidGenerationInputError) as ei:
        require_non_empty_prompt(None)  # type: ignore[arg-type]
    assert ei.value.http_status == 422


def test_sloughgpt_domain_error_is_exception() -> None:
    assert issubclass(SloughGPTDomainError, Exception)
