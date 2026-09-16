"""Backward-compatibility shim — imports from the new ``domain.errors`` package."""

from domain.errors import (
    EmptyPromptError,
    InvalidGenerationInputError,
    SloughGPTDomainError,
    require_non_empty_prompt,
)

__all__ = [
    "SloughGPTDomainError",
    "InvalidGenerationInputError",
    "EmptyPromptError",
    "require_non_empty_prompt",
]
