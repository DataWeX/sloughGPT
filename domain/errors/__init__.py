"""errors — Domain-layer exceptions for SloughGPT.

Public API:
    SloughGPTDomainError, InvalidGenerationInputError, EmptyPromptError, require_non_empty_prompt
"""
from __future__ import annotations

from domain.errors._internal.errors import (
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
