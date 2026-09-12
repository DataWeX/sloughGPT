"""errors — Domain-layer exceptions for SloughGPT.

Public API:
    SloughGPTDomainError, InvalidGenerationInputError, EmptyPromptError, require_non_empty_prompt
"""

from domain.errors._internal.errors import (
    SloughGPTDomainError,
    InvalidGenerationInputError,
    EmptyPromptError,
    require_non_empty_prompt,
)

__all__ = [
    "SloughGPTDomainError",
    "InvalidGenerationInputError",
    "EmptyPromptError",
    "require_non_empty_prompt",
]
