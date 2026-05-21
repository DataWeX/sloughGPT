"""Domain-layer exceptions for SloughGPT (``packages/core-py``).

Raised from training/inference code before or during tensor work. The HTTP layer
(``apps/api/server``) maps these to JSON responses via ``exception_handler`` —
no FastAPI imports here so domains stay portable (CLI, notebooks, workers).
"""

from __future__ import annotations


class SloughGPTDomainError(Exception):
    """Predictable failure in domains (validation, incompatible state, etc.)."""

    http_status: int = 500
    code: str = "domain_error"


class InvalidGenerationInputError(SloughGPTDomainError):
    """Input invalid before tensor preparation (shape, type, or policy)."""

    http_status = 422
    code = "invalid_generation_input"


class EmptyPromptError(InvalidGenerationInputError):
    """Prompt is empty or whitespace-only after strip."""

    code = "empty_prompt"

    def __init__(self, message: str = "prompt must not be empty"):
        super().__init__(message)


def require_non_empty_prompt(prompt: object, *, field_name: str = "prompt") -> str:
    """Return a stripped non-empty string or raise domain errors.

    Call after API sanitization (e.g. :func:`InputValidator.validate_prompt`) so
    HTTP and core share the same rule.
    """
    if not isinstance(prompt, str):
        raise InvalidGenerationInputError(f"{field_name} must be a string")
    stripped = prompt.strip()
    if not stripped:
        raise EmptyPromptError(f"{field_name} must not be empty")
    return stripped
