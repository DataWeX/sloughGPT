"""
Error Taxonomy — structured error hierarchy with codes, recovery policies,
and EventBus integration.

Every error in the system should be a subclass of AppError with:
  - code: short string identifier (e.g. "model.oom")
  - message: developer-facing detail
  - user_message: end-user friendly string or i18n key
  - recoverable: can this operation be retried?
  - details: debug payload (never user-facing)

Usage:
    raise ModelError("Model OOM on forward pass",
                     user_message="Model ran out of memory. Try a smaller model.",
                     details={"model": "gpt2", "memory_mb": 2048})

    error = AppError.from_exception(ValueError("bad thing"))
    error.recoverable  # False
"""

from __future__ import annotations

import json
import traceback
from typing import Any


# ── Base ──


class AppError(Exception):
    """Base class for all application errors."""

    code: str = "general.error"
    recoverable: bool = False
    user_message: str = "Something went wrong."
    http_status: int = 500

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        user_message: str | None = None,
        recoverable: bool | None = None,
        http_status: int | None = None,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ):
        actual_code = code if code is not None else self.code
        super().__init__(message or actual_code)
        self.message = message or actual_code
        self.code = actual_code
        if user_message is not None:
            self.user_message = user_message
        if recoverable is not None:
            self.recoverable = recoverable
        if http_status is not None:
            self.http_status = http_status
        self.details = details or {}
        self.cause = cause
        self._traceback_str = traceback.format_exc() if not cause else ""

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        code: str = "general.unhandled",
        user_message: str = "An unexpected error occurred.",
        details: dict[str, Any] | None = None,
    ) -> AppError:
        """Wrap a raw exception into an AppError."""
        return cls(
            message=str(exc),
            code=code,
            user_message=user_message,
            recoverable=False,
            details=details or {},
            cause=exc,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "user_message": self.user_message,
            "recoverable": self.recoverable,
            "http_status": self.http_status,
            "details": self.details,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.code}] {self.message}>"


# ── Concrete Error Classes ──


class RecoverableError(AppError):
    """Transient error that can be retried — network, timeout, rate limit."""

    code: str = "general.recoverable"
    recoverable: bool = True
    http_status: int = 503
    user_message: str = "A temporary error occurred. Please try again."


class FatalError(AppError):
    """Non-recoverable error — model crash, config corruption, disk full."""

    code: str = "general.fatal"
    recoverable: bool = False
    http_status: int = 500
    user_message: str = "A critical error occurred. Please restart the server."


class ValidationError(AppError):
    """Bad input — malformed request, missing fields."""

    code: str = "general.validation"
    recoverable: bool = False
    http_status: int = 400
    user_message: str = "Invalid request."


class ConfigError(AppError):
    """Misconfiguration — missing env, bad file, invalid value."""

    code: str = "general.config"
    recoverable: bool = False
    http_status: int = 500
    user_message: str = "Server configuration error."


class ModelError(AppError):
    """Model-specific — OOM, NaN weights, timeout, shape mismatch."""

    code: str = "model.error"
    recoverable: bool = False
    http_status: int = 503
    user_message: str = "Model encountered an error."


class ModelOOMError(ModelError):
    """Out of memory during inference or training."""

    code: str = "model.oom"
    recoverable: bool = True
    user_message: str = "Model ran out of memory. Try a smaller model or reduce batch size."


class ModelTimeoutError(ModelError):
    """Model generation timed out."""

    code: str = "model.timeout"
    recoverable: bool = True
    user_message: str = "Model generation timed out. Please try again."


class TaskError(AppError):
    """Task queue error — execution failure, dependency broken."""

    code: str = "task.error"
    recoverable: bool = True
    http_status: int = 500
    user_message: str = "A background task failed."


class ResourceExhaustedError(AppError):
    """Rate limit or concurrency limit hit."""

    code: str = "resource.exhausted"
    recoverable: bool = True
    http_status: int = 429
    user_message: str = "Too many requests. Please slow down."


class NotFoundError(AppError):
    """Resource not found — session, dataset, model missing."""

    code: str = "resource.not_found"
    recoverable: bool = False
    http_status: int = 404
    user_message: str = "The requested resource was not found."


class AuthError(AppError):
    """Authentication or authorization failure."""

    code: str = "auth.error"
    recoverable: bool = False
    http_status: int = 401
    user_message: str = "Authentication failed."


# ── Integration helpers ──


def emit_error_event(error: AppError, source: str = ""):
    """Emit an error event on the EventBus (fire-and-forget)."""
    try:
        from domains.infrastructure.event_bus import get_event_bus
        bus = get_event_bus()
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            loop.create_task(bus.emit("error.raised", error.to_dict(), source=source))
        else:
            bus.emit_sync("error.raised", error.to_dict(), source=source)
    except Exception as e:
        import logging
        logging.getLogger("slo.errors").warning("emit_error_event failed: %s", e, extra={
            "error_code": error.code, "source": source,
        })


def error_to_sse(error: AppError) -> dict[str, Any]:
    """Convert an AppError to an SSE-compatible dict (no stack traces)."""
    return {
        "code": error.code,
        "user_message": error.user_message,
        "recoverable": error.recoverable,
        "http_status": error.http_status,
    }


def classify_exception(exc: BaseException) -> AppError:
    """Classify a raw exception into the best-matching AppError subclass."""
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, TimeoutError):
        return ModelTimeoutError(
            message=str(exc),
            details={"original_type": type(exc).__name__},
            cause=exc,
        )
    if isinstance(exc, MemoryError):
        return ModelOOMError(
            message="Out of memory",
            details={"original_type": "MemoryError"},
            cause=exc,
        )
    if isinstance(exc, (ConnectionError, ConnectionRefusedError, ConnectionResetError)):
        return RecoverableError(
            message=str(exc),
            code="network.error",
            user_message="Network connection failed. Please check your connection.",
            details={"original_type": type(exc).__name__},
            cause=exc,
        )
    if isinstance(exc, FileNotFoundError):
        return NotFoundError(
            message=str(exc),
            user_message="File not found.",
            details={"path": str(exc) if hasattr(exc, 'filename') else ""},
            cause=exc,
        )
    if isinstance(exc, PermissionError):
        return AuthError(
            message=str(exc),
            user_message="Permission denied.",
            details={"original_type": "PermissionError"},
            cause=exc,
        )
    if isinstance(exc, ValueError):
        return ValidationError(
            message=str(exc),
            user_message="Invalid value provided.",
            details={"original_type": "ValueError"},
            cause=exc,
        )
    return AppError.from_exception(exc)
