"""
Unified Error Taxonomy — single source of truth for all application errors.

Every error in the system is an AppError subclass with:
  - code: E_UPPER_CASE identifier (e.g. "E_MODEL_OOM")
  - message: developer-facing detail (never shown to users)
  - user_message: end-user friendly string
  - recoverable: can this operation be retried?
  - http_status: HTTP status code
  - details: debug payload (stripped in production)
  - cause: original exception (for chaining)
  - source: where the error originated

Error Code Registry:
  ErrorCode enum maps every E_ code to its (AppError subclass, http_status).
  Use raise_error() from schemas.common to raise with a registered code.

Usage:
    from domains.infrastructure.errors import (
        AppError, ModelOOMError, NotFoundError, ErrorCode,
        classify_exception, emit_error_event,
    )

    # Direct raise
    raise ModelOOMError("forward pass OOM", details={"model": "gpt2"})

    # Raise by code (via schemas.common.raise_error)
    from schemas.common import raise_error
    raise_error("Not found", "E_NOT_FOUND")

    # Classify raw exception
    err = classify_exception(TimeoutError("too slow"))
    err.code  # "E_MODEL_TIMEOUT"
"""

from __future__ import annotations

import json
import traceback
from enum import Enum
from typing import Any

# ── Error Code Registry ──


class ErrorCode(str, Enum):
    """Canonical error codes. Every E_ code used in the system lives here.

    Maps to (error_class_name, http_status) via ERROR_REGISTRY.
    """

    # General
    E_INTERNAL = "E_INTERNAL"
    E_UNHANDLED = "E_UNHANDLED"
    E_DOMAIN = "E_DOMAIN"
    E_RECOVERABLE = "E_RECOVERABLE"
    E_FATAL = "E_FATAL"

    # Validation
    E_BAD_REQUEST = "E_BAD_REQUEST"
    E_VAL_REQUEST = "E_VAL_REQUEST"
    E_VAL_FIELD = "E_VAL_FIELD"
    E_NOT_IMPLEMENTED = "E_NOT_IMPLEMENTED"

    # Auth
    E_AUTH_MISSING = "E_AUTH_MISSING"
    E_AUTH_FORBIDDEN = "E_AUTH_FORBIDDEN"

    # Resource
    E_NOT_FOUND = "E_NOT_FOUND"
    E_CONFLICT = "E_CONFLICT"
    E_INFRA_BUSY = "E_INFRA_BUSY"

    # Rate limiting / pressure
    E_RATE_LIMITED = "E_RATE_LIMITED"
    E_INFRA_RATE_LIMIT = "E_INFRA_RATE_LIMIT"
    E_MEMORY_PRESSURE = "E_MEMORY_PRESSURE"

    # Infrastructure
    E_INFRA_STARTUP = "E_INFRA_STARTUP"
    E_INFRA_REGISTRY = "E_INFRA_REGISTRY"
    E_INFRA_TIMEOUT = "E_INFRA_TIMEOUT"
    E_INFRA_GENERATION = "E_INFRA_GENERATION"
    E_TIMEOUT = "E_TIMEOUT"

    # Model
    E_MODEL_ERROR = "E_MODEL_ERROR"
    E_MODEL_OOM = "E_MODEL_OOM"
    E_MODEL_TIMEOUT = "E_MODEL_TIMEOUT"
    E_MODEL_LOADING = "E_MODEL_LOADING"

    # Task / background
    E_TASK_ERROR = "E_TASK_ERROR"
    E_STATE_IDLE = "E_STATE_IDLE"
    E_CANCEL_FAILED = "E_CANCEL_FAILED"

    # Config / environment
    E_CONFIG = "E_CONFIG"
    E_ENV_MISSING = "E_ENV_MISSING"

    # Network
    E_NETWORK = "E_NETWORK"

    # Shell / security
    E_SHELL_SECURITY = "E_SHELL_SECURITY"

    # Docstore
    E_UNKNOWN_COLLECTION = "E_UNKNOWN_COLLECTION"

    # Legacy aliases (mapped to canonical codes)
    E_CREATE_FAILED = "E_BAD_REQUEST"


# ── Error Registry ──
# Maps ErrorCode -> (class_name, http_status, recoverable, default_user_message)
# Populated after class definitions at bottom of file.

ERROR_REGISTRY: dict[ErrorCode, tuple[str, int, bool, str]] = {}


# ── Base ──


class AppError(Exception):
    """Base class for ALL application errors.

    Subclasses override class-level defaults. The code should use
    E_UPPER_CASE convention. The http_response shape is:

        {"error": "...", "code": "...", "details": {...}, "correlation_id": "..."}

    Attributes:
        code: Machine-readable error code (E_UPPER_CASE).
        message: Developer-facing detail (never shown to users).
        user_message: End-user friendly string.
        recoverable: Can this operation be retried?
        http_status: HTTP status code for the response.
        details: Debug payload (stripped in production responses).
        cause: Original exception for chaining.
        source: Where the error originated (e.g. "router.chat").
    """

    code: str = "E_INTERNAL"
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
        source: str | None = None,
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
        self.source = source or ""
        self._traceback_str = traceback.format_exc() if not cause else ""

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        code: str = "E_UNHANDLED",
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
        """Full internal representation (includes message, recoverable)."""
        return {
            "code": self.code,
            "message": self.message,
            "user_message": self.user_message,
            "recoverable": self.recoverable,
            "http_status": self.http_status,
            "details": self.details,
        }

    def to_http_response(self) -> dict[str, Any]:
        """HTTP response body shape. Used by exception handlers.

        Shape: {"error": ..., "code": ..., "details": {...}}
        """
        body: dict[str, Any] = {"error": self.user_message, "code": self.code}
        if self.details:
            body["details"] = self.details
        return body

    def to_sse(self) -> dict[str, Any]:
        """SSE-compatible dict (no stack traces, no details)."""
        return {
            "code": self.code,
            "user_message": self.user_message,
            "recoverable": self.recoverable,
            "http_status": self.http_status,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.code}] {self.message}>"


# ── Concrete Error Classes ──


class RecoverableError(AppError):
    """Transient error that can be retried — network, timeout, rate limit."""

    code: str = "E_RECOVERABLE"
    recoverable: bool = True
    http_status: int = 503
    user_message: str = "A temporary error occurred. Please try again."


class FatalError(AppError):
    """Non-recoverable error — model crash, config corruption, disk full."""

    code: str = "E_FATAL"
    recoverable: bool = False
    http_status: int = 500
    user_message: str = "A critical error occurred. Please restart the server."


class ValidationError(AppError):
    """Bad input — malformed request, missing fields."""

    code: str = "E_BAD_REQUEST"
    recoverable: bool = False
    http_status: int = 400
    user_message: str = "Invalid request."


class ConfigError(AppError):
    """Misconfiguration — missing env, bad file, invalid value."""

    code: str = "E_CONFIG"
    recoverable: bool = False
    http_status: int = 500
    user_message: str = "Server configuration error."


class ModelError(AppError):
    """Model-specific — OOM, NaN weights, timeout, shape mismatch."""

    code: str = "E_MODEL_ERROR"
    recoverable: bool = False
    http_status: int = 503
    user_message: str = "Model encountered an error."


class ModelOOMError(ModelError):
    """Out of memory during inference or training."""

    code: str = "E_MODEL_OOM"
    recoverable: bool = True
    user_message: str = "Model ran out of memory. Try a smaller model or reduce batch size."


class ModelTimeoutError(ModelError):
    """Model generation timed out."""

    code: str = "E_MODEL_TIMEOUT"
    recoverable: bool = True
    user_message: str = "Model generation timed out. Please try again."


class TaskError(AppError):
    """Task queue error — execution failure, dependency broken."""

    code: str = "E_TASK_ERROR"
    recoverable: bool = True
    http_status: int = 500
    user_message: str = "A background task failed."


class ResourceExhaustedError(AppError):
    """Rate limit or concurrency limit hit."""

    code: str = "E_RATE_LIMITED"
    recoverable: bool = True
    http_status: int = 429
    user_message: str = "Too many requests. Please slow down."


class NotFoundError(AppError):
    """Resource not found — session, dataset, model missing."""

    code: str = "E_NOT_FOUND"
    recoverable: bool = False
    http_status: int = 404
    user_message: str = "The requested resource was not found."


class AuthError(AppError):
    """Authentication or authorization failure."""

    code: str = "E_AUTH_MISSING"
    recoverable: bool = False
    http_status: int = 401
    user_message: str = "Authentication failed."


class ConflictError(AppError):
    """Resource conflict — busy, locked, state mismatch."""

    code: str = "E_CONFLICT"
    recoverable: bool = True
    http_status: int = 409
    user_message: str = "Resource is busy or in a conflicting state."


class TimeoutAppError(AppError):
    """Request or operation timed out."""

    code: str = "E_TIMEOUT"
    recoverable: bool = True
    http_status: int = 408
    user_message: str = "The request timed out. Please try again."


class NotImplementedAppError(AppError):
    """Feature not implemented."""

    code: str = "E_NOT_IMPLEMENTED"
    recoverable: bool = False
    http_status: int = 501
    user_message: str = "This feature is not yet implemented."


# ── Populate Registry ──

_ERROR_CLASSES: dict[str, type[AppError]] = {
    "AppError": AppError,
    "RecoverableError": RecoverableError,
    "FatalError": FatalError,
    "ValidationError": ValidationError,
    "ConfigError": ConfigError,
    "ModelError": ModelError,
    "ModelOOMError": ModelOOMError,
    "ModelTimeoutError": ModelTimeoutError,
    "TaskError": TaskError,
    "ResourceExhaustedError": ResourceExhaustedError,
    "NotFoundError": NotFoundError,
    "AuthError": AuthError,
    "ConflictError": ConflictError,
    "TimeoutAppError": TimeoutAppError,
    "NotImplementedAppError": NotImplementedAppError,
}

for _code in ErrorCode:
    _cls = _ERROR_CLASSES.get(_code.name.replace("E_", "").title().replace(" ", ""))
    if _cls is None:
        # Fallback: use AppError defaults
        ERROR_REGISTRY[_code] = ("AppError", 500, False, "Something went wrong.")
    else:
        ERROR_REGISTRY[_code] = (
            _cls.__name__,
            _cls.http_status,
            _cls.recoverable,
            _cls.user_message,
        )

# Explicit overrides for codes where the enum name doesn't match class name
ERROR_REGISTRY[ErrorCode.E_INTERNAL] = ("AppError", 500, False, "Something went wrong.")
ERROR_REGISTRY[ErrorCode.E_UNHANDLED] = ("AppError", 500, False, "An unexpected error occurred.")
ERROR_REGISTRY[ErrorCode.E_DOMAIN] = ("AppError", 400, False, "Domain error.")
ERROR_REGISTRY[ErrorCode.E_RECOVERABLE] = ("RecoverableError", 503, True, "A temporary error occurred. Please try again.")
ERROR_REGISTRY[ErrorCode.E_FATAL] = ("FatalError", 500, False, "A critical error occurred. Please restart the server.")
ERROR_REGISTRY[ErrorCode.E_BAD_REQUEST] = ("ValidationError", 400, False, "Invalid request.")
ERROR_REGISTRY[ErrorCode.E_VAL_REQUEST] = ("ValidationError", 422, False, "Request validation failed.")
ERROR_REGISTRY[ErrorCode.E_VAL_FIELD] = ("ValidationError", 422, False, "Field validation failed.")
ERROR_REGISTRY[ErrorCode.E_NOT_IMPLEMENTED] = ("NotImplementedAppError", 501, False, "This feature is not yet implemented.")
ERROR_REGISTRY[ErrorCode.E_AUTH_MISSING] = ("AuthError", 401, False, "Authentication failed.")
ERROR_REGISTRY[ErrorCode.E_AUTH_FORBIDDEN] = ("AuthError", 403, False, "You do not have permission.")
ERROR_REGISTRY[ErrorCode.E_NOT_FOUND] = ("NotFoundError", 404, False, "The requested resource was not found.")
ERROR_REGISTRY[ErrorCode.E_CONFLICT] = ("ConflictError", 409, True, "Resource is busy or in a conflicting state.")
ERROR_REGISTRY[ErrorCode.E_INFRA_BUSY] = ("ConflictError", 409, True, "Service is busy.")
ERROR_REGISTRY[ErrorCode.E_RATE_LIMITED] = ("ResourceExhaustedError", 429, True, "Too many requests. Please slow down.")
ERROR_REGISTRY[ErrorCode.E_INFRA_RATE_LIMIT] = ("ResourceExhaustedError", 429, True, "Rate limit exceeded.")
ERROR_REGISTRY[ErrorCode.E_MEMORY_PRESSURE] = ("AppError", 503, True, "Server is low on memory.")
ERROR_REGISTRY[ErrorCode.E_INFRA_STARTUP] = ("ConfigError", 503, False, "Service is starting up.")
ERROR_REGISTRY[ErrorCode.E_INFRA_REGISTRY] = ("ConfigError", 503, False, "Model registry unavailable.")
ERROR_REGISTRY[ErrorCode.E_INFRA_TIMEOUT] = ("TimeoutAppError", 408, True, "Request timed out.")
ERROR_REGISTRY[ErrorCode.E_INFRA_GENERATION] = ("ModelError", 503, True, "Generation failed.")
ERROR_REGISTRY[ErrorCode.E_TIMEOUT] = ("TimeoutAppError", 408, True, "Operation timed out.")
ERROR_REGISTRY[ErrorCode.E_MODEL_ERROR] = ("ModelError", 503, False, "Model encountered an error.")
ERROR_REGISTRY[ErrorCode.E_MODEL_OOM] = ("ModelOOMError", 503, True, "Model ran out of memory.")
ERROR_REGISTRY[ErrorCode.E_MODEL_TIMEOUT] = ("ModelTimeoutError", 503, True, "Model timed out.")
ERROR_REGISTRY[ErrorCode.E_MODEL_LOADING] = ("AppError", 503, True, "Model is still loading.")
ERROR_REGISTRY[ErrorCode.E_TASK_ERROR] = ("TaskError", 500, True, "A background task failed.")
ERROR_REGISTRY[ErrorCode.E_STATE_IDLE] = ("ConflictError", 409, False, "No active training session.")
ERROR_REGISTRY[ErrorCode.E_CONFIG] = ("ConfigError", 500, False, "Server configuration error.")
ERROR_REGISTRY[ErrorCode.E_NETWORK] = ("RecoverableError", 503, True, "Network connection failed.")
ERROR_REGISTRY[ErrorCode.E_SHELL_SECURITY] = ("AuthError", 403, False, "Shell command blocked by security policy.")
ERROR_REGISTRY[ErrorCode.E_CANCEL_FAILED] = ("ConflictError", 409, False, "Failed to cancel the operation.")
ERROR_REGISTRY[ErrorCode.E_ENV_MISSING] = ("ConfigError", 500, False, "Required environment not available.")
ERROR_REGISTRY[ErrorCode.E_UNKNOWN_COLLECTION] = ("NotFoundError", 404, False, "Unknown collection or store.")
ERROR_REGISTRY[ErrorCode.E_CREATE_FAILED] = ("ValidationError", 400, False, "Resource creation failed.")


def get_error_info(code: str) -> tuple[str, int, bool, str] | None:
    """Look up error info by code string. Returns (class_name, http_status, recoverable, user_message)."""
    try:
        ec = ErrorCode(code)
        return ERROR_REGISTRY.get(ec)
    except ValueError:
        return None


# ── Integration helpers ──


def emit_error_event(error: AppError, source: str = "") -> None:
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
            loop.create_task(bus.emit("error.raised", error.to_dict(), source=source or error.source))
        else:
            bus.emit_sync("error.raised", error.to_dict(), source=source or error.source)
    except Exception as e:
        import logging
        logging.getLogger("slo.errors").warning("emit_error_event failed: %s", e, extra={
            "error_code": error.code, "source": source,
        })


def classify_exception(exc: BaseException) -> AppError:
    """Classify a raw Python exception into the best-matching AppError.

    Already-classified AppError instances are returned as-is.
    """
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, TimeoutError):
        msg = str(exc).lower()
        if any(k in msg for k in ("model", "generation", "infer", "generate")):
            return ModelTimeoutError(
                message=str(exc),
                details={"original_type": type(exc).__name__},
                cause=exc,
            )
        return TimeoutAppError(
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
            code="E_NETWORK",
            user_message="Network connection failed. Please check your connection.",
            details={"original_type": type(exc).__name__},
            cause=exc,
        )
    if isinstance(exc, FileNotFoundError):
        return NotFoundError(
            message=str(exc),
            user_message="File not found.",
            details={"path": str(exc) if hasattr(exc, "filename") else ""},
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
    if isinstance(exc, KeyError):
        return NotFoundError(
            message=str(exc),
            user_message="Resource not found.",
            details={"original_type": "KeyError", "key": str(exc)},
            cause=exc,
        )
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "already in progress" in msg:
            return ConflictError(
                message=str(exc),
                user_message="Training already in progress.",
                details={"original_type": "RuntimeError"},
                cause=exc,
            )
        return AppError.from_exception(exc)
    return AppError.from_exception(exc)
