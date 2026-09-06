"""
Modular Class-Based API Error Handler.

Provides a registry-based error handling system where each domain
registers its own error handler class. The central APIErrorHandler
dispatches errors to the appropriate domain handler.

Usage:
    # Register domain handlers
    handler = APIErrorHandler()
    handler.register(TrainingErrorHandler())
    handler.register(InferenceErrorHandler())

    # Handle an error
    response = handler.handle(error, request_context)

    # Or use as a FastAPI middleware
    handler.attach(app)
"""

from __future__ import annotations

import logging
from typing import Any

from domains.infrastructure.errors import (
    AppError,
    AuthError,
    ConfigError,
    ModelError,
    ModelOOMError,
    ModelTimeoutError,
    NotFoundError,
    ResourceExhaustedError,
    TaskError,
    ValidationError,
    classify_exception,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from schemas.common import error_response, get_correlation_id

logger = logging.getLogger("slo.error_handler")


# ── Base Domain Error Handler ──


class DomainErrorHandler:
    """Base class for domain-specific error handlers.

    Subclass this to handle errors from a specific domain (training,
    inference, soul, agents, etc.). Each handler declares which error
    types it handles and provides custom handling logic.

    Attributes:
        domain: Domain identifier (e.g. "training", "inference").
        handled_errors: Tuple of AppError subclasses this handler processes.
    """

    domain: str = "base"
    handled_errors: tuple[type[AppError], ...] = ()

    def __init__(
        self,
        domain: str | None = None,
        handled_errors: tuple[type[AppError], ...] | None = None,
    ) -> None:
        if domain is not None:
            self.domain = domain
        if handled_errors is not None:
            self.handled_errors = handled_errors

    def can_handle(self, error: AppError) -> bool:
        """Check if this handler can handle the given error."""
        return isinstance(error, self.handled_errors)

    def handle(
        self,
        error: AppError,
        request: Request | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Handle the error and return a response body.

        Override this method to add domain-specific logic:
        - Log domain-specific details
        - Add recovery suggestions
        - Transform error messages

        NOTE: EventBus emission is handled by the exception handler, NOT here.
        This avoids double emission.

        Args:
            error: The AppError to handle.
            request: Optional FastAPI request for context.
            context: Optional extra context from the router.

        Returns:
            dict matching the error response shape.
        """
        self._log(error, request, context)
        return self._build_response(error, request)

    def _log(
        self,
        error: AppError,
        request: Request | None,
        context: dict[str, Any] | None,
    ) -> None:
        """Log the error with domain context."""
        log_fn = logger.error if error.http_status >= 500 else logger.warning
        extra = {
            "domain": self.domain,
            "error_code": error.code,
            "recoverable": error.recoverable,
        }
        if context:
            extra["context"] = context
        if request:
            extra["method"] = request.method
            extra["path"] = request.url.path

        log_fn(
            "[%s] %s: %s",
            self.domain,
            error.code,
            error.message,
            extra=extra,
        )

    def _build_response(
        self,
        error: AppError,
        request: Request | None,
    ) -> dict[str, Any]:
        """Build the error response body."""
        details = error.details.copy() if error.details else {}

        # Add domain metadata
        details["_domain"] = self.domain
        details["_handler"] = self.__class__.__name__

        return error_response(
            message=error.user_message,
            code=error.code,
            details=details if logger.isEnabledFor(logging.DEBUG) else None,
        )


# ── Concrete Domain Handlers ──


class TrainingErrorHandler(DomainErrorHandler):
    """Handle errors from the training pipeline."""

    domain = "training"
    handled_errors = (TaskError,)

    def handle(
        self,
        error: AppError,
        request: Request | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Enrich with training-specific recovery suggestions
        if isinstance(error, TaskError):
            error.user_message = "Training task failed. Check GPU memory and dataset integrity."
        return super().handle(error, request, context)


class InferenceErrorHandler(DomainErrorHandler):
    """Handle errors from the inference/generation pipeline."""

    domain = "inference"
    handled_errors = (ModelError, ModelOOMError, ModelTimeoutError)

    def handle(
        self,
        error: AppError,
        request: Request | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Add model-specific recovery hints
        if isinstance(error, ModelOOMError):
            error.details["suggestion"] = (
                "Try a smaller model, reduce max_tokens, or lower batch size."
            )
        elif isinstance(error, ModelTimeoutError):
            error.details["suggestion"] = "Try a shorter prompt or reduce max_tokens."
        return super().handle(error, request, context)


class AuthErrorHandler(DomainErrorHandler):
    """Handle authentication and authorization errors."""

    domain = "auth"
    handled_errors = (AuthError,)

    def handle(
        self,
        error: AppError,
        request: Request | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Never log auth details in production
        if error.http_status == 401:
            error.user_message = "Invalid or missing authentication token."
        elif error.http_status == 403:
            error.user_message = "You do not have permission to perform this action."
        return super().handle(error, request, context)


class ResourceErrorHandler(DomainErrorHandler):
    """Handle resource-related errors (not found, rate limited, exhausted)."""

    domain = "resource"
    handled_errors = (NotFoundError, ResourceExhaustedError)

    def handle(
        self,
        error: AppError,
        request: Request | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(error, ResourceExhaustedError):
            error.details["retry_after"] = error.details.get("retry_after", 30)
        return super().handle(error, request, context)


class ValidationErrorHandler(DomainErrorHandler):
    """Handle validation and configuration errors."""

    domain = "validation"
    handled_errors = (ValidationError, ConfigError)

    def handle(
        self,
        error: AppError,
        request: Request | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Validation errors are expected — log at debug level
        logger.debug(
            "[%s] Validation error: %s",
            self.domain,
            error.message,
        )
        return error_response(
            message=error.user_message,
            code=error.code,
            details=error.details if logger.isEnabledFor(logging.DEBUG) else None,
        )


class DefaultErrorHandler(DomainErrorHandler):
    """Catch-all handler for unclassified errors."""

    domain = "general"
    handled_errors = (AppError,)

    def handle(
        self,
        error: AppError,
        request: Request | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return super().handle(error, request, context)


# ── Central Error Handler Registry ──


class APIErrorHandler:
    """Central error handler with domain-specific dispatch.

    Maintains a registry of DomainErrorHandler instances and dispatches
    errors to the first matching handler. Falls back to DefaultErrorHandler.

    Usage:
        handler = APIErrorHandler()
        handler.register(TrainingErrorHandler())
        handler.register(InferenceErrorHandler())

        # Handle an error
        body = handler.handle(error, request)

        # Attach to FastAPI app
        handler.attach(app)
    """

    def __init__(self) -> None:
        self._handlers: list[DomainErrorHandler] = []
        self._default = DefaultErrorHandler()
        self._attached = False

    def register(self, handler: DomainErrorHandler) -> APIErrorHandler:
        """Register a domain error handler.

        Handlers are checked in registration order. Register more
        specific handlers first.

        Args:
            handler: The DomainErrorHandler to register.

        Returns:
            self, for fluent chaining.
        """
        self._handlers.append(handler)
        logger.debug(
            "Registered error handler: %s (domain=%s, errors=%s)",
            handler.__class__.__name__,
            handler.domain,
            [e.__name__ for e in handler.handled_errors],
        )
        return self

    def unregister(self, handler_type: type[DomainErrorHandler]) -> APIErrorHandler:
        """Unregister a handler by type.

        Args:
            handler_type: The handler class to remove.

        Returns:
            self, for fluent chaining.
        """
        self._handlers = [h for h in self._handlers if not isinstance(h, handler_type)]
        return self

    def find_handler(self, error: AppError) -> DomainErrorHandler:
        """Find the first handler that can handle the given error.

        Args:
            error: The AppError to match.

        Returns:
            The matching DomainErrorHandler, or DefaultErrorHandler.
        """
        for handler in self._handlers:
            if handler.can_handle(error):
                return handler
        return self._default

    def handle(
        self,
        error: AppError | Exception,
        request: Request | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Handle an error by dispatching to the appropriate domain handler.

        Args:
            error: The error to handle. If not an AppError, it will be
                   classified first via classify_exception().
            request: Optional FastAPI request for context.
            context: Optional extra context from the router.

        Returns:
            dict matching the error response shape.
        """
        # Classify if needed
        if not isinstance(error, AppError):
            error = classify_exception(error)

        handler = self.find_handler(error)
        return handler.handle(error, request, context)

    def handle_to_response(
        self,
        error: AppError | Exception,
        request: Request | None = None,
        context: dict[str, Any] | None = None,
    ) -> JSONResponse:
        """Handle an error and return a JSONResponse.

        Args:
            error: The error to handle.
            request: Optional FastAPI request for context.
            context: Optional extra context.

        Returns:
            FastAPI JSONResponse with the appropriate status code.
        """
        if not isinstance(error, AppError):
            error = classify_exception(error)

        handler = self.find_handler(error)
        body = handler.handle(error, request, context)

        # Extract correlation ID
        cid = get_correlation_id()
        if cid:
            body["correlation_id"] = cid

        return JSONResponse(
            status_code=error.http_status,
            content=body,
        )

    def attach(self, app: FastAPI) -> None:
        """Attach this handler to a FastAPI app as the exception handler.

        Registers a catch-all Exception handler that dispatches through
        the domain handler registry.

        Args:
            app: The FastAPI application instance.
        """
        if self._attached:
            logger.warning("APIErrorHandler already attached to app")
            return

        error_handler_self = self

        async def _catch_all(request: Request, exc: Exception) -> JSONResponse:
            return error_handler_self.handle_to_response(exc, request)

        app.add_exception_handler(Exception, _catch_all)
        self._attached = True
        logger.info(
            "APIErrorHandler attached with %d domain handlers: %s",
            len(self._handlers),
            [h.domain for h in self._handlers],
        )

    @property
    def handler_count(self) -> int:
        """Return the number of registered handlers."""
        return len(self._handlers)

    @property
    def domains(self) -> list[str]:
        """Return list of registered domain names."""
        return [h.domain for h in self._handlers]


# ── Factory ──


def create_default_error_handler() -> APIErrorHandler:
    """Create an APIErrorHandler with all default domain handlers registered.

    Returns:
        Configured APIErrorHandler with training, inference, auth,
        resource, and validation handlers.
    """
    return (
        APIErrorHandler()
        .register(TrainingErrorHandler())
        .register(InferenceErrorHandler())
        .register(AuthErrorHandler())
        .register(ResourceErrorHandler())
        .register(ValidationErrorHandler())
    )
