"""
Exception handlers extracted from main.py.

Provides FastAPI exception handlers for:
- SloughGPTDomainError (internal domain errors)
- ValidationError (Pydantic)
- HTTPException (FastAPI)
- BaseException / unhandled errors (catch-all)

Register via ``register_all_handlers(app)``.
"""

from __future__ import annotations

import logging
import traceback

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger("man.exception_handlers")


async def _domain_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch domain-layer exceptions (SloughGPTDomainError subclasses)."""
    msg = str(exc) or "Domain error"
    logger.warning("Domain error on %s %s: %s", request.method, request.url.path, msg)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": msg},
    )


async def _validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Catch Pydantic validation errors."""
    errors = exc.errors()
    logger.warning("Validation error on %s: %s", request.url.path, errors)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation failed", "details": errors},
    )


async def _request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Catch FastAPI request validation errors."""
    errors = exc.errors()
    logger.warning("Request validation error on %s: %s", request.url.path, errors)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Request validation failed", "details": errors},
    )


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Re-raise HTTPExceptions as JSON."""
    from fastapi import HTTPException as _HTTPException
    h = exc  # type: _HTTPException
    logger.info("HTTP %d on %s %s: %s", h.status_code, request.method, request.url.path, h.detail)
    return JSONResponse(
        status_code=h.status_code,
        content={"error": h.detail},
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected errors — returns 500 with correlation ID."""
    corr_id = getattr(request.state, "correlation_id", "-")
    logger.exception("Unhandled error on %s %s [%s]", request.method, request.url.path, corr_id)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "correlation_id": corr_id,
        },
    )


def register_all_handlers(app: FastAPI):
    """Register all exception handlers on a FastAPI instance."""
    try:
        from domains.core.base import SloughGPTDomainError
        app.add_exception_handler(SloughGPTDomainError, _domain_error_handler)
    except ImportError:
        pass

    app.add_exception_handler(ValidationError, _validation_error_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_error_handler)

    from fastapi import HTTPException
    app.add_exception_handler(HTTPException, _http_exception_handler)

    app.add_exception_handler(Exception, _unhandled_error_handler)
