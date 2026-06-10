"""
Error logging router — accepts frontend JS errors for server-side monitoring.

- POST /errors/log: log one or more client-side errors
- GET /errors/recent: retrieve recent errors (for admin view)
"""

import logging
import time
import uuid
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger("man.errors")

router = APIRouter(prefix="/errors", tags=["errors"])

# In-memory ring buffer for recent errors
MAX_ERRORS = 500
_error_buffer: list[dict] = []
_error_count_since_clear = 0


class ErrorEntry(BaseModel):
    message: str
    source: str = "web"
    stack: Optional[str] = None
    url: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None
    timestamp: Optional[str] = None
    metadata: Optional[dict] = None


class ErrorBatch(BaseModel):
    errors: List[ErrorEntry]


@router.post("/log")
async def log_errors(batch: ErrorBatch, request: Request):
    """
    Accept frontend JS errors and store them in the ring buffer.

    Returns a 201 with the count of logged errors.
    """
    global _error_count_since_clear
    now = datetime.utcnow().isoformat()
    client_host = request.client.host if request.client else "unknown"

    for entry in batch.errors:
        error_record = {
            "id": uuid.uuid4().hex[:12],
            "message": entry.message,
            "source": entry.source or "web",
            "stack": entry.stack,
            "url": entry.url,
            "line": entry.line,
            "col": entry.col,
            "client_host": client_host,
            "timestamp": entry.timestamp or now,
            "metadata": entry.metadata or {},
        }
        _error_buffer.append(error_record)
        _error_count_since_clear += 1
        logger.error(
            "CLIENT ERROR [%s] %s | %s:%s %s",
            error_record["id"],
            entry.message[:120],
            entry.url or "?",
            entry.line or 0,
            entry.col or 0,
        )

    # Trim buffer
    while len(_error_buffer) > MAX_ERRORS:
        _error_buffer.pop(0)

    return {"status": "ok", "logged": len(batch.errors)}


@router.get("/recent")
async def get_recent_errors(limit: int = 50, offset: int = 0):
    """Return paginated client errors (newest first)."""
    total = len(_error_buffer)
    start = max(0, total - offset - limit)
    end = max(0, total - offset)
    return {
        "errors": list(reversed(_error_buffer[start:end])),
        "unread_count": _error_count_since_clear,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.delete("/clear")
async def clear_errors():
    """Clear all stored errors and reset unread counter."""
    _error_buffer.clear()
    global _error_count_since_clear
    _error_count_since_clear = 0
    return {"status": "ok", "cleared": True}


@router.get("/unread")
async def unread_count():
    """Return the number of errors logged since last clear."""
    return {"unread_count": _error_count_since_clear}
