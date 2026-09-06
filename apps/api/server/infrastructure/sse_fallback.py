"use strict"

"""
Consolidated SSE fallback utilities.

When ``domains.api.sse_envelope`` is available, this module re-exports its
functions.  When it is not available (e.g. during testing or when the
core-py package is not installed), this module provides canonical
fallback implementations so that routers do not need to duplicate them.
"""

import json
from typing import Any

try:
    from domains.api.sse_envelope import (
        sse_complete as _sse_complete,
    )
    from domains.api.sse_envelope import (
        sse_error as _sse_error,
    )
    from domains.api.sse_envelope import (
        sse_event as _sse_event,
    )
    from domains.api.sse_envelope import (
        sse_token as _sse_token,
    )

    sse_event = _sse_event
    sse_token = _sse_token
    sse_error = _sse_error
    sse_complete = _sse_complete
except ImportError:

    def sse_event(
        stream: str,
        phase: str,
        status: str,
        data: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        message: str = "",
        id: str | None = None,
    ) -> str:
        """Fallback SSE event builder."""
        env = {
            "stream": stream,
            "phase": phase,
            "status": status,
            "data": data or {},
            "meta": meta or {},
            "message": message,
        }
        if id is not None:
            env["id"] = id
        return "data: " + json.dumps(env, default=_json_safe) + "\n\n"

    def sse_token(
        stream: str,
        token: str,
        done: bool = False,
        meta: dict[str, Any] | None = None,
        elapsed_ms: float | None = None,
    ) -> str:
        """Fallback token streaming shortcut."""
        phase = "STREAMING"
        status = "complete" if done else "working"
        m = dict(meta) if meta else {}
        if done and elapsed_ms is not None:
            m["elapsed_ms"] = round(elapsed_ms, 1)
        return sse_event(stream, phase, status, {"token": token}, m, "")

    def sse_error(
        stream: str,
        phase: str,
        error: str,
        meta: dict[str, Any] | None = None,
        code: str | None = None,
        http_status: int | None = None,
    ) -> str:
        """Fallback error event builder."""
        data: dict[str, Any] = {"error": error}
        if code is not None:
            data["code"] = code
        if http_status is not None:
            data["http_status"] = http_status
        return sse_event(stream, phase, "error", data, meta or {}, f"Error: {error}")

    def sse_complete(
        stream: str,
        phase: str = "COMPLETE",
        data: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        message: str = "Done",
    ) -> str:
        """Fallback completion event builder."""
        return sse_event(stream, phase, "complete", data or {}, meta or {}, message)


def _json_safe(val: Any) -> Any:
    """Coerce non-serializable types to JSON-safe equivalents."""
    if hasattr(val, "tolist"):
        return val.tolist()
    if hasattr(val, "item"):
        return val.item()
    return val
