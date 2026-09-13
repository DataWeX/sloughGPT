"""Shared error formatting for shell commands.

Provides a single source of truth for user-friendly error messages,
including connection-error hints, permission detection, and colored output.
"""
from __future__ import annotations


def format_error(e: Exception, cmd: str = "", *, color: bool = True) -> str:
    """Format an exception into a user-friendly error message.

    Handles ConnectionError, Timeout, PermissionError, FileNotFoundError
    with contextual hints. Falls back to ``{etype}: {e}`` for other types.

    Args:
        e: The caught exception.
        cmd: Optional command name for prefixing the message.
        color: Whether to include ANSI color codes (default True).
    """
    import requests as _req
    etype = type(e).__name__

    # --- connection / network ---
    if isinstance(e, (_req.ConnectionError, ConnectionError)):
        hint = " Is the API server running? Use 'api start'."
        return f"  Connection failed ({etype}): {e}{hint}"
    if isinstance(e, (_req.Timeout, TimeoutError)):
        return f"  Request timed out ({etype}): {e}"

    # --- filesystem ---
    if isinstance(e, PermissionError):
        return f"  Permission denied: {e}"
    if isinstance(e, FileNotFoundError):
        return f"  File not found: {e}"

    # --- fallback ---
    prefix = f"  [{cmd}] " if cmd else "  "
    return f"{prefix}{etype}: {e}"
