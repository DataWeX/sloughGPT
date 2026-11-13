"""Human-readable size and duration formatting (backend side).

Frontend equivalents live in ``apps/web/lib/format-bytes.ts`` and
``apps/web/lib/formatDuration.ts`` — keep the unit ladders in sync.
"""

from __future__ import annotations

__all__ = ["format_size", "format_time"]


def format_size(size_bytes: int) -> str:
    """Format a byte count as ``"1.5 MB"`` (B → PB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_time(seconds: float) -> str:
    """Format a duration as ``"45.0s"`` / ``"12.5m"`` / ``"2.1h"`` / ``"3.0d"``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"
