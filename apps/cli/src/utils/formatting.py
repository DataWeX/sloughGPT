"""
Formatting utilities for consistent CLI output.

Provides file size formatting, time formatting, and text wrapping.
"""
from typing import Optional


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    if size_bytes < 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024.0:
            if size == int(size):
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0

    return f"{size:.1f} PB"


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "2m 30s")
    """
    if seconds < 0:
        return "0s"

    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining = seconds % 60

    if minutes < 60:
        if remaining < 1:
            return f"{minutes}m"
        return f"{minutes}m {remaining:.0f}s"

    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m"


def format_number(n: int) -> str:
    """Format number with thousands separators.

    Args:
        n: Number to format

    Returns:
        Formatted string (e.g., "1,234,567")
    """
    return f"{n:,}"


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to append when truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def wrap_text(text: str, width: int = 80) -> list[str]:
    """Wrap text to specified width.

    Args:
        text: Text to wrap
        width: Maximum line width

    Returns:
        List of wrapped lines
    """
    if not text:
        return []

    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word)
        if current_length + word_length + (1 if current_line else 0) <= width:
            current_line.append(word)
            current_length += word_length + (1 if current_line else 0)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length

    if current_line:
        lines.append(" ".join(current_line))

    return lines or [""]


def indent(text: str, spaces: int = 2) -> str:
    """Indent text by specified number of spaces.

    Args:
        text: Text to indent
        spaces: Number of spaces

    Returns:
        Indented text
    """
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))


def pad(text: str, width: int, alignment: str = "left") -> str:
    """Pad text to specified width.

    Args:
        text: Text to pad
        width: Target width
        alignment: "left", "right", or "center"

    Returns:
        Padded text
    """
    if alignment == "right":
        return text.rjust(width)
    elif alignment == "center":
        return text.center(width)
    return text.ljust(width)
