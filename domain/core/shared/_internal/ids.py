"""Random identifier generation."""

from __future__ import annotations

import random
import string

__all__ = ["generate_id"]


def generate_id(prefix: str = "") -> str:
    """Generate a random 8-char lowercase-alphanumeric ID, optionally prefixed.

    Example:
        >>> id = generate_id("run_")
        >>> id.startswith("run_") and len(id) == 12
        True
    """
    chars = string.ascii_lowercase + string.digits
    random_id = "".join(random.choices(chars, k=8))
    return f"{prefix}{random_id}" if prefix else random_id
