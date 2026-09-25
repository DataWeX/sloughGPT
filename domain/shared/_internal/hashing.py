"""String hashing helpers."""

from __future__ import annotations

import hashlib

__all__ = ["hash_string"]


def hash_string(s: str, algorithm: str = "sha256") -> str:
    """Hex-digest *s* with ``sha256``, ``md5``, or ``sha1``.

    Unknown algorithms return the input unchanged (historical behaviour —
    callers rely on it as a passthrough).
    """
    if algorithm == "sha256":
        return hashlib.sha256(s.encode()).hexdigest()
    if algorithm == "md5":
        return hashlib.md5(s.encode()).hexdigest()
    if algorithm == "sha1":
        return hashlib.sha1(s.encode()).hexdigest()
    return s
