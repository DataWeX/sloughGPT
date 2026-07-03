"""MIME type registration for AML.

Registers ``application/aml`` with Python's ``mimetypes`` module
so that ``.aml`` files are recognized system-wide.
"""

import mimetypes
import os
from pathlib import Path

_MIME = "application/aml"
_EXT = ".aml"


def register_mime() -> None:
    """Register ``application/aml`` with the system MIME database.

    Safe to call multiple times — idempotent.
    """
    if mimetypes.guess_type("x.aml") != (_MIME, None):
        mimetypes.add_type(_MIME, _EXT)

    # also register in ~/.mime.types on Unix
    mime_path = Path.home() / ".mime.types"
    if mime_path.exists():
        content = mime_path.read_text()
        if _MIME not in content:
            with open(mime_path, "a") as f:
                f.write(f"\n{_MIME}\t\taml\n")

    # register in /etc/mailcap for mail clients (read-only, skip if no perms)


def deregister_mime() -> None:
    """Remove ``application/aml`` registration (for testing cleanup)."""
    if mimetypes.guess_type("x.aml") == (_MIME, None):
        # Python 3.12+ has mimetypes.unregister
        if hasattr(mimetypes, "unregister"):
            mimetypes.unregister(_MIME, _EXT)


def detect_mime(path: str) -> str | None:
    """Guess MIME type for a file path."""
    mime, _ = mimetypes.guess_type(path)
    return mime


def is_aml(path: str) -> bool:
    """Check if a file path has AML MIME type."""
    return detect_mime(path) == _MIME
