"""files — compressed blob store (sync core, standalone-runnable).

Public API:
    FileStore — put/get/list/delete with per-class compress-at-rest:
        text/code → zstd (stored frame doubles as wire format),
        model weights → raw (proven incompressible), unknown → zstd-if-smaller.
"""
from __future__ import annotations

from domain.files._internal.store import (
    DEFAULT_ROOT,
    MODEL_EXTENSIONS,
    TEXT_EXTENSIONS,
    FileStore,
)

__all__ = ["FileStore", "DEFAULT_ROOT", "MODEL_EXTENSIONS", "TEXT_EXTENSIONS"]
