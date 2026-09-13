"""Backward-compatibility shim — imports from the new ``domain.core`` package."""

from domain.core import (
    get_db,
)

__all__ = [
    "get_db",
]
