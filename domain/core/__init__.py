"""core — Core database and utilities.

Public API:
    get_db
"""

from domain.core._internal.database import get_db

__all__ = [
    "get_db",
]
