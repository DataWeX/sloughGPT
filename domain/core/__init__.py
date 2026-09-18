"""core — Core database and utilities.

Public API:
    get_db
    get_rag_service
"""

from domain.core._internal.database import get_db
from domain.core._internal.rag_service import get_rag_service

__all__ = [
    "get_db",
    "get_rag_service",
]
