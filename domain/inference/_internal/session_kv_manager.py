"""Backward-compatibility shim — SessionKVManager now lives in
``domain.infrastructure._internal.kv_cache.session``.

Kept so existing imports of ``domain.inference._internal.session_kv_manager``
keep resolving to the canonical implementation.
"""

from __future__ import annotations

from domain.infrastructure._internal.kv_cache.session import SessionKVManager

__all__ = ["SessionKVManager"]
