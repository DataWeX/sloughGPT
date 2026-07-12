"""Minimal HTTP request object."""

import json
from typing import Any, Dict, Optional


class Request:
    """Parsed HTTP request."""

    __slots__ = ("method", "path", "headers", "body", "query", "_json_cache")

    def __init__(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: bytes,
        query: Optional[Dict[str, str]] = None,
    ):
        self.method = method
        self.path = path
        self.headers = headers
        self.body = body
        self.query = query or {}
        self._json_cache: Any = _MISSING

    async def json(self) -> Any:
        """Parse body as JSON (cached)."""
        if self._json_cache is _MISSING:
            try:
                self._json_cache = json.loads(self.body) if self.body else {}
            except json.JSONDecodeError:
                self._json_cache = {}
        return self._json_cache

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "")

    @property
    def is_disconnected(self) -> bool:
        return False  # TODO: track connection state


_MISSING = object()
