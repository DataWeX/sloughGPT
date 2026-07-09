"""Minimal ASGI-like app with route registration."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from domains.server.request import Request
from domains.server.response import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("man.server")


class App:
    """Minimal HTTP app — route registration + dispatch."""

    def __init__(self, title: str = "sloughgpt", version: str = "0.1.0"):
        self.title = title
        self.version = version
        self.routes: List[Tuple[str, str, Callable]] = []  # (method, path, handler)
        self._startup: List[Callable] = []
        self._shutdown: List[Callable] = []

    # ── Route decorators ─────────────────────────────────────────────────

    def get(self, path: str):
        """Register GET handler."""
        def decorator(fn):
            self.routes.append(("GET", path, fn))
            return fn
        return decorator

    def post(self, path: str):
        """Register POST handler."""
        def decorator(fn):
            self.routes.append(("POST", path, fn))
            return fn
        return decorator

    def put(self, path: str):
        """Register PUT handler."""
        def decorator(fn):
            self.routes.append(("PUT", path, fn))
            return fn
        return decorator

    def delete(self, path: str):
        """Register DELETE handler."""
        def decorator(fn):
            self.routes.append(("DELETE", path, fn))
            return fn
        return decorator

    def route(self, path: str, methods: List[str] = None):
        """Register handler for multiple methods."""
        def decorator(fn):
            for m in (methods or ["GET", "POST"]):
                self.routes.append((m, path, fn))
            return fn
        return decorator

    def startup(self, fn):
        """Register startup hook."""
        self._startup.append(fn)
        return fn

    def shutdown(self, fn):
        """Register shutdown hook."""
        self._shutdown.append(fn)
        return fn

    # ── Dispatch ─────────────────────────────────────────────────────────

    async def handle(self, request: Request) -> Union[Response, StreamingResponse]:
        """Route a request to its handler."""
        path = request.path.rstrip("/") or "/"

        # Exact match first
        for method, route_path, handler in self.routes:
            if route_path.rstrip("/") == path and (method == request.method or method == "*"):
                return await self._call_handler(handler, request)

        # 404
        return JSONResponse(
            {"error": f"Not found: {request.method} {request.path}"},
            status=404,
        )

    async def _call_handler(self, handler: Callable, request: Request):
        """Call a handler, normalizing its return type."""
        try:
            result = await handler(request)
        except Exception as e:
            logger.exception("Handler error: %s", e)
            return JSONResponse({"error": str(e)}, status=500)

        # Normalize return type
        if isinstance(result, (Response, StreamingResponse)):
            return result
        if isinstance(result, dict):
            return JSONResponse(result)
        if isinstance(result, tuple):
            body, status = result
            if isinstance(body, dict):
                return JSONResponse(body, status=status)
            return Response(str(body), status=status)
        return JSONResponse(result)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def run_startup(self):
        for fn in self._startup:
            try:
                result = fn()
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.warning("Startup hook failed: %s", e)

    async def run_shutdown(self):
        for fn in self._shutdown:
            try:
                result = fn()
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.warning("Shutdown hook failed: %s", e)
