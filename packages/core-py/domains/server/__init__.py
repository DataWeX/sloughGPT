"""
Minimal HTTP server — zero dependencies, own ASGI-like interface.

Built from scratch using Python asyncio. No FastAPI, no uvicorn, no aiohttp.
Just asyncio + stdlib sockets.

Usage:
    from domains.server import App, run

    app = App()

    @app.get("/health")
    async def health(req):
        return {"status": "ok"}

    run(app, host="127.0.0.1", port=8000)
"""

from domains.server.app import App
from domains.server.protocol import run
from domains.server.request import Request
from domains.server.response import Response, JSONResponse, StreamingResponse

__all__ = ["App", "run", "Request", "Response", "JSONResponse", "StreamingResponse"]
