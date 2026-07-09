"""
Minimal HTTP server — zero dependencies, own ASGI-like interface.

Built from scratch using Python asyncio. No FastAPI, no uvicorn, no aiohttp.
Just asyncio + stdlib sockets.

Usage:
    from man_server import App, run

    app = App()

    @app.get("/health")
    async def health(req):
        return {"status": "ok"}

    run(app, host="127.0.0.1", port=8000)

Or from domains.server (same package):
    from domains.server import App, run
"""

from man_server.app import App
from man_server.protocol import run
from man_server.request import Request
from man_server.response import Response, JSONResponse, StreamingResponse

__all__ = ["App", "run", "Request", "Response", "JSONResponse", "StreamingResponse"]
__version__ = "0.1.0"
