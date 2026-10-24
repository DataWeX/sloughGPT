"""FileHub HTTP app — a tiny hand-rolled ASGI projection of FileHub.

Deliberately not FastAPI: the hub is a standalone process with its own IP
and a handful of routes; the wire layer (``WireServer(asgi_app=...)``)
already owns framing, keep-alive and streaming. Async lives only at this
seam — every store/hub call is ``asyncio.to_thread``-wrapped or bounded
per-connection (the wire runs a per-request loop per connection thread,
so a chunked blocking read only ever stalls its own download).

Routes:
    GET  /health                 — status + blob count
    GET  /files                  — JSON listing (id, name, sizes, codec)
    GET  /files/{id}/download    — raw bytes; ``Accept-Encoding: zstd``
                                   serves the stored frame untouched (text
                                   class); ``Range`` supported for raw blobs
                                   (model class — resumable client pulls)
    POST /pull                   — JSON {"url": ..., "filename": ...}
                                   downcraft pull-through into the store
    POST /files?name=...         — raw-body upload, chunked to disk

Auth: ``X-FileHub-Token`` header or ``?token=`` query; open when no token
is configured (loopback/dev).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from urllib.parse import parse_qs

from domain.files import FileHub

logger = logging.getLogger("slo.filehub")

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


def _hdr(scope) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}


async def _send_json(send, status: int, obj) -> None:
    body = json.dumps(obj).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _brief(meta: dict) -> dict:
    return {
        "id": meta["file_id"],
        "name": meta.get("original_name", meta["filename"]),
        "size": meta.get("size_bytes", 0),
        "stored": meta.get("stored_size", 0),
        "codec": meta.get("codec", "raw"),
        "uploaded_at": meta.get("uploaded_at", 0.0),
        "tags": meta.get("tags", []),
    }


def make_app(hub: FileHub):
    """Build the ASGI callable bound to *hub*."""

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            # WireServer runs with lifespan="off"; answer anyway if asked.
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
                await send({"type": "lifespan.startup.complete"})
        if scope["type"] != "http":
            return

        method: str = scope["method"]
        # wire's scope is uvicorn-style: path is an unquoted str, query
        # arrives separately as query_string (bytes).
        raw_path = scope["path"]
        path = raw_path.decode("latin-1") if isinstance(raw_path, bytes) else raw_path
        qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        headers = _hdr(scope)

        token = headers.get("x-filehub-token") or (qs.get("token") or [""])[0]
        if not hub.check_token(token):
            await _send_json(send, 401, {"error": "unauthorized"})
            return

        if method == "GET" and path == "/health":
            await _send_json(
                send, 200, {"status": "ok", "files": len(hub.store.list()), "root": str(hub.store.root)}
            )
            return

        if method == "GET" and path == "/files":
            await _send_json(send, 200, {"files": [_brief(m) for m in hub.store.list()]})
            return

        if method == "GET" and path.startswith("/files/") and path.endswith("/download"):
            await _download(send, hub, path[len("/files/") : -len("/download")], headers)
            return

        if method == "POST" and path == "/pull":
            await _pull(send, hub, receive)
            return

        if method == "POST" and path == "/files":
            name = (qs.get("name") or [""])[0]  # parse_qs already unquotes
            await _upload(send, hub, receive, name)
            return

        await _send_json(send, 404, {"error": "not found"})

    return app


async def _download(send, hub: FileHub, file_id: str, headers: dict) -> None:
    try:
        meta = hub.store.get_meta(file_id)
    except FileNotFoundError:
        await _send_json(send, 404, {"error": f"no such file: {file_id}"})
        return

    codec = meta.get("codec", "raw")
    accept = headers.get("accept-encoding", "")
    rng = headers.get("range")
    raw_size = meta.get("size_bytes", 0)
    common = [
        (b"vary", b"Accept-Encoding"),
        (b"x-slo-raw-size", str(raw_size).encode()),
        (b"x-slo-codec", codec.encode()),
    ]

    # Text/code stored zstd + client advertises zstd + no Range: the stored
    # frame is the wire format — zero transcoding.
    if codec == "zstd" and "zstd" in accept and not rng:
        stored, meta = hub.store.open_stored(file_id)
        body = stored.read_bytes()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": common
                + [
                    (b"content-type", b"application/octet-stream"),
                    (b"content-encoding", b"zstd"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
        return

    # Raw blob (model class) with a Range: 206 slice, resumable pulls.
    if codec == "raw" and rng:
        m = _RANGE_RE.match(rng.strip())
        if not m or (not m.group(1) and not m.group(2)):
            await _send_json(send, 416, {"error": f"unsupported range: {rng}"})
            return
        start = int(m.group(1) or 0)
        end = int(m.group(2)) if m.group(2) else raw_size - 1
        end = min(end, raw_size - 1)
        if start >= raw_size or start > end:
            await send(
                {
                    "type": "http.response.start",
                    "status": 416,
                    "headers": [(b"content-range", f"bytes */{raw_size}".encode())],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return
        stored, meta = hub.store.open_stored(file_id)
        length = end - start + 1
        # No Content-Length: the wire adapter owns chunked framing for
        # streamed bodies (it rejects an app-declared length). Content-Range
        # still carries the slice coordinates for resumable clients.
        await send(
            {
                "type": "http.response.start",
                "status": 206,
                "headers": common
                + [
                    (b"content-type", b"application/octet-stream"),
                    (b"content-range", f"bytes {start}-{end}/{raw_size}".encode()),
                ],
            }
        )
        sent = 0
        pending: bytes | None = None
        with stored.open("rb") as f:
            f.seek(start)
            while sent < length:
                chunk = await asyncio.to_thread(f.read, min(1024 * 1024, length - sent))
                if not chunk:
                    break
                sent += len(chunk)
                if pending is not None:
                    await send({"type": "http.response.body", "body": pending, "more_body": True})
                pending = chunk
        # The LAST data frame must carry more_body=False — a trailing empty
        # frame after chunked data would emit a second terminal chunk and
        # desync a keep-alive connection.
        await send({"type": "http.response.body", "body": pending or b"", "more_body": False})
        return

    # Default: full download, decompressed on the way out for zstd blobs.
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": common + [(b"content-type", b"application/octet-stream")],
        }
    )
    pending = None
    for chunk in hub.store.iter_chunks(file_id):
        if pending is not None:
            await send({"type": "http.response.body", "body": pending, "more_body": True})
        pending = chunk
    # Single frame with more_body=False when the file is empty: the adapter
    # derives Content-Length from it instead of opening chunked framing.
    await send({"type": "http.response.body", "body": pending or b"", "more_body": False})


async def _pull(send, hub: FileHub, receive) -> None:
    raw = await _read_body(receive)
    try:
        payload = json.loads(raw or b"{}")
        url = payload["url"]
        filename = payload.get("filename")
    except (json.JSONDecodeError, KeyError, TypeError):
        await _send_json(send, 400, {"error": 'body must be JSON {"url": ..., "filename": ...}'})
        return
    try:
        meta = await asyncio.to_thread(hub.pull, url, filename)
    except ValueError as e:
        await _send_json(send, 400, {"error": str(e)})
        return
    except Exception as e:  # downcraft raises typed errors; hub maps them all
        logger.warning("pull failed: %s", e)
        await _send_json(send, 502, {"error": f"pull failed: {e}"})
        return
    await _send_json(send, 200, {**_brief(meta), "deduped": meta.get("deduped", False)})


async def _upload(send, hub: FileHub, receive, name: str) -> None:
    if not name or "." not in name.rsplit("/", 1)[-1]:
        await _send_json(send, 400, {"error": "pass ?name=<filename.ext>"})
        return
    spool = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
    try:
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            body = msg.get("body", b"")
            if body:
                spool.write(body)
            if not msg.get("more_body", False):
                break
        spool.seek(0)
        try:
            meta = await asyncio.to_thread(hub.store.put_stream, spool, name)
        except ValueError as e:
            await _send_json(send, 400, {"error": str(e)})
            return
        await _send_json(send, 201, _brief(meta))
    finally:
        spool.close()


async def _read_body(receive) -> bytes:
    parts = []
    while True:
        msg = await receive()
        if msg["type"] == "http.disconnect":
            return b""
        parts.append(msg.get("body", b""))
        if not msg.get("more_body", False):
            break
    return b"".join(parts)
