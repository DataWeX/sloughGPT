"""Transport benchmark: our own wire layer, phase-3 (card ecea49c8).

Measures what the wire costs, in isolation, on loopback:
    keepalive    sequential RTT over one connection, sync handler mode
    keepalive_asgi  same through the ASGI adapter (adapter overhead = delta)
    uvicorn_asgi  A/B baseline: the SAME app under in-process uvicorn
    uvicorn_asgi_gapped  same with 100ms idle per request (capacity row)
    fresh        connect + request + close per measurement
    concurrent   N threads over their own keep-alive connections, req/s
    sse_ttfb     streaming response: time to first byte + total time

The A/B baseline runs uvicorn in its own thread + event loop on its own
port with the identical ASGI app — lifespan off on both sides (neither app
is lifespan-aware; the wire would otherwise wait out the handshake timeout),
same raw-socket client, interleaved runs. The delta is the transport, not
the app or the load generator. No shared entrypoints involved, so this does
not need the SLO_TRANSPORT coordination window.

Measured mechanism (2026-10-20): on a TIGHT keep-alive loop, uvicorn stalls
~40ms/request — its response head and body go out as two writes (no
TCP_NODELAY anywhere in uvicorn) and the client's delayed-ACK lets Nagle
hold the body until the 40ms timer. Our wire serializes head+body in ONE
send, so it never arms that stall. The ``uvicorn_asgi_gapped`` row (100ms
idle between requests → sub-ms) separates that pathology from uvicorn's
real per-request capacity. Results are honest loopback shapes, not network
latencies.

Usage:
    python scripts/benchmark_transport.py
    python scripts/benchmark_transport.py --requests 2000 --json
    python scripts/benchmark_transport.py --no-uvicorn
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "packages" / "wire-py"))

from wire.server import Response, WireServer  # noqa: E402

# ── minimal client (raw sockets: precise TTFB, no client-side pooling) ──


def _read_response(sock: socket.socket, t0: float) -> tuple[float, int, bytes]:
    """Read one framed response; returns (ttfb, status, body)."""
    buf = b""
    ttfb = t0
    while b"\r\n\r\n" not in buf:
        data = sock.recv(65536)
        if not data:
            raise ConnectionError("closed before headers")
        if not buf:
            ttfb = time.perf_counter()
        buf += data
    head, _, rest = buf.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n", 1)[0].split()[1])
    headers = {}
    for line in head.split(b"\r\n")[1:]:
        key, _, value = line.partition(b":")
        headers[key.strip().lower()] = value.strip().lower()

    if headers.get(b"transfer-encoding") == b"chunked":
        body = rest
        while not body.endswith(b"0\r\n\r\n"):
            data = sock.recv(65536)
            if not data:
                raise ConnectionError("closed mid-chunk")
            body += data
        return ttfb, status, _chunk_decode(body)

    length = int(headers.get(b"content-length", b"0"))
    body = rest
    while len(body) < length:
        data = sock.recv(65536)
        if not data:
            raise ConnectionError("closed mid-body")
        body += data
    return ttfb, status, body[:length]


def _chunk_decode(body: bytes) -> bytes:
    out, pos = bytearray(), 0
    while True:
        end = body.find(b"\r\n", pos)
        size = int(body[pos:end].split(b";", 1)[0], 16)
        pos = end + 2
        if size == 0:
            return bytes(out)
        out += body[pos : pos + size]
        pos += size + 2


def _exchange(conn: socket.socket, payload: bytes) -> tuple[float, float]:
    """Send one request, read one response; returns (ttfb, total), both seconds."""
    t0 = time.perf_counter()
    conn.sendall(payload)
    ttfb, _status, _body = _read_response(conn, t0)
    return ttfb - t0, time.perf_counter() - t0


_GET = b"GET /bench HTTP/1.1\r\nHost: wire\r\n\r\n"


def _percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {
        "p50_ms": round(pick(0.50) * 1e3, 3),
        "p95_ms": round(pick(0.95) * 1e3, 3),
        "max_ms": round(ordered[-1] * 1e3, 3),
        "mean_ms": round(statistics.fmean(samples) * 1e3, 3),
    }


# ── workloads ─────────────────────────────────────────────────────────


def bench_keepalive(port: int, requests: int, gap: float = 0.0) -> dict:
    conn = socket.create_connection(("127.0.0.1", port), timeout=10)
    lat: list[float] = []
    try:
        for _ in range(requests):
            _ttfb, total = _exchange(conn, _GET)
            lat.append(total)
            if gap:
                time.sleep(gap)
    finally:
        conn.close()
    out = _percentiles(lat)
    out["req_s"] = round(requests / sum(lat))
    if gap:
        out["gap_ms"] = round(gap * 1e3)
    return out


def bench_fresh(server: WireServer, requests: int) -> dict:
    lat: list[float] = []
    for _ in range(requests):
        conn = socket.create_connection(("127.0.0.1", server.port), timeout=10)
        try:
            _ttfb, total = _exchange(conn, _GET)
            lat.append(total)
        finally:
            conn.close()
    return _percentiles(lat)


def bench_concurrent(server: WireServer, threads: int, per_thread: int) -> dict:
    errors: list[BaseException] = []
    durations: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            conn = socket.create_connection(("127.0.0.1", server.port), timeout=10)
            local: list[float] = []
            try:
                for _ in range(per_thread):
                    _ttfb, total = _exchange(conn, _GET)
                    local.append(total)
            finally:
                conn.close()
            with lock:
                durations.extend(local)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    wall0 = time.perf_counter()
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    wall = time.perf_counter() - wall0
    if errors:
        raise errors[0]
    out = _percentiles(durations)
    out["req_s"] = round(len(durations) / wall)
    out["threads"] = threads
    return out


def bench_sse_ttfb(server: WireServer, samples: int) -> dict:
    payload = b"GET /sse HTTP/1.1\r\nHost: wire\r\n\r\n"
    ttfbs: list[float] = []
    totals: list[float] = []
    conn = socket.create_connection(("127.0.0.1", server.port), timeout=10)
    try:
        for _ in range(samples):
            ttfb, total = _exchange(conn, payload)
            ttfbs.append(ttfb)
            totals.append(total)
    finally:
        conn.close()
    out = _percentiles(ttfbs)
    out["total_p50_ms"] = round(statistics.median(totals) * 1e3, 3)
    return out


# ── servers under test ────────────────────────────────────────────────


class _UvicornTarget:
    """In-process uvicorn serving the SAME app — the A/B transport baseline.

    Own thread, own event loop, pre-bound ephemeral port handed to
    ``serve(sockets=)``; readiness = uvicorn's ``started`` flag. Symmetric
    with the wire side: lifespan off, raw-socket client, loopback.
    """

    def __init__(self, app) -> None:
        self._app = app
        self._port = 0
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        import uvicorn  # benchmark-only dependency (the competitor itself)

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(64)
        listener.setblocking(False)
        self._port = int(listener.getsockname()[1])
        server = uvicorn.Server(uvicorn.Config(self._app, log_level="warning", lifespan="off"))
        ready, stop = self._ready, self._stop

        def runner() -> None:
            async def main() -> None:
                serve = asyncio.ensure_future(server.serve(sockets=[listener]))
                while not server.started and not serve.done():
                    await asyncio.sleep(0.005)
                if serve.done():
                    serve.result()  # crashed before startup — propagate
                ready.set()
                while not stop.is_set():
                    await asyncio.sleep(0.05)
                server.should_exit = True
                await serve

            try:
                asyncio.run(main())
            except BaseException as exc:  # noqa: BLE001 - benchmark boundary
                self._failure = exc
                ready.set()

        self._thread = threading.Thread(target=runner, name="bench-uvicorn", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=15.0):
            raise RuntimeError("uvicorn failed to start within 15s")
        if self._failure is not None:
            raise RuntimeError(f"uvicorn failed: {self._failure!r}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None


def _handler(request) -> Response:
    return Response(200, {"Content-Type": "text/plain"}, b"bench-ok")


async def _asgi_app(scope, receive, send):
    while True:
        message = await receive()
        if not message.get("more_body", False) and message["type"] == "http.request":
            break
    if scope["path"] == "/sse":
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        import asyncio

        for i in range(5):
            await send(
                {
                    "type": "http.response.body",
                    "body": f"data: {i}\n\n".encode(),
                    "more_body": i < 4,
                }
            )
            await asyncio.sleep(0.005)  # real stream shape: frames over time
        return
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": b"bench-ok"})


class _UploadApp:
    """ASGI: drains the streamed request body, replies with the byte count."""

    async def __call__(self, scope, receive, send) -> None:
        total = 0
        while True:
            message = await receive()
            total += len(message["body"])
            if not message.get("more_body", False):
                break
        payload = str(total).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})


def bench_upload(port: int, size_mib: float = 10.0, rounds: int = 5) -> dict:
    """End-to-end upload of a big Content-Length body (streamed receive).

    What this measures: total time from the client's first write to the
    byte-count response — framing + off-loop body streaming + response.
    Byte-exactness is asserted every round, so a lost or duplicated
    chunk fails the row instead of skewing it.
    """
    payload = b"u" * int(size_mib * 1024 * 1024)
    head = f"POST /upload HTTP/1.1\r\nHost: bench\r\nContent-Length: {len(payload)}\r\n\r\n".encode()
    lat: list[float] = []
    for _ in range(rounds):
        conn = socket.create_connection(("127.0.0.1", port), timeout=30)
        try:
            t0 = time.perf_counter()
            conn.sendall(head + payload)
            _ttfb, status, body = _read_response(conn, t0)
            lat.append(time.perf_counter() - t0)
            assert status == 200, status
            assert int(body) == len(payload), body  # byte-exact through the stream
        finally:
            conn.close()
    out = _percentiles(lat)
    out["mib_s"] = round(size_mib / min(lat), 1)
    out["rounds"] = rounds
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--per-thread", type=int, default=250)
    parser.add_argument("--sse-samples", type=int, default=50)
    parser.add_argument(
        "--no-uvicorn", action="store_true", help="skip the in-process uvicorn A/B baseline"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    results: dict = {}
    handler_srv = WireServer(_handler, host="127.0.0.1", port=0, access_log=False)
    # lifespan="off": _asgi_app is not lifespan-aware (it would park in
    # receive() until the auto-mode timeout) — transport is what we measure
    asgi_srv = WireServer(
        asgi_app=_asgi_app, host="127.0.0.1", port=0, lifespan="off", access_log=False
    )
    uvicorn_srv = None if args.no_uvicorn else _UvicornTarget(_asgi_app)
    handler_srv.start()
    asgi_srv.start()
    if uvicorn_srv is not None:
        uvicorn_srv.start()
    try:
        results["keepalive"] = bench_keepalive(handler_srv.port, args.requests)
        results["fresh"] = bench_fresh(handler_srv, min(args.requests, 200))
        results["concurrent"] = bench_concurrent(handler_srv, args.threads, args.per_thread)
        results["keepalive_asgi"] = bench_keepalive(asgi_srv.port, args.requests)
        if uvicorn_srv is not None:
            # adjacent to our ASGI row — same load window, interleaved A/B
            results["uvicorn_asgi"] = bench_keepalive(uvicorn_srv.port, args.requests)
            # capacity row: 100ms idle sidesteps the tight-loop Nagle stall
            results["uvicorn_asgi_gapped"] = bench_keepalive(uvicorn_srv.port, 50, gap=0.1)
        results["sse_ttfb"] = bench_sse_ttfb(asgi_srv, args.sse_samples)
        # streaming-body row: 10 MiB upload through the receive() seam
        upload_srv = WireServer(
            asgi_app=_UploadApp(), host="127.0.0.1", port=0, lifespan="off", access_log=False
        )
        upload_srv.start()
        try:
            results["upload_wire"] = bench_upload(upload_srv.port)
        finally:
            upload_srv.stop()
        if uvicorn_srv is not None:
            upload_uvicorn = _UvicornTarget(_UploadApp())
            upload_uvicorn.start()
            try:
                results["upload_uvicorn"] = bench_upload(upload_uvicorn.port)
            finally:
                upload_uvicorn.stop()
    finally:
        handler_srv.stop()
        asgi_srv.stop()
        if uvicorn_srv is not None:
            uvicorn_srv.stop()

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"wire transport benchmark (loopback, {args.requests} requests)")
    for name, stats in results.items():
        parts = ", ".join(f"{k}={v}" for k, v in stats.items())
        print(f"  {name:16} {parts}")
    asgi_delta = results["keepalive_asgi"]["p50_ms"] - results["keepalive"]["p50_ms"]
    print(f"  asgi adapter overhead (p50): {asgi_delta:+.3f} ms/req")
    if "uvicorn_asgi" in results:
        ours = results["keepalive_asgi"]["p50_ms"]
        theirs = results["uvicorn_asgi"]["p50_ms"]
        pct = (ours - theirs) / theirs * 100.0
        print(
            f"  A/B vs uvicorn (p50, same app): ours {ours:.3f} ms | "
            f"uvicorn {theirs:.3f} ms | delta {ours - theirs:+.3f} ms ({pct:+.1f}%)"
        )
        if "uvicorn_asgi_gapped" in results:
            gapped = results["uvicorn_asgi_gapped"]["p50_ms"]
            print(
                f"  uvicorn gapped (100ms idle/req): {gapped:.3f} p50 — the tight-loop "
                f"stall is head/body split + Nagle/delayed-ACK; single-write framing avoids it"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
