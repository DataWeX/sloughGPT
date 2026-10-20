"""Transport benchmark: our own wire layer, phase-3 part 1 (card ecea49c8).

Measures what the wire costs, in isolation, on loopback:
    keepalive    sequential RTT over one connection, sync handler mode
    keepalive_asgi  same through the ASGI adapter (adapter overhead = delta)
    fresh        connect + request + close per measurement
    concurrent   N threads over their own keep-alive connections, req/s
    sse_ttfb     streaming response: time to first byte + total time

No uvicorn baseline yet — the A/B harness lands with the SLO_TRANSPORT
wiring (shared entrypoint, needs the parallel-lane coordination window);
this run establishes OUR numbers for that comparison. Results are honest
loopback shapes, not network latencies.

Usage:
    python scripts/benchmark_transport.py
    python scripts/benchmark_transport.py --requests 2000 --json
"""

from __future__ import annotations

import argparse
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


def bench_keepalive(server: WireServer, requests: int) -> dict:
    conn = socket.create_connection(("127.0.0.1", server.port), timeout=10)
    lat: list[float] = []
    try:
        for _ in range(requests):
            _ttfb, total = _exchange(conn, _GET)
            lat.append(total)
    finally:
        conn.close()
    out = _percentiles(lat)
    out["req_s"] = round(requests / sum(lat))
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--per-thread", type=int, default=250)
    parser.add_argument("--sse-samples", type=int, default=50)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    results: dict = {}
    handler_srv = WireServer(_handler, host="127.0.0.1", port=0)
    asgi_srv = WireServer(asgi_app=_asgi_app, host="127.0.0.1", port=0)
    handler_srv.start()
    asgi_srv.start()
    try:
        results["keepalive"] = bench_keepalive(handler_srv, args.requests)
        results["fresh"] = bench_fresh(handler_srv, min(args.requests, 200))
        results["concurrent"] = bench_concurrent(handler_srv, args.threads, args.per_thread)
        results["keepalive_asgi"] = bench_keepalive(asgi_srv, args.requests)
        results["sse_ttfb"] = bench_sse_ttfb(asgi_srv, args.sse_samples)
    finally:
        handler_srv.stop()
        asgi_srv.stop()

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"wire transport benchmark (loopback, {args.requests} requests)")
    for name, stats in results.items():
        parts = ", ".join(f"{k}={v}" for k, v in stats.items())
        print(f"  {name:16} {parts}")
    asgi_delta = results["keepalive_asgi"]["p50_ms"] - results["keepalive"]["p50_ms"]
    print(f"  asgi adapter overhead (p50): {asgi_delta:+.3f} ms/req")
    return 0


if __name__ == "__main__":
    sys.exit(main())
