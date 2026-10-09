#!/usr/bin/env python3
"""Benchmark different IPC protocols for shell-to-domain communication.

Tests: TCP, UDP, Unix Socket, WebSocket, HTTP, Shared Memory
Metrics: latency (p50/p95/p99), throughput, overhead
"""

import socket
import struct
import tempfile
import threading
import time
from pathlib import Path

# ── Config ──
ITERATIONS = 1000
PAYLOAD_SIZES = [64, 256, 1024, 4096]  # bytes
HOST = "127.0.0.1"


def generate_payload(size: int) -> bytes:
    """Generate test payload."""
    return b"X" * size


# ── TCP ──
def run_tcp_benchmark(iterations, payload_size):
    """TCP echo benchmark."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, 0))
    port = server.getsockname()[1]
    server.listen(1)

    latencies = []

    def server_thread():
        conn, _ = server.accept()
        payload = generate_payload(payload_size)
        for _ in range(iterations):
            data = conn.recv(len(payload))
            conn.sendall(data)
        conn.close()
        server.close()

    t = threading.Thread(target=server_thread, daemon=True)
    t.start()

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, port))
    payload = generate_payload(payload_size)

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        client.sendall(payload)
        client.recv(len(payload))
        latencies.append(time.perf_counter_ns() - t0)

    client.close()
    t.join(timeout=2)
    return latencies


# ── UDP ──
def run_udp_benchmark(iterations, payload_size):
    """UDP echo benchmark."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, 0))
    port = sock.getsockname()[1]

    latencies = []

    def server_thread():
        for _ in range(iterations):
            data, addr = sock.recvfrom(4096)
            sock.sendto(data, addr)
        sock.close()

    t = threading.Thread(target=server_thread, daemon=True)
    t.start()

    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = generate_payload(payload_size)

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        client.sendto(payload, (HOST, port))
        client.recv(4096)
        latencies.append(time.perf_counter_ns() - t0)

    client.close()
    t.join(timeout=2)
    return latencies


# ── Unix Socket ──
def run_unix_benchmark(iterations, payload_size):
    """Unix socket echo benchmark."""
    sock_path = Path(tempfile.mktemp())
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)

    latencies = []

    def server_thread():
        conn, _ = server.accept()
        payload = generate_payload(payload_size)
        for _ in range(iterations):
            data = conn.recv(len(payload))
            conn.sendall(data)
        conn.close()
        server.close()
        sock_path.unlink()

    t = threading.Thread(target=server_thread, daemon=True)
    t.start()

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(sock_path))
    payload = generate_payload(payload_size)

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        client.sendall(payload)
        client.recv(len(payload))
        latencies.append(time.perf_counter_ns() - t0)

    client.close()
    t.join(timeout=2)
    return latencies


# ── WebSocket-like (length-prefixed framing) ──
def run_ws_benchmark(iterations, payload_size):
    """WebSocket-like benchmark with length-prefixed frames."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, 0))
    port = server.getsockname()[1]
    server.listen(1)

    latencies = []

    def server_thread():
        conn, _ = server.accept()
        payload = generate_payload(payload_size)
        for _ in range(iterations):
            length_bytes = conn.recv(4)
            length = struct.unpack("!I", length_bytes)[0]
            conn.recv(length)
            conn.sendall(struct.pack("!I", len(payload)) + payload)
        conn.close()
        server.close()

    t = threading.Thread(target=server_thread, daemon=True)
    t.start()

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, port))
    payload = generate_payload(payload_size)

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        client.sendall(struct.pack("!I", len(payload)) + payload)
        length_bytes = client.recv(4)
        length = struct.unpack("!I", length_bytes)[0]
        client.recv(length)
        latencies.append(time.perf_counter_ns() - t0)

    client.close()
    t.join(timeout=2)
    return latencies


# ── Shared Memory (mmap) ──
def run_shm_benchmark(iterations, payload_size):
    """Shared memory benchmark using mmap."""
    import mmap

    shm_path = Path(tempfile.mktemp())
    shm_path.write_bytes(b"\x00" * 8192)

    latencies = []
    with open(shm_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 8192)
        payload = generate_payload(payload_size)

        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            mm.seek(0)
            mm.write(payload)
            mm.seek(0)
            mm.read(payload_size)
            latencies.append(time.perf_counter_ns() - t0)

        mm.close()

    shm_path.unlink()
    return latencies


# ── HTTP ──
def run_http_benchmark(iterations, payload_size):
    """HTTP benchmark using http.server."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class EchoHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(generate_payload(payload_size))

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), EchoHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    import urllib.request

    latencies = []
    payload = generate_payload(payload_size)
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/", data=payload, method="POST")
        urllib.request.urlopen(req, timeout=30).read()
        latencies.append(time.perf_counter_ns() - t0)

    server.shutdown()
    return latencies


# ── Stats ──
def stats(latencies):
    """Calculate stats."""
    latencies.sort()
    n = len(latencies)
    return {
        "p50": latencies[n // 2] / 1000,  # μs
        "p95": latencies[int(n * 0.95)] / 1000,
        "p99": latencies[int(n * 0.99)] / 1000,
        "mean": sum(latencies) / n / 1000,
        "min": latencies[0] / 1000,
        "max": latencies[-1] / 1000,
    }


def main():
    print(
        f"{'Protocol':<15} {'Size':<8} {'p50 (μs)':<10} {'p95 (μs)':<10} {'p99 (μs)':<10} {'mean (μs)':<10}"
    )
    print("-" * 73)

    benchmarks = [
        ("tcp", run_tcp_benchmark),
        ("udp", run_udp_benchmark),
        ("unix", run_unix_benchmark),
        ("ws_frame", run_ws_benchmark),
        ("shared_memory", run_shm_benchmark),
        ("http", run_http_benchmark),
    ]

    for payload_size in PAYLOAD_SIZES:
        for name, fn in benchmarks:
            try:
                latencies = fn(ITERATIONS, payload_size)
                s = stats(latencies)
                print(
                    f"{name:<15} {payload_size:<8} {s['p50']:<10.1f} {s['p95']:<10.1f} {s['p99']:<10.1f} {s['mean']:<10.1f}"
                )
            except Exception as e:
                print(f"{name:<15} {payload_size:<8} ERROR: {e}")
        print()

    print("\nRecommendation for shell-to-domain IPC:")
    print("  - Unix socket: fastest for local, no network overhead")
    print("  - TCP: good for remote support, reliable")
    print("  - UDP: lowest latency but unreliable (drops packets)")
    print("  - WebSocket: best for streaming/token-by-token")
    print("  - Shared memory: fastest but complex, no isolation")
    print("  - HTTP: slowest, but simplest integration")


if __name__ == "__main__":
    main()
