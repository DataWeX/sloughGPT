#!/usr/bin/env python3
"""Benchmark: gateway edge admission — shed latency, pass-through cost, AIMD.

Drives a dedicated gateway instance (spawns its own, never touches the
live one) with a dead/hanging Python core so every scenario is
deterministic. Measures what the edge's optimizer actually costs:

    pass     exempt /health pass-through (access-log write included)
    stats    /gateway/stats snapshot cost (in-memory, under no load)
    rate     global rate-limit 429 fast-reject latency
    breaker  open-breaker 503 shed latency (no dial attempted)
    streams  stream-cap 429 shed latency while slots are held
    adaptive AIMD halving: N stream failures → stream_limit /= 2

Usage:
    .venv/bin/python scripts/benchmark_gateway_admission.py
    .venv/bin/python scripts/benchmark_gateway_admission.py --json /tmp/gw.json
    .venv/bin/python scripts/benchmark_gateway_admission.py --bin path/to/slough-gateway
"""

from __future__ import annotations

import argparse
import http.client
import json
import multiprocessing
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

PORT = int(os.environ.get("BENCH_GW_PORT", "8099"))
DEAD_CORE = "http://127.0.0.1:9"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_binary(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("MAN_GATEWAY_BIN"):
        candidates.append(Path(os.environ["MAN_GATEWAY_BIN"]))
    root = repo_root()
    candidates.append(root / "apps/gateway/target/release/slough-gateway")
    candidates.append(
        Path.home() / "Documents/sloughGPT-edge-cascade/apps/gateway/target/release/slough-gateway"
    )
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    sys.exit("gateway binary not found — pass --bin or set MAN_GATEWAY_BIN")


def wait_listening(port: int, timeout: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.2):
                return
        except OSError:
            time.sleep(0.05)
    sys.exit(f"gateway did not listen on :{port} within {timeout}s")


_spawn_seq = 0


def _port_owner(port: int) -> int | None:
    try:
        out = subprocess.run(
            ["ss", "-Hltnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"pid=(\d+)", out)
    return int(m.group(1)) if m else None


def spawn_gateway(binary: Path, workdir: Path, overrides: dict[str, str]) -> subprocess.Popen:
    global _spawn_seq
    _spawn_seq += 1
    # A zombie gateway from a crashed run would swallow wait_listening
    # while our own spawn dies on AddrInUse — every scenario would then
    # silently measure the WRONG instance. The bench port is ours alone.
    owner = _port_owner(PORT)
    if owner:
        try:
            os.kill(owner, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(0.3)
    env = os.environ.copy()
    env.update(
        {
            "MAN_GATEWAY_PORT": str(PORT),
            "MAN_CORE_URL": DEAD_CORE,
            "MAN_GATEWAY_RATE_LIMIT": "0",
            "MAN_GATEWAY_BREAKER_FAILURES": "0",
            "MAN_GATEWAY_MAX_STREAMS": "0",
            "MAN_GATEWAY_ACCESS_LOG": str(workdir / f"access-{_spawn_seq}.jsonl"),
            "RUST_LOG": "warn",
        }
    )
    env.update(overrides)
    log = open(workdir / f"gw-{PORT}.log", "ab")
    proc = subprocess.Popen(
        [str(binary)], cwd=str(workdir), env=env, stdout=log, stderr=log, start_new_session=True
    )
    wait_listening(PORT)
    if proc.poll() is not None:
        sys.exit(f"gateway exited immediately (rc={proc.returncode}) — bind failed?")
    return proc


def kill_gateway(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def req(method: str, path: str, conn: http.client.HTTPConnection | None = None) -> tuple[int, float]:
    own = conn is None
    if own:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
    t0 = time.perf_counter()
    try:
        conn.request(method, path, body=b"{}" if method == "POST" else None)
        resp = conn.getresponse()
        resp.read()
        return resp.status, time.perf_counter() - t0
    finally:
        if own:
            conn.close()


def pct(samples: list[float], p: float) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    return s[min(len(s) - 1, max(0, int(round(p * len(s))) - 1))] * 1e6


def line(name: str, n: int, lat: list[float], extra: str = "") -> dict:
    d = {
        "n": n,
        "p50_us": round(pct(lat, 0.50), 1),
        "p95_us": round(pct(lat, 0.95), 1),
        "p99_us": round(pct(lat, 0.99), 1),
    }
    print(
        f"  {name:<26} n={n:<5} p50={d['p50_us']:>8.1f}µs  "
        f"p95={d['p95_us']:>8.1f}µs  p99={d['p99_us']:>8.1f}µs {extra}"
    )
    return d


def server_side_ms(access_log: Path) -> list[float]:
    """Server-observed latency (access log `ms` field) — no client noise."""
    if not access_log.exists():
        return []
    return [json.loads(l)["ms"] for l in access_log.read_text().splitlines() if l.strip()]


def scenario_pass(binary: Path, wd: Path) -> dict:
    proc = spawn_gateway(binary, wd, {})
    access = wd / f"access-{_spawn_seq}.jsonl"
    try:
        for _ in range(50):
            req("GET", "/health")
        workers, per = 8, 125  # processes — a thread benchmark measures the GIL, not the gateway
        q: multiprocessing.Queue = multiprocessing.Queue()

        def worker() -> None:
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
            lat = []
            for _ in range(per):
                _, dt = req("GET", "/health", conn)
                lat.append(dt)
            conn.close()
            q.put(lat)

        procs = [multiprocessing.Process(target=worker) for _ in range(workers)]
        t0 = time.perf_counter()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        wall = time.perf_counter() - t0
        lat: list[float] = []
        for _ in procs:
            lat.extend(q.get())
        n = len(lat)
        rps = int(n / wall)
        print(
            f"\n[pass] exempt /health — {rps} req/s ({workers} procs x {per}, "
            "access log on; box loopback ceilings ~9k rps, gateway server-side ms below)"
        )
        d = line("client /health", n, lat)
        d["rps"] = rps
        sms = server_side_ms(access)
        d["server_ms_p50"] = sorted(sms)[len(sms) // 2] if sms else None
        if sms:
            ss = sorted(sms)
            print(
                f"  {'server-side (log)':<26} n={len(ss):<5} p50={ss[len(ss)//2]:>7.3f}ms  "
                f"p95={ss[int(.95*len(ss))]:>7.3f}ms"
            )
        return d
    finally:
        kill_gateway(proc)


def scenario_stats(binary: Path, wd: Path) -> dict:
    proc = spawn_gateway(binary, wd, {})
    try:
        for _ in range(20):
            req("GET", "/gateway/stats")
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
        lat = [req("GET", "/gateway/stats", conn)[1] for _ in range(200)]
        conn.close()
        print("\n[stats] /gateway/stats snapshot (sequential)")
        return line("GET /gateway/stats", 200, lat)
    finally:
        kill_gateway(proc)


def scenario_rate(binary: Path, wd: Path) -> dict:
    # local clients get ×10 → effective global budget = 100
    proc = spawn_gateway(binary, wd, {"MAN_GATEWAY_RATE_LIMIT": "10"})
    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
        rejects: list[float] = []
        admitted = 0
        for _ in range(160):
            status, dt = req("GET", "/bench", conn)
            if status == 429:
                rejects.append(dt)
            else:
                admitted += 1
        conn.close()
        print(f"\n[rate] global 100/window local — {admitted} admitted, {len(rejects)} shed 429")
        assert len(rejects) >= 50, f"expected ~60 rejects, got {len(rejects)}"
        return line("429 rate fast-reject", len(rejects), rejects)
    finally:
        kill_gateway(proc)


def scenario_breaker(binary: Path, wd: Path) -> dict:
    proc = spawn_gateway(binary, wd, {"MAN_GATEWAY_BREAKER_FAILURES": "3"})
    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
        opened = False
        for _ in range(6):
            status, _ = req("POST", "/chat/stream", conn)
            if status == 503:
                opened = True
                break
        assert opened, "breaker never opened after 3 failures"
        lat = []
        for _ in range(100):
            status, dt = req("GET", "/bench", conn)
            assert status == 503, f"expected shed 503, got {status}"
            lat.append(dt)
        conn.close()
        print("\n[breaker] open → shed without dialing upstream")
        return line("503 breaker shed", 100, lat)
    finally:
        kill_gateway(proc)


class HangServer:
    """Accepts TCP, reads the request, never answers — holds streams in-flight."""

    def __init__(self) -> None:
        self.srv = socket.create_server(("127.0.0.1", 0))
        self.port = self.srv.getsockname()[1]
        self.socks: list[socket.socket] = []
        self.stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        self.srv.settimeout(0.2)
        while not self.stop.is_set():
            try:
                c, _ = self.srv.accept()
            except OSError:
                continue
            self.socks.append(c)
            threading.Thread(target=self._drain, args=(c,), daemon=True).start()

    @staticmethod
    def _drain(c: socket.socket) -> None:
        try:
            while c.recv(4096):
                pass
        except OSError:
            pass

    def close(self) -> None:
        self.stop.set()
        for s in self.socks:
            try:
                s.close()
            except OSError:
                pass
        self.srv.close()


def scenario_streams(binary: Path, wd: Path) -> dict:
    hang = HangServer()
    proc = spawn_gateway(
        binary,
        wd,
        {
            "MAN_GATEWAY_MAX_STREAMS": "2",
            "MAN_CORE_URL": f"http://127.0.0.1:{hang.port}",
        },
    )
    held: list[http.client.HTTPConnection] = []

    def hold() -> None:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=15)
        held.append(conn)
        try:
            conn.request("POST", "/chat/stream", body=b"{}")
            conn.getresponse().read()
        except Exception:
            pass

    try:
        holders = [threading.Thread(target=hold, daemon=True) for _ in range(2)]
        for t in holders:
            t.start()
        deadline = time.perf_counter() + 3
        while time.perf_counter() < deadline:
            status, _ = req("GET", "/gateway/stats")
            if status == 200:
                conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=5)
                conn.request("GET", "/gateway/stats")
                body = json.loads(conn.getresponse().read())
                conn.close()
                if body["gateway"]["in_flight_streams"] == 2:
                    break
        else:
            raise SystemExit("streams never reached in_flight=2 against hanging core")

        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
        lat = []
        for _ in range(50):
            status, dt = req("POST", "/chat/stream", conn)
            assert status == 429, f"expected stream-cap 429, got {status}"
            lat.append(dt)
        conn.close()
        print("\n[streams] cap=2 held by hanging core → 50 shed 429")
        return line("429 stream shed", 50, lat)
    finally:
        hang.close()
        for t in holders:
            t.join(timeout=2)
        kill_gateway(proc)


def scenario_adaptive(binary: Path, wd: Path) -> dict:
    proc = spawn_gateway(binary, wd, {"MAN_GATEWAY_MAX_STREAMS": "8"})
    try:
        for _ in range(8):
            status, _ = req("POST", "/chat/stream")
            assert status == 502, f"expected 502 vs dead core, got {status}"
        time.sleep(2.2)
        status, _ = req("POST", "/chat/stream")  # admission runs the tune
        assert status == 502, f"expected 502, got {status}"
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=5)
        conn.request("GET", "/gateway/stats")
        g = json.loads(conn.getresponse().read())["gateway"]
        conn.close()
        print("\n[adaptive] 8/8 stream failures in one 2s window")
        d = {
            "stream_limit": g["stream_limit"],
            "stream_max": g["stream_max"],
            "expect": 4,
        }
        verdict = "PASS" if g["stream_limit"] == 4 else "FAIL"
        print(f"  stream_limit {g['stream_max']} → {g['stream_limit']} (expect 4)  [{verdict}]")
        return d
    finally:
        kill_gateway(proc)


def main() -> int:
    ap = argparse.ArgumentParser(description="gateway edge admission benchmark")
    ap.add_argument("--bin", default=None, help="path to slough-gateway release binary")
    ap.add_argument("--json", type=Path, default=None, help="write metrics JSON")
    args = ap.parse_args()

    binary = find_binary(args.bin)
    wd = Path(tempfile.mkdtemp(prefix="gw-bench-"))
    print(f"gateway admission benchmark — binary={binary} port={PORT} workdir={wd}")

    results: dict = {"bin": str(binary), "port": PORT}
    failures = 0
    for name, fn in [
        ("pass", scenario_pass),
        ("stats", scenario_stats),
        ("rate", scenario_rate),
        ("breaker", scenario_breaker),
        ("streams", scenario_streams),
        ("adaptive", scenario_adaptive),
    ]:
        try:
            results[name] = fn(binary, wd)
            if name == "adaptive" and results[name].get("stream_limit") != 4:
                failures += 1
        except SystemExit as e:
            print(f"[{name}] FAILED: {e}")
            results[name] = {"error": str(e)}
            failures += 1

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nmetrics → {args.json}")
    print(f"{'ALL SCENARIOS PASS' if failures == 0 else f'{failures} scenario(s) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
