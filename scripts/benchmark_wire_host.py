#!/usr/bin/env python
"""Benchmark: wire host vs uvicorn — transport parity on the GZip path.

Card ecea49c8 (flip gate). Serves ONE synthetic FastAPI app wrapped in the
real ``SelectiveGZipMiddleware`` under both servers and measures:

1. **Correctness parity** — identical decoded bodies and identical
   content-encoding on both transports, every route 200 (raw compressed
   bytes are not comparable: gzip embeds mtime in its header). The gzip
   route is
   the exact path that used to 500 under the wire host (middleware declares
   Content-Length, then streams frames with ``more_body=True`` — fixed by
   the declared-CL buffering in ``wire.asgi_host``).
2. **Latency parity** — sequential keep-alive p50/p95 per transport.
3. **Throughput parity** — concurrent request bursts (requests/second).

Run:
    PYTHONPATH="$PWD/packages/wire-py" ./scripts/python scripts/benchmark_wire_host.py
    ... --update-baseline      # store data/benchmark_wire_host_baseline.json
    ... --runs 200             # requests per sequential phase

Exit non-zero when byte-parity breaks, the burst errors, or wire's own
p95 drifts more than ``--max-regression`` (default 0.20) above the stored
baseline. The baseline JSON stores a machine-state fingerprint (loadavg +
uptime): a reboot or a materially different load at compare time prints a
``[state]`` warning — the drift may be machine load, not code — while the
verdict logic stays unchanged. The wire/uvicorn ratio is REPORTED, not
gated: the residual gap (idle-machine ~x0.8 burst, widening with
concurrency) is thread-per-connection contention and belongs to the
flip-gate decision on card ecea49c8, not to per-change regression —
track it with ``--update-baseline`` over time.

NOTE: absolute p95s are machine-state-specific. Pre-reboot runs under
background load measured wire p95 5.4-7.3ms / burst x0.63-0.66; the same
code on an idle machine measures ~2.3ms / x0.81. Re-record the baseline
whenever the machine state materially changes.
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "apps/api" / "server"
WIRE_DIR = REPO_ROOT / "packages" / "wire-py"
for path in (str(SERVER_DIR), str(WIRE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fastapi import FastAPI  # noqa: E402
from infrastructure.compression import SelectiveGZipMiddleware  # noqa: E402

BASELINE_FILE = REPO_ROOT / "data" / "benchmark_wire_host_baseline.json"


def machine_state() -> dict:
    """Load fingerprint — absolute p95s only mean something in the same state."""
    load1, load5, load15 = os.getloadavg()
    uptime_s = int(float(Path("/proc/uptime").read_text().split()[0]))
    return {
        "loadavg": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "uptime_s": uptime_s,
    }


def state_warnings(recorded: dict, current: dict) -> list[str]:
    """Why the drift number may lie: a reboot or a different load than recorded."""
    warnings: list[str] = []
    if current["uptime_s"] < recorded.get("uptime_s", 0):
        warnings.append(
            f"machine rebooted since baseline (recorded uptime {recorded['uptime_s']}s, "
            f"now {current['uptime_s']}s)"
        )
    rec_load = recorded.get("loadavg", [0.0])[0]
    cur_load = current["loadavg"][0]
    if abs(cur_load - rec_load) > 1.0:
        direction = "busier" if cur_load > rec_load else "quieter"
        warnings.append(
            f"system {direction} than at baseline recording "
            f"(loadavg {rec_load:.2f} -> {cur_load:.2f})"
        )
    return warnings


def make_app() -> FastAPI:
    """Synthetic app with the real middleware — the fixed bug path."""
    import random

    rng = random.Random(7)
    # >500 B of mixed-entropy JSON so SelectiveGZip actually engages
    payload = [
        {"id": i, "token": f"{rng.getrandbits(32):08x}", "name": f"row-{i}"} for i in range(400)
    ]
    small = {"ok": True, "n": 1}

    app = FastAPI()
    app.add_middleware(SelectiveGZipMiddleware, minimum_size=500)

    @app.get("/payload")
    def payload_route() -> list:
        return payload

    @app.get("/small")
    def small_route() -> dict:
        return small

    return app


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def serve_uvicorn(app: FastAPI) -> tuple[object, threading.Thread, int]:
    import uvicorn

    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not getattr(server, "started", False):
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn failed to start within 15s")
        time.sleep(0.02)
    return server, thread, port


def serve_wire(app: FastAPI) -> tuple[object, int]:
    from wire import WireServer

    server = WireServer(asgi_app=app, host="127.0.0.1", port=0, lifespan="off")
    server.start()
    return server, server.port


def fetch(port: int, path: str) -> tuple[int, bytes, str | None]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        conn.request("GET", path, headers={"Accept-Encoding": "gzip"})
        resp = conn.getresponse()
        return resp.status, resp.read(), resp.getheader("content-encoding")
    finally:
        conn.close()


def measure(port: int, runs: int, concurrency: int) -> dict:
    """Sequential keep-alive latencies + a concurrent burst through one conn."""
    paths = ["/payload", "/small"]
    sequential: list[float] = []
    # ONE keep-alive connection for the whole phase: real clients pool
    # connections, and a fresh TCP conn per request would charge wire's
    # thread-per-connection setup against every sample (structurally
    # unfair to the sync accept model, and invisible to uvicorn).
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        for i in range(runs):
            path = paths[i % 2]
            start = time.perf_counter()
            conn.request("GET", path, headers={"Accept-Encoding": "gzip"})
            resp = conn.getresponse()
            resp.read()
            sequential.append((time.perf_counter() - start) * 1000)
            assert resp.status == 200, f"{path} -> {resp.status}"
    finally:
        conn.close()

    # concurrent burst: `concurrency` workers x (runs//concurrency) requests,
    # each worker on its own keep-alive connection
    per_worker = max(1, runs // concurrency)
    errors: list[str] = []

    def worker() -> None:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
            for _ in range(per_worker):
                conn.request("GET", "/payload", headers={"Accept-Encoding": "gzip"})
                resp = conn.getresponse()
                resp.read()
                if resp.status != 200:
                    errors.append(f"status {resp.status}")
            conn.close()
        except Exception as exc:  # noqa: BLE001 - report, don't kill the run
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    burst_start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    burst_seconds = time.perf_counter() - burst_start

    sequential.sort()
    return {
        "p50_ms": round(statistics.median(sequential), 3),
        "p95_ms": round(sequential[int(len(sequential) * 0.95) - 1], 3),
        "mean_ms": round(statistics.fmean(sequential), 3),
        "burst_rps": round((per_worker * concurrency) / burst_seconds, 1),
        "burst_errors": errors,
    }


def correctness(app: FastAPI, wire_port: int, uvicorn_port: int) -> list[str]:
    """Both transports must serve the same decoded body; wire must not 500 gzip."""
    problems: list[str] = []
    for path in ("/payload", "/small"):
        try:
            u_status, u_body, u_enc = fetch(uvicorn_port, path)
            w_status, w_body, w_enc = fetch(wire_port, path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{path}: fetch failed {exc!r}")
            continue
        if w_status != 200:
            problems.append(f"{path}: wire returned {w_status}")
        if u_status != w_status:
            problems.append(f"{path}: status uvicorn={u_status} wire={w_status}")
        if u_enc != w_enc:
            problems.append(f"{path}: content-encoding uvicorn={u_enc!r} wire={w_enc!r}")
        # decode before comparing: gzip.compress embeds mtime in the
        # header, so raw compressed bytes legitimately differ across
        # processes/seconds — the invariant is the same DECODED body
        if u_enc == "gzip":
            u_body, w_body = gzip.decompress(u_body), gzip.decompress(w_body)
        if u_body != w_body:
            problems.append(f"{path}: body differs (uvicorn {len(u_body)}B vs wire {len(w_body)}B)")
    # the gzip route must actually be compressed — otherwise parity is vacuous
    _, _, enc = fetch(wire_port, "/payload")
    if enc != "gzip":
        problems.append(f"/payload not gzipped under wire (content-encoding={enc!r})")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=300, help="sequential requests")
    parser.add_argument("--concurrency", type=int, default=8, help="burst workers")
    parser.add_argument("--max-regression", type=float, default=0.20)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    app = make_app()
    uvicorn_server, uvicorn_thread, uvicorn_port = serve_uvicorn(app)
    wire_server, wire_port = serve_wire(app)
    print(f"uvicorn :{uvicorn_port}  wire :{wire_port}  runs={args.runs}")

    try:
        problems = correctness(app, wire_port, uvicorn_port)
        if problems:
            print("CORRECTNESS FAIL:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("correctness: PASS (identical decoded bodies, gzip engaged on both)")

        uv = measure(uvicorn_port, args.runs, args.concurrency)
        wire = measure(wire_port, args.runs, args.concurrency)
    finally:
        wire_server.stop()
        uvicorn_server.should_exit = True
        uvicorn_thread.join(timeout=10)

    for label, data in (("uvicorn", uv), ("wire", wire)):
        print(
            f"{label:8} p50={data['p50_ms']:.3f}ms p95={data['p95_ms']:.3f}ms "
            f"mean={data['mean_ms']:.3f}ms burst={data['burst_rps']:.0f} rps "
            f"errors={len(data['burst_errors'])}"
        )
    if uv["burst_errors"] or wire["burst_errors"]:
        print(f"burst errors: uvicorn={uv['burst_errors']} wire={wire['burst_errors']}")
        return 1

    p95_ratio = wire["p95_ms"] / max(uv["p95_ms"], 1e-6)
    rps_ratio = wire["burst_rps"] / max(uv["burst_rps"], 1e-6)
    print(f"wire/uvicorn (flip-gate metric): p95 x{p95_ratio:.2f}  throughput x{rps_ratio:.2f}")

    baseline = None
    if BASELINE_FILE.exists():
        baseline = json.loads(BASELINE_FILE.read_text())
    drift = None
    if baseline is not None:
        drift = wire["p95_ms"] / max(baseline["wire"]["p95_ms"], 1e-6) - 1
        verdict = "PASS" if drift <= args.max_regression else "FAIL"
        print(
            f"wire p95 vs baseline: {drift:+.1%} "
            f"(baseline {baseline['wire']['p95_ms']:.3f}ms, limit +{args.max_regression:.0%}) "
            f"[{verdict}]"
        )
        recorded_state = baseline.get("machine")
        if recorded_state:
            for warning in state_warnings(recorded_state, machine_state()):
                print(f"  [state] {warning} — drift may be machine load, not code")
    else:
        verdict = "PASS"
        print("no baseline yet — correctness gate only [PASS]; run --update-baseline")

    result = {
        "runs": args.runs,
        "concurrency": args.concurrency,
        "machine": machine_state(),
        "uvicorn": uv,
        "wire": wire,
        "p95_ratio": round(p95_ratio, 3),
        "throughput_ratio": round(rps_ratio, 3),
        "baseline_drift": None if drift is None else round(drift, 3),
        "verdict": verdict,
    }
    if args.update_baseline:
        BASELINE_FILE.write_text(json.dumps(result, indent=2) + "\n")
        print(f"baseline written: {BASELINE_FILE.relative_to(REPO_ROOT)}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
