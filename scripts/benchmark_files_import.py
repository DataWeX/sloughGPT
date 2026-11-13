#!/usr/bin/env python
"""Benchmark: /files/import pipeline — throughput, overhead, one-request invariant.

What it measures (card 20261108_094501, Import-from-URL):

  floor-*  raw ``download_file`` (no job registry, no progress) — the cost
           floor for one compressed fetch against a local origin
  pipe-*   ``FilesRouter._run_import`` — floor + job-registry ``on_chunk``
           updates + metadata upsert (the import's worker thread, called
           inline so the measurement excludes thread scheduling)
  e2e-*    the real ASGI journey: POST /files/import → poll
           GET /files/import/{job_id} (20 ms cadence) → done — adds the
           endpoint, auth guard and registry read path. The UI polls at
           500 ms, which adds *display* latency, not throughput; this row
           measures the pipeline itself.

  wan-*     opt-in (``--wan`` / ``--wan-url``): the same e2e journey against a
            real public origin — RTT, TLS handshake, TCP slow-start and the
            CDN's *actual* compression answer (loopback origins always honor
            Accept-Encoding; real ones vary). INFORMATIONAL by design:
            real-network variance never gates — the loopback rows above stay
            the recorded baseline. Integrity without a known payload: stored
            bytes must be identical across repeats (``--wan-repeat`` ≥ 2);
            one-request is not observable from the client side and stays
            pinned by the loopback rows.

Reported per row: wall seconds, decoded MiB/s (disk byte-space — the same
space the progress bar lives in), wire MiB/s, saved % (WireStats), and the
origin request count. Overhead% = pipe vs floor, e2e vs floor.

Invariants the benchmark FAILS on (they are what the feature promises):
  * one request per import — a HEAD probe or a retry shows up in the count
  * stored bytes == payload (sha256) — decoded integrity end to end
  * e2e pct == 100 only at done; gzip rows report saved > 0 + encoding
    "gzip"; identity rows report saved == 0

Run:
    PYTHONNOUSERSITE=1 ./scripts/python scripts/benchmark_files_import.py \
        [--mib 8] [--repeat 3] [--wan] [--wan-url URL] [--wan-repeat 2]

History is appended to ``scripts/benchmark_files_import.json``
(``benchmark_results.py record --kind`` has no import category — same call
as the downcraft compression benchmark).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "apps" / "api" / "server", ROOT / "packages" / "downcraft"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Loopback origin + local dev parity — the SSRF guard must let this through.
os.environ.setdefault("SLO_FILES_ALLOW_PRIVATE", "1")
logging.basicConfig(level=logging.WARNING)

from downcraft.download.http import WireStats, download_file  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from apps.api.server.infrastructure.exception_handlers import (  # noqa: E402
    register_all_handlers,
)
from apps.api.server.routers import files as files_mod  # noqa: E402
from apps.api.server.routers.files import FilesRouter  # noqa: E402

HISTORY = ROOT / "scripts" / "benchmark_files_import.json"
ENCODINGS = ("identity", "gzip")
# Our own repo via raw.githubusercontent (Fastly) — polite default: we control
# the bytes, it answers gzip, and no third party takes load from a benchmark.
DEFAULT_WAN_URL = (
    "https://raw.githubusercontent.com/DataWeX/sloughGPT/main/docs/AGENT_SYNC.md"
)


def make_payload(mib: float) -> bytes:
    """Compressible-but-not-trivial text (~55-65% gzip ratio).

    Uniform filler would gzip at 1000:1 and hide the wire tier; pure random
    bytes would not compress at all and hide the savings receipt.
    """
    rng = random.Random(42)
    target = int(mib * 1024 * 1024)
    out = bytearray()
    while len(out) < target:
        out += (
            f"line {len(out):09d} tok {rng.getrandbits(96):024x} "
            f"note import-bench\n"
        ).encode()
    return bytes(out[:target])


class _Origin:
    """Local origin: serves one payload (identity or gzip), counts GETs."""

    def __init__(self, payload: bytes, encoding: str):
        self.count = 0
        self.lock = threading.Lock()
        self.body = gzip.compress(payload) if encoding == "gzip" else payload
        self.encoding = encoding
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                with outer.lock:
                    outer.count += 1
                self.send_response(200)
                self.send_header("Content-Length", str(len(outer.body)))
                if outer.encoding == "gzip":
                    self.send_header("Content-Encoding", "gzip")
                self.end_headers()
                self.wfile.write(outer.body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/bench.txt"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stored_blobs(directory: Path) -> list[Path]:
    """Stored blobs in an uploads dir — any name (WAN filenames aren't .txt).

    Skips ``_``-prefixed bookkeeping files and downcraft ``.sgpart``
    partials: only real stored payloads count for sha verification.
    """
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and not p.name.startswith("_") and not p.name.endswith(".sgpart")
    )


def run_floor(url: str, workdir: Path, tag: str, payload_sha: str | None) -> dict:
    """Raw download_file — the cost floor (no job, no progress callback)."""
    dest = workdir / f"floor-{tag}.bin"
    stats = WireStats()
    t0 = time.perf_counter()
    download_file(url, dest, compressed=True, skip_if_exists=False, stats=stats)
    wall = time.perf_counter() - t0
    ok_sha = None if payload_sha is None else sha256_file(dest) == payload_sha
    dest.unlink(missing_ok=True)
    return {
        "wall_s": wall,
        "wire_bytes": stats.wire_bytes,
        "saved_pct": stats.saved_pct,
        "encoding": stats.content_encoding,
        "sha_ok": ok_sha,
    }


def run_pipe(url: str, workdir: Path, tag: str, payload_sha: str | None) -> dict:
    """FilesRouter._run_import inline — registry + on_chunk + metadata upsert."""
    router = FilesRouter()
    router.UPLOADS_DIR = workdir / f"pipe-{tag}"
    router.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"bench-{tag}"
    FilesRouter._job_put(
        job_id,
        {
            "job_id": job_id,
            "state": "running",
            "filename": "bench.txt",
            "done": 0,
            "total": 0,
            "file_id": None,
            "wire_bytes": None,
            "saved_bytes": None,
            "content_encoding": "identity",
            "error": None,
        },
    )
    t0 = time.perf_counter()
    router._run_import(job_id, url, "bench.txt")
    wall = time.perf_counter() - t0
    job = files_mod._IMPORT_JOBS[job_id]
    stored = stored_blobs(router.UPLOADS_DIR)
    ok_sha = None if payload_sha is None else (
        bool(stored) and sha256_file(stored[0]) == payload_sha
    )
    return {
        "wall_s": wall,
        "wire_bytes": job["wire_bytes"],
        "saved_bytes": job["saved_bytes"],
        "state": job["state"],
        "pct": 100.0 if job["state"] == "done" else 0.0,
        "encoding": job["content_encoding"],
        "sha_ok": ok_sha,
    }


def run_e2e(url: str, workdir: Path, tag: str, payload_sha: str | None) -> dict:
    """Full ASGI journey: POST start → poll status → terminal state.

    ``payload_sha`` may be ``None`` (WAN rows fetch an arbitrary URL): then
    ``sha_ok`` is ``None`` and ``stored_sha`` is returned instead, so the
    caller can compare stored bytes across repeats.
    """
    router = FilesRouter()
    router.UPLOADS_DIR = workdir / f"e2e-{tag}"
    router.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(router.router)

    with TestClient(app) as client:
        t0 = time.perf_counter()
        resp = client.post("/files/import", json={"url": url})
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        deadline = time.time() + 120
        status = None
        while time.time() < deadline:
            status = client.get(f"/files/import/{job_id}").json()
            if status["state"] != "running":
                break
            time.sleep(0.02)
        wall = time.perf_counter() - t0

        if status is None or status["state"] == "running":
            raise SystemExit(f"e2e-{tag}: import never finished (timeout)")

        # Poll cost of the status read the UI performs every 500 ms.
        samples = []
        for _ in range(20):
            p0 = time.perf_counter()
            client.get(f"/files/import/{job_id}")
            samples.append((time.perf_counter() - p0) * 1e6)
        samples.sort()

    stored = stored_blobs(router.UPLOADS_DIR)
    stored_sha = sha256_file(stored[0]) if stored else None
    ok_sha = None if payload_sha is None else stored_sha == payload_sha
    return {
        "wall_s": wall,
        "wire_bytes": status["wire_bytes"],
        "saved_bytes": status["saved_bytes"],
        "state": status["state"],
        "pct": status["pct"],
        "done": status["done"],
        "total": status["total"],
        "encoding": status["content_encoding"],
        "poll_us_mean": sum(samples) / len(samples),
        "poll_us_p95": samples[int(len(samples) * 0.95) - 1],
        "stored_sha": stored_sha,
        "sha_ok": ok_sha,
    }


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def git_short() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mib", type=float, default=8.0, help="payload size in MiB")
    ap.add_argument("--repeat", type=int, default=3, help="runs per row (median kept)")
    ap.add_argument(
        "--wan",
        action="store_true",
        help="also run an informational e2e row against a real public origin",
    )
    ap.add_argument(
        "--wan-url",
        default=None,
        help="real-network URL to fetch (implies --wan; default: our own repo)",
    )
    ap.add_argument(
        "--wan-repeat",
        type=int,
        default=1,
        help="WAN fetches; >=2 enables the cross-repeat sha-stability check",
    )
    args = ap.parse_args()
    wan_enabled = args.wan or args.wan_url is not None

    payload = make_payload(args.mib)
    payload_sha = hashlib.sha256(payload).hexdigest()
    size = len(payload)
    checks: list[tuple[str, bool, str]] = []
    rows_out: list[dict] = []
    poll_out: dict[str, float] = {}
    overhead: dict[str, str] = {}

    print(
        f"files import benchmark — {args.mib:g} MiB payload × {args.repeat} repeat(s)"
    )
    header = f"{'row':<18} {'wall_s':>8} {'dec MiB/s':>10} {'wire MiB/s':>11} {'saved%':>7} {'reqs':>5}"
    print(header)
    print("-" * len(header))

    try:
        for encoding in ENCODINGS:
            origin = _Origin(payload, encoding)
            workdir = Path(tempfile.mkdtemp(prefix=f"bench-import-{encoding}-"))
            try:
                # Aggregate across repeats: median wall, summed request count.
                agg: dict[str, dict] = {}
                for row_name, runner in (
                    (f"floor-{encoding}", run_floor),
                    (f"pipe-{encoding}", run_pipe),
                    (f"e2e-{encoding}", run_e2e),
                ):
                    walls, reqs = [], 0
                    last: dict = {}
                    for i in range(args.repeat):
                        before = origin.count
                        last = runner(origin.url, workdir, f"{encoding}-{i}", payload_sha)
                        reqs += origin.count - before
                        walls.append(last["wall_s"])
                    wall = median(walls)
                    dec_mibs = size / wall / (1024 * 1024)
                    wire = last.get("wire_bytes") or 0
                    wire_mibs = wire / wall / (1024 * 1024) if wire else 0.0
                    saved_pct = last.get("saved_pct")
                    if saved_pct is None and last.get("saved_bytes") is not None:
                        w, s = last["wire_bytes"] or 0, last["saved_bytes"]
                        saved_pct = 100.0 * s / max(1, w + s)
                    agg[row_name] = {
                        "row": row_name,
                        "wall_s": round(wall, 4),
                        "dec_mibs": round(dec_mibs, 1),
                        "wire_mibs": round(wire_mibs, 1),
                        "saved_pct": None if saved_pct is None else round(saved_pct, 1),
                        "reqs": reqs,
                        "state": last.get("state", "done"),
                        "pct": last.get("pct"),
                        "encoding": last.get("encoding"),
                        "sha_ok": last["sha_ok"],
                    }
                    # Invariants ------------------------------------------------
                    checks.append(
                        (
                            f"{row_name} one-request",
                            reqs == args.repeat,
                            f"{reqs} GET(s) for {args.repeat} run(s)",
                        )
                    )
                    checks.append(
                        (
                            f"{row_name} sha256",
                            last["sha_ok"],
                            "stored bytes == payload" if last["sha_ok"] else "MISMATCH",
                        )
                    )
                    if row_name.startswith("e2e"):
                        poll_out[f"{encoding}_mean_us"] = round(
                            last["poll_us_mean"], 1
                        )
                        poll_out[f"{encoding}_p95_us"] = round(last["poll_us_p95"], 1)
                        checks.append(
                            (
                                f"{row_name} pct==100 at done",
                                last["pct"] == 100.0,
                                f"pct={last['pct']}",
                            )
                        )
                    if encoding == "gzip":
                        ok = (last.get("saved_bytes") or 0) > 0 or (
                            last.get("saved_pct") or 0
                        ) > 0
                        checks.append(
                            (f"{row_name} gzip savings", ok, f"saved={last.get('saved_bytes')}")
                        )
                        checks.append(
                            (
                                f"{row_name} gzip encoding",
                                "gzip" in (last.get("encoding") or ""),
                                f"encoding={last.get('encoding')}",
                            )
                        )
                    else:
                        sv = last.get("saved_bytes")
                        ok = sv == 0 if sv is not None else (last.get("saved_pct") or 0) == 0
                        checks.append(
                            (f"{row_name} identity saved==0", ok, f"saved={sv}")
                        )

                floor_w = agg[f"floor-{encoding}"]["wall_s"]
                pipe_w = agg[f"pipe-{encoding}"]["wall_s"]
                e2e_w = agg[f"e2e-{encoding}"]["wall_s"]
                overhead[encoding] = (
                    f"pipe {100 * (pipe_w / floor_w - 1):+.1f}%  "
                    f"e2e {100 * (e2e_w / floor_w - 1):+.1f}%"
                )

                for row_name in (f"floor-{encoding}", f"pipe-{encoding}", f"e2e-{encoding}"):
                    r = agg[row_name]
                    rows_out.append(r)
                    saved = "-" if r["saved_pct"] is None else f"{r['saved_pct']}"
                    print(
                        f"{r['row']:<18} {r['wall_s']:>8.4f} {r['dec_mibs']:>10.1f} "
                        f"{r['wire_mibs']:>11.1f} {saved:>7} {r['reqs']:>5}"
                    )
            finally:
                origin.stop()
                shutil.rmtree(workdir, ignore_errors=True)
    finally:
        # Per-encoding cleanup (origin + tempdir) happens in the inner finally.
        pass

    # ── WAN row (opt-in, informational) ────────────────────────────────────
    # Real-network variance never gates: the loopback rows above are the
    # recorded baseline. Integrity without a known payload = stored bytes
    # identical across repeats (--wan-repeat >= 2).
    wan_out: dict | None = None
    if wan_enabled:
        wan_url = args.wan_url or DEFAULT_WAN_URL
        wan_repeat = max(1, args.wan_repeat)
        print(f"\nwan-e2e (informational): {wan_url} × {wan_repeat}")
        wan_workdir = Path(tempfile.mkdtemp(prefix="bench-import-wan-"))
        wan_runs: list[dict] = []
        try:
            for i in range(wan_repeat):
                wan_runs.append(run_e2e(wan_url, wan_workdir, f"wan-{i}", None))
        finally:
            shutil.rmtree(wan_workdir, ignore_errors=True)

        last = wan_runs[-1]
        wall = median([r["wall_s"] for r in wan_runs])
        done = last["done"] or 0
        dec_mibs = done / wall / (1024 * 1024) if wall > 0 else 0.0
        wire = last.get("wire_bytes") or 0
        wire_mibs = wire / wall / (1024 * 1024) if wire and wall > 0 else 0.0
        saved_pct = None
        if last.get("saved_bytes") is not None:
            saved_pct = 100.0 * last["saved_bytes"] / max(1, wire + last["saved_bytes"])

        # Invariants (state/pct/integrity/savings — one-request is NOT
        # observable from the client side; it stays pinned by loopback rows).
        checks.append(
            ("wan state==done", last["state"] == "done", f"state={last['state']}")
        )
        checks.append(("wan pct==100", last["pct"] == 100.0, f"pct={last['pct']}"))
        checks.append(
            (
                "wan done==total>0",
                last["done"] == last["total"] and last["total"] > 0,
                f"done={last['done']} total={last['total']}",
            )
        )
        if wan_repeat >= 2:
            shas = [r["stored_sha"] for r in wan_runs]
            stable = None not in shas and len(set(shas)) == 1
            checks.append(
                (
                    "wan repeat sha stable",
                    bool(stable),
                    f"{len(set(shas))} distinct sha(s) across {wan_repeat} fetch(es)",
                )
            )
        enc = last.get("encoding")
        if enc is None:
            pass  # no content-encoding reported — savings shown honestly as "-"
        elif enc == "identity":
            sv = last.get("saved_bytes")
            ok = sv == 0 if sv is not None else (saved_pct or 0) == 0
            checks.append(("wan identity saved==0", ok, f"saved={sv}"))
        else:
            ok = (last.get("saved_bytes") or 0) > 0 or (saved_pct or 0) > 0
            checks.append(
                (f"wan {enc} savings", ok, f"saved={last.get('saved_bytes')}")
            )

        saved = "-" if saved_pct is None else f"{saved_pct:.1f}"
        print(
            f"{'wan-e2e':<18} {wall:>8.4f} {dec_mibs:>10.1f} "
            f"{wire_mibs:>11.1f} {saved:>7} {'-':>5}"
        )
        wan_out = {
            "url": wan_url,
            "repeat": wan_repeat,
            "row": {
                "row": "wan-e2e",
                "wall_s": round(wall, 4),
                "dec_mibs": round(dec_mibs, 1),
                "wire_mibs": round(wire_mibs, 1),
                "saved_pct": None if saved_pct is None else round(saved_pct, 1),
                "state": last["state"],
                "pct": last["pct"],
                "done": last["done"],
                "total": last["total"],
                "encoding": last.get("encoding"),
                "stored_sha": last["stored_sha"],
            },
        }

    print()
    for enc in ENCODINGS:
        print(f"overhead vs floor ({enc}): {overhead[enc]}")
    if poll_out:
        print(
            "status poll cost: "
            + ", ".join(f"{k}={v} µs" for k, v in sorted(poll_out.items()))
        )

    failed = [c for c in checks if not c[1]]
    print(f"checks: {len(checks) - len(failed)}/{len(checks)} ok")
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")

    record = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": git_short(),
        "mib": args.mib,
        "repeat": args.repeat,
        "rows": rows_out,
        "overhead": overhead,
        "poll_us": poll_out,
        "wan": wan_out,
        "checks_ok": not failed,
    }
    history = []
    if HISTORY.exists():
        try:
            history = json.loads(HISTORY.read_text())
        except (OSError, ValueError):
            history = []
    history.append(record)
    HISTORY.write_text(json.dumps(history[-100:], indent=2) + "\n")
    print(f"history: {HISTORY.relative_to(ROOT)} (record #{len(history)})")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
