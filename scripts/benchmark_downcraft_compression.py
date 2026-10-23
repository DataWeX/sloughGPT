#!/usr/bin/env python3
"""Downcraft compression benchmark — throughput, ratio, and resume cost.

Measures the *client tier* of downcraft compression (what
``download_file`` actually pays), in two tiers:

Tier 1 — codec (in-process, downcraft's own stream functions):
  * identity stream-copy baseline MiB/s — the uncompressed reference
  * SLZ4 (44-byte header + LZ4 frame) at levels 1 / 6 / 16:
    encode MiB/s, decode MiB/s, wire ratio; decode is SHA-256-verified
    against the source (a mismatch aborts the benchmark — throughput
    numbers are meaningless without integrity)
  * edge gateway codecs gzip-6 / zstd-3 as ratio/throughput reference
    rows (gateway-side codec measurements live in
    ``scripts/benchmark_gateway_compression.py``)

Tier 1b — external baselines (the way codec teams actually benchmark:
matched budgets, same corpus, SHA-256-gated):
  * speed class — each reference CLI at its fast setting
    (gzip -1, zstd -1, lz4 -1, xz -1, bzip2 -1, 7z -mx=1)
  * ratio class — each at its max practical setting
    (gzip -9, zstd -19, lz4 -9 HC, xz -9, bzip2 -9, 7z -mx=9)
  * WinRAR (RAR) is proprietary and not installed; ``7z`` is the
    reproducible stand-in for the RAR/WinZip solid-LZMA school, and
    ``gzip`` is the same DEFLATE algorithm as minigzip.
  * missing tools are skipped, not fatal; every produced payload is
    decoded and SHA-256-verified against the source.

Tier 2 — end-to-end through ``downcraft.download.http.download_file``
against a local Range server (``tests/helpers.RangeHandler`` — the same
SLZ4-protocol server the test-suite uses, reused rather than copied):
  * identity one-shot (baseline wall time / throughput)
  * SLZ4 one-shot (``application/x-lz4``, streamed decode to identity)
  * identity interrupted (server truncates mid-body; client must resume
    with a Range derived from the on-disk ``.sgpart``)
  * SLZ4 interrupted (mid-wire crash; resume must stay in decoded
    byte-space)
  Every run passes a SHA-256 checksum into ``download_file``; a
  mismatch/failed download exits non-zero.  Note the SLZ4 rows include
  the local origin's on-the-fly compression (``slz4_wire`` runs per full
  GET) — the pure decode cost is the tier-1 dec figure.

Knobs applied for measurement:
  * download chunk = 64 KiB — with the production 8 MiB default a small
    payload dies at truncation before the first chunk is yielded, so the
    "resume" would measure a 0-byte restart instead of a real resume
  * retry delay = 0 — the production 2 s backoff would dominate wall time

History is appended to ``scripts/benchmark_downcraft_compression.json``
(``benchmark_results.py record --kind`` has no compression category).

Usage:
    ./scripts/python scripts/benchmark_downcraft_compression.py [--mib 8]
    ./scripts/python scripts/benchmark_downcraft_compression.py --corpus code
    ./scripts/python scripts/benchmark_downcraft_compression.py --file path/to/artifact
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from http.server import HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "downcraft" / "tests"))
sys.path.insert(0, str(ROOT / "packages" / "downcraft"))

from downcraft.download import http as http_mod  # noqa: E402
from downcraft.download.compress import compress_stream, decompress_stream  # noqa: E402
from downcraft.download.http import download_file  # noqa: E402
from helpers import RangeHandler, _range_url, slz4_wire  # noqa: E402

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None

# Measurement chunk: small enough that truncation lands after at least one
# full yielded chunk (real resume), aligned cut points keep it deterministic.
DL_CHUNK = 64 * 1024


def make_payload(mib: int) -> bytes:
    """Semi-structured records with a deterministic hash token per line.

    Mid-entropy on purpose: pure padding would compress to nothing (no
    throughput signal), pure randomness would never compress (no ratio
    signal).  sha256-derived tokens keep runs reproducible.
    """
    target = mib * 1024 * 1024
    out = bytearray()
    i = 0
    while len(out) < target:
        token = hashlib.sha256(str(i).encode()).hexdigest()[:32]
        out.extend(f"id={i:08d} tok={token} status=ok blob={'y' * 120}\n".encode())
        i += 1
    return bytes(out[:target])


def make_code_corpus(max_bytes: int = 8 * 1024 * 1024) -> bytes:
    """Deterministic tar of this repo's source — the "hosting my code" corpus.

    Text (code/markdown/config) behaves nothing like the synthetic records
    above, so ratio claims about hosting source files must be measured here.
    Deterministic by construction: sorted paths, mtime 0, uid/gid 0, USTAR.
    """
    exts = {".py", ".ts", ".tsx", ".md", ".json", ".yaml", ".yml", ".sh", ".toml"}
    skip = {".git", "node_modules", ".venv", "__pycache__", "dist", "build", ".next", "data"}
    files: list[tuple[str, Path]] = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix not in exts:
            continue
        rel = p.relative_to(ROOT)
        if any(part in skip for part in rel.parts):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if 0 < size < 2 * 1024 * 1024:
            files.append((str(rel), p))
    buf = io.BytesIO()
    total = 0
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, p in files:
            if total >= max_bytes:
                break
            blob = p.read_bytes()
            total += len(blob)
            info = tarfile.TarInfo(name=name)
            info.size = len(blob)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(blob))
    return buf.getvalue()


def _mibs(nbytes: int, secs: float) -> float | None:
    if secs <= 0:
        return None
    return round(nbytes / (1024 * 1024) / secs, 1)


# ─────────────────────────── Tier 1: codec ───────────────────────────


def bench_identity(data: bytes) -> dict:
    """Stream-copy baseline — the uncompressed reference."""
    src = io.BytesIO(data)
    dst = io.BytesIO()
    t0 = time.perf_counter()
    while True:
        chunk = src.read(1024 * 1024)
        if not chunk:
            break
        dst.write(chunk)
    elapsed = time.perf_counter() - t0
    return {
        "codec": "identity",
        "raw_bytes": len(data),
        "wire_bytes": len(data),
        "ratio": 1.0,
        "enc_mibs": _mibs(len(data), elapsed),
        "dec_mibs": None,
    }


def bench_slz4(data: bytes, level: int) -> dict:
    """Compress via downcraft ``compress_stream`` (SLZ4 header + frame),
    then decompress and verify the SHA-256 — integrity is a hard gate."""
    src = io.BytesIO(data)
    dst = io.BytesIO()
    t0 = time.perf_counter()
    compress_stream(src, dst, compression_level=level)
    enc = time.perf_counter() - t0
    wire = dst.getvalue()

    out = io.BytesIO()
    t0 = time.perf_counter()
    decompress_stream(io.BytesIO(wire), out, verify_header=True)
    dec = time.perf_counter() - t0
    got = out.getvalue()
    if hashlib.sha256(got).hexdigest() != hashlib.sha256(data).hexdigest():
        raise RuntimeError(f"SLZ4 level {level}: decoded bytes fail SHA-256")

    return {
        "codec": f"slz4-lv{level}",
        "raw_bytes": len(data),
        "wire_bytes": len(wire),
        "ratio": round(len(wire) / len(data), 4),
        "enc_mibs": _mibs(len(data), enc),
        "dec_mibs": _mibs(len(data), dec),
    }


def bench_gzip(data: bytes, level: int = 6) -> dict:
    """Edge codec reference — gzip as the gateway/client would see it."""
    buf = io.BytesIO()
    t0 = time.perf_counter()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=level, mtime=0) as f:
        f.write(data)
    enc = time.perf_counter() - t0
    wire = buf.getvalue()
    t0 = time.perf_counter()
    gzip.decompress(wire)
    dec = time.perf_counter() - t0
    return {
        "codec": f"gzip-{level}",
        "raw_bytes": len(data),
        "wire_bytes": len(wire),
        "ratio": round(len(wire) / len(data), 4),
        "enc_mibs": _mibs(len(data), enc),
        "dec_mibs": _mibs(len(data), dec),
    }


def bench_zstd(data: bytes, level: int = 3) -> dict | None:
    if zstd is None:
        return None
    cctx = zstd.ZstdCompressor(level=level)
    t0 = time.perf_counter()
    wire = cctx.compress(data)
    enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    zstd.ZstdDecompressor().decompress(wire)
    dec = time.perf_counter() - t0
    return {
        "codec": f"zstd-{level}",
        "raw_bytes": len(data),
        "wire_bytes": len(wire),
        "ratio": round(len(wire) / len(data), 4),
        "enc_mibs": _mibs(len(data), enc),
        "dec_mibs": _mibs(len(data), dec),
    }


def run_tier1(data: bytes) -> list[dict]:
    rows = [bench_identity(data)]
    for level in (1, 6, 16):
        rows.append(bench_slz4(data, level))
    rows.append(bench_gzip(data))
    zrow = bench_zstd(data)
    if zrow:
        rows.append(zrow)
    return rows


# ─────────────────── Tier 1b: external reference CLIs ───────────────────

# Matched budgets, one speed-class and one ratio-class setting per tool.
EXTERNAL_LEVELS: dict[str, tuple[str, str]] = {
    "gzip": ("1", "9"),
    "zstd": ("1", "19"),
    "lz4": ("1", "9"),  # 9 = HC mode (the lz4 CLI's "best")
    "xz": ("1", "9"),
    "bzip2": ("1", "9"),
    "7z": ("1", "9"),  # -mx=N; 7z is the RAR/WinZip-school stand-in
}


def bench_external_cli(tool: str, data: bytes, level: str) -> dict | None:
    """Compress via the system CLI at *level*, decode, SHA-256-gate.

    Returns ``None`` when the tool is not installed (skipped, not fatal).
    Encode/decode walls include process spawn (~ms — noise at 8 MiB).
    """
    exe = shutil.which(tool)
    if exe is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src.bin"
        src.write_bytes(data)
        if tool == "7z":
            arc = Path(td) / "src.7z"
            t0 = time.perf_counter()
            subprocess.run(
                [exe, "a", "-t7z", f"-mx={level}", "-y", str(arc), str(src)],
                check=True, capture_output=True,
            )
            enc = time.perf_counter() - t0
            wire = arc.read_bytes()
            t0 = time.perf_counter()
            got = subprocess.run(
                [exe, "x", "-so", "-y", str(arc)], check=True, capture_output=True
            ).stdout
            dec = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            wire = subprocess.run(
                [exe, "-c", f"-{level}", str(src)], check=True, capture_output=True
            ).stdout
            enc = time.perf_counter() - t0
            t0 = time.perf_counter()
            got = subprocess.run(
                [exe, "-d", "-c"], input=wire, check=True, capture_output=True
            ).stdout
            dec = time.perf_counter() - t0
    if hashlib.sha256(got).digest() != hashlib.sha256(data).digest():
        raise RuntimeError(f"external {tool}-{level}: decoded bytes fail SHA-256")
    return {
        "codec": f"ext-{tool}-{level}",
        "raw_bytes": len(data),
        "wire_bytes": len(wire),
        "ratio": round(len(wire) / len(data), 4),
        "enc_mibs": _mibs(len(data), enc),
        "dec_mibs": _mibs(len(data), dec),
    }


def run_tier1b(data: bytes) -> list[dict]:
    rows: list[dict] = []
    for tool, (speed, ratio) in EXTERNAL_LEVELS.items():
        for level in (speed, ratio):
            row = bench_external_cli(tool, data, level)
            if row:
                rows.append(row)
    return rows


# ────────────────────── Tier 2: end-to-end download ──────────────────────


def run_download(
    server: HTTPServer,
    path: str,
    payload: bytes,
    *,
    compressed: bool,
    truncate_at: int | None,
) -> dict:
    """One ``download_file`` run; reports wall time, throughput, attempts.

    Passes the payload SHA-256 as *checksum* so downcraft itself verifies
    end-to-end integrity; the benchmark also re-verifies and flags it.
    """
    sha = hashlib.sha256(payload).hexdigest()
    mark = len(RangeHandler.requests_log)
    if truncate_at is not None:
        RangeHandler.truncate_once[path] = truncate_at
    try:
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / path.rsplit("/", 1)[-1]
            t0 = time.perf_counter()
            download_file(_range_url(server, path), dest, checksum=sha, compressed=compressed)
            wall = time.perf_counter() - t0
            actual = hashlib.sha256(dest.read_bytes()).hexdigest()
    finally:
        RangeHandler.truncate_once.pop(path, None)
    attempts = sum(1 for e in RangeHandler.requests_log[mark:] if e["path"] == path)
    return {
        "scenario": path.rsplit("/", 1)[-1] + ("-interrupted" if truncate_at else "-oneshot"),
        "bytes": len(payload),
        "attempts": attempts,
        "wall_s": round(wall, 3),
        "mibs": _mibs(len(payload), wall),
        "sha_ok": actual == sha,
    }


def run_tier2(server: HTTPServer, payload: bytes, wire_len: int) -> list[dict]:
    cut = lambda n: max(DL_CHUNK, (n // 2 // DL_CHUNK) * DL_CHUNK)  # noqa: E731
    rows = [
        run_download(server, "/file.bin", payload, compressed=False, truncate_at=None),
        run_download(server, "/file.lz4", payload, compressed=True, truncate_at=None),
        run_download(
            server, "/file.bin", payload, compressed=False, truncate_at=cut(len(payload))
        ),
    ]
    if wire_len >= 2 * DL_CHUNK:
        rows.append(
            run_download(server, "/file.lz4", payload, compressed=True, truncate_at=cut(wire_len))
        )
    return rows


# ─────────────────────────────── reporting ───────────────────────────────


def print_tables(tier1: list[dict], tier1b: list[dict], tier2: list[dict]) -> None:
    print("\nTier 1 — codec (in-process, SHA-256-gated):")
    print(
        f"  {'codec':<10} {'wire B':>12} {'ratio':>8} "
        f"{'enc MiB/s':>10} {'dec MiB/s':>10}"
    )
    for r in tier1:
        print(
            f"  {r['codec']:<10} {r['wire_bytes']:>12} {r['ratio']:>8.4f} "
            f"{str(r['enc_mibs']):>10} {str(r['dec_mibs'] or '-'):>10}"
        )

    if tier1b:
        print("\nTier 1b — external baselines (system CLIs, SHA-256-gated):")
        print(
            f"  {'codec':<14} {'wire B':>12} {'ratio':>8} "
            f"{'enc MiB/s':>10} {'dec MiB/s':>10}"
        )
        for r in tier1b:
            print(
                f"  {r['codec']:<14} {r['wire_bytes']:>12} {r['ratio']:>8.4f} "
                f"{str(r['enc_mibs']):>10} {str(r['dec_mibs'] or '-'):>10}"
            )

    if not tier2:
        return
    base = tier2[0]["wall_s"] or 1e-9
    print("\nTier 2 — end-to-end download_file (Range server, checksum-verified):")
    print(
        f"  {'scenario':<24} {'requests':>8} {'wall s':>8} "
        f"{'MiB/s':>8} {'vs 1x':>7} {'sha256':>7}"
    )
    for r in tier2:
        rel = f"{r['wall_s'] / base * 100:.0f}%"
        print(
            f"  {r['scenario']:<24} {r['attempts']:>8} {r['wall_s']:>8.3f} "
            f"{str(r['mibs']):>8} {rel:>7} {'ok' if r['sha_ok'] else 'FAIL':>7}"
        )


def print_verdict(tier1: list[dict], tier1b: list[dict]) -> None:
    """Pareto standing — where SLZ4 actually ranks. Summarize, don't dump."""
    competitors = [r for r in tier1 + tier1b if r["codec"] != "identity"]
    if not competitors:
        return

    def _num(v: float | None) -> float:
        return v if isinstance(v, (int, float)) else 0.0

    best_ratio = min(competitors, key=lambda r: r["ratio"])
    best_enc = max(competitors, key=lambda r: _num(r["enc_mibs"]))
    best_dec = max(competitors, key=lambda r: _num(r["dec_mibs"]))
    by_ratio = sorted(competitors, key=lambda r: r["ratio"])
    n = len(by_ratio)
    print("\nVerdict (Pareto standing, this corpus):")
    print(f"  best ratio : {best_ratio['codec']:<12} {best_ratio['ratio']:.4f}")
    print(f"  best encode: {best_enc['codec']:<12} {_num(best_enc['enc_mibs'])} MiB/s")
    print(f"  best decode: {best_dec['codec']:<12} {_num(best_dec['dec_mibs'])} MiB/s")
    for rank, r in enumerate(by_ratio, start=1):
        if r["codec"].startswith("slz4"):
            print(f"  {r['codec']} ratio rank: {rank}/{n}")
    print("  note: encode cost is paid PER REQUEST for on-the-fly wire")
    print("  compression (Tier 2); compress-at-rest (the archive model)")
    print("  pays it once per file, at upload time.")


def main() -> int:
    ap = argparse.ArgumentParser(description="downcraft compression benchmark")
    ap.add_argument("--mib", type=int, default=8, help="synthetic payload size in MiB")
    ap.add_argument("--file", type=Path, default=None, help="benchmark a real artifact instead")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_downcraft_compression.json",
        help="JSON run history path",
    )
    ap.add_argument("--skip-e2e", action="store_true", help="codec tier only")
    ap.add_argument(
        "--corpus",
        choices=("synthetic", "code"),
        default="synthetic",
        help="payload class: synthetic records (default) or a deterministic "
        "repo-source tar (the hosting-my-code corpus)",
    )
    args = ap.parse_args()

    if args.file is not None:
        data = args.file.read_bytes()
        label = str(args.file)
    elif args.corpus == "code":
        data = make_code_corpus()
        label = "deterministic repo-source tar (code corpus)"
    else:
        data = make_payload(args.mib)
        label = f"{args.mib} MiB synthetic download records"
    print(f"payload: {label} ({len(data)} bytes)")

    tier1 = run_tier1(data)
    tier1b = run_tier1b(data)
    wire_len = len(slz4_wire(data))

    tier2: list[dict] = []
    server = None
    if not args.skip_e2e:
        # Measurement knobs — see module docstring.
        http_mod.CHUNK_SIZE = DL_CHUNK
        http_mod.RETRY_DELAY = 0.0
        # Injected truncations log a WARNING per attempt; the table's
        # requests column is the resume signal — keep output readable.
        logging.getLogger("downcraft.download.http").setLevel(logging.ERROR)
        RangeHandler.payloads = {"/file.bin": data, "/file.lz4": data}
        RangeHandler.lz4_paths = {"/file.lz4": True}
        server = HTTPServer(("127.0.0.1", 0), RangeHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            tier2 = run_tier2(server, data, wire_len)
        finally:
            server.shutdown()

    print_tables(tier1, tier1b, tier2)
    print_verdict(tier1, tier1b)

    history: list[dict] = []
    if args.out.exists():
        try:
            history = json.loads(args.out.read_text())
        except (json.JSONDecodeError, OSError):
            history = []
    history.append(
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "payload": label,
            "bytes": len(data),
            "tier1": tier1,
            "tier1b": tier1b,
            "tier2": tier2,
        }
    )
    args.out.write_text(json.dumps(history, indent=2))

    bad = [r for r in tier2 if not r["sha_ok"]]
    if bad:
        print(f"\nFAIL: {len(bad)} run(s) failed SHA-256 verification")
        return 1
    print(f"\nhistory: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
