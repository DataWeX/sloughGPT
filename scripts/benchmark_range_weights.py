"""
Range-weight loading benchmark (card 20261125_134544).

Compares four ways of materialising N tensors from a real safetensors file:

* **A full** — one big read of the whole file (today's eager byte behaviour:
  every tensor ends up resident, exactly like ``_load_safetensors_bf16``).
* **B seek** — header → offset index → sequential ``os.pread`` of only the N
  selected tensors (the existing loader's transport, made lazy).
* **C parallel** — same N tensors, concurrent preads (ThreadPoolExecutor).
* **D mmap** — same N tensors via mapped pages, held as zero-copy
  ``memoryview`` slices (no second copy; the mapping IS the memory).

Metrics per variant and N (default sweep 1, 10, 100):

* **time-to-first-tensor** (headline) + total wall time for N
  — A reports no first-tensor time (it is not lazy).
  — B/C: first tensor materialised by ``pread``; C uses earliest completion.
  — D: first tensor materialised when its pages are first accessed (faults
    are the read, so the hash update of slice 0 is included).
* **bytes requested** (logical) vs file size, plus ``/proc/self/io``
  ``rchar`` (syscall bytes) and ``read_bytes`` (physical device reads) —
  warm passes should show logical ≫ physical (page cache doing its job).
* **resident MB** with the materialised tensors held + peak RSS (VmHWM):
  B/C/D must never approach the full file size.
* **hash identity** — sha256 over (name, bytes) of the selected tensors in
  file order. A == B == C == D on every N, cold and warm. **This is the
  gate**: exit code is non-zero on any mismatch.

Each (variant, N) runs in its **own child process**: VmHWM is a per-process
high-water mark, and a second load in the same process measures the page
cache, not the code (same lesson as ``benchmark_lazy_autoload.py``). Each
child takes a cold pass first — best-effort eviction via
``posix_fadvise(POSIX_FADV_DONTNEED)``, no sudo — then a warm pass.

Complementary to, NOT a duplicate of:
* ``benchmark_lazy_autoload.py`` — .slnc header-only vs full materialisation
* ``benchmark_model_load.py`` — serial vs parallel inflate of the archive
Neither measures partial N-tensor reads, bytes touched, or cross-path hash
identity.

Scope: file-based transports only (read/seek/mmap). HTTP/socket is out of
scope by decision (2026-11-25) — networking, if ever needed, gets its own
protocol then.

Usage:
    ./scripts/python scripts/benchmark_range_weights.py
    ./scripts/python scripts/benchmark_range_weights.py --json out.json
    ./scripts/python scripts/benchmark_range_weights.py --sweep 1,10,100 --workers 8
Exit code is non-zero when hash identity fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_FILE = str(
    _REPO / "models/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/model.safetensors"
)
DEFAULT_SWEEP = "1,10,100"
VARIANTS = ("A", "B", "C", "D")
VARIANT_LABELS = {"A": "full", "B": "seek", "C": "parallel", "D": "mmap"}


# --------------------------------------------------------------------------- #
# /proc readers
# --------------------------------------------------------------------------- #
def _status_mb(field: str) -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _io_mb(name: str) -> float:
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            if line.startswith(name + ":"):
                return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 0.0


# --------------------------------------------------------------------------- #
# safetensors header
# --------------------------------------------------------------------------- #
def _parse_header(path: str) -> tuple[int, list[tuple[str, int, int]]]:
    """Return (header_len, [(name, rel_begin, rel_end), ...]) in file order."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    tensors = [
        (name, spec["data_offsets"][0], spec["data_offsets"][1])
        for name, spec in header.items()
        if name != "__metadata__" and isinstance(spec, dict) and "data_offsets" in spec
    ]
    tensors.sort(key=lambda t: t[1])
    return header_len, tensors


def _select(count: int, n: int) -> list[int]:
    """Evenly-spread deterministic selection of n indices over file order."""
    n = max(1, min(n, count))
    if n == 1:
        return [count // 2]
    last = count - 1
    return sorted({round(i * last / (n - 1)) for i in range(n)})


# --------------------------------------------------------------------------- #
# hash — identical construction across every variant
# --------------------------------------------------------------------------- #
def _digest_updates(h: hashlib._Hash, items: list[tuple[str, int, int]], blobs) -> None:
    for (name, _, _), blob in zip(items, blobs):
        h.update(name.encode())
        h.update(blob)


# --------------------------------------------------------------------------- #
# the four passes
# --------------------------------------------------------------------------- #
def _pass_full(fd: int, items: list[tuple[str, int, int]], _workers: int):
    """A — read the entire file, then hash the selected tensors out of it."""
    started = time.perf_counter()
    data = os.pread(fd, os.fstat(fd).st_size, 0)  # one slurp — single allocation
    read_done = time.perf_counter() - started
    h = hashlib.sha256()
    # Zero-copy memoryview slices, one at a time: transient views only — same
    # incremental, no-copy hashing the lazy paths do (a materialised slice list
    # would coexist with the buffer and double-count A's peak RSS).
    mv = memoryview(data)
    _digest_updates(h, items, (mv[a:b] for _, a, b in items))
    digest = h.hexdigest()
    mv.release()
    total_s = time.perf_counter() - started
    # Not lazy: the "first tensor" moment IS the full read.
    return read_done, total_s, digest, [data]


def _pread_timed(fd: int, a: int, b: int, t0: float):
    blob = os.pread(fd, b - a, a)
    return blob, time.perf_counter() - t0


def _pass_seek(fd: int, items: list[tuple[str, int, int]], _workers: int):
    """B — sequential pread of exactly the selected tensors."""
    started = time.perf_counter()
    h = hashlib.sha256()
    held = []
    first_s = None
    for i, (name, a, b) in enumerate(items):
        blob = os.pread(fd, b - a, a)
        if i == 0:
            first_s = time.perf_counter() - started
        held.append(blob)
        h.update(name.encode())
        h.update(blob)
    return first_s, time.perf_counter() - started, h.hexdigest(), held


def _pass_parallel(fd: int, items: list[tuple[str, int, int]], workers: int):
    """C — concurrent preads; digest still assembled in file order."""
    workers = max(1, min(workers, len(items)))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_pread_timed, fd, a, b, started) for _, a, b in items]
        results = [f.result() for f in futures]
    first_s = min(t for _, t in results)
    held = [blob for blob, _ in results]
    h = hashlib.sha256()
    _digest_updates(h, items, held)
    return first_s, time.perf_counter() - started, h.hexdigest(), held


def _pass_mmap(fd: int, items: list[tuple[str, int, int]], _workers: int):
    """D — mapped pages, zero-copy memoryview slices (faults are the read)."""
    started = time.perf_counter()
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    view = memoryview(mm)
    h = hashlib.sha256()
    held = []
    first_s = None
    for i, (name, a, b) in enumerate(items):
        sl = view[a:b]
        held.append(sl)
        h.update(name.encode())
        h.update(sl)  # first access faults slice i's pages in
        if i == 0:
            first_s = time.perf_counter() - started
    digest = h.hexdigest()
    # Keep the mapping (and its resident pages) alive until RSS is measured;
    # the caller releases `held` first.
    return first_s, time.perf_counter() - started, digest, held


PASSES = {"A": _pass_full, "B": _pass_seek, "C": _pass_parallel, "D": _pass_mmap}


# --------------------------------------------------------------------------- #
# one measured pass (cold or warm)
# --------------------------------------------------------------------------- #
def _one_pass(
    variant: str, path: str, items: list[tuple[str, int, int]], workers: int, cold: bool
) -> dict:
    fd = os.open(path, os.O_RDONLY)
    try:
        if cold:
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        rchar_before = _io_mb("rchar")
        phys_before = _io_mb("read_bytes")
        first_s, total_s, digest, held = PASSES[variant](fd, items, workers)
        resident_mb = _status_mb("VmRSS")
        rchar_mb = round(_io_mb("rchar") - rchar_before, 3)
        phys_mb = round(_io_mb("read_bytes") - phys_before, 3)
        logical_mb = round(sum(b - a for _, a, b in items) / 1e6, 3)
        del held  # release materialised tensors before moving on
        return {
            "first_s": None if first_s is None else round(first_s, 4),
            "total_s": round(total_s, 4),
            "logical_mb": logical_mb,
            "rchar_mb": rchar_mb,
            "phys_mb": phys_mb,
            "resident_mb": round(resident_mb, 1),
            "digest": digest,
        }
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- #
# child
# --------------------------------------------------------------------------- #
def _child(variant: str, n: int, path: str, workers: int) -> int:
    header_len, tensors = _parse_header(path)
    selected = _select(len(tensors), n)
    items = [
        (tensors[i][0], 8 + header_len + tensors[i][1], 8 + header_len + tensors[i][2])
        for i in selected
    ]
    baseline_mb = _status_mb("VmRSS")
    out = {
        "variant": variant,
        "n": len(selected),
        "requested_n": n,
        "file_mb": round(os.path.getsize(path) / 1e6, 1),
        "tensor_count": len(tensors),
        "baseline_rss_mb": round(baseline_mb, 1),
        "passes": {},
    }
    for label, cold in (("cold", True), ("warm", False)):
        out["passes"][label] = _one_pass(variant, path, items, workers, cold)
    out["peak_rss_mb"] = round(_status_mb("VmHWM"), 1)
    print(json.dumps(out))
    return 0


# --------------------------------------------------------------------------- #
# parent
# --------------------------------------------------------------------------- #
def _spawn(variant: str, n: int, path: str, workers: int, timeout_s: float) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        variant,
        "--n",
        str(n),
        "--file",
        path,
        "--workers",
        str(workers),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{variant}/N={n} child failed ({proc.returncode}):\n{proc.stderr[-2000:]}"
        )
    payload = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if not payload:
        raise RuntimeError(f"{variant}/N={n} child printed nothing:\n{proc.stdout[-2000:]}")
    return json.loads(payload[-1])


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}" if value < 10 else f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def _print_pass(label: str, n: int, records: list[dict]) -> None:
    print(f"\n  N={n} — {label} pass")
    print(
        f"    {'variant':<10} {'first_s':>8} {'total_s':>8} {'logical':>9} "
        f"{'rchar':>9} {'phys':>9} {'resident':>9} {'peakRSS':>9}  hash"
    )
    for rec in records:
        p = rec["passes"][label]
        print(
            f"    {rec['variant'] + ' ' + VARIANT_LABELS[rec['variant']]:<10} "
            f"{_fmt(p['first_s']):>8} {_fmt(p['total_s']):>8} {_fmt(p['logical_mb'], 'M'):>9} "
            f"{_fmt(p['rchar_mb'], 'M'):>9} {_fmt(p['phys_mb'], 'M'):>9} "
            f"{_fmt(p['resident_mb'], 'M'):>9} {_fmt(rec['peak_rss_mb'], 'M'):>9}  "
            f"{p['digest'][:12]}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--file", default=DEFAULT_FILE, help="safetensors file to benchmark")
    ap.add_argument("--sweep", default=DEFAULT_SWEEP, help="comma list of tensor counts")
    ap.add_argument("--workers", type=int, default=8, help="parallel readers for variant C")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-child timeout (s)")
    ap.add_argument("--json", default="", help="write results to this path")
    ap.add_argument(
        "child_variant",
        nargs="?",
        default="",
        choices=[""] + list(VARIANTS),
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--n", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        return _child(args.child_variant, args.n, args.file, args.workers)

    if not Path(args.file).is_file():
        print(f"error: file not found: {args.file}")
        return 2

    sweep = [int(x) for x in args.sweep.split(",") if x.strip()]
    file_mb = os.path.getsize(args.file) / 1e6
    print(f"file: {args.file}")
    print(f"size: {file_mb:.1f} MB   sweep N={sweep}   variants A/B/C/D (full/seek/parallel/mmap)")

    results: list[dict] = []
    for n in sweep:
        for variant in VARIANTS:
            results.append(_spawn(variant, n, args.file, args.workers, args.timeout))

    # ---- report ----
    gate_failures: list[str] = []
    for n in sweep:
        group = [r for r in results if r["requested_n"] == n]
        for label in ("cold", "warm"):
            _print_pass(label, n, group)
            digests = {r["variant"]: r["passes"][label]["digest"] for r in group}
            if len(set(digests.values())) != 1:
                gate_failures.append(f"N={n}/{label}: {digests}")

    # ---- gate ----
    print()
    if gate_failures:
        print("GATE FAIL — hash identity across variants:")
        for failure in gate_failures:
            print(f"  {failure}")
    else:
        print("GATE PASS — A == B == C == D on every N, cold and warm (byte-identical).")

    if args.json:
        payload = {
            "file": args.file,
            "file_mb": round(file_mb, 1),
            "sweep": sweep,
            "results": results,
            "hash_identity": not gate_failures,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")

    return 1 if gate_failures else 0


if __name__ == "__main__":
    sys.exit(main())
