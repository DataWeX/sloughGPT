"""
Soul Filename-Grammar + Checkpoint I/O Benchmark

Measures the cost the .soul filename grammar adds to the checkpoint path
(domain/inference/_internal/slo_format.py: soul_path / soul_read_candidates /
load_soul's candidate probe).

What this is guarding:
  - save_soul() now canonicalizes the destination before writing, and
    load_soul() probes an ordered candidate list instead of opening one path.
    Both are new work on a hot path (every trainer checkpoint, every model
    load), so the per-call cost has to stay negligible next to the I/O.
  - The probe has a worst case: the canonical name is absent and the legacy
    double-appended spelling is what exists on disk. That path does two
    os.path.isfile() calls and one open instead of one open.

Benchmarks:
  soul_path_write      pure canonicalization (string work only, no I/O)
  read_candidates      ordered probe-list construction (string work only)
  save_canonical       save_soul() to an already-canonical name (adds soul_path)
  save_legacy_name     save_soul() from a legacy .soul.soul name (canonicalizes)
  load_canonical_hit   load_soul() where the canonical file exists (1 probe)
  load_legacy_hit      load_soul() where only .soul.soul exists (2 probes + miss)
  sidecar_small        read_sidecar() on a normal sidecar (no training blob)
  sidecar_bloated      read_sidecar() where metadata.training_state is 40MB
  detail_enrich        identity fill for one checkpoint detail request

Interpretation: the four string benchmarks should read as microseconds; the
I/O benchmarks should be dominated by the read itself. A regression shows up
as load_legacy_hit pulling away from load_canonical_hit by more than one
stat() call's worth (~0.01ms), which means the probe loop is doing real work
per candidate rather than a single failed stat.

The two sidecar benchmarks guard a defect that shipped: a sidecar carries
`metadata.training_state` — the optimizer state training resumes from — which
reaches 40MB inside a 69MB file. Readers that json.load()'d it whole cost
3.85s for the worst single file, 81.7s to list models/, and 17.9s to serve
GET /training/checkpoints. `read_sidecar` skips the blob, so the two reads
should land in the same sub-millisecond band; a large ratio means the size
guard has stopped working and the blob is being parsed again. This benchmark
exits 1 when it does.

The detail benchmark guards the opposite edge of the same split. Opening one
checkpoint now runs classify_soul() so the row always names its container
(all 23 files on this machine self-describe, ~7ms each), but that probe must
stay a per-request cost: if enrichment ever grows into parsing a whole
sidecar or hashing a whole file, opening a single checkpoint starts paying
what listing all of them once cost — the 81.74s cmd_models defect. Exits 1
past 50ms, ~50x the expected probe and still under what a 40MB blob parse
(3.85s) or a 44MB whole-file hash would cost.

Usage:
    python scripts/benchmark_soul_paths.py
    python scripts/benchmark_soul_paths.py --json
    python scripts/benchmark_soul_paths.py --repeats 2000
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class BenchResult:
    name: str
    repeats: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    ops_per_s: float


def _summarize(name: str, lat: list[float], repeats: int) -> BenchResult:
    lat = sorted(lat)
    return BenchResult(
        name=name,
        repeats=repeats,
        mean_ms=round(1000 * statistics.fmean(lat), 6),
        p50_ms=round(1000 * lat[len(lat) // 2], 6),
        p95_ms=round(1000 * lat[int(len(lat) * 0.95)], 6),
        ops_per_s=round(1.0 / statistics.fmean(lat)) if statistics.fmean(lat) else 0.0,
    )


def _time(fn, repeats: int) -> list[float]:
    """Wall-clock each call in seconds. First call is warmed up and dropped."""
    fn()
    lat = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        lat.append(time.perf_counter() - t0)
    return lat


# Bloated-sidecar guard. A sidecar whose metadata.training_state is skipped
# parses only the ~2KB above the blob, so its p50 stays sub-millisecond even
# with the whole file on disk. One that gets parsed whole moves ~40MB of JSON
# through json.loads: tens of milliseconds, 100x the correct path. 10ms sits
# two orders of magnitude above the pass case and well under the fail case,
# so machine contention can't flip the verdict.
_SIDECAR_P50_MS = 10.0

_SIDECAR_BLOB_BYTES = 40 * 1024 * 1024  # matches the real 40.9M-char blobs


def _sidecar_exit(bloated: BenchResult) -> int:
    """0 when read_sidecar() skipped the resume-state blob, 1 when it parsed it."""
    return 1 if bloated.p50_ms > _SIDECAR_P50_MS else 0


# Detail-enrichment guard. classify_soul() probes the spelling candidates and
# reads a handful of KB, so it lands well under a millisecond; 50ms is ~50x
# that headroom (concurrent runs here swing the load average 18-30) while
# staying far below the failure cases — parsing a 40MB resume blob whole cost
# 3.85s and hashing a 44MB checkpoint would cost hundreds of ms.
_DETAIL_P50_MS = 50.0


def _detail_exit(detail: BenchResult) -> int:
    """0 when identity fill stayed a per-request cost, 1 when it grew heavy."""
    return 1 if detail.p50_ms > _DETAIL_P50_MS else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=500, help="iterations per benchmark")
    ap.add_argument("--io-repeats", type=int, default=150, help="iterations for I/O benchmarks")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    from domain.inference import load_soul, save_soul, soul_path, soul_read_candidates
    from domain.inference._internal.slo_format import read_sidecar
    from domain.training._internal.checkpoints import _describe_identity

    results: list[BenchResult] = []

    # ── Pure string work: the grammar itself, no filesystem ──────────────
    results.append(
        _summarize(
            "soul_path_write",
            _time(lambda: soul_path("models/v2.3/gpt2.5-model.soul.soul"), args.repeats),
            args.repeats,
        )
    )
    results.append(
        _summarize(
            "read_candidates",
            _time(lambda: soul_read_candidates("m/x.soul.soul"), args.repeats),
            args.repeats,
        )
    )

    tmp = Path(tempfile.mkdtemp(prefix="soul_bench_"))
    try:
        weights_only = {"weights_only": True}

        # ── Write path: canonicalization cost against real I/O ───────────
        def save_canonical():
            save_soul(None, str(tmp / "canon.soul"), **weights_only)

        def save_legacy():
            save_soul(None, str(tmp / "legacy.soul.soul"), **weights_only)

        results.append(
            _summarize("save_canonical", _time(save_canonical, args.io_repeats), args.io_repeats)
        )
        results.append(
            _summarize("save_legacy_name", _time(save_legacy, args.io_repeats), args.io_repeats)
        )

        # ── Read path: 1 probe vs the 2-probe legacy worst case ─────────
        save_soul(None, str(tmp / "hit.soul"), **weights_only)
        save_soul(None, str(tmp / "miss.soul"), **weights_only)
        # Legacy-only layout: file is at the double-appended name, canonical
        # is absent, so load_soul must fail the first isfile() before it
        # reaches the real file on the second candidate.
        save_soul(None, str(tmp / "legacyhit.soul"), **weights_only)
        (tmp / "legacyhit.soul").rename(tmp / "legacyhit.soul.soul")

        results.append(
            _summarize(
                "load_canonical_hit",
                _time(lambda: load_soul(str(tmp / "hit.soul")), args.io_repeats),
                args.io_repeats,
            )
        )
        results.append(
            _summarize(
                "load_legacy_hit",
                _time(lambda: load_soul(str(tmp / "legacyhit.soul")), args.io_repeats),
                args.io_repeats,
            )
        )

        # ── Sidecar read: the resume-state blob must be skipped ─────────
        # Two byte-identical documents except for metadata.training_state,
        # the optimizer state training resumes from (real ones reach 40MB
        # inside a 69MB file). read_sidecar() parses the whole document for
        # the small one and only the head for the big one, so the two latencies
        # have to land in the same band: if the size guard ever stops firing,
        # the blob is being json.parse()'d again and this ratio blows up.
        sc_small = {
            "name": "bench",
            "born_at": "2026-10-01T00:00:00Z",
            "final_train_loss": 1.5,
            "integrity_hash": "ab12cd34ef56",
            "metadata": {"vocab_size": 31, "config": {"n_layer": 4}},
        }
        sc_big = json.loads(json.dumps(sc_small))
        sc_big["metadata"]["training_state"] = {
            "step": 1,
            "blob": "x" * _SIDECAR_BLOB_BYTES,  # ~40MB, past the 4MB parse limit
        }
        for stem, doc in (("sidec_small", sc_small), ("sidec_big", sc_big)):
            (tmp / f"{stem}.soul").write_bytes(b"")
            (tmp / f"{stem}.soul.meta.json").write_text(json.dumps(doc), encoding="utf-8")

        sc_repeats = max(20, min(args.io_repeats, 100))
        results.append(
            _summarize(
                "sidecar_small",
                _time(lambda: read_sidecar(str(tmp / "sidec_small.soul")), sc_repeats),
                sc_repeats,
            )
        )
        results.append(
            _summarize(
                "sidecar_bloated",
                _time(lambda: read_sidecar(str(tmp / "sidec_big.soul")), sc_repeats),
                sc_repeats,
            )
        )

        # ── Detail enrichment: classify one file, fill only what's missing ─
        # checkpoint_info() runs this on every detail request so a file whose
        # sidecar merely names it can't leave the dialog with no format at
        # all. The contract it must keep: one file per request, and gaps only
        # — a stored integrity_hash / tier / provenance is never overwritten,
        # or read-time derivation would drift identity between sessions.
        (tmp / "detail.soul").write_bytes(b"not a soul container, just bytes")
        detail_row = {"name": "detail", "model_path": str(tmp / "detail.soul")}
        results.append(
            _summarize(
                "detail_enrich",
                _time(lambda: _describe_identity(dict(detail_row)), sc_repeats),
                sc_repeats,
            )
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    by_name = {r.name: r for r in results}
    hit, legacy = by_name["load_canonical_hit"], by_name["load_legacy_hit"]
    sc_small_r, sc_big_r = by_name["sidecar_small"], by_name["sidecar_bloated"]
    detail_r = by_name["detail_enrich"]

    if args.json:
        print(
            json.dumps(
                {
                    "results": [asdict(r) for r in results],
                    "probe_overhead_ms": round(legacy.p50_ms - hit.p50_ms, 6),
                    "sidecar_bloat_ratio": round(sc_big_r.p50_ms / sc_small_r.p50_ms, 4)
                    if sc_small_r.p50_ms
                    else None,
                    "detail_p50_ms": detail_r.p50_ms,
                },
                indent=2,
            )
        )
        return _sidecar_exit(sc_big_r) | _detail_exit(detail_r)

    print("\nSoul filename grammar + checkpoint I/O")
    print("=" * 66)
    print(f"{'benchmark':<22}{'p50 ms':>12}{'p95 ms':>12}{'ops/s':>14}")
    print("-" * 66)
    for r in results:
        print(f"{r.name:<22}{r.p50_ms:>12.4f}{r.p95_ms:>12.4f}{r.ops_per_s:>14,.0f}")
    print("-" * 66)
    print(f"probe overhead (2nd candidate miss): {legacy.p50_ms - hit.p50_ms:+.4f} ms median")
    print(
        "guard: overhead should stay within noise of one failed stat() "
        "(~0.01ms). A larger gap means the probe loop is doing per-candidate "
        "work beyond the existence check."
    )
    ratio = sc_big_r.p50_ms / sc_small_r.p50_ms if sc_small_r.p50_ms else 0.0
    print(
        f"sidecar bloat ratio (40MB training_state / plain): "
        f"{sc_small_r.p50_ms:.4f} -> {sc_big_r.p50_ms:.4f} ms = {ratio:.1f}x"
    )
    rc = _sidecar_exit(sc_big_r)
    print(
        "guard: reading a sidecar that carries metadata.training_state must "
        f"stay under {_SIDECAR_P50_MS} ms — it skips the blob and parses the "
        "head instead. Over it means size-guard stopped firing and every "
        "model listing pays the resume-state parse again."
    )
    print("sidecar guard: " + ("PASS" if rc == 0 else "FAIL"))
    drc = _detail_exit(detail_r)
    print(
        f"detail enrichment (classify one file, fill missing fields): {detail_r.p50_ms:.4f} ms p50"
    )
    print(
        "guard: opening one checkpoint may classify its bytes and nothing "
        f"heavier — past {_DETAIL_P50_MS} ms it is parsing a whole sidecar or "
        "hashing a whole file, so one detail click starts costing what "
        "listing every model once cost."
    )
    print("detail guard: " + ("PASS" if drc == 0 else "FAIL"))
    print()
    return rc | drc


if __name__ == "__main__":
    raise SystemExit(main())
