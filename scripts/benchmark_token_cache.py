#!/usr/bin/env python3
"""Benchmark the content-addressed token cache: cold vs warm prepare_data.

Measures the wall-clock cost of ``prepare_data`` on a synthetic corpus of
``--mb`` megabytes (default 64, above the 50 MB SOP floor):

* cold — first call, no cache on disk: full read + char-loop/tokenize + store
* warm — second call, cache hit: memmap open + meta read

Pass criterion (SOP): warm >= 5x faster than cold on >= 50 MB. Also prints
the on-disk cache footprint (uint16 compact vs the int64 the trainer used
to carry). Writes nothing outside a throwaway tmp dir and ``runs/tokencache``
(overridable via ``--cache-root`` / ``--corpus``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _make_corpus(path: Path, target_bytes: int) -> None:
    """Deterministic filler text that clears validate_training_data floors."""
    sentence = (
        "The quick brown fox jumps over the lazy dog while the sun sets "
        "behind the distant mountains and the river reflects the evening light "
    )
    chunk = (sentence + "\n") * 64
    written = 0
    with path.open("w", encoding="utf-8") as f:
        while written < target_bytes:
            f.write(chunk)
            written += len(chunk)


def run(corpus_mb: int = 64, cache_root: Path | None = None) -> dict:
    import numpy as np  # noqa: F401 — ensures env matches trainer deps

    from domain.training._internal import token_cache
    from domain.training._internal.train_pipeline import prepare_data

    tmp = Path(tempfile.mkdtemp(prefix="tokcache-bench-"))
    try:
        corpus = tmp / "corpus.txt"
        _make_corpus(corpus, corpus_mb * 1024 * 1024)
        actual_mb = corpus.stat().st_size / (1024 * 1024)

        root = cache_root or (tmp / "tokencache")
        token_cache.DEFAULT_CACHE_ROOT = root

        t0 = time.perf_counter()
        d1, vocab1, *_ = prepare_data(str(corpus))
        cold_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        d2, vocab2, *_ = prepare_data(str(corpus))
        warm_s = time.perf_counter() - t0

        assert vocab1 == vocab2 and len(d1) == len(d2)

        bins = list(root.glob("*.bin"))
        cache_bytes = sum(p.stat().st_size for p in bins)
        naive_int64_bytes = len(d1) * 8

        speedup = cold_s / warm_s if warm_s > 0 else float("inf")
        return {
            "corpus_mb": round(actual_mb, 1),
            "tokens": int(len(d1)),
            "vocab": int(vocab1),
            "cold_s": round(cold_s, 3),
            "warm_s": round(warm_s, 4),
            "speedup": round(speedup, 1),
            "cache_mb": round(cache_bytes / (1024 * 1024), 1),
            "naive_int64_mb": round(naive_int64_bytes / (1024 * 1024), 1),
            "pass": speedup >= 5.0,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--mb", type=int, default=64, help="corpus size in MB (>=50 per SOP)")
    parser.add_argument("--cache-root", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run(args.mb, args.cache_root)
    print("token cache benchmark (cold vs warm prepare_data)")
    print(
        f"  corpus        : {result['corpus_mb']} MB, {result['tokens']} tokens, vocab {result['vocab']}"
    )
    print(f"  cold (miss)   : {result['cold_s']} s")
    print(f"  warm (hit)    : {result['warm_s']} s")
    print(f"  speedup       : {result['speedup']}x  (SOP >= 5x)")
    print(
        f"  cache on disk : {result['cache_mb']} MB  (naive int64 would be {result['naive_int64_mb']} MB)"
    )
    print(f"  verdict       : {'PASS' if result['pass'] else 'FAIL'}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
