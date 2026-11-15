"""Tokenizer encode throughput benchmark.

Measures what the /chat time-to-first-token path spends on BPE: the pure
``MorphTokenizer.encode`` cost at growing prompt sizes, with no model, no
HTTP and no SSE in the way.

Why it exists (card 20261102_001616): the merge loop inside ``_bpe_encode``
was O(n²) — rescanning every adjacent pair and rebuilding the list on each
merge — so a 6000-char prompt cost 3.88s to tokenize (fitted exponent 2.02)
against 0.12s for 1024. The heap + linked-list ``_bpe_merge`` replaced it
with O(n log n), taking 6000 chars to 9.4ms. This benchmark is the gate that keeps it there: a
quadratic regression shows up as ms/1k chars growing with input size, which
``--ci`` fails on via the ratio check.

Reports, per tokenizer and per size:
  ms_total     wall time to encode the whole prompt
  ms_per_1k    normalized cost (the number that must stay flat as size grows)
  tokens       token count (for tok/s context)
  ratio_vs_prev  ms/1k relative to the previous size — flat == linear

Usage:
    python scripts/benchmark_tokenizer_encode.py
    python scripts/benchmark_tokenizer_encode.py --reps 5 --json-out run.json
    python scripts/benchmark_tokenizer_encode.py --update   # store baseline
    python scripts/benchmark_tokenizer_encode.py --ci       # fail on regression
    python scripts/benchmark_tokenizer_encode.py --model gpt2

Feed a run into the results store with:
    python scripts/benchmark_results.py record --kind inference --json-file run.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

BASELINE_FILE = Path("data/benchmark_tokenizer_encode_baseline.json")

# Prompt sizes in characters. The small/large pair is what the scaling gate
# compares; the rest of the curve shows where per-1k cost starts to climb.
SIZES = [256, 1024, 3120, 6000]

# A linear encoder keeps ms/1k flat, so 4x the input costs ~4x the time.
# Quadratic costs ~16x. 8x sits between and fails only the quadratic case.
SCALING_LIMIT = 8.0

BASE_TEXT = "The quick brown fox jumps over the lazy dog. "


def _cached_models() -> list[str]:
    """Tokenizers present in the local cache (loader skips uncached ones)."""
    from domain.infrastructure._internal.model_resolver import get_model_dir

    out = []
    for mid in ("gpt2", "Qwen/Qwen2.5-0.5B-Instruct"):
        try:
            if (get_model_dir(mid) / "tokenizer.json").exists():
                out.append(mid)
        except Exception:  # noqa: BLE001 - resolver may not know the layout
            continue
    return out


def measure(model_id: str, reps: int) -> dict:
    from domain.infrastructure._internal.morph_tokenizer import MorphTokenizer

    tok = MorphTokenizer.from_pretrained(model_id)

    sizes: list[dict] = []
    prev_ms_per_1k = None
    for n in SIZES:
        text = (BASE_TEXT * (n // len(BASE_TEXT) + 1))[:n]
        tok.encode(text)  # warm: numba JIT + weight flattening

        samples = []
        n_tokens = 0
        for _ in range(reps):
            t0 = time.perf_counter()
            ids = tok.encode(text)
            samples.append((time.perf_counter() - t0) * 1000)
            n_tokens = len(ids)

        ms = statistics.median(samples)
        ms_per_1k = ms / n * 1000
        ratio = (ms_per_1k / prev_ms_per_1k) if prev_ms_per_1k else None
        sizes.append(
            {
                "chars": n,
                "ms_total": round(ms, 3),
                "ms_per_1k": round(ms_per_1k, 3),
                "tokens": n_tokens,
                "tok_per_s": round(n_tokens / (ms / 1000), 1) if ms > 0 else 0,
                "ratio_vs_prev": round(ratio, 3) if ratio else None,
            }
        )
        prev_ms_per_1k = ms_per_1k

    # Scaling gate: smallest vs largest size, normalized per 1k chars.
    first, last = sizes[0], sizes[-1]
    scale_ratio = last["ms_per_1k"] / first["ms_per_1k"] if first["ms_per_1k"] else 0.0

    return {
        "model": model_id,
        "reps": reps,
        "sizes": sizes,
        "scaling_ratio": round(scale_ratio, 3),
        "scaling_limit": SCALING_LIMIT,
        "linear": scale_ratio < SCALING_LIMIT,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="single tokenizer to measure")
    ap.add_argument("--reps", type=int, default=3, help="repeats per size (median)")
    ap.add_argument("--json-out", default=None, help="write full results JSON")
    ap.add_argument("--update", action="store_true", help="store as baseline")
    ap.add_argument("--ci", action="store_true", help="exit non-zero on regression")
    args = ap.parse_args()

    models = [args.model] if args.model else _cached_models()
    if not models:
        print("[FAIL] no cached tokenizers found", file=sys.stderr)
        return 1

    results = []
    print(f"[BENCH] tokenizer encode — {len(models)} tokenizer(s), reps={args.reps}")
    print(
        f"{'tokenizer':<30} {'chars':>7} {'ms':>9} {'ms/1k':>9} "
        f"{'tokens':>7} {'tok/s':>9} {'ratio':>7}"
    )
    for mid in models:
        r = measure(mid, args.reps)
        results.append(r)
        short = mid.split("/")[-1]
        for s in r["sizes"]:
            ratio = f"{s['ratio_vs_prev']:.2f}x" if s["ratio_vs_prev"] else "-"
            print(
                f"{short:<30} {s['chars']:7d} {s['ms_total']:9.2f} "
                f"{s['ms_per_1k']:9.3f} {s['tokens']:7d} {s['tok_per_s']:9.1f} "
                f"{ratio:>7}"
            )
        verdict = "LINEAR" if r["linear"] else "QUADRATIC"
        print(
            f"{short:<30} scaling {r['scaling_ratio']:.2f}x per 1k across "
            f"{SIZES[0]}->{SIZES[-1]} chars  [{verdict}]\n"
        )

    payload = {"results": results}

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"[BENCH] JSON written to {args.json_out}")

    if args.update:
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_FILE.write_text(json.dumps(payload, indent=2, default=str))
        print(f"[BASELINE] saved to {BASELINE_FILE}")

    if args.ci:
        bad = [r for r in results if not r["linear"]]
        if bad:
            names = ", ".join(r["model"] for r in bad)
            print(f"[FAIL] quadratic encode scaling: {names}", file=sys.stderr)
            return 1
        print("[OK] encode scaling within linear bounds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
