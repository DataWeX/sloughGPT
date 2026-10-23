"""Decode slope benchmark — in-process int8 tokens/sec for the GEMV hot path.

Measures what card 093 (and card 084 before it) use as the headline decode
number: load a cached .slnc model, quantize int8 (symmetric), then generate
N tokens for several N and fit time = prefill + N * decode_sec. The inverse
of the fitted slope is decode tok/s — prefill and load time are absorbed by
the intercept, so the number isolates the per-token path (matmul_int8_f32_c).

Usage:
    python scripts/benchmark_decode_slope.py                      # resolver default model
    python scripts/benchmark_decode_slope.py --slnc /path/model.slnc
    python scripts/benchmark_decode_slope.py --target 25          # exit 1 below target

Gates (card 093): slope >= 25 tok/s.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _resolve_slnc(model: str) -> Path:
    from domain.infrastructure._internal.model_resolver import get_model_dir

    slnc = get_model_dir(model) / "model.slnc"
    if not slnc.exists():
        raise SystemExit(
            f"no model.slnc at {slnc} — pass --slnc explicitly "
            "(model ids resolve via HF_HOME or repo models/hf-cache/)."
        )
    return slnc


def _load_model(slnc: Path, model_id: str):
    from domain.inference._internal.slonet_provider import SloNetChatProvider

    t0 = time.perf_counter()
    provider = SloNetChatProvider.from_slnc(str(slnc), model_id=model_id, quantize=True)
    return provider, time.perf_counter() - t0


def _gen_once(provider, input_ids, n: int) -> float:
    t0 = time.perf_counter()
    provider._model.generate_numpy(
        input_ids,
        max_new_tokens=n,
        temperature=0.0,
        eos_token=provider._tokenizer.eos_token_id or 0,
    )
    return time.perf_counter() - t0


def _slope(points: list[tuple[int, float]]) -> float:
    """Least-squares slope (sec/token) of time over token count."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in points)
    var = sum((x - mx) ** 2 for x in xs)
    return cov / var


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--slnc", default=None, help="explicit model.slnc path")
    ap.add_argument("--sizes", default="32,64,128,256", help="token counts to time")
    ap.add_argument("--prompt", default="The quick brown fox jumps over", help="fixed prompt")
    ap.add_argument("--warmup", type=int, default=16, help="warmup tokens before timing")
    ap.add_argument("--target", type=float, default=None, help="gate: min decode tok/s")
    ap.add_argument("--json", action="store_true", help="machine-readable summary line")
    args = ap.parse_args()

    slnc = Path(args.slnc) if args.slnc else _resolve_slnc(args.model)
    provider, load_s = _load_model(slnc, args.model)
    tokenizer = provider._tokenizer

    from domain.infrastructure._internal.quant_core import wrapper as qw

    flags = []
    for name in ("HAS_AVX2", "HAS_AVX512", "HAS_CBLAS"):
        if hasattr(qw, name):
            flags.append(f"{name}={getattr(qw, name)}")

    input_ids = tokenizer.encode(args.prompt)
    import numpy as np

    input_ids = np.array([input_ids], dtype=np.int64)

    # Warmup: first calls calibrate per-shape kernel dispatch + touch pages.
    _gen_once(provider, input_ids, args.warmup)

    points: list[tuple[int, float]] = []
    rows = []
    for n in [int(s) for s in args.sizes.split(",")]:
        secs = min(_gen_once(provider, input_ids, n) for _ in range(2))
        points.append((n, secs))
        rows.append((n, secs, n / secs))

    slope = _slope(points)
    tok_s = 1.0 / slope if slope > 0 else 0.0

    if args.json:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "load_s": round(load_s, 2),
                    "decode_tok_s": round(tok_s, 2),
                    "points": [
                        {"n": n, "s": round(s, 3), "naive_tok_s": round(v, 2)} for n, s, v in rows
                    ],
                    "flags": flags,
                    "target": args.target,
                }
            )
        )
    else:
        load1, load5, load15 = os.getloadavg()
        print(f"model: {args.model}  ({flags})")
        print(f"loadavg: {load1:.1f} {load5:.1f} {load15:.1f}  ({os.cpu_count()} cpus)")
        print(f"load:  {load_s:.1f}s   int8 symmetric")
        for n, secs, naive in rows:
            print(f"  gen {n:4d} tok: {secs:7.2f}s   {naive:6.2f} tok/s (naive)")
        print(f"decode slope: {tok_s:.2f} tok/s  ({1000 / tok_s:.1f} ms/token)")

    if args.target is not None and tok_s < args.target:
        print(f"FAIL: decode slope {tok_s:.2f} < target {args.target}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
