"""Decode profile — cProfile attribution of the per-token hot path.

Companion to benchmark_decode_slope.py (card 20261024_111415): the gate
measures 15.3 tok/s against a 25 target and this script says WHERE each
token's time goes. It runs generate_numpy — the exact loop the slope
benchmark times — under cProfile and reports:

- top functions by tottime (inline numpy ops land in generate_numpy itself,
  ctypes/numba calls land in their Python-level callees),
- a stage bucket table (fused quantized matmul, numba kernels, KV
  quant/dequant, LM head, other forward passes),
- profiler-slowed wall time vs the unprofiled baseline for distortion
  context.

No source instrumentation — the hot loop stays untouched.

Usage:
    python scripts/profile_decode.py
    python scripts/profile_decode.py --slnc /path/model.slnc --n 64 --top 30
    python scripts/profile_decode.py --json profile.json
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

# Stage buckets: (bucket, fn_keys, file_keys) — a row matches when every
# non-empty key list has at least one substring hit. First match wins.
BUCKETS: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("kv_quant", ("quantize_kv_tensor", "dequantize_kv_tensor"), ()),
    ("quantized_matmul", ("__call__",), ("quantization.py",)),  # QuantizedLinear
    ("lm_head", ("lm_head_argmax",), ()),
    ("forward_numpy", ("forward_numpy",), ()),
    ("numba_kernels", ("nb_", "numba", "njit"), ()),
    ("generation_loop", ("generate_numpy",), ("slonet.py",)),
]


def _bucket(fn: str, filename: str) -> str:
    for name, fn_keys, file_keys in BUCKETS:
        if fn_keys and not any(k in fn for k in fn_keys):
            continue
        if file_keys and not any(k in filename for k in file_keys):
            continue
        if fn_keys or file_keys:
            return name
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--slnc", default=None, help="explicit model.slnc path")
    ap.add_argument("--n", type=int, default=64, help="tokens to generate")
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--prompt", default="The quick brown fox jumps over")
    ap.add_argument("--top", type=int, default=30, help="functions to print")
    ap.add_argument("--json", default=None, metavar="PATH", help="write full stats JSON")
    args = ap.parse_args()

    if args.slnc:
        slnc = Path(args.slnc)
    else:
        from domain.infrastructure._internal.model_resolver import get_model_dir

        slnc = get_model_dir(args.model) / "model.slnc"
        if not slnc.exists():
            raise SystemExit(f"no model.slnc at {slnc} — pass --slnc explicitly")

    from domain.inference._internal.slonet_provider import SloNetChatProvider

    t0 = time.perf_counter()
    provider = SloNetChatProvider.from_slnc(str(slnc), model_id=args.model, quantize=True)
    load_s = time.perf_counter() - t0
    tokenizer = provider._tokenizer
    import numpy as np

    input_ids = np.array([tokenizer.encode(args.prompt)], dtype=np.int64)

    # Unprofiled baseline first — profiler distortion context.
    provider._model.generate_numpy(input_ids, max_new_tokens=args.warmup, temperature=0.0)
    t0 = time.perf_counter()
    provider._model.generate_numpy(
        input_ids,
        max_new_tokens=args.n,
        temperature=0.0,
        eos_token=tokenizer.eos_token_id or 0,
    )
    base_s = time.perf_counter() - t0

    # Profiled run — same args.
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    provider._model.generate_numpy(
        input_ids,
        max_new_tokens=args.n,
        temperature=0.0,
        eos_token=tokenizer.eos_token_id or 0,
    )
    prof.disable()
    prof_s = time.perf_counter() - t0

    stats = pstats.Stats(prof)
    total_profiled = sum(v[2] for v in stats.stats.values())  # total time in profiled calls

    # Stage buckets: exclusive time (tottime) + inclusive (cumtime) per bucket.
    buckets: dict[str, dict[str, float]] = {}
    rows = []
    for (filename, _line, fn), (cc, nc, tt, ct, _callers) in stats.stats.items():
        b = _bucket(fn, filename)
        d = buckets.setdefault(b, {"tottime": 0.0, "cumtime": 0.0, "calls": 0})
        d["tottime"] += tt
        d["cumtime"] += ct
        d["calls"] += nc
        rows.append((tt, ct, nc, fn, filename))

    print(f"model: {args.model}   load {load_s:.1f}s")
    print(f"tokens: {args.n}   baseline {base_s:.2f}s ({args.n / base_s:.1f} tok/s unprofiled)")
    print(
        f"profiled {prof_s:.2f}s ({args.n / prof_s:.1f} tok/s, {prof_s / base_s:.1f}x distortion)"
    )

    print("\n── stage buckets (exclusive time inside profiled call) ──")
    print(f"  {'bucket':<20} {'tottime':>8} {'share':>7} {'calls':>7}")
    for name, d in sorted(buckets.items(), key=lambda kv: -kv[1]["tottime"]):
        share = d["tottime"] / total_profiled * 100 if total_profiled else 0
        print(f"  {name:<20} {d['tottime']:7.3f}s {share:6.1f}% {d['calls']:7d}")

    print(f"\n── top {args.top} by tottime ──")
    for tt, ct, nc, fn, filename in sorted(rows, reverse=True)[: args.top]:
        short = filename.rsplit("/", 3)[-3:] if "/" in filename else [filename]
        print(f"  {tt:7.3f}s  {ct:7.3f}s  {nc:7d}x  {fn:<34} {'/'.join(short)}")

    if args.json:
        payload = {
            "model": args.model,
            "n_tokens": args.n,
            "baseline_s": round(base_s, 3),
            "profiled_s": round(prof_s, 3),
            "distortion_x": round(prof_s / base_s, 2),
            "loadavg": list(__import__("os").getloadavg()),
            "buckets": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in buckets.items()},
            "top": [
                {
                    "tottime": round(tt, 4),
                    "cumtime": round(ct, 4),
                    "calls": nc,
                    "fn": fn,
                    "file": filename,
                }
                for tt, ct, nc, fn, filename in sorted(rows, reverse=True)[: args.top]
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\njson → {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
