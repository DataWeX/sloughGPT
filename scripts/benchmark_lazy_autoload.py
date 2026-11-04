"""
Lazy-guard autoload benchmark (card 20261101_144013, scope B).

Measures the two halves of the boot model-load path on the real ``.slnc``:

* **header-only construction** — ``SloNetChatProvider.lazy_from_slnc``: config +
  tensor table only, no weight pages faulted in. This is what readiness now
  waits for.
* **materialisation** — ``materialize_model()``: the full 2.5 GB read, moved off
  the readiness path onto a background thread 15 s later.

Each case runs in its **own child process** because peak RSS (``VmHWM``) is a
per-process high-water mark and the second load in a process inherits the first
one's page cache — comparing them in one process would measure the cache, not
the code.

The gate is the ratio that must hold regardless of machine: header-only
construction has to be far cheaper than an eager load (default 5×), and
materialisation has to land within 1.5× of the eager load that does the same
work (it is the same call). Peak RSS is reported, not gated: both paths peak at
the same 7.4 GB on this artifact, which is the point — B buys time, not memory.

Usage:
    ./scripts/python scripts/benchmark_lazy_autoload.py
    ./scripts/python scripts/benchmark_lazy_autoload.py --json out.json
    ./scripts/python scripts/benchmark_lazy_autoload.py --model Qwen/Qwen2.5-0.5B-Instruct
Exit code is non-zero when a gate fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (
    str(_REPO),
    str(_REPO / "packages" / "downcraft"),
    str(_REPO / "packages" / "core-py"),
    str(_REPO / "packages" / "mogdb" / "src"),
    str(_REPO / "apps" / "api" / "server"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _peak_rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _provider_kwargs(model_id: str) -> dict:
    """Production flags, identical to ``startup._try_lazy_guard_autoload``."""
    return {
        "model_id": model_id,
        "quantize": True,
        "quant_bits": 8,
        "quant_mode": "symmetric",
        "free_quantized_originals": True,
    }


def _case_eager(model_id: str) -> dict:
    from domain.inference._internal.slonet_provider import SloNetChatProvider
    from domain.infrastructure.model_resolver import get_model_dir

    slnc = str(get_model_dir(model_id) / "model.slnc")
    started = time.perf_counter()
    provider = SloNetChatProvider.from_slnc(slnc, **_provider_kwargs(model_id))
    elapsed = time.perf_counter() - started
    assert provider._model is not None, "eager load produced no model"
    return {
        "case": "eager",
        "slnc_mb": round(os.path.getsize(slnc) / 1e6, 1),
        "load_s": round(elapsed, 3),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "resident_rss_mb": round(_rss_mb(), 1),
    }


def _case_lazy(model_id: str) -> dict:
    from domain.inference._internal.slonet_provider import SloNetChatProvider
    from domain.infrastructure.model_resolver import get_model_dir

    slnc = str(get_model_dir(model_id) / "model.slnc")

    started = time.perf_counter()
    provider = SloNetChatProvider.lazy_from_slnc(slnc, **_provider_kwargs(model_id))
    construct_s = time.perf_counter() - started
    if provider._model is not None:
        raise AssertionError("lazy_from_slnc materialised weights — it must be header-only")
    construct_rss = _rss_mb()

    started = time.perf_counter()
    provider.materialize_model()
    materialise_s = time.perf_counter() - started
    assert provider._model is not None, "materialize_model produced no model"

    return {
        "case": "lazy",
        "slnc_mb": round(os.path.getsize(slnc) / 1e6, 1),
        "construct_s": round(construct_s, 3),
        "construct_rss_mb": round(construct_rss, 1),
        "materialise_s": round(materialise_s, 3),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "resident_rss_mb": round(_rss_mb(), 1),
    }


def _spawn(case: str, model_id: str, timeout_s: float) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--case", case, "--model", model_id]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{case} case failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
    payload = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if not payload:
        raise RuntimeError(f"{case} case printed no result:\n{proc.stdout[-2000:]}")
    return json.loads(payload[-1])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--case", choices=["eager", "lazy"], default="", help=argparse.SUPPRESS)
    ap.add_argument(
        "--min-ratio", type=float, default=5.0, help="header vs eager gate (default 5×)"
    )
    ap.add_argument("--max-materialise-ratio", type=float, default=1.5)
    ap.add_argument("--timeout", type=float, default=600.0, help="per-case timeout in seconds")
    ap.add_argument("--json", default="", help="write results to this path")
    args = ap.parse_args()

    if args.case == "eager":
        print(json.dumps(_case_eager(args.model)))
        return 0
    if args.case == "lazy":
        print(json.dumps(_case_lazy(args.model)))
        return 0

    try:
        eager = _spawn("eager", args.model, args.timeout)
        lazy = _spawn("lazy", args.model, args.timeout)
    except Exception as e:  # pragma: no cover - environment failure
        print(f"FAIL: {e}", file=sys.stderr)
        return 2

    ratio = eager["load_s"] / lazy["construct_s"] if lazy["construct_s"] > 0 else 0.0
    print(f"model: {args.model}   .slnc: {eager['slnc_mb']:.0f} MB")
    print()
    print(f"{'phase':<26} {'seconds':>8} {'peak RSS':>10}")
    print(f"{'eager from_slnc':<26} {eager['load_s']:>8.2f} {eager['peak_rss_mb']:>9.0f}M")
    print(
        f"{'lazy construct (header)':<26} {lazy['construct_s']:>8.2f} {lazy['construct_rss_mb']:>9.0f}M"
    )
    print(
        f"{'lazy materialise (deferred)':<26} {lazy['materialise_s']:>8.2f} {lazy['peak_rss_mb']:>9.0f}M"
    )
    print()
    print(
        f"ready-before-weights: {lazy['construct_s']:.2f}s vs {eager['load_s']:.2f}s eager "
        f"({ratio:.1f}× cheaper, gate >= {args.min_ratio:.1f}×)"
    )
    print(
        f"peak RSS is shared: eager {eager['peak_rss_mb']:.0f} MB vs lazy {lazy['peak_rss_mb']:.0f} MB "
        "(B buys time, not memory)"
    )

    failures: list[str] = []
    if ratio < args.min_ratio:
        failures.append(f"header-only {ratio:.2f}× cheaper than eager, gate {args.min_ratio:.2f}×")
    if lazy["materialise_s"] > eager["load_s"] * args.max_materialise_ratio:
        failures.append(
            f"materialise {lazy['materialise_s']:.2f}s vs eager {eager['load_s']:.2f}s "
            f"(> {args.max_materialise_ratio:.1f}× — same call, so this is overhead)"
        )

    print()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
    else:
        print("PASS: lazy construction is header-only and materialisation is same-cost")

    payload = {
        "model": args.model,
        "ratio": round(ratio, 2),
        "eager": eager,
        "lazy": lazy,
        "failures": failures,
        "passed": not failures,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
