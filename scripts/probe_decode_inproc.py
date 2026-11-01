#!/usr/bin/env python3
"""In-process decode probe — no HTTP, no SSE.

Boots the same provider the server uses, generates N tokens, reports
prefill and steady-state decode tok/s plus the per-token gap series.
Isolates model compute from transport.

Usage: ./scripts/python scripts/probe_decode_inproc.py [--tokens 64]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Match the server's boot: HF_HOME is pinned to the repo cache (main.py sets it
# before any model import) or the probe loads from ~/.cache and finds nothing.
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf-cache"))
for p in (ROOT, ROOT / "packages" / "core-py", ROOT / "apps" / "api" / "server"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _is_async_iter(obj) -> bool:
    return hasattr(obj, "__aiter__")


async def _aiter(obj):
    if _is_async_iter(obj):
        async for x in obj:
            yield x
    else:
        for x in obj:
            yield x


async def _load_provider(timeout_s: int = 180):
    from domain.models import get_provider

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        p = get_provider("default")
        # A "no model" fallback provider is still an object — require a real
        # model behind it or the probe measures the error path.
        if p is not None and getattr(p, "_server", None) is not None:
            return p
        if p is not None and getattr(p, "_get_model", None) is not None:
            try:
                if p._get_model() is not None:
                    return p
            except Exception:  # noqa: BLE001
                pass
        await asyncio.sleep(1)
    return None


async def _ensure_model(model_id: str, timeout_s: int = 300) -> None:
    """Load through the server's own controller so the probe matches e2e."""
    from controllers.models import get_models_controller
    from domain.models import get_provider

    if get_provider("default") is not None:
        return
    ctl = get_models_controller()
    print(f"loading {model_id} via ModelsController ...", flush=True)
    t0 = time.perf_counter()
    await asyncio.to_thread(ctl.load_model, model_id)
    print(f"load_model returned in {time.perf_counter() - t0:.1f}s", flush=True)


async def _amain(args) -> int:
    print("loading model ...", flush=True)
    t_load = time.perf_counter()
    try:
        await _ensure_model(args.model)
    except Exception as e:  # noqa: BLE001
        print(f"ensure_model failed ({e}); falling back to provider wait")
    provider = await _load_provider()
    if provider is None:
        print("FATAL: provider never became available (model not loaded)")
        return 1
    print(f"provider ready in {time.perf_counter() - t_load:.1f}s", flush=True)

    msgs = [{"role": "user", "content": args.prompt}]
    stamps: list[float] = []
    tokens: list[str] = []
    t0 = time.perf_counter()
    try:
        gen = provider.chat_stream(msgs, max_tokens=args.tokens, temperature=0.8)
        async for tok in _aiter(gen):
            if not tok:
                continue
            stamps.append(time.perf_counter())
            tokens.append(tok)
    except Exception as e:  # noqa: BLE001
        print(f"stream aborted: {type(e).__name__}: {e}")
        return 1
    total = time.perf_counter() - t0
    n = len(tokens)
    print()
    print("=" * 60)
    print(f"tokens={n} total_s={total:.2f}")
    if stamps:
        print(f"prefill_ms={(stamps[0] - t0) * 1000:.0f}")
    if n >= 2:
        span = stamps[-1] - stamps[0]
        print(f"decode_tok_s={(n - 1) / span:.2f}  ({span:.2f}s for {n - 1} tok)")
        ms = [round((b - a) * 1000) for a, b in zip(stamps, stamps[1:])]
        s = sorted(ms)
        print(f"gap_ms min={min(ms)} med={s[len(s) // 2]} p90={s[int(0.9 * len(s))]} max={max(ms)}")
        print(f"gaps={ms}")
    print(f"text={''.join(tokens)[:160]!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="Write a short paragraph about online presence.")
    args = ap.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
