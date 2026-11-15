"""Benchmark one training step across compute devices (card 20261104_084408).

What is measured: a full compute step on the real ``SloughGPTModel`` through the
real hand-written autograd engine — forward, backward, AdamW update, gradient
clip. Not measured: dataset loading, checkpointing, feed refresh, progress
callbacks. Those are device-independent, and including them would hide the
number this script exists to produce.

Per device it reports ms/step, tokens/s, matmul dispatch attribution (gpu / cpu
/ fallbacks), host<->device traffic, and the loss drift against the CPU
reference run. Drift is printed, not asserted — the gate lives in
``tests/core-py/test_train_device_parity.py`` — so a machine that is merely
slow still yields data instead of failing.

Usage::

    ./scripts/python scripts/benchmark_train_device.py
    ./scripts/python scripts/benchmark_train_device.py --devices cpu auto torch
    ./scripts/python scripts/benchmark_train_device.py --steps 10 --json /tmp/devices.json

Exit codes: 0 measured; 2 a requested device is unavailable (declared, not
silently replaced).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from domain.slolib._internal.gpu import (
    accelerator_for,
    dispatch_stats,
    pop_accelerator,
    push_accelerator,
)

_KEEP = ("gpu_matmuls", "cpu_matmuls", "fallbacks", "h2d_bytes", "d2h_bytes")


def _build_model(n_embed: int, n_layer: int, n_head: int, block: int):
    from domain.models import SloughGPTModel

    np.random.seed(0)
    return SloughGPTModel(
        vocab_size=65,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block,
        dropout=0.0,
    )


def _snapshot(model) -> list[np.ndarray]:
    return [np.array(p.data, copy=True) for p in model.parameters()]


def _restore(model, snapshot: list[np.ndarray]) -> None:
    from domain.training._internal.training_handler import zero_grads

    for param, saved in zip(model.parameters(), snapshot, strict=True):
        np.copyto(param.data, saved)
    zero_grads(list(model.parameters()))


def _step(model, optimizer, batches, index) -> float:
    from domain.training._internal.training_handler import clip_gradients, zero_grads

    x, y = batches[index % len(batches)]
    model.train()
    _logits, loss = model(x, y)
    loss.backward()
    params = [p for p in model.parameters() if p.grad is not None]
    clip_gradients(params, 1.0)
    optimizer.step(params)
    zero_grads(list(model.parameters()))
    return float(loss.item())


def _measure(device: str, args, batches, initial) -> dict:
    from domain.training._internal.slonet import SloAdamW

    accelerator = accelerator_for(device)
    model = _build_model(args.n_embed, args.n_layer, args.n_head, args.block)
    _restore(model, initial)
    optimizer = SloAdamW(lr=1e-3, weight_decay=0.0)

    previous = push_accelerator(accelerator)
    try:
        for step in range(args.warmup):
            _step(model, optimizer, batches, step)
        accelerator.sync()
        before = dispatch_stats()
        started = time.perf_counter()
        losses = [_step(model, optimizer, batches, step) for step in range(args.steps)]
        accelerator.sync()
        elapsed = time.perf_counter() - started
        after = dispatch_stats()
    finally:
        pop_accelerator(previous)

    dispatch = {key: after[key] - before[key] for key in _KEEP}
    tokens = args.steps * args.batch * args.block
    return {
        "requested": device,
        "resolved": accelerator.name,
        "kind": accelerator.device_type,
        "ms_per_step": round(elapsed / args.steps * 1000, 2),
        "tokens_per_sec": round(tokens / elapsed, 1),
        "losses": [round(value, 6) for value in losses],
        "dispatch_per_step": {key: round(value / args.steps, 1) for key, value in dispatch.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["cpu", "auto"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--n-embed", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    batches = [
        (
            rng.integers(0, 65, (args.batch, args.block)),
            rng.integers(0, 65, (args.batch, args.block)),
        )
        for _ in range(4)
    ]

    initial = _snapshot(_build_model(args.n_embed, args.n_layer, args.n_head, args.block))

    results: list[dict] = []
    for device in args.devices:
        try:
            results.append(_measure(device, args, batches, initial))
        except (RuntimeError, ValueError) as error:
            print(f"device {device!r} unavailable: {error}")
            return 2

    reference = results[0]["losses"]
    print(
        f"\nmodel: n_embed={args.n_embed} n_layer={args.n_layer} "
        f"batch={args.batch} block={args.block} steps={args.steps}\n"
    )
    print(
        f"{'device':<10}{'resolved':<10}{'kind':<6}{'ms/step':>9}{'tok/s':>10}{'gpu mm':>8}{'cpu mm':>8}{'MB xfer':>9}{'drift':>9}"
    )
    for result in results:
        dispatch = result["dispatch_per_step"]
        drift = max(
            abs(a - b) / max(abs(a), 1e-9) for a, b in zip(reference, result["losses"], strict=True)
        )
        megabytes = (dispatch["h2d_bytes"] + dispatch["d2h_bytes"]) / 1e6
        print(
            f"{result['requested']:<10}{result['resolved']:<10}{result['kind']:<6}"
            f"{result['ms_per_step']:>9}{result['tokens_per_sec']:>10}"
            f"{dispatch['gpu_matmuls']:>8}{dispatch['cpu_matmuls']:>8}"
            f"{megabytes:>9.1f}{drift:>9.1e}"
        )

    verdict = ""
    if len(results) > 1 and results[0]["ms_per_step"] > 0:
        best = min(results, key=lambda item: item["ms_per_step"])
        speedup = results[0]["ms_per_step"] / max(best["ms_per_step"], 1e-9)
        verdict = f"fastest: {best['resolved']} ({speedup:.2f}x vs {results[0]['resolved']})"
        print(f"\n{verdict}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "model": {
                        "n_embed": args.n_embed,
                        "n_layer": args.n_layer,
                        "n_head": args.n_head,
                        "batch": args.batch,
                        "block": args.block,
                        "steps": args.steps,
                    },
                    "devices": results,
                    "verdict": verdict,
                },
                indent=2,
            )
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
