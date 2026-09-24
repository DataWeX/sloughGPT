#!/usr/bin/env python3
"""Benchmark the owned path (goal 12): soul load vs soul→slnc→load.

Measures:
- train tiny soul (optional, --skip-train reuses fixture soul)
- soul_to_slnc bridge time
- from_soul load time + first-token latency
- from_slnc load time + first-token latency
- weight-identity pass/fail (bridge fidelity)

Usage:
    python scripts/benchmark_owned_path.py --json /tmp/owned_path.json
    python scripts/benchmark_results.py record --kind training --json-file /tmp/owned_path.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "packages" / "core-py"))


def _train_tiny(tmp: Path) -> Path:
    from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig

    corpus = "".join(chr(32 + (i % 95)) for i in range(400)) * 4
    data = tmp / "corpus.txt"
    data.write_text(corpus, encoding="utf-8")

    cfg = TrainerConfig(
        vocab_size=0,
        n_embed=32,
        n_layer=1,
        n_head=4,
        block_size=32,
        batch_size=4,
        epochs=1,
        max_steps=4,
        learning_rate=1e-3,
        warmup_steps=1,
        checkpoint_dir=str(tmp / "ck"),
        checkpoint_interval=10_000,
        log_interval=100,
        eval_interval=10_000,
        min_data_quality=0.0,
        max_toxicity_rate=1.0,
    )
    trainer = SloughGPTTrainer(data_path=str(data), config=cfg, soul_name="bench-owned")
    trainer.train()
    out = tmp / "owned"
    trainer.save(str(out), include_optimizer_state=False, is_final=True)
    return Path(str(out) + ".soul")


def _first_token_ms(provider) -> float:
    t0 = time.perf_counter()
    try:
        if hasattr(provider, "generate"):
            provider.generate("Hi", max_new_tokens=1, temperature=0.0)
        elif hasattr(provider, "_model") and provider._model is not None:
            # Fallback: raw forward via encode
            tok = getattr(provider, "_tokenizer", None)
            ids = tok.encode("Hi") if tok else [1, 2]
            import numpy as np

            provider._model.forward_numpy(np.asarray([ids], dtype=np.int64))
    except Exception:
        pass
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None, help="write metrics JSON for benchmark_results.py")
    ap.add_argument("--soul", type=Path, default=None, help="reuse an existing .soul")
    args = ap.parse_args()

    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="owned_path_"))
    metrics: dict = {"config": {"n_embed": 32, "n_layer": 1, "n_head": 4, "block_size": 32}}

    # 1) Obtain a soul
    if args.soul and args.soul.exists():
        soul_path = args.soul
        metrics["train_s"] = 0.0
        metrics["trained"] = False
    else:
        t0 = time.perf_counter()
        soul_path = _train_tiny(tmp)
        metrics["train_s"] = round(time.perf_counter() - t0, 3)
        metrics["trained"] = True

    # 2) Bridge
    from domain.inference._internal.slo_format import load_soul
    from domain.infrastructure._internal.soul_to_slnc import soul_to_slnc

    slnc_path = tmp / "owned.slnc"
    t0 = time.perf_counter()
    soul_to_slnc(soul_path, slnc_path)
    metrics["bridge_s"] = round(time.perf_counter() - t0, 4)
    metrics["slnc_bytes"] = slnc_path.stat().st_size

    # 3) Weight identity
    _, sd = load_soul(str(soul_path))
    from domain.inference._internal.slonet_provider import SloNetChatProvider

    t0 = time.perf_counter()
    p_soul = SloNetChatProvider.from_soul(str(soul_path), model_id="bench-soul")
    metrics["load_soul_s"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    p_slnc = SloNetChatProvider.from_slnc(str(slnc_path), model_id="bench-slnc")
    metrics["load_slnc_s"] = round(time.perf_counter() - t0, 4)

    loaded = dict(p_slnc._model._named_parameters())
    mismatches = []
    for k, arr in sd.items():
        if k not in loaded:
            mismatches.append(f"missing:{k}")
        else:
            got = loaded[k].data
            if got.shape != arr.shape or not (got == arr).all():
                mismatches.append(f"diff:{k}")
    metrics["weight_identity_pass"] = len(mismatches) == 0
    metrics["weight_mismatches"] = mismatches[:10]
    metrics["n_tensors"] = len(sd)

    # 4) First-token latency (best-effort)
    metrics["first_token_soul_ms"] = round(_first_token_ms(p_soul), 3)
    metrics["first_token_slnc_ms"] = round(_first_token_ms(p_slnc), 3)

    # 5) Peak memory during bridge+loads (approx)
    tracemalloc.start()
    soul_to_slnc(soul_path, tmp / "owned2.slnc")
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics["bridge_peak_mb"] = round(peak / (1024 * 1024), 2)

    passed = metrics["weight_identity_pass"]
    metrics["passed"] = passed

    print(json.dumps(metrics, indent=2, default=str))
    if args.json:
        args.json.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {args.json}", file=sys.stderr)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
