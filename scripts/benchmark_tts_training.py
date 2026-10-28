"""TTS training benchmark — does real-gradient espeak distillation actually learn?

Replaces the era of `_manual_backward` (random noise, no .grad) with a measured
gate on `SpectrogramDecoder.train_step`. Measures, on a seeded fixed corpus:

  - train loss drop: mean(first 5 steps) vs mean(last 5 steps)
  - held-out loss: `eval_loss` (forward-only) before vs after — never trains
    on what it measures (espeak_pairs.DEFAULT_CORPUS tail split)
  - weight movement: fraction of parameters whose data changed since init
  - throughput: optimizer steps/sec
  - optional A/B audio: untrained vs trained synthesis (--samples-dir)

What it measures: the LEARNING SIGNAL of the training path (gradients real,
loss descends, hold-out improves). It does NOT measure perceived audio
quality — there is no offline perceptual/ASR metric in the tree yet.

Gold thresholds:
  - train reduction: final5 / first5 <= 0.95
  - held-out improves: final < initial (strict, forward-only eval)
  - finite losses only (no NaN/inf)
  - weight movement: >= 50% of parameters changed
  - steps/sec reported, not gated (watch history for regressions)

Usage:
    ./scripts/python scripts/benchmark_tts_training.py
    ./scripts/python scripts/benchmark_tts_training.py --steps 300 --seed 7
    ./scripts/python scripts/benchmark_tts_training.py --samples-dir /tmp/tts_ab
Record:
    ./scripts/python scripts/benchmark_results.py record --kind training --json-file <this --json>

Exit codes: 0 PASS, 1 FAIL, 2 espeak missing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "packages" / "core-py"))

GOLD = {
    "max_train_reduction": 0.95,  # last5 / first5 of train loss
    "min_weight_move": 0.90,  # fraction of parameters changed (all must be live)
    "min_steps_per_sec": 0.0,  # informational only
}

AB_PHRASE = "The quick brown fox jumps over the lazy dog."


def _write_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(wav, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=200, help="optimizer steps (default 200)")
    parser.add_argument("--lr", type=float, default=1e-3, help="SloAdam lr (default 1e-3)")
    parser.add_argument("--max-frames", type=int, default=200, help="mel frame cap per sample")
    parser.add_argument("--holdout", type=int, default=0, help="hold-out phrases (default DEFAULT_HOLDOUT)")
    parser.add_argument("--seed", type=int, default=0, help="numpy RNG seed (default 0)")
    parser.add_argument(
        "--json",
        default="/tmp/tts-training-results.json",
        help="metrics JSON output (feed to benchmark_results record --kind training)",
    )
    parser.add_argument(
        "--samples-dir",
        default="",
        help="if set, write untrained_vs_trained .wav A/B pair here",
    )
    args = parser.parse_args(argv)

    from domain.voice._internal.espeak_pairs import (
        DEFAULT_CORPUS,
        DEFAULT_HOLDOUT,
        build_pairs,
        espeak_available,
    )

    if not espeak_available():
        print("[ERR] espeak/espeak-ng not installed", file=sys.stderr)
        return 2

    np.random.seed(args.seed)
    holdout_n = max(0, min(args.holdout or DEFAULT_HOLDOUT, len(DEFAULT_CORPUS) - 2))
    train_texts, holdout_texts = DEFAULT_CORPUS[:-holdout_n], DEFAULT_CORPUS[-holdout_n:]
    t_render = time.perf_counter()
    train = build_pairs(train_texts)
    holdout = build_pairs(holdout_texts)
    print(
        f"data: {len(train)} train + {len(holdout)} holdout "
        f"(render {time.perf_counter() - t_render:.1f}s, seed={args.seed})"
    )

    from domain.voice._internal.tts import TTSEngine

    np.random.seed(args.seed)
    tts = TTSEngine(lr=args.lr)
    init_snapshot = [p.data.copy() for p in tts.parameters()]
    hold_init = tts.eval_loss(holdout, max_frames=args.max_frames)

    train_losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        text, wav = train[step % len(train)]
        train_losses.append(tts.train_step(text, wav, max_frames=args.max_frames))
    elapsed = time.perf_counter() - t0
    hold_final = tts.eval_loss(holdout, max_frames=args.max_frames)

    first5 = float(np.mean(train_losses[:5]))
    last5 = float(np.mean(train_losses[-5:]))
    reduction = last5 / first5 if first5 > 0 else float("inf")
    moved = sum(
        not np.allclose(a, p.data) for a, p in zip(init_snapshot, tts.parameters(), strict=True)
    )
    weight_move = moved / max(len(init_snapshot), 1)
    sps = args.steps / max(elapsed, 1e-9)
    finite = bool(np.isfinite(train_losses).all())

    checks = {
        "finite_losses": finite,
        "train_reduction": reduction <= GOLD["max_train_reduction"],
        "holdout_improved": hold_final < hold_init,
        "weight_movement": weight_move >= GOLD["min_weight_move"],
    }
    passed = all(checks.values())

    if args.samples_dir:
        sdir = Path(args.samples_dir)
        sdir.mkdir(parents=True, exist_ok=True)
        fresh = TTSEngine()
        _write_wav(sdir / "untrained.wav", fresh.text_to_waveform(AB_PHRASE), fresh.sample_rate)
        _write_wav(sdir / "trained.wav", tts.text_to_waveform(AB_PHRASE), tts.sample_rate)
        print(f"samples: {sdir / 'untrained.wav'} vs {sdir / 'trained.wav'}")

    results = {
        "kind": "tts_training",
        "passed": passed,
        "checks": checks,
        "initial_loss": round(first5, 6),
        "final_loss": round(last5, 6),
        "reduction_ratio": round(reduction, 6),
        "holdout_initial": round(hold_init, 6),
        "holdout_final": round(hold_final, 6),
        "weight_move_frac": round(weight_move, 4),
        "steps": args.steps,
        "steps_per_sec": round(sps, 3),
        "elapsed_s": round(elapsed, 2),
        "seed": args.seed,
        "lr": args.lr,
        "gold": GOLD,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(results, indent=2))

    print("\n=== tts training benchmark ===")
    for key in (
        "initial_loss",
        "final_loss",
        "reduction_ratio",
        "holdout_initial",
        "holdout_final",
        "weight_move_frac",
        "steps_per_sec",
    ):
        print(f"{key:<18}: {results[key]}")
    for name, ok in checks.items():
        print(f"[{'OK' if ok else 'FAIL'}] {name}")
    print(f"json: {args.json}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
