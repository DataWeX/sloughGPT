"""Espeak distillation driver — trains the domain TTS decoder on (text, audio) pairs.

Pair source: domain/voice/_internal/espeak_pairs.py (espeak synthesizes a
waveform per corpus phrase). Training is the ONE loop that already exists —
TTSEngine.train_step / eval_loss — so this is a dataset driver, not a second
trainer (AGENTS: dataset adapter, never duplicate the training loop).

Usage:
    ./scripts/python scripts/train_tts_espeak.py                     # 300 steps, default out
    ./scripts/python scripts/train_tts_espeak.py --steps 500 --lr 5e-4
    ./scripts/python scripts/train_tts_espeak.py --out models/tts/seed7.npz --seed 7

Reports: pair build time, hold-out loss before/after (eval_loss — forward
only, never trains on the hold-out), train-loss curve every ~10%, steps/sec,
checkpoint path. Exit 0 = finite losses and train loss decreased; 1 = diverged.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "packages" / "core-py"))


def build_data(holdout_n: int, max_frames: int):
    from domain.voice._internal.espeak_pairs import (
        DEFAULT_CORPUS,
        DEFAULT_HOLDOUT,
        build_pairs,
        espeak_available,
    )

    if not espeak_available():
        print("ERROR: espeak/espeak-ng binary not found — install espeak", file=sys.stderr)
        raise SystemExit(2)
    holdout_n = max(0, min(holdout_n, len(DEFAULT_CORPUS) - 2))
    holdout_n = holdout_n or DEFAULT_HOLDOUT
    train_texts = DEFAULT_CORPUS[:-holdout_n] if holdout_n else list(DEFAULT_CORPUS)
    holdout_texts = DEFAULT_CORPUS[-holdout_n:] if holdout_n else []
    t0 = time.perf_counter()
    train = build_pairs(train_texts)
    holdout = build_pairs(holdout_texts) if holdout_texts else []
    print(
        f"pairs: {len(train)} train + {len(holdout)} holdout "
        f"(espeak render {time.perf_counter() - t0:.1f}s, max_frames={max_frames})"
    )
    return train, holdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=300, help="optimizer steps (default 300)")
    parser.add_argument("--lr", type=float, default=1e-3, help="SloAdam lr (default 1e-3)")
    parser.add_argument("--max-frames", type=int, default=200, help="mel frame cap per sample")
    parser.add_argument(
        "--holdout", type=int, default=0, help=f"hold-out phrases (default DEFAULT_HOLDOUT)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models/tts/espeak_distilled.npz",
        help="checkpoint output path",
    )
    parser.add_argument("--seed", type=int, default=0, help="numpy RNG seed (default 0)")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    train, holdout = build_data(args.holdout, args.max_frames)
    if not train:
        print("ERROR: empty training set", file=sys.stderr)
        return 1

    from domain.voice._internal.tts import TTSEngine

    tts = TTSEngine(lr=args.lr)
    hold_before = tts.eval_loss(holdout, max_frames=args.max_frames) if holdout else 0.0
    print(f"holdout loss (init): {hold_before:.6f}")

    losses: list[float] = []
    t0 = time.perf_counter()
    report_every = max(1, args.steps // 10)
    for step in range(args.steps):
        text, wav = train[step % len(train)]
        loss = tts.train_step(text, wav, max_frames=args.max_frames)
        losses.append(loss)
        if (step + 1) % report_every == 0 or step + 1 == args.steps:
            recent = float(np.mean(losses[-min(10, len(losses)) :]))
            print(f"step {step + 1:>5}/{args.steps}  last10={recent:.6f}")
    elapsed = time.perf_counter() - t0

    hold_after = tts.eval_loss(holdout, max_frames=args.max_frames) if holdout else 0.0
    first5 = float(np.mean(losses[: min(5, len(losses))]))
    last5 = float(np.mean(losses[-5:]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tts.save(str(out))

    print("\n=== summary ===")
    print(f"steps           : {args.steps}  ({args.steps / max(elapsed, 1e-9):.2f} steps/sec)")
    print(f"train first5    : {first5:.6f}")
    print(f"train last5     : {last5:.6f}")
    print(f"holdout before  : {hold_before:.6f}")
    print(f"holdout after   : {hold_after:.6f}")
    print(f"checkpoint      : {out}")

    finite = bool(np.isfinite(losses).all())
    improved = last5 < first5
    if not finite:
        print("RESULT: FAIL (non-finite loss)")
        return 1
    if not improved:
        print("RESULT: FAIL (train loss did not decrease)")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
