#!/usr/bin/env python
"""Distill run vs held-out slice — the corpus-spec card's step 4 (card 20261126_105302).

Trains a small SloughGPT student to imitate a Shakespeare-trained SloTransformer
teacher on the ``distill_corpus`` TRAIN split, then reports CE / perplexity on
the held-out slice for both student and teacher — the number the corpus-spec
plan line ("distill run vs held-out slice = a number") promised.

Entry is in-process (same process → function call), mirroring
``apps/api/server/training/distill.py``'s data path exactly: char encode in the
teacher's vocab → ``block_size`` windows with ``block_size // 2`` stride →
``DistillationTrainer`` (gradient-fixed ``step``) → NaN-guarded batches →
per-epoch average loss. Seeded 42 like the production job.

No downloads: the teacher is an on-disk checkpoint of ours
(``checkpoints/tinyshakespeare_1794493048.soul`` by default).

Usage:
    ./scripts/python scripts/distill_corpus_eval.py [--epochs 2] [--teacher PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from domain.models import SloughGPTModel
from domain.training import DistillationConfig, DistillationTrainer, import_from_sou
from domain.training._internal.distill_corpus_adapter import _render
from domain.training._internal.slonet import Tensor, cross_entropy, no_grad

DEFAULT_TEACHER = "checkpoints/tinyshakespeare_1794493048.soul"
DEFAULT_DATA = "data/distill_corpus"


def load_teacher(path: Path) -> tuple[object, dict[str, int]]:
    """Load the teacher model + its char stoi (from the checkpoint meta)."""
    model = import_from_sou(str(path))
    meta_file = Path(str(path) + ".meta.json")
    if not meta_file.exists():
        raise SystemExit(f"checkpoint meta not found: {meta_file}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    stoi = (meta.get("metadata") or {}).get("stoi")
    if not stoi:
        raise SystemExit(f"no metadata.stoi in {meta_file} — cannot encode in teacher vocab")
    return model, {str(k): int(v) for k, v in stoi.items()}


def encode(text: str, stoi: dict[str, int]) -> tuple[np.ndarray, int]:
    """Char-encode in the teacher's vocab; drop chars outside it (reported)."""
    ids = [stoi[c] for c in text if c in stoi]
    return np.asarray(ids, dtype=np.int64), len(text) - len(ids)


def make_blocks(ids: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """distill.py's exact windowing: stride block_size // 2, next-char targets."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in range(0, len(ids) - block_size, block_size // 2):
        x = ids[i : i + block_size]
        y = ids[i + 1 : i + block_size + 1]
        if len(x) == block_size and len(y) == block_size:
            xs.append(x)
            ys.append(y)
    if not xs:
        return np.empty((0, block_size), dtype=np.int64), np.empty((0, block_size), dtype=np.int64)
    return np.stack(xs), np.stack(ys)


def eval_ce(model: object, xs: np.ndarray, ys: np.ndarray, batch: int = 32) -> float:
    """Mean next-char CE (nats) of ``model`` over blocks, in no_grad."""
    if len(xs) == 0:
        return float("nan")
    total = 0.0
    n = 0
    with no_grad():
        for start in range(0, len(xs), batch):
            bx = xs[start : start + batch]
            by = ys[start : start + batch]
            out = model(bx)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(logits, Tensor):
                logits = Tensor(np.asarray(logits, dtype=np.float32))
            loss = cross_entropy(logits, Tensor(by))
            total += float(np.asarray(loss.data)) * len(bx)
            n += len(bx)
    return total / max(n, 1)


def train(
    trainer: DistillationTrainer,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    epochs: int,
    batch: int,
    seed: int,
) -> list[float]:
    """distill.py's loop: seeded shuffle, NaN-guarded batches, epoch averages."""
    rng = random.Random(seed)
    n = len(xs)
    epoch_avgs: list[float] = []
    for epoch in range(epochs):
        order = list(range(n))
        rng.shuffle(order)
        loss_sum = 0.0
        steps = 0
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            losses = trainer.step(xs[idx], ys[idx])
            loss = float(losses.get("total_loss", 0.0))
            if not math.isfinite(loss):
                continue  # NaN guard — same as distill.py
            loss_sum += loss
            steps += 1
        avg = loss_sum / max(steps, 1)
        epoch_avgs.append(avg)
        print(f"  epoch {epoch + 1}/{epochs}: avg_total_loss={avg:.4f} ({steps} steps)", flush=True)
    return epoch_avgs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--teacher", default=DEFAULT_TEACHER, help="path to teacher .soul checkpoint")
    ap.add_argument("--data", default=DEFAULT_DATA, help="dataset dir (input.txt + heldout.jsonl)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--embed", type=int, default=64, help="student n_embed (teacher is 128)")
    ap.add_argument("--layers", type=int, default=2, help="student n_layer")
    ap.add_argument("--heads", type=int, default=4, help="student n_head")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    args = ap.parse_args()

    t0 = time.time()
    np.random.seed(args.seed)  # distill.py seeds both RNGs at job start

    teacher, stoi = load_teacher(Path(args.teacher))
    vocab_size = len(stoi)
    print(f"teacher: {args.teacher} (vocab {vocab_size})", flush=True)

    data_dir = Path(args.data)
    train_text = (data_dir / "input.txt").read_text(encoding="utf-8")
    held_records = [
        json.loads(line)
        for line in (data_dir / "heldout.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    held_text = "\n".join(_render(r.get("messages", [])) for r in held_records)

    train_ids, train_dropped = encode(train_text, stoi)
    held_ids, held_dropped = encode(held_text, stoi)
    train_x, train_y = make_blocks(train_ids, args.block)
    held_x, held_y = make_blocks(held_ids, args.block)
    print(
        f"train: {len(train_text):,} chars -> {len(train_ids):,} ids "
        f"({train_dropped:,} outside vocab), {len(train_x):,} blocks",
        flush=True,
    )
    print(
        f"heldout: {len(held_text):,} chars -> {len(held_ids):,} ids "
        f"({held_dropped:,} outside vocab), {len(held_x):,} blocks "
        f"({len(held_records)} records)",
        flush=True,
    )
    if len(train_x) == 0 or len(held_x) == 0:
        raise SystemExit("no blocks — not enough data")

    student = SloughGPTModel(
        vocab_size=vocab_size,
        n_embed=args.embed,
        n_layer=args.layers,
        n_head=args.heads,
        block_size=args.block,
    )
    trainer = DistillationTrainer(
        teacher,
        student,
        DistillationConfig(
            temperature=args.temperature,
            alpha=args.alpha,
            beta=args.beta,
        ),
    )

    print(
        f"training: student={args.embed}d/{args.layers}L/{args.heads}H, "
        f"epochs={args.epochs}, batch={args.batch} (trainer's own lr, as production)",
        flush=True,
    )
    epoch_avgs = train(
        trainer, train_x, train_y, epochs=args.epochs, batch=args.batch, seed=args.seed
    )

    print("evaluating...", flush=True)
    s_train = eval_ce(student, train_x, train_y)
    s_held = eval_ce(student, held_x, held_y)
    t_held = eval_ce(teacher, held_x, held_y)

    print()
    print("== RESULT ==")
    print(f"  train loss curve      : {' -> '.join(f'{v:.4f}' for v in epoch_avgs)}")
    print(f"  student CE train      : {s_train:.4f} nats  (ppl {math.exp(s_train):.2f})")
    print(f"  student CE heldout    : {s_held:.4f} nats  (ppl {math.exp(s_held):.2f})")
    print(f"  teacher CE heldout    : {t_held:.4f} nats  (ppl {math.exp(t_held):.2f})")
    print(
        f"  compression gap       : teacher ppl {math.exp(t_held):.2f} vs student {math.exp(s_held):.2f}"
    )
    print(f"  wall                  : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
