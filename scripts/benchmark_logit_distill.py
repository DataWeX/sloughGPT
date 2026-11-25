"""
Quick benchmark: the logit-distill seam (card 20261125_143810).

Answers four questions about the teacher-student logit-distillation design,
numpy only, zero downloads:

  1. parallel teacher-forced forward  vs  autoregressive single-token loop
     (the "text needs sampling, logits don't" claim — measured, not argued)
  2. KL(teacher || student) between two DISTINCT SloTransformer weight sets:
     exact copy (expect ~0 — the zero-delta trap) vs 1%-perturbed
     init-from-teacher (expect > 0 — usable training signal)
  3. logits memory at batch x seq x vocab (what actually occupies RAM/VRAM)
  4. one real training step: does DistillationTrainer.step() move the
     student's weights, and how long does the in-loop cost run?

Usage:
    ./scripts/python scripts/benchmark_logit_distill.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "packages/core-py")

import os
import time

# Before numpy: OpenBLAS reads these once at library load (same as benchmark_slonet).
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("SLO_OPENBLAS_NUM_THREADS", "1"))
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLO_OMP_NUM_THREADS", "1"))

import numpy as np

from domain.training._internal.distillation import (
    DistillationConfig,
    DistillationLoss,
    DistillationTrainer,
)
from domain.training._internal.slonet import (
    SloAdam,
    SloTransformer,
    Tensor,
    kl_div_loss,
    log_softmax,
    no_grad,
    softmax,
)

VOCAB = 256
N_EMBED = 128
N_LAYER = 4
N_HEAD = 4
BLOCK = 128
TEMP = 4.0


def fmt_time(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:.1f} µs"
    if s < 1.0:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.2f} s"


def build_model() -> SloTransformer:
    return SloTransformer(
        vocab_size=VOCAB,
        n_embed=N_EMBED,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        block_size=BLOCK,
        dropout=0.0,
        tie_weights=False,
    )


def copy_with_noise(dst: SloTransformer, src: SloTransformer, scale: float, rng) -> None:
    """dst := src (+ noise). scale=0 is an exact copy (the zero-delta control)."""
    for p_d, p_s in zip(dst.parameters(), src.parameters()):
        noise = rng.normal(0.0, scale, p_s.data.shape).astype(p_s.data.dtype)
        p_d.data[...] = p_s.data + noise


def weight_delta(before: list[np.ndarray], model: SloTransformer) -> float:
    deltas = [float(np.mean(np.abs(p.data - w))) for p, w in zip(model.parameters(), before)]
    return max(deltas) if deltas else 0.0


def snapshot(model: SloTransformer) -> list[np.ndarray]:
    return [p.data.copy() for p in model.parameters()]


def forward_parallel(model: SloTransformer, ids: np.ndarray) -> np.ndarray:
    model.clear_kv_cache()
    out, _ = model.forward(ids, use_cache=False)
    return out.data if isinstance(out, Tensor) else np.asarray(out)


def forward_ar_incremental(model: SloTransformer, ids: np.ndarray) -> list[np.ndarray]:
    """Autoregressive with KV cache: one token per step, use_cache + start_pos."""
    model.clear_kv_cache()
    T = ids.shape[1]
    steps = []
    for i in range(T):
        out, _ = model.forward(ids[:, i : i + 1], use_cache=True, start_pos=i)
        steps.append(out.data if isinstance(out, Tensor) else np.asarray(out))
    return steps


def forward_ar_prefix(model: SloTransformer, ids: np.ndarray) -> list[np.ndarray]:
    """Autoregressive without cache: re-forward the growing prefix each step."""
    T = ids.shape[1]
    steps = []
    for i in range(T):
        out, _ = model.forward(ids[:, : i + 1], use_cache=False)
        steps.append(out.data if isinstance(out, Tensor) else np.asarray(out))
    return steps


def timed(fn, reps: int, warmup: int = 1):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    return (time.perf_counter() - t0) / reps, out


def section(title: str) -> None:
    print()
    print(f"[{title}]")
    print("-" * 56)


def main() -> None:
    rng = np.random.default_rng(1234)
    ids = np.random.default_rng(99).integers(0, VOCAB, size=(1, BLOCK)).astype(np.int64)
    labels = np.random.default_rng(7).integers(0, VOCAB, size=(1, BLOCK))

    teacher = build_model()
    student_exact = build_model()
    student_pert = build_model()
    copy_with_noise(student_exact, teacher, 0.0, rng)
    copy_with_noise(student_pert, teacher, 0.01, rng)

    print("=" * 56)
    print("Logit-Distill Seam Benchmark")
    print(f"model: {N_LAYER}L/{N_EMBED}d/{N_HEAD}h vocab={VOCAB} seq={BLOCK}")
    print("=" * 56)

    # ---------------------------------------------------------------- 1
    section("1. parallel teacher-forced vs autoregressive")
    par_t, par_logits = timed(lambda: forward_parallel(teacher, ids), reps=5, warmup=2)
    print(f"  parallel (1 x {BLOCK} positions, no cache): {fmt_time(par_t)}")

    ar_mode = "incremental (kv cache)"
    try:
        steps = forward_ar_incremental(teacher, ids)
        # correctness spot-check: cached last-token logits vs parallel last position
        close = np.allclose(steps[-1][0, 0], par_logits[0, -1], atol=1e-4)
        print(f"  cache path matches full forward: {'yes' if close else 'NO — timing only'}")
    except Exception as e:  # noqa: BLE001 — fallback is the point
        ar_mode = f"prefix re-forward (cache failed: {type(e).__name__})"
        steps = forward_ar_prefix(teacher, ids)
    ar_t, _ = timed(
        lambda: (forward_ar_incremental if "incremental" in ar_mode else forward_ar_prefix)(
            teacher, ids
        ),
        reps=3,
        warmup=1,
    )
    print(f"  autoregressive ({BLOCK} single-token steps, {ar_mode}): {fmt_time(ar_t)}")
    print(f"  -> parallel is {ar_t / par_t:.1f}x faster than autoregressive")

    # ---------------------------------------------------------------- 2
    section("2. KL(teacher || student): exact copy vs perturbed init")
    with no_grad():
        t_logits = forward_parallel(teacher, ids)
    loss_fn = DistillationLoss(DistillationConfig(temperature=TEMP, alpha=0.5, beta=0.5))
    for name, student in (("exact copy", student_exact), ("1% perturbed", student_pert)):
        s_logits = forward_parallel(student, ids)
        total, losses = loss_fn(s_logits, t_logits, labels=labels)
        print(
            f"  {name:14s} soft(KL*T^2)={losses.get('soft_loss', 0.0):.6f}  "
            f"hard(CE)={losses.get('hard_loss', 0.0):.4f}  total={total:.6f}"
        )

    # ---------------------------------------------------------------- 3
    section("3. logits memory (batch x seq x vocab, fp32)")
    for b, t, v in [(1, BLOCK, VOCAB), (1, 512, 50257), (4, 512, 50257), (8, 1024, 50257)]:
        mb = b * t * v * 4 / 1e6
        note = "  (teacher + student = x2 in a distill step)" if v > 1000 else ""
        print(f"  B={b:<2d} T={t:<5d} V={v:<6d} {mb:9.1f} MB{note}")

    # ---------------------------------------------------------------- 4a
    section("4a. DistillationTrainer.step() — does the student actually move?")
    cfg = DistillationConfig(temperature=TEMP, alpha=0.5, beta=0.5)
    try:
        trainer = DistillationTrainer(teacher, student_pert, cfg)
        before = snapshot(student_pert)
        t0 = time.perf_counter()
        stats = trainer.step(ids, labels)
        step_t = time.perf_counter() - t0
        delta = weight_delta(before, student_pert)
        print(f"  wall time: {fmt_time(step_t)}   losses: {stats}")
        print(f"  max mean |delta w| after 1 step: {delta:.3e}")
        if delta == 0.0:
            print("  -> NO-OP: weights unchanged — gradient path severed (numpy detach)")
        else:
            print("  -> student moved: gradient path works")
    except Exception as e:  # noqa: BLE001 — report, don't crash
        print(f"  step raised {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- 4b
    section("4b. manual graph-preserving step (candidate fix path)")
    student_fix = build_model()
    copy_with_noise(student_fix, teacher, 0.01, rng)
    try:
        before = snapshot(student_fix)
        s_logits_t, _ = student_fix.forward(ids)
        with no_grad():
            t_logits_t, _ = teacher.forward(ids)
        s_logp = log_softmax(s_logits_t / TEMP, dim=-1)
        t_prob = softmax(Tensor(t_logits_t.data / TEMP), dim=-1)
        soft = kl_div_loss(s_logp, t_prob, reduction="batchmean") * (TEMP**2)
        t0 = time.perf_counter()
        soft.backward()
        SloAdam(lr=1e-4).step(student_fix.parameters())
        step_t = time.perf_counter() - t0
        delta = weight_delta(before, student_fix)
        print(f"  wall time: {fmt_time(step_t)}  KL*T^2={float(np.asarray(soft.data)):.6f}")
        print(f"  max mean |delta w| after 1 step: {delta:.3e}")
        print("  -> student moved: graph preserved to params" if delta else "  -> NO-OP")
    except Exception as e:  # noqa: BLE001
        print(f"  manual step raised {type(e).__name__}: {e}")

    print()


if __name__ == "__main__":
    main()
