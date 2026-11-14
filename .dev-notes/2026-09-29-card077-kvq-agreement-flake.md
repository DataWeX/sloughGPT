---
id: 20260929_077_kvq_agreement_flake
title: "Flake — test_fp32_model_kvq_agrees_with_fp32_cache fails in suite context (random tiny_model, no seed) — FIXED"
status: done
date: 2026-09-29
tags: [tests, flaky, quantization]
---

# Card 077 — KVQ agreement test is structurally flaky in suite context

`packages/core-py/tests/test_quantization_integration.py::
TestInt8QuantizedKvCache::test_fp32_model_kvq_agrees_with_fp32_cache`
asserts greedy agreement ≥ 0.7 between `quantize_kv=False/True` on
`tiny_model` — a **randomly initialized, untrained** SloTransformer
with **no np.random.seed**. Near-tie random logits + int8 KV noise flip
early greedy tokens → agreement can collapse to 0.0.

## Fix (commit 067c01271, branch feat/ux-flows-library) — DONE 2026-09-29

Seed the `tiny_model` fixture with `np.random.seed(1)` (option 1) and
document why in the fixture docstring:

- `seed=0` deterministically trips the flip (agreement 0.000 — verified);
  seeds 1-5 all give agreement 1.000, so the assertion keeps its power
  to catch real KV-quant corruption — not seed-hacking, the init is the
  test subject's input distribution.
- Ruling out global-state flake: a second independent cause would have
  survived the seed — it didn't.

## Gates

- Full `test_quantization_integration.py`: 2/2 runs green.
- Wide batch (10 files, the context that flaked 1/2 on BOTH tips):
  **3/3 green** post-fix (logs `~/.cache/slog-journeys/card077-wide-r*.log`).
- ruff check + format clean (lint-staged passed at commit).

## Original evidence (2026-09-29, branch feat/ux-flows-library @ cd1574bf1)

- Alone: 5/5 pass on BOTH pre-fix (60ce12608) and post-fix tips.
- Wide batch (10 files): fails 1/2 runs on BOTH pre-fix and post-fix
  tips — **pre-existing, not caused by any branch work**; suite-context
  RNG-stream sensitivity (any new test consuming np.random shifts
  downstream inits).

## Alternatives considered (not taken)

- (2) train the fixture — meaningful but adds per-test cost (fixture is
  function-scoped, ~20 users) and changes the test subject beyond the
  flake's cause.
- (3) margin-aware xfail — heavier, hides real regressions behind skip
  logic.

## Coordination (second session, `fix/journey-test-gates`)

Reached the same fix independently while this card still read `todo`
(about 20 minutes behind `067c01271`), then **reverted** the duplicate
edit — no second copy of the seed lands. Evidence worth keeping from
that run:

- 9-seed probe: 0/7/510 → agreement 0.000, 1/2/42/77/1234/20260929 →
  1.000. So `seed=0` really does fail (matches their note) and `seed=1`
  sits at first-token top1−top2 margin 0.408 — comfortable.
- Generation proven RNG-independent: with the fixture seeded, ran
  fp32-vs-kvq under global `np.random.seed` 0 / 7 / 123456 / 999999 →
  byte-identical tokens every time. Supports their "no second global-
  state cause" claim with a direct measurement.
- Alternate variant considered: seed **and** `np.random.get_state()` /
  `set_state()` restore around construction, so downstream tests keep
  their own draw points (a bare `seed(1)` fixes the stream for the rest
  of the session). Worth folding in if it ever flakes again.
