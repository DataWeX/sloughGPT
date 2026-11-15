---
id: 20261102_001621_fixinference-chatgenerate-repeats-the-completed-block-instea
status: done
board: 20261102_001621_fixinference-chatgenerate-repeats-the-completed-block-instea
title: "fix(inference): chat/generate repeats the completed block instead of stopping"
tags: ["inference", "quality", "generation"]
sprint: ""
gh: null
---

# fix(inference): chat/generate repeats the completed block instead of stopping

FINDING (2026-11-02, observed while verifying card 20261101_110410 on the live server :8000): the canonical prompt "Count 1 to 10, one per line. Just numbers." streams 1..10 clean (zero leak markers) and then REPEATS the block: 1..10 x3 within max_tokens=64. POST /inference/generate {"prompt":"count 1 to 5","max_tokens":48} returns 1..5 x5, and at max_tokens=256 returns the same block ~9 times.

NOT an overrun past 10 (never emits 11+) and NOT a leak (no <|, endoftext, 三明治, <|im_start|>, User:, Assistant:). The model completes the task correctly and then never stops — so the defect is at the stop/sampling seam, not in the weights.

INVESTIGATE (in order):
1. Is the generation config actually applying an EOS/stop set? Qwen2.5-0.5B-Instruct uses <|im_end|> (token 151645); if the sampler never checks it, no prompt can terminate early. Check the generate loop stop predicate and whether max_tokens is the ONLY exit.
2. Repetition penalty / no_repeat_ngram present in the config surface but unwired (or wired to 1.0)?
3. Chat template: is the prompt rendered with the instruct template (im_start/im_end) or raw? A raw prompt explains weak instruction-following but not non-termination.

SCOPE: sampling/stop seam only. Do NOT retrain. Do NOT touch model weights. Fix the max_tokens default too if it is the only exit.

VERIFY: canonical count-1-to-10 request ends after 10 with no repeated block; generate count 1 to 5 -> exactly 1..5; no leak markers; decode throughput not regressed (scripts/benchmark_latency.py).
