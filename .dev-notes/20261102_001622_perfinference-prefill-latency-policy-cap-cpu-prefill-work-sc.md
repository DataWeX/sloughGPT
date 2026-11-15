---
id: 20261102_001622_perfinference-prefill-latency-policy-cap-cpu-prefill-work-sc
status: todo
board: 20261102_001622_perfinference-prefill-latency-policy-cap-cpu-prefill-work-sc
title: "perf(inference): prefill latency policy — cap CPU prefill work (scoped out of card 20261101_110410)"
tags: ["inference", "performance", "area:inference"]
sprint: ""
gh: null
---

# perf(inference): prefill latency policy — cap CPU prefill work (scoped out of card 20261101_110410)

SCOPED OUT of card 20261101_110410 item 3 (2026-11-02), deliberately, not silently.

WHY IT CHANGED SHAPE: item 3 was originally "clamp prefill total_len to a sane CPU limit" as OOM defence. The OOM is already gone — the numpy fallback is chunked over the query axis under SLO_PREFILL_FALLBACK_BUDGET_MB (a6c8f3b31, proven by tests/core-py/test_prefill_fallback.py), so prefill memory is O(seq), not O(seq^2). What remains is LATENCY, not safety: an 8k-token prefill is ~a minute of CPU inside a 120 s request timeout.

WHY THAT IS NOT A DRIVE-BY FIX: a token clamp means silently truncating the user prompt. Which end gets dropped (head for recency, tail for instructions), and whether truncation is announced, is a context-window product decision (docs/UX_FLOWS.md owns it) — not a kernel flag.

INVESTIGATE: measure prefill wall-clock vs prompt length on this box (seq 1k/2k/4k/8k, H=16, E_head=64); find where the 120 s timeout actually bites; then propose truncation semantics with the numbers attached.

SCOPE: prefill path only. Generation already streams fine and must NOT be affected (card 20261101_110410 scope correction).
