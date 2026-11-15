---
id: 20261102_023933_latency-regression-254x-since-sep-30-238s-606s-valid-measure
status: todo
board: 20261102_023933_latency-regression-254x-since-sep-30-238s-606s-valid-measure
title: "Latency regression 2.54x since Sep 30 (2.38s -> 6.06s valid measure; 5.6x pre-11-02 fixes) — within-run variance 1.5-11.6s = prime lead"
tags: ["perf", "latency", "inference", "regression", "area:inference"]
sprint: ""
gh: null
---

# Latency regression 2.54x since Sep 30 (2.38s -> 6.06s valid measure; 5.6x pre-11-02 fixes) — within-run variance 1.5-11.6s = prime lead



## Update 2026-11-02 03:31 — first VALID measurement on current main: 2.54x, not 5.6x

Harness fixed (4ed0e6592, this card's DoD item 3): THREE traps, not one — (1) the recorded rate-limit trap (25 back-to-back vs 10 req/60s, 429s silently dropped, fake 8.8ms min), (2) the script as committed had NO spacing at all, (3) NEW and worse: the clock stopped at urlopen() return, which on SSE /chat is response HEADERS (~12ms TTFB) while generation streamed on for seconds — one run reported '-98.8% improvement' while still mid-stream. Now: body drained to EOF (TTFB kept separately), paced at SPACING_S = 60/10 + 1s, ANY 429 fails the run loudly, host snapshot recorded, --json-out feeds benchmark_results.py record.

Valid run (15 samples, 7s spacing, 0x429, body drained, main 4b31e3e51+): mean **6058.8ms**, min 1559.1, p50 4695.8, p95/max 11602, ttfb mean 12.1ms. vs Sep 30 baseline 2381.5ms = **+154% = 2.54x**. Down from 5.6x: tonight's landings (boot prewarm, session KV reuse, identical-prompt resume, /v1 response-cache replay) recovered ~55% of the gap. The regression is REAL and remains.

Recorded: canonical store latency/Qwen--Qwen2.5-0.5B-Instruct_2026-11-02T023408277275Z.json (first entry in that series).

Environmental leads WEAKENED: current box 79C pkg / swap 2.4 of 3.8Gi / load 2.3 — BETTER than baseline's recorded conditions (97C during the Sep 30 run) yet 2.54x slower. Cool box, slower code.

NEW PRIME LEAD — within-run spread 1.56s to 11.6s (7.5x) at constant code and payload shape: this is not a uniform slowdown. Suspect per-request state or per-prompt path (5 distinct prompts; retrieval/workflow stage; queueing). TTFB 12ms constant means headers flush immediately — next probe is time-to-FIRST-EVENT and per-prompt labels, then fast-vs-slow request diff.

DoD status: spacing + rate-limit trap FIXED; recorded in history DONE; bisect/attack decision OPEN (variance-first instrumentation vs git-bisect Sep 30 - Nov 1).
