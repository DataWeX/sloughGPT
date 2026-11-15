---
id: 20261103_085029_perfstartup-soul-inventory-scan-feedback-workflow-must-not-g
status: done
board: 20261103_085029_perfstartup-soul-inventory-scan-feedback-workflow-must-not-g
title: "perf(startup): soul inventory scan + feedback workflow must not gate readiness"
tags: ["perf", "startup", "inference"]
sprint: ""
gh: null
---

# perf(startup): soul inventory scan + feedback workflow must not gate readiness

Readiness is gated on inventory + background workflows. Measured cold start 2026-11-03
(journal 09:16:57 -> 09:17:37, 40s to model_loaded):
  imports                 3.1s   (526 modules, fastapi 1.86s)      unavoidable
  soul inventory scan    11-23s  <- THE BLANK: 09:17:00 -> 09:17:11
  feedback workflow       9s     <- 09:17:11 -> 09:17:20
  fp32 .slnc weights      2.9s   (2.5GB, mmap+parallel, already good)
  quant npz DEFLATE       8s     (separate card 20261103_082215)

DEFECT A: SloManager.__init__ -> _scan_souls() parses every candidate soul
synchronously on the event loop. _parse_soul_info reads the embedded JSON
config header and json.loads it: measured 2.21s for ONE 67MB config
(read 0.20s), and 23.34s for the real 25 candidates in models/
(10 files at 2.3-2.7s each). Deduped to 5 souls -> the observed 11s window.
A 67MB config is NOT needed to answer 'what souls exist'.
Fix: lazy scan (no scan in __init__), persisted (path, mtime, size) -> SloInfo
index at data/.soul_index.json, and a daemon warm thread from startup so the
first /souls request never pays it.

DEFECT B: main.py:178 calls _start_feedback_workflow() on the loop; it does
adapter aggregation (workflow.py:569, observed 9s) before returning. It is
background work riding the critical path.
Fix: run it on a daemon thread, same as the adjacent rag-init thread.

Also noted (not fixed here): 'Stage CRITICAL: complete (3/3 hooks OK, 0.0s
elapsed)' reports 0.0s for a 13s window - the stage timer is broken.

Gate: scripts/benchmark_startup.py cold-start -> /health 200, plus soul-scan
wall time from a cold index. Expect ~40s -> ~14s with card 20261103_082215.
