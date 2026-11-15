---
id: 20261001_health_payload_trim
title: "Health payload: drop per_tensor from /health, /health/detailed, /health/stream"
status: done
date: 2026-10-01
tags: [backend, performance, api, health]
board_id: d7492f1a-9066-4012-9f64-dbef1930bbfa
board: d7492f1a-9066-4012-9f64-dbef1930bbfa
hash: b3c72d0fd04e589277c778ac9da17ced97e5c5dc7fb6130861959e4e747634f3
---

# Health payload: drop per_tensor from /health, /health/detailed, /health/stream

Measured 2026-10-01: GET /health = 10,099,672 B raw (1.14 MB gzip); GET
/health/detailed = 10,107,676 B; SSE /health/stream = 10,566,666 B every ~3.3s
per client, uncompressed (gzip skips text/event-stream) ≈ 3.2 MB/s.
quantization.per_tensor = 10,098,416 B of it (lm_head alone 3.38 MB); UI reads
only quantized/bits/mode/summary ≈ 300 B. Root:
slonet_provider.quantization_report() -> error_report() wired in at
apps/api/server/controllers/health.py:517 (/health) and :694 (/health/detailed),
SSE snapshot at apps/api/server/routers/health.py:854. Fix: health requests
report WITHOUT per_tensor (provider kwarg include_per_tensor=False, default True
keeps packages/core-py provider tests green); on-demand per-tensor endpoint
apps/api/server/routers/models.py:1455 and QuantizeCard action result MUST stay
unchanged. Acceptance: SSE event < 10 KB verified live, regression test
asserting no per_tensor in health quantization payload + summary present,
lint+typecheck+pytest green, scripts/benchmark_latency.py run, before/after
numbers reported. Branch: fix/health-payload-trim.

SHARED WORKSPACE
- index updated: no index change required (fix only — no new app/package/doc)
- shared copy reused: root .venv + node_modules, no new deps
- coordination named: card B "TraitRadar: SVG path syntax passed to <polygon
  points>" (7c1bfccb) — same branch fix/health-payload-trim

## Resolution

MERGED to main 2026-10-02.

- Commit: `2c460cbaf` "fix(health): drop per_tensor from health payloads
  (10 MB -> ~300 B); radar polygon points" (branch `fix/health-payload-trim`).
- Merge commit `e1fbcee08` (parents `952e59571` + `2c460cbaf`). Main advanced
  `a1ebb7f2d` -> `952e59571` (WIP: preserves another session's 30-file staged
  work; reversal = `git reset --soft HEAD^` per its commit message) ->
  `e1fbcee08`.
- Verification on merged tree: byte-identical to the tested branch for all
  touched paths; pytest `tests/server/test_health_controller.py` +
  `packages/core-py/tests/test_slonet_provider.py` = 138 passed; ruff clean;
  worktree clean.
- Residual (noted, not blocking): live slough-api service still runs the root
  checkout — after-numbers (SSE size + `scripts/benchmark_latency.py`) captured
  when the API next restarts on the fixed code.
