---
id: 20261102_011051_fixgpu-salvage-bounded-accelerator-probe-boot-prewarm-v1-ses
status: done
board: 20261102_011051_fixgpu-salvage-bounded-accelerator-probe-boot-prewarm-v1-ses
title: "fix(gpu): salvage bounded accelerator probe + boot prewarm + /v1 session_id from the 680-behind fork (landed)"
tags: ["inference", "gpu", "startup", "api", "port"]
sprint: ""
gh: null
---

# fix(gpu): salvage bounded accelerator probe + boot prewarm + /v1 session_id from the 680-behind fork (landed)

LANDED on main (2026-11-02), recorded here because the originating cards lived only in the stale repo-local .kanban and never reached the canonical board.

WHY THIS CARD EXISTS: feat/provenance-lineage is 29 ahead / 680 behind origin/main. Port-don't-rebuild was applied file by file; these three commits proved ABSENT from main (grep prewarm_accelerator = 0 hits, session_id in routers/llm_v1.py = 0 hits, get_accelerator() still calling _detect_best_backend() inline), so they were cherry-picked rather than rebuilt.

WHAT LANDED (main, in order):
- d391ef04 fix(gpu): bounded, probe-once accelerator detection. The 2026-10-29 05:06 incident: the lazy cupy first-import hung natively UNDER a live /health request, poisoned the import lock, and because the backend cache only filled AFTER detection returned, every concurrent get_accelerator() caller re-entered detection and piled onto it — :8000 served nothing for 4+ minutes. Now one daemon probe per reset epoch, callers bounded by SLO_ACCEL_PROBE_TIMEOUT (default 5s) then CPU fallback + latch, late result supersedes (generation-guarded), reset bumps the generation to discard in-flight probes, and _CUDABackend.is_available() does a ctypes driver check BEFORE importing cupy.
- 4ef278cf fix(startup): prewarm kick moved to the executed staged_loader path. It had been in _phase_config — a hook with no runner on that line (proved by boot logs producing no probe) — so the first import still happened under live traffic.
- 3dbcac51 perf(api): /v1 chat/completions accepts session_id. ChatBody had no field and pydantic ignores extras, so external multi-turn clients had the key SILENTLY DISCARDED while SessionKVManager (LRU+TTL) sat unused below the seam and the whole prefix re-prefilled per request. Field + one injection point (stream inherits it) + CHAT_SPEC param, so the descriptor stays the contract source of truth.
- bbcfafba fix(gpu): reset_accelerator() never rebound _PROBE_DONE — MY BUG FOUND IN THE PORTED CODE, not in the fork. The rebind was written without _PROBE_DONE in the global statement, so it created a function-local and the module kept the previous epoch's already-SET event. Every post-reset get_accelerator() then saw wait() return instantly with _BACKEND still None and latched the CPU fallback — CUDA silently dropped for the process life. Measured: epoch 1 cuda 0.33s, epoch 2 cpu 0.0014s before; cuda 0.33/0.03/0.03 after. The imported suite was green because test_reset_allows_fresh_reprobe counts probe CALLS, not which backend was SERVED; the new test uses a _Sentinel(_CPUBackend) subclass plus a 0.05s probe because a trivial fake detector won the race against the stale event and hid the bug.

GATES (re-run, not trusted from commit messages): 209 passed across gpu-probe + /v1 router + prefill fallback + memory pressure + config + config router + process-guard default + startup preload + startup routers; tests/contract 43 passed; apps/api/server/tests/test_ts_contracts.py 6 passed (so the fork's regenerated contracts.gen.ts is byte-correct against main's generator); gen_router_manifest.py --check up to date (55 modules); ruff check + format clean.

NOT PORTED, deliberately: the selfgen block in train_pipeline.py (main has it; the fork copy is a merge artifact that DUPLICATES it) and apps/web/app/(app)/vm/page.tsx (+1631 lines = the pre-decomposition monolith). UI-only cosmetic diffs (globals.css gradients, navigation/redirects, package.json) left unported — no journey behind them.

SIDE EFFECTS: SLO_AUTH_REQUIRED and SLO_CORS_ORIGINS joined _SKIP_ENV_KEYS (both are read via os.environ on main and warned "Unknown config key" at boot). The fork's comment justifying them by a "fail-closed default" was NOT imported — main has default "false"; whether that default should flip remains the separate decision in card fcc6ebfe8.

── VERIFIED LIVE 2026-11-02 (added after the fact by the session that landed it) ──

RESTART: systemctl --user restart slough-api (the declarative path; the unit is byte-identical to scripts/systemd/slough-api.service). New MainPID 267247. Boot journal: probe complete 02:14:07 INSIDE "Stage CRITICAL", "Application startup complete" 02:14:18 — the probe now finishes 11s before anything can call into it. /health 200.

SESSION REUSE — IT WORKS, measured (Qwen2.5-0.5B-Instruct on :8000, 482-token prompt, temperature=0):
  turn 1 (cold), session=ext-A ......... 16.48s
  turn 2 EXTENDS turn 1, same session ... 1.96s   <- 17.4s saved by the resume
  turn 2 same messages, COLD session ... 19.38s
That is SloTransformer._resolve_kv_state resuming at start=482 instead of re-prefilling.

CORRECTION TO AN EARLIER READING OF THIS SAME VERIFICATION: `usage.prompt_tokens` is NOT evidence. routers/llm_v1.py:~332 computes it by estimating the request messages, so a warm turn reports the full 482 regardless. An initial probe replayed the SAME prompt and saw ~16s every time — that is NOT a broken cache: an identical prompt is an explicit degenerate case (`if 0 < s < prompt_len` with s == prompt_len -> start = 0), pinned by tests/core-py/test_slonet_kv_cache.py::test_resolve_identical_prompt_falls_back. Resume only fires when the new prompt EXTENDS the cached prefix, which is what session_id is for. The cost of that degenerate case is now card 20261102_012822_perfinference-a-repeated-identical-prompt-re-prefills-from-s.

REMAINING GAP (carded, not fixed): SloNetServer.generate() drops session_id on the ProcessGuard branch, so a DIRECT provider.chat() with a live guard silently loses reuse. /v1 is unaffected because ProviderRouter.chat delegates to chat_stream -> generate_stream -> pump, which honours it — proven by GET /models/process-guard reporting requests_served == 1 (boot warmup only) across 8 /v1 requests. Card 20261102_012822_fixguard-slonetservergenerate-drops-session-id-on-the-proces.

GATES (re-run on the landed tree, not trusted): 209 passed across gpu-probe + /v1 router + prefill fallback + memory pressure + config + config router + process-guard default + startup preload + startup routers; tests/contract 43 passed; apps/api/server/tests/test_ts_contracts.py 6 passed; gen_router_manifest.py --check up to date (55 modules); ruff check + format clean. Separate post-hoc KV gate: tests/core-py/test_slonet_kv_cache.py + apps/api/server/tests/test_llm_v1_router.py = 41 passed.
