---
id: 20260924_056_fix_preexisting_server_failures
status: done
title: 86 pre-existing tests/server failures — stale _internal patch targets + env gaps
tags: [test, debt]
---
# Card 056 — pre-existing tests/server failure debt

## Scope (from card)
Baseline on main: **89** tests/server failures (86 legacy `fail-old.txt` − 1 fixed
+ 4 new auto-train 404s). `fail-old.txt` was wiped by another session wiping
`/tmp/opencode` — recovered baseline lives at `~/.cache/card056/fail-baseline-89.ids`.
Categories:
1. Stale `_internal`/`_api` patch targets after facade moves — retarget to
   `domain.<pkg>.<name>` (consumer late-imports facade).
2. Real impl bugs the tests caught (5 found — see Result).
3. Stale test expectations vs deliberate changes (no-provider 503→500 in
   8b1617dd4, deleted `/auto-train/*` routes, response-shape drift).
4. Order-dependent pollution (readiness_gate 2, startup config_validated 1,
   multimodal DPO 1 — pass alone).

## Approach
- Per-file batches, count kept monotonic vs baseline; run each file after edits.
- Work in worktree `/home/mana/Documents/sloughgpt-srv-tests` on branch
  `feat/server-test-debt` (isolated off main; shared tree is a stale diverged
  branch with other sessions' WIP). Worktree got a `.venv` symlink (endpoint
  env checks resolve `repo_root/.venv`).
- Diagnostic: `--tb=line` + zip ordered `path:line:` output with ordered
  `FAILED` lines to pair failures with messages.

## Constraints
- Do not touch card 055 db_pool semantics (done + merged as 6c5c0453b).
- Only stage files belonging to this card when committing.

## Result
**DONE — full suite green: `1895 passed, 0 failed, 68 deselected` on main at
c8fb631dd (worktree sloughGPT-fix-recovery, pytest tests/server, 2026-09-25).**
Baseline 89 → 0. Final batch commits on main: 1ef601bd6 (kb+auth),
39d5c0fd0 (startup stub), 2443bb441/8b1617dd4-era (benchmark), e29c0d88c
(health), de29ffbe9 (models), e6c1dd8d2 (workflow), c8fb631dd (readiness +
DPO facade + startup facade).

Order-dependent root cause (the stubborn last 4):
`apps/api/server/tests/test_support.py` replaces
`ReadinessGateMiddleware.dispatch` with a no-op **at import time** (so its
schema-validation tests get 422 not 503). `tests/server/test_server_api.py`
imports `get_test_client` from it at module level → collected before any test
runs → process-wide neutered gate → readiness tests got 200 in full suite but
passed in isolation. Fix: stash `_real_dispatch` on the class in test_support;
`test_readiness_gate.py` captures pristine dispatch at import + autouse
fixture restores it per-test. Related: fixture also clears core
`ServerState.model/tokenizer` (other tests load them), multimodal DPO test
patches facade `domain.feedback.HFDPOTrainer` (function-local late import),
startup tests patch facade `resource_manager.get_resource_manager`.

Impl bugs fixed (test-driven):
1. `controllers/models.py::_resolve_device` — underscore names not re-exported
   by ml_types facade star-import → import from `_internal`.
2. `routers/security.py` — `require_auth_if_enabled` returns None when auth
   disabled (unguarded `.get()` → 500); `status=403` kwarg → `status_code=403`
   (status kwarg → TypeError → 500).
3. `domain/knowledge/engine.py` — missing public `get_ingestor()`; router
   called it → AttributeError → `/knowledge/ingest-url` always 500.
4. `domain/training/__init__.py` — facade missing `RepoImporter`,
   `HuggingFaceImporter`, `URLImporter`, `ISBNImporter`, `DataImporter`,
   `BooksSearch`, `GitHubSearch` (router + engine late-import from facade) →
   added as lazy `_LAZY_IMPORTS`.
5. `domain/training/engine.py` — `extract_pairs_from_sessions/logs` had wrong
   signatures (`sessions`/`logs` positional) vs router kwargs + underlying
   pair_extractor kwargs → `/mobile/train/from-sessions` TypeError, empty
   stream; also added passthrough `write_training_text`.
6. `routers/infer.py::infer_embed` — `model_name` used in NotImplementedError
   handler before assignment → NameError → 500 instead of ngram fallback.
7. `infrastructure/startup.py` — prewarm entry `domains.api.sse_envelope`
   (stale pre-move path) → `domain.api._internal.sse_envelope`.
8. `domain/voice/engine.py` — `get_speech_recognizer(use_server=False)` gives
   BrowserSpeechRecognizer (no `.recognize()`) → server transcribe always 500
   → `use_server=True`.
9. `domain/voice/_internal/tts.py::TTSEngine.__init__` — wrong unpack index
   (`_, _, SloAdam, ...` picked SloLSTM) → `SloAdam(lr=)` TypeError → server
   TTS always 500.

Test fixes: facade retargets (companion/feedback/health/traits/server_state/
model_registry/datasets controller/user_adapters/models.get_provider/
multimodal.VideoProcessor/mobile get_notification_service/mobile_training
store), dylib skipif via native engine target, stale route retargets
(`/auto-train/*` → `/settings/training/auto-train/status` + `/training/checkpoints`,
`/metrics` → `/system/metrics`), stale 503→500 (deliberate 8b1617dd4),
voice-engine patch.object wiring (transcribe/synth response contract),
kb stats mock retarget (`memory.list_all`, engine derives stats), auth
expected key set + None-guard, status env native-engine patches.
