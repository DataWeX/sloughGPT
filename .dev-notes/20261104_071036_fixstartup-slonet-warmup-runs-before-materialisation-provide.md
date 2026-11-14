---
id: 20261104_071036_fixstartup-slonet-warmup-runs-before-materialisation-provide
status: done
board: 20261104_071036_fixstartup-slonet-warmup-runs-before-materialisation-provide
title: "fix(startup): SloNet warmup runs before materialisation — 'provider missing _tokenizer/_model' on every lazy boot"
tags: [startup, slonet, warmup, lazy-guard]
sprint: ""
gh: null
---

# fix(startup): SloNet warmup runs before materialisation

Session 2026-11-10 (slonet-warmup lane, built directly after Gen hit the free-model limit; batch approved by user 2026-11-09).

## Problem

Every lazy-guard boot logged an unactionable WRN:
`[MODEL] slonet_server SloNet warmup failed: ThreadWorker generate error: ... provider missing _tokenizer/_model`
— `_run_warmup` fired at guard start while the `lazy_from_slnc` provider is header-only by design (weights on the preload timer), so the attempt always failed and trained operators to ignore a boot WRN.

## Fix (branch `fix/slonet-warmup-ordering`, from origin/main 07fdd0c2f)

- `SloNetServer._run_warmup(forced=False)`: lazy mode DEFERS (`_warmup_deferred` — no generate, no warning, never claims `completed`) instead of attempting-and-failing against a header-only provider.
- new `SloNetServer.trigger_warmup()`: idempotent post-materialisation hook; spawns the forced run; pending-trigger handoff covers firing before the init-spawned run has executed (spawn is async — found by stress, not the first green run).
- `startup._preload_parent_weights`: after the guard stop, calls `provider._server.trigger_warmup()` fail-open (own try/except, gated on `model is not None`) once the resident model is published.

## Evidence

- Pushed: `63598c379` (fix) + `f89b44912` (domain rehash) + `eef591455` (AGENT_SYNC) — local == remote, 0 ahead. NOT merged (user sanction required).
- TDD: RED 4/4 watched → GREEN; stress 10× → 0 fails after replacing fixed `time.sleep` waits with 5s-deadline `_poll` (fixed sleeps flaked 5/10).
- Gates: slonet+startup combo 269 passed ×3; `tests/server/` 2135 passed / 1 skipped (excl. pre-existing phoneme `ModuleNotFoundError`, already ruled note-only); ruff check+format clean; post-commit domain rehash (`infrastructure c5419209 → f58f724b`) + `--check` + `test_domain_baseline` 8/8.
- Diff review caught own insertion stealing `core.model.set.assert_called_once_with` from `test_preload_publishes_materialised_model` — restored.

## Not touched

Preload timing/order semantics beyond warmup placement; `tests/core-py/test_results/*.json`; notification surfaces (frozen).
