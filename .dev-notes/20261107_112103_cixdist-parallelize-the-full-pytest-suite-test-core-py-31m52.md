---
id: 20261107_112103_cixdist-parallelize-the-full-pytest-suite-test-core-py-31m52
status: done
board: 20261107_112103_cixdist-parallelize-the-full-pytest-suite-test-core-py-31m52
title: "ci(xdist): parallelize the full pytest suite — test-core-py 31m52s is the PR queue's bottleneck"
tags: ["ci", "performance", "testing", "area:tooling"]
sprint: ""
gh: null
---

# ci(xdist): parallelize the full pytest suite — test-core-py 31m52s is the PR queue's bottleneck

# ci(xdist): parallelize the full pytest suite

**Goal:** test-core-py takes ~31m52s single-process (measured 2026-11-04) and is the long pole of every PR's matrix; E2E 26m27s, test-web 10m20s. pytest-xdist 3.8.0 is installed in the sloughgpt env but UNUSED — no `-n` flag anywhere in CI or pytest.ini. Parallelize the full suite so the merge-to-main backstop (Test Rule clause 3 — full suite stays, never weakened) gets cheaper.

**Scope:**
- Enable xdist for test-core-py (start `-n auto` or a fixed `-n 4` on the runner's cores) and measure wall clock vs the 31m52s baseline.
- Hazards to handle explicitly (why this is a card, not a one-liner):
  - **Order-dependent tests** — we've paid for these already (self_train stop order-dependence #296, learner fixture join #295, response_tracker global order). xdist distributes tests across workers and WILL surface new ones. `-p no:randomly` stays; each xdist run's distribution is deterministic per worker count only if a grouping dist mode (`--dist loadscope` / `loadfile`) is considered.
  - **Shared ports / sockets** — journey tests hit localhost:5173; parallel workers collide (the X-Slo-Web-Tree foreign-tree problem gets worse with N servers).
  - **Subprocess tests** (self_train stop fixture, benchmark e2e CLI runs) — CPU contention + timeouts.
  - **Disk**: each worker imports the tree; ~/.cache is at ~90%.
- Decide placement: pytest.ini addopts vs CI workflow flag vs scripts/ wrapper (the repo's Command Rule says flags are declared once).

**Not in scope:** the affected-closure work (card 20261104_085623 — run LESS on PRs); E2E parallelization (separate runner class); weakening the merge-to-main full run.

**Evidence/baseline:** test-core-py 31m52s (2026-11-04), 42823 passed/116 skipped typical run (2026-11-07 logs). Target: 2x+ wall-clock reduction with zero flake introduction; gate = full suite green x3 consecutive runs on the parallel config before it becomes the default.

## Pickup findings (2026-11-24)

- **Premise partly stale:** `-n auto` already landed (card 20261112_062436). Measured **1.45x** (50m19s serial → 34m32s), NOT 2x — the CI log says **"created: 2/2 workers"**: the public runner has **2 CPUs**, so `-n auto` inside one job is ceilinged. The lever that adds hardware is **N shards on N runners** (public repo = free minutes).
- Remaining scope taken: 4-way sharding + flake hazards + the ×3-consecutive-green gate.

## Landed design (pushed 79b12de96, run 45355582144)

- **Owned split, no new deps** — `tests/shard_split.py` (round-robin over the collected list: disjoint + complete by construction; spreads sibling files so one pathological file can't strand a quarter on one runner) + `tests/conftest.py` hook. `SLO_TEST_SHARD=k/n`; unset/empty = exact no-op; malformed spec = pytest exit 4 (fail loud, never silently run whole suite in one shard). Build-from-scratch rule applied to CI plumbing (vs a floating unpinned split plugin in CI's install line).
- **Matrix on `test-core-py`** — push → `[1,2,3,4]`, PR → `[1]` (PR keeps the single unsharded affected-closure run unchanged). `fail-fast: false` (a red shard must not cancel siblings — keep the evidence). Each shard runs **BOTH passes** over its quarter (kept in one job so `test_workflow_testmon_split.py`'s job-scoped pin keeps exact teeth; the ~7m-faster two-job overlap variant is an evidence-gated follow-up).
- **Coverage** — `COVERAGE_FILE=.coverage.shard{k}` per shard; pass 1 gets `--cov-report=` (4 partial percentages would read as red) + `--cov-fail-under=0` (pyproject `fail_under=50` would be inherited per shard — pytest_cov/plugin.py:270) + `--durations=25` (straggler profile; local box can't instrument). New `test-core-py-coverage` job: `coverage combine` + `coverage report` — the 50% floor is enforced on the COMBINED data; pyproject stays the single declaration.
- **Testmon single-writer preserved** — shards upload `.testmondata*` artifacts (include-hidden-files: dotfiles; glob catches WAL sidecars). New `test-core-py-record-merge`: `scripts/merge_testmon.py` (stdlib sqlite only; target = byte-copy of shard 1 so schema comes from a real recording, never a hand DDL) merges shards → ONE `.testmondata` PRs restore. pytest-testmon 2.3.1 has no merge path (verified vs wheel); `suite_execution_file_fsha` is dead DDL (zero readers/writers). **Strict verification**: row counts must equal sum (proves no loss AND no test in two shards), fingerprint union, `PRAGMA foreign_key_check`, schema `user_version` equality — output **unlinked on any violation** (an under-selecting DB would silently skip affected tests on PRs; strictly worse than no DB where run_affected over-selects — the safe direction). Cache save stays verbatim: main only, continue-on-error.
- **Needs wiring** — `build` + `benchmark` now need the two new jobs (both are main-only, matching their consumers' own `if:` — no skipped-needs trap on PRs).

**Measurements:** shard split over real collection = 5+5+4+4=18 disjoint; merge E2E green (bogus spec → exit 4; `coverage combine` "Combined 4 files", TOTAL 72% exit 0); 36 pins+new tests green; test-root closure **1944 passed** (2 pre-existing foreign-server 404s on :8000 — reproduce on pristine origin/main; they SKIP in CI, no server there). Expected wall 54–66m → **~21–26m (2.5–3x)**.

**Gate remaining:** ×3 consecutive green CI runs on main.

### Run 1 — `45361359978` on `12ed64882` (2026-11-24, attempt 1): red on flakes, sharding proven

- **Scale proof:** pass-1 per shard = 10,722 / 10,723 / 10,724 / 10,722 collected = **42,891 ≈ full suite, disjoint+complete**; TestPersonas tests scattered across shards 2/4 = the round-robin hook fired in CI.
- **Wall:** pass 1 = 7m36s–14m06s per shard, pass 2 ≈ 6m31s (shard 3) → **test-core-py ≈ 20–25m vs 54–66m pre-shard ≈ 2.5x** (contended morning, other runs queued).
- **Shard 3 green both passes** with the exact new flags → flags/coverage config exonerated.
- **5 failures across shards 1/2/4, pass 1 only** (failed step ⇒ pass 2 + uploads skipped ⇒ coverage/record-merge jobs skipped with it): `test_infra_thread_teardown::test_b_no_leaked_threads_after_teardown` (assert `_leaked_workflow` None — setup ran on another xdist worker, `--dist load` class-split), `test_pugqeep_engine::test_query_methods_survive_concurrent_spawns` (**120s CI timeout vs 5.3s local** — 2-CPU runner + coverage overhead), 3× `test_consciousness_router::TestPersonas` **404 == 200** (documented flake class — AGENT_SYNC 11-23 "2× consciousness (flake, passed on re-run)").
- **Three-strike:** all 3 sites green locally on this branch (39 passed / 8 passed / 1 passed 5.3s); yesterday's pre-shard run `45204414933` (old config) failed on a DIFFERENT flake (`test_wgpu_backend::TestLayerNorm`, 42,777 passed) — these tests pass under the old config too; identical flags green on shard 3. **Conclusion: dynamic-distribution order/timing flakes (the card's predicted hazard), not split-logic defects.**
- **Mitigation if reruns keep reddening (evidence-gated, not yet applied):** `--dist loadscope` (groups module → keeps `TestConftestTeardown::test_a/b` on one worker — the card's own suggestion).
- Attempt 2 (`gh run rerun --failed`) re-runs shards 1/2/4 **and** gives `test-core-py-coverage` + `test-core-py-record-merge` their first CI exercise.

### Strike 2 — attempt 2 (2026-11-24): byte-identical → deterministic, not flake

- Rerun reproduced the **same 5 tests, same shards, same counts** (`10695/10692/10691` passed both attempts) → deterministic given shard membership. The flake conclusion in Run 1 above was **wrong** — three-strike corrected it.
- Local slice repro (old split): shard 2 fails `test_b` + pugqeep locally too; slice 3 green. Yesterday's old-config run failed a DIFFERENT test (`test_wgpu_backend`) → main has independent flake history; these 3 sites are placement-dependent, not time-dependent.

### Root causes (code-level, 2026-11-25)

1. **`TestPersonas` 404s — disk-chain coupling; the mock never worked.** `_PM` patches `...personality.PersonalityManager` (source module) but every handler does `from domain.cognition import PersonalityManager` (facade binding) → `@patch(_PM)` NEVER applied. Tests pass only because real `save` writes `data/consciousness/personas/test.json` and real `get/activate/delete` find it in collection order. Proof: worktree `personas/` is empty because `delete` ran last and cleaned up; slice 2 (save, 12:34) → slice 3 (get, 12:39) passed across processes on the SAME disk; CI = separate runners → deterministic 404.
2. **`test_a/b` — item round-robin splits the deliberate handoff.** `test_a` sets `global _leaked_workflow`; `test_b` asserts it non-None (class docstring: "Order-dependent"). Consecutive items → different shards → different processes → guard fails. Within a shard, `--dist load` could still split them across workers (module globals are per-process).
3. **pugqeep race test — contention, not deadlock.** 60k spawns: 5.3s alone; >120s in CI (2/2 attempts); locally the SAME slice measured >150s once and <10s once → bimodal, sibling-load-dependent. Old config was serial (`42777 passed in 36:32`, no workers) so it never contended. Its own `hammer_t.join(120)` would fail the asserts even with a bigger outer timeout → both must scale together.

### Fix (one identity: the split respects file boundaries) — 2026-11-25

- `tests/shard_split.py`: **file-grouped LPT** — group items by nodeid prefix, largest file first into the currently smallest shard, name tie-break. Contract now: disjoint + complete AND a file never straddles shards. Balance measured **10,472×4 exact over 41,888 items** (naive file-RR was 1.30x — rejected). Name tie-break makes placement independent of filesystem collection order → local slice runs predict CI placement exactly (round-robin did not).
- `tests/test_shard_split.py`: nodeid-shaped items; new pins — file-never-straddles, placement-ignores-collection-order, LPT balance (20 pass).
- `.github/workflows/ci_cd.yml`: `--dist loadfile` on BOTH test-core-py passes — one file = one worker in collection order; module globals AND the save→load disk chain need it (without it `get` could race ahead of `save` on the shared job disk).
- `tests/core-py/test_pugqeep_engine.py`: `@pytest.mark.timeout(600)` + `hammer_t.join(120→480)` (reader join 60 stays; 480+60 < 600 outer). No assertion weakened. CI `--durations=25` reports the real number for an evidence-based bump if 600 ever proves tight.
- Untouched: router + teardown test files (fixed structurally, assertions intact); test-root untouched (green as-is); no workflow pins assert command flags (verified — 26 seam gates pass).
- **Deferred (follow-up card):** the `_PM` wrong-target defect — a real latent test bug (12 tests pass for the wrong reason); fixing it changes 12 tests' behavior = separate test-correctness identity, after the gate.
- Gates before push: ruff check+format ✓; YAML ✓; 26 seam gates (split + workflow pins) ✓; 4 local slices with the exact CI command ✓ (see below).

### Landed (2026-11-25): `f82b98c14` (fix) + `e6ef2396d` (rehash) pushed `ef2e448d8..e6ef2396d` → main

- **Local 4-slice verification** (exact CI command, `--dist loadfile`): slice1 `10876 passed` (×2), slice2 `10549 passed` (critical sites **0 failures** — test_b + TestPersonas green in new placement), slice3 `10533 passed` (pugqeep timeout fix in place), slice4 `10850 passed` exit 0. Sum ≈ **43,067 ≈ full suite** (disjoint+complete at scale). Two environmental flakes triaged, neither placement: quant CLI ×3 died at exactly 150s while my own orphaned slice4 ran concurrently at box load 12 (green in the clean re-run + CI ×2), hf_lora top_p failed once in-shape (numerical NaN warning) then passed standalone ×3 and passed in the SAME placement on the previous run.
- **Baseline main-red repair** (`e6ef2396d`): placement reviewed first — `16f886905` (BatchArray) + `555e760b0` (prefill) both stayed inside `domain/training/_internal/` (3 modified files, no adds/moves; only 2 private module helpers added, no class/facet changes) → rehash moved 1 hash (`training 9f75e850 -> 3abce1a7`), `--check` = "no drift". This also cleared main's OTHER reds (`test_gates_cli`, `test_clean_tree_is_green`, `TestDirectoryHashDrift` — all the same root cause).
- **Gate run 1 = `45612603707`** (CI/CD on `e6ef2396`) → **FAILED** (1 red job): shards 1/3/4 + all 8 other jobs green (incl. `test-root` → rehash repair verified), shard 2 red with 6 training failures.

### Run-1 red: triaged to root cause + fixed (2026-11-25)

- **Symptom:** shard 2 = `6 failed, 10700 passed` — `test_training_export_presets` ×2 + `test_training_robustness::TestConfigValidation` ×4, all `run_full_cycle → assert result.success` with `error='learning_rate must be > 0, got 0.0'` (all on gw0, shard took 1060s vs 171–248s siblings).
- **Not the split's placement rules** (three-strike): passed standalone (41/41), passed full shard-2 exact-CI-command local repro (10641/10641) — because my box's `~/.config/sloughgpt/training_history.jsonl` has months of good entries diluting the average.
- **Root cause (reproduced byte-identically):** fresh-HOME run of just the 2 files → **the same 6 failures**, 79 history lines written. Chain: `TrainingConfig.adaptive` defaults **on** → `_configure` reads the **HOME-shared** `training_history.jsonl` (common to all workers/runs on a machine) → tests record `TrainingOutcome(...)` with dataclass default `learning_rate=0.0` → once ≥4 "similar" records exist (confidence = (0.3+0.1n)×consistency > 0.6), the recommendation **overrides lr to 0** → `PipelineConfig.__post_init__` raises. CI's fresh VM starts empty (poison accumulates mid-run); local passes by dilution. `--dist loadfile` changed writer/reader ordering so it manifests deterministically.
- **Fix (4 files, one identity: training-history state must not leak across tests/runs):**
  1. `tests/core-py/conftest.py` — session autouse `_isolate_training_history`: points `outcome_tracker.DEFAULT_HISTORY_PATH` at a per-worker tmp file (same precedent as `_reset_log_correlation_context`).
  2. `domain/training/_internal/adaptive_config.py` — `recommend()` drops `lr <= 0` sentinel records before matching (all-sentinel → heuristic fallback, always lr > 0).
  3. `domain/training/_internal/comprehensive_trainer.py` — `_configure` applies `rec.learning_rate` only when `> 0` (contract guard at the application seam).
  4. `tests/core-py/test_adaptive_infra.py` — regression test `test_recommend_ignores_default_lr_sentinels` (5 sentinel records → fallback, `based_on_runs == 0`, `lr > 0`, `confidence == 0.3`).
- **Verified:** fresh-HOME repro 6-failed → **41 passed**; targeted closure **166 passed**; ruff+format clean; full affected closure (run_affected, full record) → running.
- **Follow-ups carded:** `20261125_150613` per-model line-of-best-fit lr estimator (user proposal — trend up/down, total estimator); `20261125_150614` `_PM` wrong-target (12 persona tests pass for the wrong reason).

### Fix pushed — PR #387 (2026-11-25)

- Commit `45b6ae63e` on `feat/ci-shard-core-py` (pushed after a stale `index.lock` from the 10:57 server restart was cleared — 0 bytes, no live git process).
- **All 4 local slices × exact CI command green**: 10,876 / 10,538 / 10,533 / 10,850 passed. Slice 4's first attempt was killed at 97% by the restart; clean rerun exit=0.
- Slice 2 rerun (started by another session, overlapped my slice 4 → 2× `-n auto` contention) failed `test_hf_lora_finetune::test_generate_numpy_lora_with_top_p`; passes standalone in 1.45s → contention flake, unrelated test, two-strike (original slice 2 green + standalone green).
- Gates: ruff check+format ✓, YAML ✓, 29 seam-gate tests ✓.
- **Remaining: ×3 consecutive green CI runs on main after merge.**

### Turn of events (2026-11-25, evening) — shard split landed on main via parallel session; PR 387 repurposed

- A **parallel session took over `.wt-ci-shard`** mid-day: rebased my `45b6ae63e` onto main (→ `f82b98c14`), pushed **the shard split + --dist loadfile directly to main** (lane push), plus its own baseline rehash `e6ef2396d`. **The card's fix is therefore ON MAIN.**
- It then wrote the `--dist loadfile` fallout fix (shared training-history lr=0 sentinels → 6 `run_full_cycle` failures, run 45612603707) but **died at the 16:05 restart with it uncommitted** (last activity 16:05, pytest killed).
- Taken over after confirmation: committed its fix as `42fd636ca` (ruff clean, 64-test closure green), rehash after it (`0caf4e793`, --check no drift, 16 gate tests green).
- **PR 387 repurposed**: force-pushed (old head's content already on main), title/body now = the lr-sentinel fix (REST API — `gh pr edit` is broken repo-wide by the projectCards GraphQL deprecation).
- **×3 green gate status: 0 of 3.** main is on a 6-run failure streak (lint, test-root baseline drift, test-core-py shards — exactly what this PR fixes). Gate starts once main goes green.



### lr-sentinel fix rebased + pushed to main (2026-11-25, night)

- Root cause of gate run #1 (45612603707) shard-2 red **proven by fresh-HOME repro**: `TrainingConfig.adaptive` defaults on -> `_configure` reads HOME-shared `~/.config/sloughgpt/training_history.jsonl` -> tests record `TrainingOutcome` with default `learning_rate=0.0` -> >=4 similar records => confidence>0.6 => lr overridden to 0 => `PipelineConfig.__post_init__` raises. Local passed by history dilution; CI fresh VM poisons mid-run.
- Fix (one identity: training-history state must not leak): conftest session-autouse `_isolate_training_history` (per-worker tmp `DEFAULT_HISTORY_PATH`); `adaptive_config.recommend()` drops `lr<=0` sentinels; `comprehensive_trainer._configure` applies `rec.learning_rate` only if `>0`; regression test `test_recommend_ignores_default_lr_sentinels`.
- Verified: fresh-HOME repro 6-failed -> **41 passed**; targeted closure 166 passed; ruff+format clean.
- Rebased onto origin/main twice (kanban syncs, then origin's distill fix `2563e6101`), baseline rehashed for `domain/training` (`ca433a865`, `2d2a6b939`), pre-push gates green (baseline --check 0, ruff, 58 seam tests, 41 fresh-HOME).
- **Pushed `2d2a6b939` to main.** Gate run #2 = CI/CD `45687529659`. Watcher `.ci-watch/watch-45687529659.sh`.
- Local RUN_ALL fast phase: 46,948 passed / 17 failed — all 17 files pass in isolation (387 tests green); documented order-dependent pollution class (card `20261125_133509`), none in my closure.
- **Gate: 0 of 3 green.**

### GATE MET — card done (2026-11-26)

- Pushed `2d2a6b939` to main (4 commits: lr-sentinel fix + 2 baseline rehashes + format fix).
- **×3 consecutive green CI/CD runs on main:** `45687529659` (2d2a6b939), `45692374565` (e0f657f07), `45697989951` (77489b486) — all 17 jobs green incl. `test-core-py-coverage` (≥50% floor) and `test-core-py-record-merge`, both previously skipped.
- AGENT_SYNC.md entry sealed `a1bb769d4df1223e (re-sealed after rebase onto origin's new entries)`, ledger gate exit 0.

### Follow-up: PR-pass loadfile gap + main lint fix (2026-11-26)

- **Gap found while triaging PR #387 run `45682075668`** (`test-core-py (1)` red, 43m): the card's `--dist loadfile` fix covered only the two main-push passes; the **PR path** (`run_affected.sh … -n auto`, ci_cd.yml line ~187) still used xdist's default `load`, re-splitting same-file tests across workers → deterministically failed the placement-sensitive set (warmth traits, RAG `%d`, `test_a`/`test_b` handoff, TestPersonas save→get, pugqeep spawner). Evidence: same tests green on main's loadfile shards (run `45630172556`), red on the PR's `load`; local 4-slice runs green with the identical CI command.
- **Fix `b41a5c729`:** `--dist loadfile` added to the PR pass + comment. All 3 workflow pytest invocations now carry it (verified: count = 3).
- **Main-side lint red, three-strike:** `c19a46de9` (session exporter) landed unformatted; main's own CI/CD `45744168758` failed the same `ruff format --check` gate. Fix `24e7d5ed5`: ruff format `session_exporter.py` + `test_session_exporter.py` (12 exporter tests green, full-repo format check green).
- **PR #387 retitled/re-bodied** (REST `gh api -X PATCH` — `gh pr edit` still broken repo-wide by the GraphQL projectCards deprecation) to carry exactly these 2 commits: `ci(xdist): --dist loadfile on the PR run_affected pass too`.
- **PR #387 MERGED 2026-11-26 ~10:28** (by the parallel session during a restart), all checks green incl. `test-core-py (1)`. Verified on main: loadfile count = 3, `session_exporter.py` ruff-clean.
- Note: this card itself was sealed done earlier (`a1bb769d4df1223e`, gate ×3 green) — the two commits above are follow-ups on the same lane, not a reopening of the gate.
