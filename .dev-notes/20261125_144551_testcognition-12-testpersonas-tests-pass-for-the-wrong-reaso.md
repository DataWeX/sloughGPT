---
title: test(cognition): 12 TestPersonas tests pass for the wrong reason — _PM patches the source module, handlers bind the facade
created: 2026-11-25T14:45:51.506467+00:00
updated: 2026-11-25T14:45:51.506467+00:00
tags: testing, cognition, area:domain
status: todo
---


# TestPersonas mocks never worked — tests pass only via a save→load disk chain

**Origin:** triage of CI run 45361359978 (card 20261107_112103, 2026-11-25). The `TestPersonas` 404s across shards had two independent causes; the placement half was fixed by the file-grouped split + `--dist loadfile`, but the **test-correctness half is deferred to this card**.

**Defect:** `TestPersonas` uses `_PM` to patch the *source module* of `PersonalityManager`, but the handlers under test do `from domain.cognition import PersonalityManager` — they bind the **facade**, so the patch never reaches them. The tests therefore pass only through an accidental save→load disk chain (a `personas/` file written by one test is read by another): proven when worktree `personas/` was empty (delete ran last) yet slice2-save → slice3-get passed on the same disk across processes. The mocks are dead code — the tests assert nothing about what they think they assert.

**Fix direction:**
- Patch where the name is *bound* (the facade attribute the handlers resolve, or patch before the `from ... import` site), not where it is *defined* — or drop the mocks and make the save→load dependency explicit and hermetic (tmp_path, per-test).
- Keep the incident-derived guard: the a/b module-global handoff (`_leaked_workflow`) test must stay — it is what caught the item round-robin split breaking file-level invariants.

**Acceptance:** each TestPersonas test passes with the disk chain removed (fresh tmp persona store per test), and fails if the handler's manager is stubbed to return 404 — i.e. the test now actually exercises the mock.

---

## Resolution — 2026-11-26 (branch `fix/testpersonas-facade-patch`, commit `968033689`)

**Root cause confirmed empirically, not inferred.** Handlers do function-local
`from domain.cognition import X` (routers/consciousness.py:499, 512, 552, 264…),
which reads the **facade** attribute at call time. The facade bound these names
*eagerly at import* from `_internal.consciousness`, so patching the `_internal`
source module swapped that module's attribute while the facade kept its original
reference — **every mock was dead code**.

Measured (identity probe, not assertion):

```
patching _internal…personality.PersonalityManager → facade id unchanged  → DEFECT CONFIRMED
patching domain.cognition.PersonalityManager       → facade IS the mock   → FIX VERIFIED
```

**The disk chain was load-bearing — proven by falsification:**
`test_get_persona` run in isolation (no prior `test_save_persona`) → **1 failed,
exit 1**. In a full run it passed only because save writes `personas/test.json`,
get reads it, delete wipes it last — which is why the dir reads empty afterward.

**Fix:** retargeted `_PM`, `_PP`, `_CE` → `domain.cognition.*` (the read point).
`_GO` already used the correct pattern and is untouched.

**Correction to this card:** says *12 tests*; actual is **8 `TestPersonas` +
7 `TestPersonality` = 15**. Also, the `_leaked_workflow` guard the card says
"must stay" lives in `tests/core-py/test_infra_thread_teardown.py` — a
different file; satisfied by not touching it.

### Evidence (polygraph objection: measured externally, not self-certified)

| # | Measurement | Expected | Result |
|---|---|---|---|
| 1 | baseline | 15 pass | **15 passed, exit 0** |
| 2 | post-fix | 15 pass | **15 passed, exit 0** |
| 3 | fault injection — mock returns 404 where test expects 200 | **must FAIL** | **FAILED, exit 1** ✓ |
| 4 | disk chain severed — fresh tree, no `data/consciousness` | 0 bytes written | **15 passed, dir never created** |

Measurement 3 is the sensitivity check: pre-fix this fault was silently ignored
(dead mock → real manager answered from disk). Post-fix the mock controls the
outcome, so the suite is no longer vacuous. Measurement 4 is the independence
check: pre-fix the run *created* `personality.json` + `personas/`; post-fix it
writes nothing.

**Gates:** ruff check ✓ / format ✓ · full file 39 passed · adjacent cognition 62
passed · `run_affected.sh` closure 39 passed (8m) · `test_domain_baseline` 8
passed (diff is tests-only, no rehash needed) · `check_feature_tags --validate`
✓ · `check_contract --static` ✓ (616 descriptors) · `gen_router_manifest --check`
✓ (46 modules).

**Honest limit:** measurement 3 establishes *sensitivity* (can catch the fault),
not *specificity* (no false positives).

**Incident:** the `.wt-personas` worktree directory was `rm -rf`'d **three times**
mid-session (dangling `prunable` admin entry = raw deletion, not
`prune-worktrees.sh`, which ends in `git worktree prune` and would leave no
entry; no log trace; 37 other `.wt-*` untouched; disk at 91%). Recovered each
time via the `WORKTREE_RECOVERY.md` runbook (`prune` + `re-add`). **Zero work
lost because the fix was committed before running gates** — the lesson from the
`AGENT_SYNC` TTS incident.
