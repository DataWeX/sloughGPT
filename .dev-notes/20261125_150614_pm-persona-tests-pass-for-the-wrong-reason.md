---
id: 20261125_150614_pm-persona-tests-pass-for-the-wrong-reason
status: todo
board: 20261125_150614_pm-persona-tests-pass-for-the-wrong-reason
title: "test(cognition): 12 TestPersonas tests pass for the wrong reason — _PM patches the source module, handlers bind the facade"
tags: ["testing", "cognition", "area:domain"]
sprint: ""
gh: null
---

# TestPersonas mocks never worked — tests pass only via a save→load disk chain

**Origin:** triage of CI run 45361359978 (card 20261107_112103, 2026-11-25). The `TestPersonas` 404s across shards had two independent causes; the placement half was fixed by the file-grouped split + `--dist loadfile`, but the **test-correctness half is deferred to this card**.

**Defect:** `TestPersonas` uses `_PM` to patch the *source module* of `PersonalityManager`, but the handlers under test do `from domain.cognition import PersonalityManager` — they bind the **facade**, so the patch never reaches them. The tests therefore pass only through an accidental save→load disk chain (a `personas/` file written by one test is read by another): proven when worktree `personas/` was empty (delete ran last) yet slice2-save → slice3-get passed on the same disk across processes. The mocks are dead code — the tests assert nothing about what they think they assert.

**Fix direction:**
- Patch where the name is *bound* (the facade attribute the handlers resolve, or patch before the `from ... import` site), not where it is *defined* — or drop the mocks and make the save→load dependency explicit and hermetic (tmp_path, per-test).
- Keep the incident-derived guard: the a/b module-global handoff (`_leaked_workflow`) test must stay — it is what caught the item round-robin split breaking file-level invariants.

**Acceptance:** each TestPersonas test passes with the disk chain removed (fresh tmp persona store per test), and fails if the handler's manager is stubbed to return 404 — i.e. the test now actually exercises the mock.
