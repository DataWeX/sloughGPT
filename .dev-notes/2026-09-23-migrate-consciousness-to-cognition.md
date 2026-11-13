# Migrate consciousness → cognition (unravel into _internal)

**Status:** done
**Sprint:** —
**gh:** —

## Task
Merge `domain/consciousness` into `domain/cognition`: decompose modules
(engine, qualia, self_model, config, evaluation, meta_cognition, narrative,
personality, training) under `domain/cognition/_internal/`, re-integrate
public exports on `domain.cognition`, keep `domain.consciousness` as a
backward-compat shim.

## Body
- Target: `domain/cognition/_internal/consciousness/` (subpackage — see expert review).
- Branch: `feat/consciousness-to-cognition`.
- Preserve: router `_get_trainer()` → `domain.consciousness.training`,
  `domain/core` lazy map, `packages/core-py/domains/consciousness` shim.
- No new training loop — `training.py` stays a dataset adapter.
- Out of scope: collapsing duplicate `domain/cognitive`.

## Result
Canonical: `domain/cognition/_internal/consciousness/` (9 modules + training adapter).
Facades: `domain.cognition` re-exports API; `domain.consciousness.*` shims re-export same objects.
Verified: singleton identity across 4 import paths; 285 related tests green; ruff clean.
3 failures in tests/test_consciousness.py are pre-existing (confirmed on HEAD worktree).

## Merge
Branch `feat/consciousness-to-cognition` is green (ruff + 235 related tests, 3 pre-existing failures only).
`main` is locked to worktree `/home/mana/Documents/sloughGPT-fix-recovery` — cannot `git checkout main` here.
Merge from that worktree:
  git -C /home/mana/Documents/sloughGPT-fix-recovery merge feat/consciousness-to-cognition
Note: branch also contains feat/vite-phase0 ancestry (main is ancestor). Prefer `--no-ff` merge or cherry-pick 3aa5338b+d598b73e if vite work should not land yet.

---

## MERGED (2026-09-23)

- Branch `feat/consciousness-to-cognition` merged to `main` as `39f579a5`.
- Reflection loop fix `a630ca3f` included: `engine.reflect(apply=True)` commits+persists deltas; managers use `reflect(apply=False)` + `.narrative`; router no longer double-applies.
- board.jsonl union corrected in follow-up commit after empty-ours first resolve.
- Verification on main: 146 consciousness-related tests green; ruff clean; singleton identity OK across domain.cognition / domain.consciousness / domain.core.
