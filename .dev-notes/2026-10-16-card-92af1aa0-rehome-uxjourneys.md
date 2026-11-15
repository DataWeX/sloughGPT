---
id: 92af1aa0
title: "REMOVE the UX journey harness entirely — we don't need it; mole keeps 4 probes; all uxjourneys branches must not land"
status: in_progress
board_id: Main
date: 2026-10-16
branch: refactor/uxjourneys-rehome
tags: [rehome, ux-journeys, mole, structure, process-boundary]
---

# Card 92af1aa0 — REHOME domain/journeys -> packages/uxjourneys

User-directed placement rule (docs/STRUCTURE.md): domain/ = product logic;
packages/<name>/ = shared libraries with a real import name; tests/ = suites
only; scripts/ = ops entrypoints; docs/ = specification.

## Pickup verification (2026-10-16, session .wt-gpuops)

- Collision screen re-run with UNIQUE-CONTENT diffs (the name-only signal
  false-positives on stale-behind branches): only 2 branches carry real
  edits on the target paths — `feat/aml-stream-loader` (34 lines) and
  `feat/domain-consolidation-phase0` (34 lines) — BOTH dormant (no
  worktree). The one live branch touching them (`feat/persona-journey-tests`
  @ wt-persona-journeys, clean) carries 0 diff lines.
- `domain/journeys` still present on main; `packages/uxjourneys` absent →
  card genuinely open (path-based screen after the 85e15cc4 miss).

## Scope (from card)

1. `git mv domain/journeys -> packages/uxjourneys` — import name
   `uxjourneys`, module API unchanged (Flow, FLOWS, get_flow, list_flows,
   main; CLI `python -m uxjourneys`).
2. mole/probes/journey.py:71 — replace importlib import of the runner with
   a SUBPROCESS to the harness entrypoint; deletes the :36 env-freezing
   ordering hack. mole already reads the report via an explicit mirrored
   path (probe._report_path :41).
3. scripts/site_doctor_probe.py — docstrings + the `python -m
   domain.journeys` hint at :583 → strings only.
4. scripts/run_ux_flows.py — legacy duplicate (second Flow dataclass :151,
   all 13 flows, NO spec field → escapes the content gate, reports its own
   green; declared legacy in docs/FEATURES.md:302). Port to thin
   delegation onto uxjourneys; the duplicate registry goes.
5. Hygiene allowlist keys: tests/test_code_hygiene.py:74,
   tests/test_domain_print_allowlist.py:41.
6. Register packages/uxjourneys in packages/README.md + docs/INDEX.md;
   record the placement rule in docs/STRUCTURE.md.

NOT TOUCHED: packages/avion (stays generic; proven domain-free by
tests/test_stdlib_only.py).

## Status

Branch `feat/uxjourneys-rehome` cut from main 65f22d5eb. **Implemented,
gates green — awaiting landing** (main is checked out at
/home/mana/Documents/sloughgpt-api-std; land with the other ready
branches: fix/seam-holes-main, feat/credential-authority,
feat/web-terminal-phase1).

### Landed on branch (2026-10-16)

- `4a53bdb71` fix(hygiene): bare-except ratchet parses AST, not docstring
  prose — drive-by with a proven pre-existing root cause: `test_no_bare_except`
  was RED ON MAIN (`atomic_write.py` landed by 8de5784da carries
  `except: unlink` prose in its docstring; both files byte-identical to
  main). Standalone commit so any session can cherry-pick it independently.
- `44b3a6754` feat(uxjourneys): the rehome — 19 files, +170/-521,
  renames preserved by git (50-100% similarity).

### Implementation notes / measured deviations from the card's citations

- Item 2 (probe): subprocess boundary — `python -m uxjourneys` spawned
  with SLO_JOURNEY_REPORT pinned (parent read == child write by
  construction), PYTHONPATH handed over from find_repo_root(),
  SLO_JOURNEY_SWEEP_TIMEOUT (default 1800s) converts a hung child into a
  probe error. The env-freezing ordering hack + its 3 comment sites are
  gone (child freezes its own env). `test_sweep_uses_stub_runner`
  (sys.modules stub) replaced by REAL boundary tests per the real-engine
  directive: closed loopback port → honest exit-2 surfacing; zero-width
  timeout → killed-child error; plus a real `python -m uxjourneys
  --list` spawn test (13 flows, no mocks, no browser).
- Item 3: **no-op** — `scripts/site_doctor_probe.py` does not exist
  (citation drift); the real probe is
  domain/core/_internal/doctor/probes/journey.py, done in item 2.
- Item 4: 448-line duplicate registry → 27-line thin delegation onto
  `uxjourneys.main`; `--list` output diff-proven byte-identical to
  `python -m uxjourneys --list`. The unproven "reports its own green"
  clause from the card was dropped from the docstring (no self-check
  exists in the old script — verified via git show).
- Item 5: allowlist keys **removed, not re-pathed** — both print
  ratchets are domain/-only (test_code_hygiene print test roots
  ("domain",); test_domain_print_allowlist rglobs domain/), so a
  packages/ key would be inert rot. The runner's stdout-is-the-report
  contract is documented in its own docstring instead.
- Item 6: packages/README.md registration + STRUCTURE.md uxjourneys
  bullet + placement rule appended (rehome = git mv, never a copy; keep
  CLI flags; src-layout rule). docs/INDEX.md **not touched** — measured:
  it is a docs-only navigation table (Doc|Purpose), no new doc was
  created, so there is nothing correctly-shaped to register there.

### Gates (all run on 44b3a6754, PYTHONNOUSERSITE=1)

- pytest targeted: **60 passed, 1 deselected** (slow live test) —
  tests/test_ux_flows_library.py + packages/core-py/tests/test_doctor.py
  + tests/test_code_hygiene.py + tests/test_domain_print_allowlist.py +
  tests/core-py/test_launcher_repo_copy.py.
- ruff: clean on every touched file.
- whole-tree collection: 46146/46856 tests collected, **0 import errors**
  (710 slow deselected).
- real CLI: `./scripts/python -m uxjourneys --list` → 13 flows rc=0
  (launcher auto-scan picks up packages/uxjourneys/src);
  `scripts/run_ux_flows.py --list` byte-identical (diff-proven).
- real doctor: `python -m domain.core._internal.doctor --no-sweep` →
  journey=ok, reads the on-disk report (stale 296h → INFO finding),
  overall WARN rc=0.
- benchmark: n/a — no training/inference path touched (tests/docs only).

Land-risk note stands: `feat/aml-stream-loader` and
`feat/domain-consolidation-phase0` carry 34-line diffs on the target
paths — coordinate if either revives.

## SUPERSESSION (2026-10-16, session .cache/wt-uxjourneys) — read before merging any uxjourneys branch

The user rejected minting a package: **"we don't need another library for our
journeys!"** Placement is therefore `tests/ux_journeys/`, NOT
`packages/uxjourneys`. `tests/` is already a package with subpackages
(`server/`, `contract/`, `fixtures/`), so this is the established shape, and
`python -m tests.ux_journeys` from the repo root needs no PYTHONPATH.

Landed: **`2beb966b0`** on `refactor/uxjourneys-rehome` (clean 20-file diff on
origin/main). Mole crosses a **process boundary** — it spawns
`python -m tests.ux_journeys` with PYTHONNOUSERSITE=1 (the `tests` name is
shadowed by a stray user-site package) and reads the report file. The env-freeze
race documented at `probes/journey.py:36` is deleted, not ordered around.

**`feat/uxjourneys-rehome` @ `44b3a6754` (.wt-gpuops) implements the REJECTED
`packages/uxjourneys` design and must not land as-is.** Its gates were green on
its own design, so nothing is broken — it is simply superseded. `.wt-gpuops`
should stand down on this card rather than continue; porting its per-flow
findings onto `2beb966b0` is cheap because the file moves are renames (R092–R100).

`feat/rehome-uxjourneys` is empty (no diff vs origin/main) — delete it.

Merge order vs `test/journey-content-gate` (`79b788b95`): land the rehome
FIRST. That branch edits `domain/journeys/flows.py`, which this renames —
git's rename detection carries its `must_see`/`must_not_see` additions onto
`tests/ux_journeys/flows.py`; the reverse order forces a rename-vs-modify
conflict.

### Gates on 2beb966b0 (refactor/uxjourneys-rehome, PYTHONNOUSERSITE=1)

- broad: `pytest tests -q -m "not slow"` -> **3248 passed, 13 skipped, 190 deselected**, rc=0.
- targeted: **85 passed** (test_ux_flows_library + test_mole_probes + test_code_hygiene +
  test_domain_print_allowlist + tests/server/test_mole_router).
- ruff: clean on every touched file. py_compile: clean.
- real CLI from repo root, no PYTHONPATH: `python -m tests.ux_journeys --list` -> 13 flows rc=0;
  `scripts/run_ux_flows.py --list` rc=0; `--flow nope` rc=2; no-stack rc=2.
- every flow builds: 47 steps across 13 flows, avion resolved from packages/avion/src.
- mutation-verified 4/4 (import-instead-of-spawn, exit-1-as-error, PYTHONNOUSERSITE dropped,
  exit-2-as-success), each caught by its intended test. First pass caught two VACUOUS tests —
  an assertion satisfied by the ambient shell, and one that asserted run_probe()'s outcome
  (which discards sweep_error when a report loads) instead of _sweep()'s return.
- benchmark: n/a — no training/inference path touched (docs/tests/mole plumbing only).

Status stays **in_progress**: the work is committed on a branch, not merged to main.

## FINAL OUTCOME (2026-10-16) — this SUPERSEDES everything above

The user's sequence: "we don't need another library for our journeys!" →
re-homed to `tests/ux_journeys` (no new package) → "we don't need it".
**The harness is deleted outright**, not relocated. The rehome commit
`2beb966b0` stands in history; `8b77470d8` removes it and the seam it forced.

Deleted: `tests/ux_journeys/` (546 lines), `scripts/run_ux_flows.py`,
`domain/core/_internal/mole/probes/journey.py`, `tests/test_ux_flows_library.py`.
Mole keeps **system, gates, benchmarks, http, sse**. Site doctor keeps its
read-only contract test, reworded to guard the rule rather than the journey case.

**No uxjourneys branch may land in any form:**
- `feat/uxjourneys-rehome` @ 44b3a6754 (.wt-gpuops) — `packages/uxjourneys`; user rejected.
- `refactor/uxjourneys-rehome` @ 2beb966b0 — `tests/ux_journeys`; superseded by the cut.
- `feat/rehome-uxjourneys` — empty, delete.

The `domain/journeys` → `tests/ux_journeys` move is therefore **no longer a
merge-order constraint** against `test/journey-content-gate` (`79b788b95`) in
the "rehome must land first" sense — that branch's `must_see`/`must_not_see`
additions to `domain/journeys/flows.py` are moot now that the file is gone
either way. Whatever it still wants has to be re-expressed against the spec
text in `docs/UX_FLOWS.md`, not the deleted registry.

### Evidence on the cut (2026-10-16)

- `8b77470d8` backend — harness, probe, forwarder, its tests, docs. Net −1454.
- `a2d494fe0` frontend — /mole types, fixtures, RunMoleButton docstring, page subtitle.
- targeted: **81 passed** (test_mole_probes + test_mole_router +
  test_site_doctor_probe + test_code_hygiene + test_domain_print_allowlist).
- web: **31 passed** (5 mole files) · `tsc --noEmit` clean · `ruff` clean on every
  touched file.
- source-wide sweep: zero surviving references to the harness, its report path,
  or its env vars (remaining `SLO_WEB_URL` hits are pre-existing live-stack
  suites unrelated to it).
- status stays **in_progress** until the branch is merged to main.
