---
id: 9b8d051b
title: "Converge: two §7 AST reachability implementations + 5 stale baseline entries on merge"
status: done
board_id: Main
date: 2026-10-19
branch: feat/ast-union-land
tags: [ast, convergence, port, landing]
---

# Card 9b8d051b — land the stranded union on the main line

Full card id: `9b8d051b-a08c-4970-aff0-81fb290afafe` (board column: in_progress).

## LANDED 2026-10-19 (Gene batch of 3, done inline — subagent nesting blocked at depth 1)

- **Commit**: `23fb2ba3b` on branch `feat/ast-union-land` (worktree
  `.wt-astland`, cut from HEAD 681bca18e). Pathspec commit, 4 new files,
  2,285 insertions — `git show --stat` matches exactly, no foreign hunks.
- **Method**: port (not rebuild) — file state checked out read-only from
  `feat/ast-union-report` tip (= `1ae19db9c` + `042b53287`); base tool
  introduced at `f68470262` on `chore/ast-reachability` (cited in commit).
  `.wt-astunion` untouched. `scripts/check_docs_api_parity.py` NEVER edited —
  CALLERLESS_BASELINE export stays read-only (`git show` + stdlib AST).
- **Adaptation vs source**: `ruff format` only (4 files reformatted to this
  tree's ruff style); tests green before (45) and after (45).
- **Gates**: pytest `tests/test_module_reachability.py
  tests/test_export_callerless_baseline.py` = 45 passed; ruff check +
  format clean; export rc=0, 52 entries; safety gate PASS.
- **Union figures reproduced (delta reported, not fudged)** — on base HEAD
  681bca18e: `5 overlap / 14 mine-only / 2 theirs-only / 19 disagreement /
  26 unresolved`. Overlap identical to the source run
  (`domain.core._internal.{base,core,grounding,knowledge_graph_v2,math_rag}`).
  Deltas vs the source branch's 5/10/3/21/23 are base-tree drift: HEAD
  (feat/generation-merge) vs the branch base 0d350ad1e — the 52 baseline
  entries repartition (5+2+19+26 = 5+3+21+23 = 52) and the card's
  "theirs-side 47" is stable (47 = 52 − 5 overlap on both). Mine-only +4 =
  `domain.cognitive._internal.*` + `domains.knowledge._internal.*` dead
  clusters visible on HEAD's restructured tree.
- **5 stale baseline entries**: resolved by parity with `537416d29`
  (already deleted those files); documented in the commit — nothing deleted,
  `feat/generate-consolidation` baseline untouched (drops its 5 stale rows
  when that branch merges, per the card).
- **LANDED on main 2026-10-19**: fast-forward merge, main HEAD = `23fb2ba3b`.
  Foreman independently re-ran every gate (not trusting the report):
  - 45/45 tests green on `feat/ast-union-land` **and** 45/45 on the source
    branch `feat/ast-union-report` (same suite, so the port didn't weaken
    a single assertion — test diff vs source is `ruff format` re-wraps only,
    verified hunk by hunk).
  - Tool diff vs `042b53287`/`1ae19db9c`: 25+/22− after
    `--ignore-all-space`, every hunk a line re-wrap — zero semantic change.
  - `scripts/check_docs_api_parity.py` output on the branch is byte-identical
    to unmodified main (exit 1 with the pre-existing `controllers_lines`
    ALARM on both) → **zero parity delta**; file count in commit = 0.
  - Union arithmetic independently confirmed: 5+19+26+2 = 52 and
    5+21+23+3 = 52, so card's "theirs-side 47 = 52 − 5 overlap" holds on both
    trees; mine-only 10 → 14 is the `domain.cognitive.*` / `domains.knowledge.*`
    dead cluster on the restructured tree.
  - Fast-forward merge left the 8 foreign staged files byte-identical
    (staged-content sha256 `5c94f82e39c68f6d` before and after).
- Status: **done** — landed sha `23fb2ba3b`, all gates re-run by the foreman.
