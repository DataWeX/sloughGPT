---
id: f37f9516
title: "Compose the two §7 AST reachability tools — union, not reconciliation"
status: review
board_id: Main
date: 2026-10-18
branch: feat/ast-union-report
tags: [ast, reachability, dead-code, scripts, composition]
---

# Card f37f9516 — compose the two §7 AST reachability passes into one report

Full card id: `f37f9516-1c90-4b34-a91d-ca733f9c079b`.

## Dispatch (2026-10-18, Gene batch of 3)

- **Worktree**: `.wt-astunion`
- **Branch**: `feat/ast-union-report`, cut FROM `chore/ast-reachability`
  tip `1ae19db9c` (that tip already carries `--union-baseline` +
  8 union tests, commit message "Refs f37f9516" — port, don't rebuild)
- **Routed to**: general (scripts + AST)
- **Boundaries**: do NOT edit `feat/generate-consolidation` or its
  `scripts/check_docs_api_parity.py` (card ownership clause), do NOT merge
  either branch, do NOT touch `domain/generation`, do NOT resolve the
  "stale baseline entries" half (card 9b8d051b owns that).

## Measured baseline (card c708da74, done)

- theirs caller-based (feat/generate-consolidation, CALLERLESS_BASELINE 52
  entries): 47 callerless-but-unique
- mine root-based + twin (chore/ast-reachability --deletion-plan): 10
  dead-cluster members
- overlap 5 / mine-only 10 / theirs-only 47 — union for coverage,
  intersection for high confidence

## Evidence (2026-10-18, Gene — composed + verified)

Branch `feat/ast-union-report` (worktree `.wt-astunion`), stacked on
`chore/ast-reachability` tip `1ae19db9c` which ALREADY carried the
composition core (`--union-baseline FILE`, OVERLAP/MINE-ONLY/THEIRS-ONLY/
DISAGREEMENT/UNRESOLVED, 8 union tests — commit message "Refs f37f9516").
Port-don't-rebuild: extended, did not rewrite.

**New commit `042b53287`** (pathspec-limited, quoting card id):
- `scripts/export_callerless_baseline.py` — reproducible AST export of the
  caller-half input (`CALLERLESS_BASELINE`, 52 sealed entries) read-only from
  `feat/generate-consolidation:scripts/check_docs_api_parity.py` via
  `git show` — that file is never edited (card ownership clause). Regex over
  the source yields 21 entries / 8 unique of f-string prose; AST refuses
  non-string fragments instead of guessing. `--ref main` after their merge.
- `tests/test_export_callerless_baseline.py` — 6 tests: source order
  preserved, comment prose excluded, non-string refusal, missing-symbol
  refusal, sealed-52 ratchet (skips if the source branch is absent),
  cwd-independence.

### One combined report (end-to-end, the card's deliverable)
```
python scripts/export_callerless_baseline.py --out callerless.txt   # 52 entries, rc=0
python scripts/check_module_reachability.py --deletion-plan --union-baseline callerless.txt  # rc=0
```
→ `5 overlap 10 mine-only 3 theirs-only 21 disagreement 23 unresolved`
(theirs-side 47 = card number; overlap 5 = exactly c708da74's
`domain.core._internal.{base,core,grounding,knowledge_graph_v2,math_rag}`;
mine-only 10 = the dead-cluster members; deletion-plan safety gate PASS).
Logs: `~/.cache/gene-batch-20261018/{union_end_to_end,union_final}.log`,
baseline `…/callerless_baseline.txt`.

### Gates
- `ruff check scripts/export_callerless_baseline.py
  tests/test_export_callerless_baseline.py` → clean (reachability files clean
  at base too); `py_compile` → OK.
- `pytest tests/test_export_callerless_baseline.py tests/test_module_reachability.py -q`
  → **45 passed** (39 reachability incl. 8 union + 6 new).

### Boundaries kept
`check_docs_api_parity.py` untouched; no branch merged; no module deleted;
the "5 stale baseline entries" half left to sibling card **9b8d051b**
(overlap noted — coordinate at review). No docs added (the §7 doc
`docs/DOMAIN_CONSOLIDATION.md` referenced by the tool does not exist in the
tree; FORMAL_SPECS.md only names the tool in passing).

## Closure (2026-10-18, reviewed by parent session)

Reviewed and **closed to done — branch-landed, gates verified
independently**.

- Landed sha: **042b53287** on `feat/ast-union-report` (`.wt-astunion`,
  worktree clean), stacked on `chore/ast-reachability` tip `1ae19db9c`
  (which already carried `--union-baseline` + 8 union tests — extended,
  not rebuilt).
- Parent-session verification (not taken on the subagent's word):
  `ruff check` clean; `pytest tests/test_export_callerless_baseline.py
  tests/test_module_reachability.py -q` → **45 passed**; end-to-end
  `export_callerless_baseline.py --ref feat/generate-consolidation
  --path scripts/check_docs_api_parity.py` → **rc=0, 52 entries**;
  `check_module_reachability.py --union-baseline <file>` → **rc=0**,
  combined report generated.
- Main merge deferred: stacked on chore/ast-reachability; landing tracks
  through the usual LAND machinery once main is quiet.
- Sibling **9b8d051b** still owns the stale-baseline half (coordinate at
  landing; overlap acknowledged on the baseline-staleness seam).
