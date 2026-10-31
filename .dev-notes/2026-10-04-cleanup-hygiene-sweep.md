---
id: 20261004_cleanup_hygiene_sweep
title: "code-hygiene sweep: print() = 0 real violations + future-import debt 860 files"
status: done
date: 2026-10-04
tags: [cleanup, code-quality, python]
board_id: 7d5e5a7c-7b63-4208-97fe-62d8bfd857d6
board: 7d5e5a7c-7b63-4208-97fe-62d8bfd857d6
hash: 1803bed42adbc48ecde79252e8985484b4300016e0ed049fb16385dec62e62a4
---

FINDING (2026-10-04, cleanup-agent skill run — user said "continue building"). Six-pattern sweep over domain/ + packages/ + apps/ + scripts/ (2277 py files; note: `rg` is NOT installed — all ripgrep-based skill commands silently no-op; GNU grep used instead):

CLEAN (0 findings each): (1) f-string logger calls — 0; (2) bare except — 0; (3) TODO/FIXME/HACK/XXX — 0 (py+ts, pruned node_modules); (6) unused imports — 0 (ruff 0.16.8 --select F401 on domain + core-py). Prior gate-green campaigns left the codebase genuinely hygienic.

BUCKET A — print(): grep 115 → AST-classified 92 real calls (23 = docstring examples), concentrated in 7 files:
- 71x domain/shell/_internal/world_driver.py — TUI/world renderer: print IS its output channel
- 16x domain/multimodal/_internal/phoneme_encoder_cli.py — CLI tool stdout (its job)
- 1x inference_engine.py:613 `ENGINE_READY port=` — parent-process protocol handshake (legit)
- 1x logging/config.py:1316 — print(..., file=sys.stderr) warning during logging bootstrap (can't log while configuring the logger — legit)
- 1x input_device.py:104 readline_echo — terminal echo (TUI channel)
- 1x realm_live.py:83 argparse CLI run summary; 1x lm_eval_char.py:170 --json machine-readable stdout
VERDICT: 92/92 legitimate. ROOT CAUSE of the "finding": the skill's print() pattern is too blunt — it flags stdout-channel code (TUI/CLI/protocol/stderr) where print belongs; converting these to logger would be WRONG (stdout is the product surface). Pattern needs refinement (exempt CLI/TUI/stderr/protocol contexts).

BUCKET B — future-import: 860 of 1665 py files lack `from __future__ import annotations` (805 comply, ~48%): core-py tests 502/800, core-py/domains 208/268, domain/* ~55, misc ~95 (scripts/conftests/init). NO explicit mandate anywhere: PYTHON_FIRST.md:124 only EXEMPLIFIES it; ruff config (select E,F,W,I,N,UP,B,C4) has no require-future rule (ruff cannot enforce it). ROOT CAUSE: convention-by-example with zero enforcement → silent 52% debt.

STRUCTURAL NOTE: domain/ (588 py) and packages/core-py/domains/ (268 py) are SEPARATE trees — possible legacy duplication worth its own look.

PROPOSED ACTIONS (pending user scope decision):
1. Ratchet test tests/test_code_hygiene.py encoding the sweep patterns with proper exemptions (CLI/TUI/stderr/protocol prints allowed; py-scope pruned) so regressions can't creep back; also fixes skill pattern #4 wording.
2. Future-import batch — scope options: (a) domains only (domain/ + core-py/domains ≈ 256 misses, honors the skill's original target), (b) all 860 incl. tests, (c) defer/never (drop pattern from skill if convention is not actually wanted).
3. cleanup-agent skill: note `rg` absence (use grep fallback) + print-pattern exemption guidance.

ACCEPTANCE: user picks scope → implement on branch, gates = ruff clean + py_compile + backend test subset → merge; card → done.

SCOPE DECISION (user, 2026-10-04): "Ratchet + skill only" — future-import stays OPEN (options a/b/c above).

LANDED (2026-10-04): branch `feat/hygiene-ratchet` → commit `bb7162041` → merged main `a189d78eb`. Built:
1. tests/test_code_hygiene.py — 4 zero-tolerance tests (f-string logger incl. single-quote/logging. variants, bare except, task markers py+ts+tsx, library prints). Exemptions all reasoned: stdout-channel file allowlist (7 + journeys/runner.py), `__main__.py` entries, `__main__` guards, `file=` calls, FSTRING_EXEMPT for avion StructuredLogger (single-positional contract). Mutation-verified: planted violations fail all 4, revert green.
2. Real fixes the ratchet found ON MAIN (my first sweep had run on the root checkout's OTHER branch = branch skew, plus a regex missing single-quote/logging. variants): lr_schedulers.py:130 f-string → lazy %s. avion/session.py:95 initially "fixed" → BROKE 3 avion tests (StructuredLogger.info(message, **fields) takes no positional varargs) → reverted to f-string + reasoned exemption — the test run caught my own regression.
3. .opencode/skills/cleanup-agent/SKILL.md hardened: rg-missing tooling note (all 6 original commands silently no-oped → phantom zeros; scope-control discipline added), validated grep/ruff commands, context-aware print rule, ratchet-first workflow, future-import open-decision note.
4. tests/README.md registration.

GATES: 381 passed / 18 skipped / 0 failed (training + avion + ratchet subsets); ruff clean on all changed files; py_compile clean.

ENV FINDING (candidate doc fix): AGENTS' pinned PYTHONPATH omits `packages/mogdb/src` — tests importing apps/api/server/training/job_store.py (test_lora_ship_journey, test_training_router_kwds) die with ModuleNotFoundError unless it is added. Also: `rg` (ripgrep) is not installed machine-wide.

SCOPE PICK (user, 2026-10-04, question tool): (a) domains-only.

SCOPE REALITY CHECK: packages/core-py/domains tree NO LONGER EXISTS on main (dual-tree step-4 deletion) — the ~256 estimate came from the branch-skewed first sweep. Actual scope on main = domain/ tree: 124 of 603 files missing the import (grep counted 481 'containing the string' but 2 are comment/docstring mentions, 479 ast-verified real imports).

PLAN: (1) ast-based batch insert — verified on a rehearsal copy incl. edge cases (shebang stays line 1, BOM/CRLF preserved, docstring/other-future/bytes-expr/empty placements), 0 compile errors; (2) ratchet test #5 added to tests/test_code_hygiene.py enforcing domain/ (fixes ROOT CAUSE: convention-by-example with zero enforcement); (3) PYTHON_FIRST.md mandate line; (4) gates = ruff + py_compile + backend suite baseline-vs-post comparison (baseline invalid first attempt: 3 collection errors were MY PYTHONPATH missing packages/mogdb/src — known AGENTS gap; relaunched).


## LANDED — 2026-10-05 (scope (a) executed)

- Branch commit `9a1b4c492` merged to main as `4e7594a67` (api-std --no-ff): 126 files, +190/-16.
- Batch: `from __future__ import annotations` added to 124 domain files (added=124, broken=0); ruff --fix 18 (UP037 de-quote, UP045 Optional drop, I001); ruff domain clean; py_compile 125 files.
- Ratchet: `tests/test_code_hygiene.py::test_domain_files_carry_future_annotations` (FUTURE_ROOTS = domain + legacy packages/core-py/domains) red -> green; hygiene suite 5 passed; docstring moved future-import from NOT-enforced to enforced #5; PYTHON_FIRST.md mandate subsection 5.
- Backend suite (same worktree pre/post): post = 566 failed / 44937 passed / 89 errors. All 430 "NEW vs 04:09 detail2b baseline" bucketed: (1) GPU/hardware/native-engine ~250 — F-runs already present in pre-batch partial log (future_baseline_partial.log, same worktree, exact F-count matches e.g. mobile_router 16/16); (2) missing playwright dep ~89 — E rows pre-batch same positions (journeys/computer_use/e2e_training_trigger); (3) order/state-dependent 7 — pass in isolation on BOTH pristine-HEAD and post-batch trees; (4) system_router 4 — identical FAILED sets full-file pre vs post; (5) stale-baseline shape — detail2b ran 8.5k tests vs 45.7k (api-std tree state at 04:09). Annotation-eval signature errors (get_type_hints/ForwardRef/PydanticUserError): **zero** — batch risk class absent.
- Evidence: ~/.cache/sweep-logs/{future_post.log, future_baseline_partial.log, compare_out.txt, baseline_named.txt}.