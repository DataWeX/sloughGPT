---
id: 20260928_067_journey_library
status: doing
title: "Card 067 — Journey library: spec anchors, contract tests, browser/live gating (merge pending)"
tags: [testing, journeys, browser, pytest]
sprint: current
gh: null
---
# Card 067 — Journey library

## Delivered (branch `feat/journey-library`, worktree sloughgpt-router-wiring)

1. `scripts/run_ux_flows.py` — `Flow.spec` anchors: each of the 13 flows carries
   its verbatim `docs/UX_FLOWS.md` heading; `--list` prints it; `_chat_flow(spec=...)`.
   Formatted (`ruff format` — was unformatted on main too).
2. `packages/core-py/tests/test_ux_flow_registry.py` (new, 4 tests): unique ids,
   spec verbatim-in-doc, bidirectional `### N.` number match, non-empty labels.
3. `packages/core-py/tests/test_ux_flows_browser.py` (new): module fixture runs
   all 13 FLOWS once (Firefox, `RUNNER.run`), per-flow parametrized asserts;
   `pytestmark = [browser, live]`; skips if stack down. Runner loaded via
   importlib with `sys.modules[name]=mod` (dataclass requirement).
4. Marker gating: `browser`/`live` registered in BOTH pytest.ini files;
   addopts now `-m "not slow and not browser"` (root) and core-py equivalent.
   `test_user_journeys.py` + `test_comprehensive_training_journeys.py` marked.
5. Deprecation headers (`run_journey_tests.py`, `domain/testing/_internal/journeys.py`)
   + `packages/avion/tests/test_journeys.py` pointer + `docs/ROADMAP.md:30` line.

## Concurrent-session reconciliation (main's `30fed5792`)

While this card was in flight, the concurrent session landed a journey
consolidation: deleted the whole legacy `domain/testing/` harness (fake-green
`passed=True`), deleted `run_journey_tests.py`, deleted 5 duplicate journey
test files (incl. `test_comprehensive_training_journeys.py` — my marker target),
rewrote `test_user_journeys.py` (route-level Playwright suite, env URLs), and
pointed ROADMAP at `run_ux_flows.py` as THE spec runner.

Merge `main → feat/journey-library` (commit `6e06a9d25`) resolved:
- **Accept deletions** of the 3 files I had modified (deprecation/marker edits moot).
- `pytest.ini`: main's pythonpath (`../chargectl/src`) + my `-m "not slow and not browser"`.
- `test_user_journeys.py`: their env-based `BASE`/`API` + my `pytestmark = [browser, live]`.
- avion docstring: kept "(the spec-linked registry)".
- ROADMAP auto-merged both intents (line 30: "13 spec-linked … browser/live opt-in").
- Kept my unique pieces: spec anchors (main had none), both new test files
  (main had none), root pytest.ini markers.
No duplication: single spec runner story holds — `run_ux_flows.py` = library,
`test_ux_flows_browser.py` = pytest gate, `test_user_journeys` = route smoke.

## Gates (all on `feat/journey-library`)

- ruff: all checks passed (touched files).
- registry + avion: **7 passed**.
- marker smoke: browser files under default addopts → **exit 5, 0 ran** (deselected).
- **live 13-flow suite: EXIT:0, 13 dots, 0 skips — three separate runs**
  (pre-merge `live5`, post-resolution `live6`, final branch `4d02d8a21`).
- Full suite gate: **543 unique FAILED** across all 790 files + cli, 0 browser.
  Recovery: batches 03/07 (reporter INTERNALERROR) → reporter-less reruns (13+39);
  batches 11/13/18 (deterministic kills: OOM-ish 137 at 69%/62%, 1800s timeout at
  62%) → per-file `-v` runs with `-o addopts=` + `-k 'not test_logs_follow'`
  (nodeid `--deselect` never matched — rootdir-relative nodeid mismatch; this
  is also why original batch.13 timed out). Gap files closed via retry2 (2/2 EXIT:0/1).
  Old baseline 646 unique (incl 116 browser) → 543 now; only 2 of 543 sit in
  files main has since deleted ⇒ merged-state expectation ≈541.

## Environment findings (this card)

- The concurrent session repeatedly kills/gets killed: tool interrupts take down
  detached processes too (supervisors, uvicorn, vite all died repeatedly).
  Pattern: relaunch + resume markers (`.done`, `FILE_DONE`) everywhere.
- `pkill/pgrep -f` self-match is relentless: any plain occurrence of the target
  string ANYWHERE in the same command kills own shell (bracket-trick insufficient
  if the plain name also appears, e.g. in a later `sed path`).
- Journey stack on non-default ports needs: `SLO_CORS_ORIGINS` incl 5177 AND
  `NEXT_PUBLIC_API_URL=http://localhost:8001` for vite (client calls API direct,
  not through proxy); boot overlay = `[aria-label="Startup progress"]`,
  dismisses <5s when healthy.
- `apps/web` in this worktree needed BOTH `node_modules` symlinks (app + repo
  root, deps hoisted to root) — untracked (gitignore misses symlink form).
- `/tmp/opencode` wiped repeatedly mid-session → logs to `~/opencode-logs/`.

## Gate evidence (exact commands + numbers)

| Gate | Command | Result |
|---|---|---|
| ruff | `ruff check scripts/run_ux_flows.py packages/core-py/tests/test_ux_flow_registry.py packages/core-py/tests/test_ux_flows_browser.py packages/core-py/tests/test_user_journeys.py packages/avion/tests/test_journeys.py` | All checks passed |
| registry contract | `pytest packages/core-py/tests/test_ux_flow_registry.py packages/avion/tests/test_journeys.py -o testpaths= -p no:cacheprovider` | **7 passed** (4 registry + 3 avion) |
| marker smoke | `pytest packages/core-py/tests/test_user_journeys.py packages/core-py/tests/test_ux_flows_browser.py` (default addopts) | **exit 5**, 0 ran — browser suites deselected |
| live journeys ×3 | `SLO_API_URL=http://127.0.0.1:8001 SLO_WEB_URL=http://127.0.0.1:5177 pytest -m browser packages/core-py/tests/test_ux_flows_browser.py` | **EXIT:0, `.............` 13 dots, 0 skips** — runs: `live5` (pre-merge 2eeaf1095), `live6` (post-resolution 6e06a9d25), final (4d02d8a21) |
| full suite | `driver3.sh` (20×40-file batches + cli) + recovered re-runs | **543 unique FAILED**, 0 browser, full 790-file coverage |
| card 070 (merged) | contract 14/14 · `tests/server` 1892/4 · root 1145/9F/5E · ruff clean | exact pre-existing baselines |

## Coverage-recovery chronology (why 543 is trustworthy)

Driver marked `.done` **regardless of exit code** (`touch "$b.done"` after every run),
so crashed batches looked complete. Recovery pipeline:

| Batch | Crash | Root cause | Recovery |
|---|---|---|---|
| 03 | EXIT:3 INTERNALERROR | reporter's `open(TEST_RESULT_LOG)` collides with `test_cpu_topology`'s builtin-open patch (call-phase report) | reporter-less rerun → 13 FAILED, EXIT:1 |
| 07 | EXIT:3 `_FakeTty` leak | `test_io*` tty-patch leak hits reporter phase | reporter-less rerun → 39 FAILED, EXIT:1 |
| 11 | EXIT:137 @69% | deterministic external SIGKILL at same % twice (OOM-curve or test kills pg) | per-file `-v` runs |
| 13 | EXIT:124 @62% | **`--deselect ...test_logs_follow` never matched** (nodeid is rootdir-relative: `tests/test_shell_repl_more.py::…`, not `packages/core-py/tests/…`) → the known hang ran → 1800s timeout | per-file + `-k 'not test_logs_follow'` |
| 18 | EXIT:137 @30% | same deterministic kill | per-file `-v` runs |
| gap files | `test_quantization_benchmark_e2e` EXIT:137, `test_vm_benchmark` EXIT:143 (600s cap) | heavy benchmark files | `retry2` with 1200s cap → 2/2 complete (EXIT:0/1) |

Per-file runs needed `-o addopts=` (wipes core-py `-q`, which cancelled my `-v`
into net-verbosity 0 → no FAILED lines) **plus** re-added `-m "not slow and not browser"`
(first attempt without it would have inflated totals with slow/browser failures).
Totals union: 15 clean batches' `.out` FAILED + 03/07 reruns + `vfiles.11/13/18` +
`retry2.log` = **543 unique nodeids** (deduped; only 2 in files main deleted ⇒ ≈541 post-merge).

## Runbook (resume this card fast)

```bash
# stack (ports are non-default on purpose; concurrent session owns 8000/5175/5176)
cd /home/mana/Documents/sloughgpt-router-wiring
SLO_CORS_ORIGINS="http://localhost:5177,http://127.0.0.1:5177" \
  setsid nohup .venv/bin/python -m uvicorn apps.api.server.main:app --host 127.0.0.1 --port 8001 > ~/opencode-logs/api8001.log 2>&1 &
cd apps/web && NEXT_PUBLIC_API_URL=http://localhost:8001 \
  setsid nohup node node_modules/vite/bin/vite.js --port 5177 --strictPort > ~/opencode-logs/vite5177.log 2>&1 &
# both required: client calls API DIRECT (no proxy hop) → missing CORS origin = overlay stuck

# live suite (detached; tool interrupts kill foreground runs)
bash ~/opencode-logs/live-suite.sh        # writes browser-live*.log, EXIT:0 + .done marker
# healthy boot: [aria-label="Startup progress"] unmounts <5s; empty text + >8s = investigate

# totals
bash ~/opencode-logs/totals.sh | sort -u  # + retry2.log union (see table above)

# crash-safe batch pattern: per-file, -v (incremental lines survive kills),
# FILE_DONE markers for resume, ALWAYS carry -m filter + -k for hang tests
```

## Relatedness vs card 070 (EM question, answered in writing)

**Unrelated at feature level; one pattern at the contract layer.** 070 pins
API.md claims to `schemas/common.py` + route AST; 067 pins UX_FLOWS.md headings
to `FLOWS` + browser execution. Both: (1) contract-test the docs↔code seam,
(2) shrink/delete stale baselines (56-entry `_internal` set / legacy journey
harness), (3) live under the same pytest marker + `-m` deselection regime.
Shared runtime code: **none** (`common.py` untouched by 067; journey code
untouched by 070).

## File inventory (branch diff vs pre-merge base)

**Mine (unique, kept through reconciliation):**
- `scripts/run_ux_flows.py` — `spec` field/anchors (13×), `--list` output, formatted
- `packages/core-py/tests/test_ux_flow_registry.py` — **new**
- `packages/core-py/tests/test_ux_flows_browser.py` — **new** (browser+live)
- `pytest.ini` (root) — browser/live markers + `-m "not slow and not browser"`
- `packages/core-py/pytest.ini` — same markers/filter merged with main's pythonpath
- `docs/ROADMAP.md:30`, `packages/avion/tests/test_journeys.py` docstring

**Mine but superseded by concurrent session's `30fed5792` (deletions accepted):**
- `scripts/run_journey_tests.py` (deprecation header), `domain/testing/_internal/journeys.py`
  (deprecation header), `packages/core-py/tests/test_comprehensive_training_journeys.py` (marker)

**Theirs, merged with my marker:** `packages/core-py/tests/test_user_journeys.py`
(their env-URL rewrite + `pytestmark = [browser, live]`)

## Risks / follow-ups

1. `--deselect packages/core-py/tests/test_shell_repl_more.py::TestCmdLogsExplainV2::test_logs_follow`
   in `driver3.sh` **has never worked** (rootdir-relative nodeid) — fix driver or
   switch to `-k 'not test_logs_follow'` before trusting future batch runs.
2. Driver marks `.done` regardless of exit code — future gate runs MUST check
   `EXIT:` lines (expect EXIT:0/1 only; 3/124/137 = partial).
3. `node_modules` symlinks in this worktree are untracked — gitignore pattern
   `node_modules/` (trailing slash) misses symlink entries; add `node_modules`
   if symlinks become standard worktree setup.
4. Concurrent-session kills are ambient: never foreground long runs; always
   detached + resume markers; `/tmp/opencode` gets wiped → `~/opencode-logs/`.
5. Old dev-note `2026-09-28-api-envelope-contract.md` (card 070) is the companion
   record for the merged card — keep both, no third doc.
6. Expected merged-state failures ≈541 (2 of 543 live in files main deleted);
   re-run only the affected subset post-merge instead of the full 790.

## Report log (supervisor SOP)

- `2026-09-29 00:35Z` final-branch gates completed: contract 14/14, tests/server 1892/4 (exact
  baseline) on `4d02d8a21`; driver3.sh deselect fixed (`-k 'not test_logs_follow'`, both sites).
  Full matrix green — only the main ref-move remains.
- `2026-09-29 00:30Z` tranche report delivered in-session: shipped 070+AGENTS.md
  section+067 gates; relatedness verdict given; blocked-merge options (a/b/c)
  put to EM. Awaiting EM decision to close 067.

## BLOCKED — final step (needs EM decision)

`git merge feat/journey-library` into main requires the `sloughgpt-api-std`
worktree (holds `main`) — parked dirty with 29 staged files by the concurrent
session (their revert/fix operation on the journey consolidation + training
work). Not touching their state. Options for EM: (a) wait for their commit,
(b) authorize `--autostash` merge, (c) move main to a clean worktree.
