---
id: 20261008_eslint_flat_config_testweb
title: "ESLint flat-config lane (card 9729733c option 1) + test-web lane (card 3f7c9b21) — PR #130"
status: done
date: 2026-10-08
tags: [lint, ci, frontend, eslint, test-web]
---

# ESLint flat-config + test-web lanes — PR #130

**Kanban:** cards `9729733c` (wip, lane owner) + `3f7c9b21` (wip, folded in) +
baseline `ca647de2` (wip, ledger — appended per sync).
**PR:** https://github.com/DataWeX/sloughGPT/pull/130
**Branch:** `fix/eslint-flat-config` (worktree `/tmp/opencode/wt-transport`).

## Scope

- **Card 9729733c option 1 (user ruling):** hand-translate strui + sdk-ts from
  legacy `.eslintrc.json` to ESLint 9 flat config. No new npm deps.
  `apps/web` stays legacy (deliberate `eslint ^8.57.0` pin).
- **Card 3f7c9b21 (test-web, never-passed bucket):** un-gate + green the
  `test-web` CI job.

## Root causes fixed

1. **Flat migration**: `packages/eslint-config/flat.mjs` = pure factories
   (zero `require()`s) because npm installs the shared config as a `file:`
   symlink and CI does an isolated per-package `npm ci` — plugin deps must
   resolve from the *consumer's* tree. Consumers ship ESM
   `eslint.config.js` (`"type": "module"`) that injects them.
2. **sdk typecheck broken 26 days**: `tsconfig` `extends` `../../` →
   `../../../` (f60c71a60 wrote a root `tsconfig.base.json`, sdk pointed one
   level short → `TS5083` everywhere; CI never noticed — lint died first).
   `noEmit: false` (publishable `files: ['dist','src']`).
3. **Strict fallout in `client.ts`**: duplicate `getSecurityKeys` (removed the
   FIRST — JS later-wins ⇒ runtime byte-identical), 8 untyped
   `this.request(...)` → `<any>` matching each section's style;
   `useSloughGPT` type-import split. `cn.test.ts` Boolean() wrap for eslint
   9's `no-constant-binary-expression` (invisible to local 8.57).
4. **test-web exit-2-since-Aug** (corrected diagnosis — original card text
   claimed source react errors; the job never reached rule checks):
   - `npm ci` in a workspace subdir climbs to the workspace ROOT and installs
     there; the job's split (apps/web, then packages/strui) meant the strui
     tree **wiped the web tree** — `next` vanished → `eslint-config-next`'s
     parser could not resolve `next/dist/compiled/babel/eslint-parser`.
     Reproduced byte-for-byte in a scratch replica; one root
     `npm ci --ignore-scripts` validates the root lock (906 pkgs) and
     resolves the whole chain. `ci.yml` already carried this exact lesson
     (comment dated 2026-10-05) — `ci_cd.yml` now matches it.
   - `consistent-type-imports` (error in `index.js`) would crash at rule
     load: `eslint-config-next` injects a parser that does not forward
     `@typescript-eslint` services → parser pinned at chain level in
     `next.js` (covers `apps/web` + `apps/lib`).
   - last 2 errors = `no-require-imports` on legitimate CJS configs →
     JS-files override in `index.js`.
   - Aug-era react source errors already fixed on main.

## Evidence

- Local (mirrors CI steps): apps/web lint **0 errors** / tsc clean /
  vitest **8076/8076** / `next build --webpack` exit 0; strui 0 errors /
  902 tests; sdk 0 errors / 227 tests.
- CI run 37736822261 (head 21c84730f): test-strui ✅ test-sdk-ts ✅
  Contract/lint/test-root/test-sdk-py/test-server ✅; test-web ❌ = pre-fix
  head (expected).
- Post-merge local: board union 948 cards chain-verified; merged journey
  file 49 passed/58 skipped + ruff clean; app-planner 203/203
  (PYTHONPATH = `$PWD:$PWD/packages/mogdb/src:$PWD/packages/app-planner/src`).

- **CI-exact replica replay (pre-push proof):** full `git archive HEAD` into a
  scratch dir + fresh root `npm ci --ignore-scripts` = byte-for-byte job
  layout (zero version drift vs the shared install). All four test-web
  steps green: eslint 0 errors / tsc clean / `next build --webpack` exit 0
  (native modules resolve fine under `--ignore-scripts`) / vitest initially
  8074/8076 — two load-races that the worktree run (8076/8076) never
  showed: MemoryCard's switch clicked while `disabled={stats === null}`,
  TokenTreeMergesCard's lineage DOM asserted before React commit. Both
  de-raced (`e2198e7c0`, wait-for-readiness) → replica full rerun
  **8076/8076 exit 0**. Lesson: replicate the CI install *layout*, not just
  the steps — shared long-lived node_modules hides timing defects.

## Lane queue (2026-10-09, 07:20Z) — training-ui-slim approved

User verdict: training UI too heavy + too much code + too cluttered.
Measured: 13.7k impl/5.7k test lines, runs page 825, recharts eager (2
dynamic imports total). Approved 3-phase approach (card `6b14be03`):
(1) weight = next/dynamic recharts consumers, (2) size = split runs/job
pages behavior-preserving, (3) clutter = home aligned to UX_FLOWS Train
My Model with FoldSection. Sequencing: PR #130 merges first, then branch
off main; three separate PRs. Meanwhile PR #130 hit the recurring board
conflict again (their Run5 close `bf0be96b9`) — union-resolved at merge
`6e9329c76` + main's 3-commit chip series absorbed with a third
format-only ruff fix (`305fb10d5`, their `023d35842` unformatted →
escalation card `ca7d10ff` pattern confirmed 3×).

## State (2026-10-09, 05:45Z) — PRESENTATION HEAD

**`e05f30d7c` = the presentation head**: dispatch `37889085874` green on
every job (test-web ✅ lint ✅ Contract/root/sdk-py/sdk-ts/server/strui ✅),
ci.yml `37889052038` = 7/7, main 0 commits ahead (their 15 absorbed at
CI-exact ruff scope → post-merge main heals lint+strui+sdk-ts+test-web).
Escalation card `ca7d10ff` filed (ruff-format process gap). Only
informational test-core-py still running — under timeout=120 it paces to
~40 min, no 6h hangs. **timeout-minutes 120 validated on data**: test-web
job durations across the 4 green runs = 12.2 / 11.7 / 7.5 / 7.6 min
(6–10% of budget) — 10× headroom, the bound only ever fires on hangs.
PR #130 = MERGEABLE, 24+ commits, presentation
next.

## State (2026-10-09, 05:30Z) — FULL GREEN ROW + absorb

**Presentation head `892086434` = every job green**: dispatch `37887369174` +
PR `37887334910` both show `test-web: success` (second-ever, now on the final
head) + lint/Contract/root/sdk-py/sdk-ts/server/strui all success; ci.yml
`37887334929` = 7/7. Main meanwhile moved +15 (shell-repl chip series
`04c74046a`) — absorbed locally: merge + ruff-format fix for its 2 files at
CI's exact scope (`ruff==0.16.8`, `packages/core-py/ + apps/api/server/` →
1040/1040 formatted, 4156 tests collect clean). Process finding: parallel
session commits Python unformatted under the CI pin twice now
(`42dea1635`, `04c74046a`) → escalation card candidate. Push absorb →
final gate → present PR #130.

## State (2026-10-09) — PUSHED

Gate 37736822261 finished at the 6h ceiling (test-core-py cancelled
12:18:33Z, zero logs; everything else green on that head). Sequence
executed: `origin/main` merged fresh (`0f2e962fc`, board 4-way union,
chains clean) → **one push of all 45 commits** → auto `pull_request`
events fired for both workflows (the flakiness cooperated) + manual
dispatch. Results on `0f2e962fc`: **test-web SUCCESS — first green in
CI history**, ci.yml **7/7**, strui/sdk/root/server/sdk-py/Contract green.
New red found: `lint` = main's own pre-existing ruff-format gap
(`42dea1635` committed `test_journey_playwright_capability.py`
unformatted — main run 37769605911 red too) → format-only fix
`965032f5a` (tests 2/2) pushed; fresh gate `37886673550` + PR
`37886652068` + ci.yml `37886652060` running, watcher `ci-watch7.txt`.
Merging PR #130 will also heal main's lint red.

## State (2026-10-08)

Local-only, held back for **one push** after gate run 37736822261
completes (concurrency `cancel-in-progress`): test-web fix `22df2bc58`,
test de-race `e2198e7c0` (replica-caught load races), shared-index
registration `9d356b6c9`, board/journal syncs (`17645e920` `2411af3b9`
`64f46af62` `eeaa1785d` `d17135402` `ba9f038ca`), and two origin/main
merges — `a8e971bef` (+15) and `7c173da45` (+14: pytest `timeout=120`
default, conftest `parents[3]` fix, 5473eb90 closed on main). Board = 948
cards, `verify_chain` clean after every sync.

On gate completion: merge `origin/main` fresh (kanban churn lands every
10–30 min) → ONE push → `gh workflow run ci_cd.yml --ref
fix/eslint-flat-config` (PR events are flaky) → verify **test-web green**
(+ strui/sdk/other jobs) → request merge approval. `test-core-py` =
informational (chip lane); CI skips the journey suite entirely (no
playwright in the CI pip list); local journey reds while another
session's `:3000` dev server runs = environmental (ledger-documented).
The `timeout=120` default should stop CI stragglers from stalling the
run toward the 6h ceiling.

## State — 2026-10-09 ~11:10Z (training-ui-slim Phases 1+2 built during gate freeze)

- **Phase 1 extended** (`b4bf860bf` on `feat/training-ui-slim`): +TrainingHistoryView+test, TurboCard deleted (803 lines, zero repo-wide refs). Training-tree total removed: **3,620 lines / 19 files** across both deletion commits (5f711e127 + b4bf860bf). Training tests 244/244, tsc clean, eslint 0 errors.
- **Phase 2 done** (`043bd2ea9` on `feat/training-ui-slim-split`, separate PR): runs page 825→240, job page 555→180. Extraction: `hooks/useTrainingRuns.ts`, `hooks/useTrainingJob.ts` (job action closures were inline JSX), components RunList/RunDetailsPanel/JobStatusCard/JobLossCard/JobDetailsCard/training-run. Proof: **full web suite 841 files / 8,076 tests green**, +18 new seam tests (card→panel, notes save, bulk select, compare retarget-clears-result, tag entry, badge map, resume routing, try-in-chat), tsc + eslint clean. Test-mock fixes learned: Card mock must forward `onClick`; 'gpt2' text is ambiguous (card vs filter option) → click 'converged'.
- **Gate 0220048b7**: 7/8 green (lint/root/server/sdk/strui/web/Contract ✅), **test-core-py still in_progress** → freeze holds (watch14 `sh_1200aa4a` + miner armed). PR shows CONFLICTING (main +4: packaging fix b53487b3b pyproject/setup.py + kanban) → round-8 absorb rides the SAME push once core-py completes.
- Kanban 6b14be03 updated with both phase results (local commit on lane).
- Awaiting user: PR #130 merge approval; Phase 3 product sign-off; 3 PRs to open post-merge (A=deletions, B=splits, C=Phase 3).

## 2026-10-09 (post-restart round: inventory BANKED + round-8 absorb)

- Server restarts wiped /tmp (both worktrees + helper scripts + watcher outputs) and killed the
  watcher + full-suite run #2. Branch refs live in the main repo .git — nothing lost.
- **Gate run 37908614696 COMPLETED (5h58m, natural exit 1): first-ever complete test-core-py
  inventory — 298 failures / 87 files, coverage 97.08% PASSED the 50% gate.** Root-cause buckets
  written into ledger card ca647de2: assertion/logic 147 (much of it broken CLI/TUI output capture
  on runner), API-drift AttributeError 82 (43 = _CUDAAccelerator old-API in gpu_accelerator),
  response-shape KeyError 21, host-tooling missing 20 (lscpu/sysctl), moved module 11, GPU env 10,
  exception-contract 7. Next-feature candidates listed on the card. Raw log = GitHub job
  113748466177. Freeze lifted after this.
- Rebuilt /tmp/opencode/wt-transport (fix/eslint-flat-config) + 3 node_modules symlinks.
- Discovered our side's board had LOST the 2 trailing meta lines (schema header +
  slot_history/tray_roots) to an earlier resolver; theirs still had them. Rewrote
  /tmp/opencode/board_union.py: 3-way union with id-less meta line preservation (ours-header-wins
  + union of extra meta). Both meta lines restored; chain verified.
- Round-8 absorb: main +15 through 76f61cc71 (logs-unification CLI fold monitor→logs, header
  screenshot tests, trait-weights test sync, packaging egg_info fix). Board-only conflict, union
  clean (their_added=0, BOTH=25 = chain artifacts only + my ca647de2 edit), ruff CI-parity 1040
  files clean. Merge c431629f0.
- Their new web screenshot tests are CI-unverified on main (their runs in_progress/cancelled) —
  local test-web parity (eslint+tsc+vitest+next build) running on the merged tree before push.
- GOTCHA (local≠CI eslint): worktree lint via shared root node_modules loads the MAIN checkout's
  @sloughgpt/eslint-config (Sep-12 symlink), not the branch's — my parser-pin fix in
  packages/eslint-config/next.js is invisible locally, producing 78 phantom errors
  (no-extra-semi/triple-slash-reference across 68 test files) that CI never sees. Same files,
  same lock versions, only the config copy differs. CI = the verdict for eslint; tsc/vitest/
  build are symlink-safe (no config-copy dependency).
- Merged-tree vitest: 840/841 files, 8075/8076 tests passed — 1 unidentified failure (tail
  capture cut the failure block). Merge touched zero apps/web files and 0220048b7 was vitest-green,
  so suspect environmental flake (heavy parallel load during run); re-running with full capture
  once next build completes.

## 2026-10-10 (phase 3 executed; round-9 absorb; merge-ready)

- Round-9 absorb: main +2 (incl. ffdb5151 dup-close adjudication — theirs won the union via
  newer updated_at). Board-only conflict, union clean, meta preserved, ruff parity 1040 clean.
  Merge 132bd9156 pushed; PR #130 conflict CLEARED (MERGEABLE/UNSTABLE). PR run 38008988902
  watching; user pre-authorized: merge when the 7 required jobs are green (test-core-py
  informational — inventory already banked, so the superseded 37976333741 run was expendable).
- PHASE 3 SIGNED OFF + IMPLEMENTED on feat/training-ui-slim-phase3 (7c84eb87b, stacked on
  slim-split): FoldSection "Show advanced options" (default-closed) wraps Epochs/Batch/LR,
  LoRA, Tokenizer, Resume-checkpoint in ConfigureStep. ASSUMPTIONS (user skipped the two
  sub-questions; conservative defaults): Base model picker STAYS in main flow; TrainStep recap
  rows KEPT; hpErrors render OUTSIDE the fold (visible-while-collapsed, gates Next). 4 new
  tests, 12/12 green targeted; tsc + training-dir suite queued behind the slim-split full run.
  Card note deferred to the post-merge board flow (6b14be03 lives on the lane board only).
- Durable artifacts (survive /tmp wipes): ~/.local/share/opencode/lane/{board_union.py,
  pr-a-body.md, pr-b-body.md}; PR bodies updated (sign-off noted in PR B body).

## 2026-10-10 — LANDED (lane closed)

- **PR #130 merged: a96ceecdc (2026-10-10T01:03:41Z).** Gate 38008988902: all 7 required jobs
  green (Contract/lint/test-root/test-sdk-py/test-sdk-ts/test-server/test-strui/test-web);
  test-core-py informational — first-ever completed inventory banked in ca647de2
  (298 failures / 87 files / 8 root-cause buckets, coverage 97.08% passed).
- Cards 9729733c + 3f7c9b21 moved to done with evidence (board commit on
  feat/training-ui-slim, riding PR #132's push). This journal note: status wip → done.
- Follow-on lanes already in flight: PR #132 (training slim Phase 1, deletions),
  Phase 2 stacked (slim-split suite running), Phase 3 implemented on
  feat/training-ui-slim-phase3 (7c84eb87b, sign-off 6b14be03).

## 2026-10-10 02:38Z — PHASE 1 LANDED (PR #132 merged 3b1056fe5)

- PR #132 fully green (15 required checks) and merged with user approval — 3,620 dead lines off
  main. PR #130 close-out rode its board commit; journal note closed (status done).
- Automation built (user ask): ~/.local/share/opencode/lane/watch-pr.sh — polls a PR to a
  terminal state (green / required-fail / conflicting) with informational-check policy as a
  parameter (test-core-py default; check NAMES drift between workflows — exclusionary rule so
  new/renamed checks block by default), --merge only with explicit flag, never auto-absorbs
  conflicts. Smoke-tested on #132 (caught a `[ "$var" ]` truthiness bug on first run — fixed).
- Stack rebased onto 3b1056fe5: slim-split = f0b102c7f (diff vs main = only the split commit),
  phase3 = 022a69922 (stacked). Suite #4 (full capture this time — suite #3's tail-pipe ate the
  2 failure identities) + phase3 tsc/dir-verify running. PR B targets main directly now.

### 2026-10-10 04:30 — suite #4 root-cause + PR #133 opened
- Full suite on rebased slim-split (`f0b102c7f`): **835 files / 8,020 tests, 1 failed** — `useTrainingJob > deletes the job…` 30s timeout (suite #3's "2 failures" was the pre-rebase tree; identities tail-cut there).
- **Root cause (solo-repro bisection)**: NOT load flake, NOT toolchain drift (react 19.3.0/RTL 16.3.3/jsdom 29.1.1 all match lock). `handleDelete` provably completes (delete+toast+push+`setShowDelete(false)` all observable) but `await act(async () => …)` never resolves — React 19.3 async-act deadlock flushing the handler's post-promise `finally` state update. V1–V5 variants: empty async act alone fine, sync act alone fine, combination hangs; act-free fire+`waitFor` passes in 317ms.
- **Fix**: `useTrainingJob.test.ts` observe via `waitFor` (pattern the passing resume tests use) — commit `76217e03b`; hook tests 13/13 green.
- **PR #133 opened** (base main): slim-split = `76217e03b`, split commit `f0b102c7f` + test fix. Watcher running (approval needed before merge). PR C body staged at `~/.local/share/opencode/lane/pr-c-body.md`.

### 2026-10-10 05:05 — PR #133 MERGED, PR #134 opened
- **PR #133 merged** → `0ea581a3f` (user-approved, merge commit): Phase 2 splits on main. 15/15 required checks green (test-core-py informational).
- Phase 3 restacked: fold commit rebased onto `0ea581a3f` → `0dd5d9291` (split+fix commits dropped as upstream; fold-only diff, 2 files +61/−7). Verify: ConfigureStep 12/12, tsc rc=0.
- **PR #134 opened** (base main): "expert controls behind a default-closed advanced toggle". Watcher running (approval before merge).
- Evidence draft: `~/.local/share/opencode/lane/card-6b14be03-evidence.md`.
