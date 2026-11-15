---
id: 20261001_line_reconciliation_plan
board: df8ec757-d592-4010-bf65-0f776bca75eb
title: "Line reconciliation plan — main (A) vs aec2b (B)"
status: done
date: 2026-10-01
tags: [git, reconciliation, coordination]
---

# Line reconciliation plan — main (A) vs aec2b (B)

Study only. **No repo file was modified, no branch/ref/index was touched.** Git usage was
restricted to `rev-parse / merge-base / rev-list / diff / show / cat-file / log / grep /
cherry / merge-tree --write-tree / worktree list / status`. The only file written is this note.
Working scratch (merge-tree payload + parsed tables) lived in `~/.local/state/sloughgpt-gene/`
and was deleted at the end of the study so that this note is the single new file.

**Nesting note:** one `git`-specialist dispatch was attempted and rejected
(`Subagent depth limit reached (1)`), so per instruction the whole analysis was done directly
with my own shell/git tools; no retry was made.

## 0. Verified ground truth (re-measured, not assumed)

| Fact | Value |
|---|---|
| merge-base | `f27654e54` (2026-09-25) |
| Line A | `main` = `a1ebb7f2d` = `origin/main`, 208 commits past base (155 non-merge) |
| Line B | `aec2b37ec` → tip `ae6c5a1db`, 165 commits past base (155 non-merge), no remote ref |
| `git merge-tree --write-tree --messages main aec2b37ec` | exit 1, merged tree `e4657a605`, **75 conflicted paths** = 72 `(1,2,3)` content + 3 `(1,3)` modify/delete |
| `git cherry main aec2b37ec` | 10 B commits patch-equivalent in A, 155 unique to B |
| `git cherry aec2b37ec main` | 10 A commits patch-equivalent in B, 145 unique to A |
| Dirty state | main tree **560** porcelain entries; **33 worktrees** |

The 10↔10 patch-equivalent pairs are the same work cherry-picked both ways (import-dialog fix +
its 2 tests, chargectl ×4, planner verify ×2, `backup_critical.sh`, layout-overflow audit).
**Neither set contributes a single conflicted file** (all 10 touch `packages/chargectl`,
`packages/app-planner`, `packages/strui`, `scripts/` — none appear in the 75).

### 0.1 Caveat discovered while measuring — line B is **not frozen** at `aec2b37ec`

- This session's primary tree is **not on `main`**: `HEAD` = `feat/design-system-tables` @
  `8085f117e`, with `aec2b37ec` **and** `ae6c5a1db` as ancestors → HEAD is line B **+19 commits**
  (through 2026-10-01), 184 commits past base, and **489 tracked files dirty** on top of it.
  A is *not* an ancestor of HEAD (each line owns its own 145/155 unique non-merge commits).
- Consequence: the conflict inventory is ref-dependent. Re-running the same measurement against
  the newest B tip gives **`git merge-tree --write-tree -z main 8085f117e` → 78 conflicts
  (75 content + 3 modify/delete)**, tree `5b01fc86aa9` — the 75 in §1 plus:
  - `docs/routers.md` — A: `+1/-1` line; B(+19): **+761/-365** regenerated "code truth" table
    (commit `d57c68cbc`); a 697-row regenerated doc vs a 1-line A edit → regenerate after merge.
  - `docs/API.md` — A `+28/-8`; B `+2/-1` → union, take A's rows + B's delta.
  - `packages/strui/src/components/ui/dropdown-menu.tsx` — A `+65/-12` (context-hoisting refactor,
    drops `focusPrev` from the destructure in one branch); B `+104/-3` adds `style` to props →
    real signature union, take care (hook order / focus ring).
- Several other local branches are dated **2026-10-01** (`feat/shared-governance`,
  `feat/systemd-units`, `feat/workspace-search-q`, `fix/http-retry-amplification`,
  `fix/journey-test-gates`, `fix/tps-measurement`, `feat/magazine-palette-generated`,
  `feat/card-hash-chain`, `feat/conda-env`, `feat/ui-screenshot-tests`) → **the merge source must
  be chosen at merge time, not today**: either the scoped `aec2b37ec` (75 conflicts, matches this
  study) or a nominated B tip (≥78). Pick a *frozen* source ref before Phase 1.

## 1. Conflict classification — 75 files

Buckets used:
- **(1) SAME** — both sides made the same logical change; only formatting/tokens/naming differ.
  Resolution = take one side wholesale (side noted). No decision needed.
- **(2) DISJOINT** — both sides changed different things in the same region (or one side is
  empty). Resolution = mechanical union, keep both. No decision needed.
- **(3) TRUE** — same behaviour/contract, divergent designs. Needs a named decision.

**Counts: (1) SAME = 29 · (2) DISJOINT = 29 · (3) TRUE = 17 · total 75.**
The 3 modify/delete files all fall in bucket (3): in every case **A deleted the file and B
modified it** (stage set `{1,3}` = ours/A side missing).

### 1.1 Bucket (1) SAME — 29 files, mechanical, take-the-side noted

| File | Same change, difference | Take |
|---|---|---|
| `apps/api/server/routers/docstore.py` | line-wrap only (A wrapped, B single-line) | either (A) |
| `apps/api/server/routers/infer.py` | identical 503 `E_INFRA_REGISTRY` guard, wrap | either (A) |
| `apps/api/server/routers/inference.py` | identical 503 guard, wrap | either (A) |
| `apps/api/server/routers/security.py` | same auth/forbidden guard; A `status_code=403`, B `status=403` (B's kwarg is the TypeError A already fixed) | **A** |
| `apps/api/server/routers/tokens.py` | import wrap | either (A) |
| `apps/api/server/routers/user_adapters.py` | same rating→signal map, var name `feedback_signal` vs `signal` | either |
| `apps/api/server/routers/benchmark.py` | same import retarget ×3; `domain.` vs `domain.benchmark.` — **both spellings exist on A and on the merged tree** | either |
| `apps/api/server/tests/test_docstore_router.py` | line-wrap | either (A) |
| `apps/api/server/tests/test_datasets_router.py` | same cache-isolation fixture, `_isolated_cache_root(tmp_path,monkeypatch)` vs `_isolate_cache_root(monkeypatch,tmp_path_factory)` | either, **fix call sites with it** |
| `apps/api/server/tests/test_knowledge_router.py` | same cwd-dependent-400 fix: A `path=str(tmp_path)` (hermetic, test just wrote `sample.py` there) vs B absolute `…/routers` path; post-block assert is `len(results)>=1` for both | **A (hermetic)** |
| `apps/api/server/tests/test_support.py` | same readiness-dispatch stash: `_real_dispatch` vs `_ORIGINAL_DISPATCH` | must match bucket-2 `test_readiness_gate.py` |
| `apps/web/app/(app)/consciousness/dashboard/page.tsx` | `&quot;` vs `"` | either |
| `apps/web/app/(app)/consciousness/master/page.tsx` | `stroke="#6366f1"`/`"#22c55e"` vs `rgb(var(--chart-1/2))` ×2 | **B (design tokens)** |
| `apps/web/app/(app)/magazine/page.tsx` | `rgb(124, 82, 196)` vs `rgb(var(--primary))` | **B (token)** |
| `apps/web/components/phoneme/DailyGoal.tsx` | `catch { /* best-effort */ }` vs `catch {}` | either |
| `apps/web/components/phoneme/PhonemeTypingMode.tsx` | regex `[\s,-]` vs `[\s,\-]` (equivalent) + `border-green-500/20` vs `border-success/20` | regex either, **token B** |
| `apps/web/components/phoneme/PronunciationDictation.tsx` | `border-green-500/20` vs `border-success/20` | **B (token)** |
| `apps/web/components/tokenizer/TokenTreeVocabCard.test.tsx` | same race fix: `{timeout: 5000}` vs comment + no timeout | either (B reads better) |
| `apps/web/cypress/e2e/chat-inference.cy.ts` | `.to.equal(true)` vs `.to.be.true` | either |
| `apps/web/cypress/e2e/datasets-page.cy.ts` | `.to.equal(true)` vs `.to.be.true` | either |
| `apps/web/features/chat/components/panels/ChatToolPanel.tsx` | className order only | either |
| `apps/web/lib/http-client.test.ts` | same stale-while-revalidate flake fix: real timers with wide margins (A) vs `vi.useFakeTimers` (B) | **B (deterministic)** |
| `docs/policies/CONTRIBUTING.md` | `**_x_**` vs `_x_` markup + path `domain/ui/api_server.py` vs `packages/core-py/domains/ui/api_server.py` | **A (path matches deletion)** |
| `domain/training/engine.py` | 2 docstring wordings only — B's `+505 TrainingEngine`/`get_training_engine` merge cleanly elsewhere | either |
| `tests/server/test_audit_instrumentation.py` | patch target `apps.api.server.training.jobs_api.get_training_executor` vs `domain.training.get_training_engine` | depends on **D2** |
| `tests/server/test_benchmark_router.py` | same 11-line patch-target retarget | either (mirror router choice) |
| `tests/server/test_companion_router.py` | `@patch(CREATE_COMPANION_TARGET)` vs literal ×3, `respond.return_value` vs `Mock()` style ×6 | either (A is tidier) |
| `tests/server/test_souls_router.py` | fixture value `no_such_ckpt_404` vs `no-such-checkpoint-for-test` | either |
| `tests/server/test_startup_routers.py` | patch-target import wrap | either (A) |

### 1.2 Bucket (2) DISJOINT — 29 files, union, no decision

| File | A does / B does |
|---|---|
| `.kanban/board.jsonl` | A flips 3 card statuses; B adds/rewrites 111 lines of cards → **union, then per-id `updated_at`-newest wins** for the 3 shared ids; reconcile with notes (notes are source of truth), never hand-curate |
| `apps/api/server/tests/test_job_store.py` | A changes `_create` statuses; B adds `data_path`+`utc_now_iso()` **and** a 240-line `discard_live`/`set_live`/`store_row_to_job` suite (A side empty) |
| `apps/api/server/training/jobs_api.py` | A adds `rank`/`alpha`/`result`/`loss_history` fields; B adds `recoverable` field from `recovery_incompatibility(job)` |
| `apps/api/server/training/job_store.py` | A rewrites the recoverable→job mapping line; B wraps it in a `recovery_incompatibility` filter → union, **verify `recovery_policy` + `store_row_to_job` both exist post-merge** |
| `apps/api/server/tests/test_companion_router.py` | A resets 3 module caches; B's `_reset()` helper resets 2 → take A (superset) |
| `apps/web/app/(app)/training/job/[id]/page.test.tsx` | A: router mocks + failed-badge test; B: `mockRecover` + resume/"Try in chat" tests |
| `apps/web/app/api/planner/helpers.test.ts` | A: commented `try/catch` cleanup; B: `require('fs')` cleanup + `persists acceptance fields` test |
| `apps/web/app/(app)/training/runs/page.test.tsx` | A's `PageContainer` mock takes `subtitle`; B's renders `headerRight` → keep both props |
| `apps/web/components/consciousness/ConsciousnessNotificationsPanel.tsx` | A adds `import type { JSX }`; B inlines `type ReactElement` → union |
| `apps/web/components/monitoring/TrainingHistory.test.tsx` | A mocks `next/navigation`; B adds toast+recover mocks and 54 lines of Resume tests |
| `apps/web/cypress.d.ts` | A none; B adds `screenshotPage/Interaction/Element/Sequence` chainables |
| `apps/web/features/chat/components/panels/KnowledgeTab.tsx` | Link import (`@/vite/next-compat/link` vs `next/link` — **both targets exist on both lines, `next` is a dep**) + `p-2 pr-10` padding (A) + inline date vs `formatMonthDay` (B) |
| `apps/web/features/chat/components/panels/MemoryTab.tsx` | same Link choice + B's `max-h-60 overflow-y-auto` on the `<ul>` |
| `apps/web/features/chat/components/panels/MemoryItemList.tsx` | B adds `max-h-60 overflow-y-auto` + prettier split; A keeps one-line `<ul>` |
| `apps/web/lib/db.ts` | A adds `session.messages ?? []` guard + simple timestamp coercion; B filters non-object messages + validates `Date` → **union, keep both guards** |
| `apps/web/lib/training-controller.ts` | A requires `failed_at`, uses arrow; B makes fields optional + adds `completed_at`/`checkpoint_path`/`dataset`, uses `const self` → union fields, keep A's arrow |
| `apps/web/app/(app)/session/page.tsx` | A adds `?? []` null-guard on `inspector.session.messages`; B reflows the JSX → **keep A's guard** |
| `docs/PRODUCT_ENGINEERING.md` | A's table = planned deletions; B's table = landed status ("✅ removed", "collections alive") + Build Order wording → union: B's status + A's rows B doesn't cover |
| `docs/VM_CONSOLE.md` | A's corrected paths (`domain/shell/_internal/…`); B's table adds 5 rows (console session, `vm_builtins.py`, `useVmConsole`, benchmark) → union, **B's `packages/core-py/domains/…` paths must be retargeted if D4 keeps the deletion** |
| `domain/auth/__init__.py` | A-only: `_LAZY_IMPORTS` repos + `__getattr__` (B side empty) |
| `domain/infrastructure/ml_types.py` | A-only comment (B side empty) |
| `domain/voice/engine.py` | A-only comment (B side empty) |
| `domain/models/__init__.py` | disjoint export names on the same line: `update_personality_traits` vs `setup_providers` (+B's 2 extra) → both |
| `domain/training/__init__.py` | disjoint lazy-export entries + `__all__` names: A's `executor`/`peek_training_executor` vs B's importer/presets/outcome-tracker block → union both APIs |
| `domain/training/_internal/slonet.py` | A adds `align_dim_ff` to the import; B adds `utc_now_iso` → union |
| `packages/strui/src/components/ui/select.tsx` | B-only `labels, registerLabel` destructured props |
| `scripts/verify.sh` | A: drop `packages/core-py/domains/ui/api_server.py`, retarget ruff to `domain/training/_internal/checkpoint_utils.py`; B: swap required file to `packages/core-py/domains/api/__init__.py` + add the `tests/test_shared_indexes.py` governance block → union, **⚠ drop B's required-file entry if D4 keeps the deletion, else verify.sh fails on day one** |
| `tests/server/test_multimodal_router.py` | A: voice-cache reset docstring + `language=en`/`elapsed_ms` asserts; B: `confidence` assert → union |
| `tests/server/test_readiness_gate.py` | A clears `core.model`/`core.tokenizer`; B restores `_RGM.dispatch` from `_ORIGINAL_DISPATCH` → union (and align with `test_support.py` naming) |

(75 = 29 + 29 + 17; the list above plus buckets 1 and 3 is exhaustive.)

### 1.3 Bucket (3) TRUE semantic conflicts — 17 files, one line each

1. **`apps/api/server/main.py`** — **A** keeps uvicorn `reload_includes`/`reload_excludes` glob
   pruning (adds `tests/**`, `apps/web/**`, …); **B** replaces scoping with
   `reload_dirs=[_SERVER_ROOT, domain, core-py/domains]`, documenting that glob patterns cost
   `Config()` >60s / 100% CPU and that `reload_excludes` is ignored by StatReload.
2. **`apps/api/server/routers/system.py`** — **A** 5 sites via `domain.training.peek_training_executor()`
   with explicit `None → initialized:false / E_INFRA_STARTUP`; **B** 5 sites via
   `domain.training.engine.get_training_engine().executor_status/executor_job/executor_job_result/purge_executor_jobs/cancel_executor_job`.
3. **`apps/api/server/routers/dashboard.py`** — **A** `from domain.training import get_turbo_lock, get_turbo_state`
   + `TrainingOutcomeTracker`; **B** `from domain.training.engine import get_training_engine`.
4. **`apps/api/server/routers/models.py`** — **A** `ExportConfig/export_model/list_export_formats`
   from the `domain.training` facade; **B** export calls through `get_training_engine()`.
5. **`apps/api/server/training/router.py`** — **A** inline `CheckpointManager.is_resumable` /
   `load_from_path` / `load_latest_with_path` with two 422 raises; **B**
   `get_training_engine().resolve_resume_checkpoint(...)` (`ValueError → 422`), plus de-duplicated
   "Data file not found" wording and `_sync_original_live("failed", ...)`.
6. **`apps/api/server/routers/workspaces.py`** — **A** imports `DatasetRepository`/`KnowledgeRepository`
   classes; **B** imports `get_dataset_repository`/`get_knowledge_repository` factories.
7. **`domain/infrastructure/__init__.py`** — **A** minimal `_LAZY_IMPORTS` with module-attr `None`
   support (`artifact_registry`, buffers, 2 repositories); **B** eager error-class surface +
   `get_error_class` + entity-repo/model-registry/server-state getters + `_LAZY_INFRA`
   (`SLNCCompiler`, `try_register`). Union is mechanical but the export contract is a design choice.
8. **`tests/contract/test_router_boundary.py`** — **A** baseline is now `frozenset()` ("Build Order #4
   done, only allows shrink"); **B** pins 23 `_internal` entries (`shell.py`, `vm.py`, `memory.py`, …).
9. **`tests/server/test_status_router.py`** — **A** per-test inline
   `monkeypatch.setattr(native_engine, "get_engine", …)`; **B** autouse
   `_native_engine_available` fixture (macOS `.dylib` on Linux ⇒ readiness always False).
10. **`apps/web/lib/session-controller.ts`** — **A** `apiPost('/chat/{sessionId}/regenerate')` (JSON);
    **B** `streamSSE('/session/{sessionId}/regenerate')`, draining the stream and surfacing error
    events specifically to stop apiPost JSON.parse→retry→triple-regen.
11. **`tests/server/test_session_router.py`** — **A** deletes 120 lines of `/session/{id}/regenerate`
    tests (commit `c2379138e` retargeted regenerate to `POST /chat/{id}` and moved coverage to
    `tests/server/test_chat_router.py`, A-only); **B** adds 134 lines of `/session/{id}/regenerate`
    tests (thinking event, disconnect, no-context, no-provider).
12. **`apps/web/index.html`** — **A** `Man - AI Platform` title + persona description (rebrand);
    **B** `Slo - AI Platform` + enterprise copy **plus `<meta name="referrer" content="no-referrer">`**
    (v86 boot-CDN 403 fix — must survive regardless of copy).
13. **`apps/web/lib/document-meta.ts`** — **A** `Man` title/description; **B** `Slo` title/description.
14. **`apps/web/components/startup/StartupOverlay.tsx`** — **A** `<ManMark className="…"/>`; **B**
    gradient `<div>` with hard-coded `S`.
15. **`apps/api/server/tests/test_collections_router.py`** *(modify/delete)* — **A** deletes it
    (342 lines, part of the dead-router purge); **B** extends it `+92/-2`. B also adds
    `apps/api/server/routers/collections.py`, which merges in **clean** — so A's deletion leaves a
    landed router with no test (and A's own frontend still calls it: `collections/page.tsx`,
    `datasets/collections-content.tsx` exist on **both** lines ⇒ A is 404ing today).
16. **`packages/core-py/domains/infrastructure/pugqeep/engine.py`** *(modify/delete)* — **A** deletes it
    (part of the ~300-file `packages/core-py/domains` purge); **B** edits it in place `+40/-1`.
17. **`packages/core-py/domains/infrastructure/quant_core/wrapper.py`** *(modify/delete)* — **A**
    deletes it; **B** edits it `+31/-8`.

### 1.4 Silent hazards — divergences that do **not** show up as conflicts

These are the dangerous ones: `git merge` reports clean, so any mechanical resolution misses them.

- **`apps/api/server/routers/chat.py` — duplicate `thinking` event (verified in merged blob
  `e4657a605`).** A inserted its `yield … "thinking"` **inside** `try:` after the SessionCore 400
  and provider 503 guards; B inserted the **identical** yield **before** `try:`. Both survive ⇒
  every regenerate emits `thinking` twice. A's guards + `messages=msgs` and B's
  `from __future__ import annotations` all survive. Also: A's endpoint is `/chat/{id}/regenerate`,
  B's `/session/{id}/regenerate` — both land, so **two regenerate endpoints exist post-merge** (ties
  to decisions D1). Fix regardless of direction: keep exactly one yield (after the guards).
- **`AGENTS.md`** — A never touched it (`git diff f27654e54..main -- AGENTS.md` is empty), so B's
  SOP-1 acceptance clause + shared-workspace rules land cleanly. Consequence:
  **`tests/test_shared_indexes.py` (B-only) becomes a live gate on the merged tree** — it will fail
  if line A added any app/package/doc without an index entry since base. Run it first in Phase 0/5.
- **`apps/web/vitest.config.ts`** — B made zero changes; A's react-alias fix + `vite/**/*.test.ts`
  include lands untouched (no conflict, no risk).
- **Two implementations of the same recovery/resume feature** (A: inline checkpoint validation;
  B: `recovery_policy` + `recovery_incompatibility` + resume UI): after the union both test suites
  can be green while the user-visible behaviour (error wording, which rows offer Resume) differs.
  Needs one integration pass over the journey, not just tests.
- **`docs/VM_CONSOLE.md` + `scripts/verify.sh` still point at `packages/core-py/domains/…` on B**
  while A deleted that tree — see D4/R3.

## 2. Recommended sequence

**Direction: staged B→A on a *new* integration branch/worktree.** A is the remote-backed trunk
(`main` = `origin/main`) that all 33 worktrees and the dirty main tree build on; B has no remote
ref and 155 unique commits. Porting A→B would mean replaying 145 A-only commits for nothing, and
merging directly inside the main tree would put 560 dirty entries at risk. So: *merge B into a
branch off main, resolve there, gate there, land on main only when green.* Nothing is ever
checked out or reset in an existing worktree.

### Phase 0 — baselines (read-only, before any merge)
Create a **clean throwaway worktree of `main`** (new worktree, never the dirty main tree) and
record failing-test **IDs**, not just counts:
```
cd <clean worktree of main>
export PYTHONPATH="$PWD/packages/downcraft:$PWD/packages/core-py:$PWD/apps/api/server"
PY=/home/mana/miniconda3/envs/sloughgpt/bin/python
$PY -m pytest tests/server -q                 # expected baseline: exactly 4 pre-existing test_system_router failures
$PY -m pytest apps/api/server/tests -q        # record IDs (not in root testpaths)
$PY -m pytest -q                              # root testpaths: packages/core-py/tests tests apps/cli/tests
$PY -m ruff check .
cd apps/web && npm run typecheck && npm run lint && npm run test
```
Also snapshot `git merge-tree --write-tree --messages -z main aec2b37ec` output (this study's
`e4657a605`) as the reproducible conflict inventory.

### Phase 1 — integration branch
**First freeze the source ref** (today B advances daily — see §0.1): nominate the B tip to merge,
record its sha + its `merge-tree` conflict count, and re-run that count immediately before Phase 2.
Then: `.wt-reconcile` + `feat/line-reconcile` off `main`; `git merge --no-commit <B-tip-sha>` there.
Never `checkout`/`reset` in main tree, `.wt-strui`, `.wt-units`, `.wt-decode`, `.wt-rt`, or any
of the 31 other worktrees.

### Phase 2 — mechanical ports first (kills ~29 of 75 conflicts before anyone thinks)
Order matters; these are decided by convention, not debate:
1. **Python formatting normalization** on the conflicting `apps/api/server/**` files → resolves
   `docstore.py`, `infer.py`, `inference.py`, `tokens.py`, `test_docstore_router.py`
   (take A's wrap), `test_startup_routers.py`, `test_companion_router.py`, `test_support.py`.
2. **Design-token sweep** → take B for `master/page.tsx`, `magazine/page.tsx`, `PhonemeTypingMode.tsx`,
   `PronunciationDictation.tsx`, `consciousness/dashboard/page.tsx`.
3. **Patch-target/facade spelling** → resolve `routers/benchmark.py` + `tests/server/test_benchmark_router.py`,
   `test_companion_router.py`, `test_souls_router.py`, `test_audit_instrumentation.py` *as a pair*
   (router + its patch targets in the same commit), pending D2.
4. **Test-flake fixes** → take B for `lib/http-client.test.ts`, either for `TokenTreeVocabCard.test.tsx`,
   A (hermetic) for `test_knowledge_router.py`.
5. **Cross-port evidence:** the 10↔10 `git cherry` equivalents mean **no substantive fix needs
   pre-porting** — both directions already carry that work, and none of it is in the 75. What
   crosses over first is *style and spelling only*.

### Phase 3 — union resolutions (29 files, mechanical splice)
Resolve every one-sided block by keeping the non-empty side; for both-sided blocks keep both
(`db.ts`, `training-controller.ts`, `session/page.tsx`, `job_store.py`, `jobs_api.py`,
`PRODUCT_ENGINEERING.md`, `VM_CONSOLE.md`, `domain/*/__init__.py`, `slonet.py`, `verify.sh`,
`board.jsonl` union + newest-`updated_at` tie-break). Gate after this phase with the *targeted*
suites for touched areas (`tests/server`, `apps/api/server/tests`, web `lib/` + `components/`).

### Phase 4 — the 17 true conflicts, grouped into 8 clusters
`main.py` reload scoping (recommend **B's `reload_dirs` + keep A's excludes**) ·
training-execution API (D2) · infra export surface (recommend **union: B's superset + A's
module-attr `None` support**) · regenerate contract (D1) · branding (D3) ·
`packages/core-py/domains` deletion (D4) · collections router (D5) ·
`test_router_boundary` baseline (recommend **regenerate from the merged tree** using the recipe
in the file's own comment, never hand-pick) + `test_status_router` (recommend **B's autouse
fixture**, Linux-correct).

### Phase 5 — full gates before anything lands on main
```
export PYTHONPATH="$PWD/packages/downcraft:$PWD/packages/core-py:$PWD/apps/api/server"
PY=/home/mana/miniconda3/envs/sloughgpt/bin/python
$PY -m pytest tests/test_shared_indexes.py -q     # B-only governance gate, new on A
$PY -m pytest tests/server -q                     # new failures vs Phase 0 IDs = regressions; the 4 test_system_router failures stay
$PY -m pytest apps/api/server/tests -q
$PY -m pytest -q                                  # root suites
$PY -m ruff check .
./scripts/verify.sh                               # ruff smoke + governance + web ci
cd apps/web && npm run typecheck && npm run lint && npm run test && npm run build
cd ../.. && python scripts/benchmark_slonet_training.py   # AGENTS: training/inference change landed (engine.py +505, training/router.py)
```
Only then `git checkout main && git merge feat/line-reconcile` (or ask the user to), and only
push after an explicit ask — **never** force-push.

## 3. Non-negotiables (any plan must respect these)

- **No force-push, ever** — `origin/main` is line A; history rewrite would orphan every worktree.
- **No resets / checkouts / stashes / ref changes in the 33 worktrees** — `.wt-strui`,
  `.wt-units`, `.wt-decode`, `.wt-rt` and the main tree (560 dirty entries) hold uncommitted user
  work; a single `git checkout .` there is unrecoverable. All resolution happens in a **new**
  worktree/branch.
- **Uncommitted state preserved** — nobody stages, resets, or cleans another session's tree.
- **Notes + board stay source of truth** — `.kanban/board.jsonl` is derived from
  `~/.config/dev-notes/*.md`; the merge only unions cards, then the normal sync reconciles
  columns. This note's status (`in_progress → done`) is authoritative for this task.
- Toolchain: conda `/home/mana/miniconda3/envs/sloughgpt/bin/python`; pytest runs carry the
  `PYTHONPATH` prefix above (stale site-packages `downcraft` shadows the repo copy).
- No installs, no commits, no merges to `main` without an explicit ask.

## 4. Risks, rollback, effort

**Risks**
- R1 *Silent* duplicates like the verified `chat.py` double `thinking` — a clean merge ≠ a correct
  merge. Mitigate with a post-merge review of `git diff main..<merge>` for every file both lines
  touched, not just the 75.
- R2 `tests/test_shared_indexes.py` lands as a new hard gate; A may have un-indexed additions.
- R3 B's `packages/core-py/domains` references (`verify.sh` required file, `VM_CONSOLE.md` paths,
  2 modified files) vs A's deletion → `verify.sh` fails immediately if kept as-is.
- R4 Duplicate recovery/resume implementations can both be green while the journey differs.
- R5 `test_router_boundary` must be regenerated from the merged tree (its own comment explains how).
- R6 Board union can resurrect stale columns/duplicate ids → re-sync from notes, don't hand-edit.
- R7 After landing, the 33 worktrees sit on pre-merge A commits → each owner must `git merge`
  (never reset) when they return; expect a second wave of conflicts *in their dirty trees*.
- R8 **Line B keeps moving** (19 commits added to `aec2b37ec` the day of this study, +3 conflicts;
  more branches dated today) → the conflict inventory here is a snapshot. Freeze the source ref
  and re-measure `merge-tree` before Phase 2; if B advances again, either re-freeze or defer.

**Rollback**
- Everything happens on `feat/line-reconcile` in `.wt-reconcile`. A failed phase ⇒
  `git merge --abort` / `git checkout -- .` **inside that worktree only**; phases are separate
  commits so one phase unwinds with `git revert <sha>`.
- If a bad merge already reached main ⇒ `git revert -m 1 <merge-sha>`. **Never `reset --hard`,
  never force-push** — 33 worktrees share this history.
- Board state rolls back by re-checking-out the pre-merge blob of `.kanban/board.jsonl`
  (`e4657a605`'s parents both exist in history); notes are outside the repo and untouched.
- Regression proof comes from the Phase 0 failing-ID lists (diff of sets, not of counts).

**Effort (rough)**
- Phase 0 baselines: 0.5–1 h (long test runs dominate).
- Phase 1 + inventory: 0.5 h.
- Phase 2 mechanical (29 conflicts): **1–2 h**, largely scripted (`ruff format`/take-side rules).
- Phase 3 unions (29 conflicts): 3–4 h — one-sided blocks are fast, `db.ts`/`session/page.tsx`/
  `board.jsonl`/`PRODUCT_ENGINEERING.md` need eyes.
- Phase 4 (17 conflicts / 8 clusters): ~0.5 h of user decisions + **4–6 h** implementation.
- Phase 5 full gates: 2–4 h wall-clock.
- **Total ≈ 1.5–2 days.** Conflicts that vanish without a single discussion: **29 (bucket 1) +
  ~10 of the disjoint one-sided blocks ≈ 40/75.** Only **17 files / 8 clusters** require a decision.

## 5. Decisions — RESOLVED 2026-10-01 (user: "use common sense"; coordinator decided with evidence)

- **D1 · Regenerate contract** — canonical endpoint & transport: A's `POST /chat/{id}/regenerate`
  (JSON, SessionCore 400 + no-provider 503, tests in `test_chat_router.py`) vs B's
  `POST /session/{id}/regenerate` (SSE + `thinking` event, `streamSSE` client). Both land post-merge
  unless one is retired, and the duplicated `thinking` yield must go either way.
- **D2 · Training-execution API standard** — routers/tests standardize on A's
  `domain.training.peek_training_executor()` facade (Build Order #4, contract baseline already
  empty) or B's `domain.training.engine.get_training_engine()` (`TrainingEngine`, +505 lines).
  Recommendation: facade-facing router API, with `TrainingEngine` re-exported behind it — but the
  standard must be chosen before resolving `system/dashboard/models/training/router` +
  3 patch-target tests.
- **D3 · Branding/copy** — A's "Man" rebrand (title, meta description, `ManMark` in the startup
  overlay) vs B's "Slo" copy. `no-referrer` meta and any tokenised colours from B should be kept
  regardless.
- **D4 · `packages/core-py/domains` deletion** — A deleted ~300 files; B modified 2 of them
  (`pugqeep/engine.py` +40, `quant_core/wrapper.py` +31/-8) and still references the tree in
  `verify.sh` and `docs/VM_CONSOLE.md`. Keep the deletion (then port/drop B's two edits + fix the
  references) or restore the tree?
- **D5 · Collections router** — A purged the router and its test as "dead", B restored the router
  (`+` test, merges clean) because the frontend calls `/collections/*` — those callers exist on
  **both** lines, so line A 404s today. Keep B's restore (recommend) or kill the page + callers too?

### §5.1 — Resolutions (2026-10-01, user delegated: "use common sense")

- **D1 → A's route contract, B's client transport.** Canonical endpoint stays
  `POST /chat/{id}/regenerate` (A: SessionCore 400 + no-provider 503 guards, 8 tests in
  `test_chat_router.py` incl. thinking/disconnect/405). Swap the *frontend client* to B's
  `streamSSE` drain pattern retargeted to that path — A's current `apiPost` against the SSE
  success path is the JSON.parse→retry→triple-regen hazard B already diagnosed. Retire B's
  duplicate `POST /session/{id}/regenerate` route; fold any B-only test assertions into
  `test_chat_router.py`. In the merged `chat.py`, keep exactly **one** `thinking` yield (the
  clean merge produces a duplicate silently — no conflict marker will show it).
- **D2 → facade standard.** Routers/tests standardize on A's `peek_training_executor()`
  facade (Build Order #4 contract); re-export B's `TrainingEngine` (+505) *behind* the facade
  rather than adopting it router-facing. Resolve `training/router.py` + its 3 patch-target
  tests as pairs in Phase 4.
- **D3 → Slo wins — newest deliberate rename; finish it.** B's `1b0011aaa` (2026-09-29,
  "brand rename Man → Slo": sidebar `app.name` ×5 locales, document title, metadata) postdates
  A's Man work (`d8a2d15ab` 09-27 brand surfaces, `7dd5cfe22` 09-28 SEO/OG). Newest-decision
  tie-break (same convention as the board's `updated_at` rule), and Slo matches repo-wide
  identity (sloughgpt env, `slo.*` logger, `slough-*` units, `@sloughgpt`). Keep regardless:
  from A the OG/Twitter meta structure + og.png pipeline + 3-way parity drift-guard
  (retargeted to Slo copy) + robots discipline; from B the `no-referrer` meta (v86 boot-CDN
  fix) + tokenised colours. Retire `ManMark`/`ManLockup` + M favicon/PWA manifest.
  **Post-merge sweep:** lock `layout.tsx`/`index.html`/`document-meta` to a single
  `Slo - AI Platform` via the parity guard (A's `index.html` still says Man today). Reversible
  with one retitle commit if the user later prefers Man.
- **D4 → keep A's deletion.** The ~300-file `packages/core-py/domains` tree was a documented
  dual-tree defect (dev-note `2026-09-24-dual-domain-tree-finding.md`); restoring it reopens
  the debt. Port B's two edits (`pugqeep/engine.py` +40, `quant_core/wrapper.py` +31/-8) onto
  their migrated A-side paths under `domain/`, and fix the two stale references (`verify.sh`,
  `docs/VM_CONSOLE.md`) in the same Phase-4 cluster. The modify/delete trio resolves as
  delete + port.
- **D5 → keep B's restore.** Frontend callers of `/collections/*` exist on both lines (A 404s
  today — the "dead code" purge was falsified by live callers); router + test merge clean
  per §0. Killing it would require also removing the page + callers, a strictly bigger change
  with no payoff.

Execution is now unblocked (Phases 0–5 below); Phase 0 baselines first.

**B-line drift watch (live):** B advanced +2 during this session —
`789c97f81` (09:13, router tests: the card-f71f9f9e double-implementation,
see that note's collision section) and `5005ad8a0` (09:42, kanban flip) on
`feat/design-system-tables`. Tips since: `177d6cc76` (10:03,
`feat/chat-owns-prompting` — main tree sat on it) → `a4b79798d` (10:16,
`feat/workspace-search-q`/`feat/core-general-search`) → `9e7e0d7bc` (17:49,
`docs/truth-up-2026-10-01`) → `d386dcec1` (19:15, `feat/kanban-sync-hardening`).
Drift log of
`git merge-tree` conflicts vs current `main` (`a1ebb7f2d`): 75 (study,
`aec2b37ec`) → 78 (`8085f117e`) → 77 @ 10:27 vs `a4b79798d` → 79 @ 17:50
vs `9e7e0d7bc` = 76 content + 3 modify/delete (cached at
`~/.config/dev-notes/evidence/mergetree-Btip2.txt`, tree `279b41c4b`) → **79 @
19:33 vs `d386dcec1`**, output at `evidence/mergetree-Btip3.txt` (tree
`f0928a45dd`) → **79 @ 2026-10-02 08:45 vs `1b90d6bcc`** (08:30, `docs:
refresh stale controller numbers…`) while A advanced to `e1fbcee08`
(`fix/health-payload-trim` merge), output at `evidence/mergetree-Btip4.txt`
(tree `7448a5917`) — both lines moved, count flat. Counts do not grow
monotonically — B sometimes fixes the
same files A fixed. Freeze nomination happens at Phase 1; re-run the count at
freeze per §0.1 — never carry an old number forward as current.
