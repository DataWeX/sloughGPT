---
id: 20261001_102
title: "test: backend router coverage — chat, model_stack, phoneme (parity audit gap)"
status: done
date: 2026-10-01
tags: [tests, backend, routers, coverage]
board_id: f71f9f9e-be9c-4004-9de5-9b27ab508c4f
board: f71f9f9e-be9c-4004-9de5-9b27ab508c4f
hash: ad91ed2fc254bd2e7fd3b16fb66537da728c1d91489e974d67493da78a60b20b
---

# test: backend router coverage — chat, model_stack, phoneme (parity audit gap)

## Background

Board card `f71f9f9e` (title-only, no body, no note → unowned): the parity
audit flagged `chat`, `model_stack`, and `phoneme` as backend routers with
no test file. Confirmed: `tests/server/` holds 54 `test_*_router.py` files
(auth, inference, voice, mobile, images, …) — and exactly those three are
missing.

Coordination checked: card 071 (parallel session) owns `docs/routers.md`
truth-up + contract test — docs layer, not `tests/server/`. Card 092 owns
acyclic autograd. No session owns router behavior tests. No new app/
package/doc → no shared-index changes expected.

## Plan

1. Survey the established pattern (`tests/server/test_inference_router.py`):
   bare `FastAPI()` + `register_all_handlers` + include the router +
   `TestClient`; envelope is `{"status": "success", "data": …}`.
2. Write three test files, each with:
   - a **route-parity pin** (exact path+method set — drift shows up as a
     failing assertion, same spirit as 071's 674-route doc pin);
   - behavior tests per endpoint.
   - `chat`: respond (success envelope / 422 / manager-exception classified),
     stream SSE (tokens + done + error event), regenerate SSE, cancel,
     active, health, addons — via `patch("domain.chat.get_chat_manager")`.
   - `model_stack`: get/push/remove/clear lifecycle on tmp paths,
     set_base preserves layers, duplicate-name + missing-name errors —
     reset module `_stack` per test.
   - `phoneme`: real engine (pure-numpy rule tables, probe-verified fast):
     encode→decode roundtrip, batch, score==1.0 identical input, detect→fr,
     status, languages; 422 validations; mock-engine failure → 5xx codes.
3. Fix the production bug probing uncovered: `GET /phoneme/languages`
   500s — `UnifiedPhonemeEncoder.supported_languages` is a `@property`
   returning `list[str]`, but `domain/voice/_internal/phoneme.py:178`
   calls it with `()` → `'list' object is not callable`. Other consumers
   (`multimodal` router, encoder CLI) correctly treat it as an attribute —
   fix the caller.
4. Gates: new tests green; `tests/server/` full green; full `tests/`
   failure delta equals the documented pre-existing set.

## Open questions (recommendation-taken)

- Real vs mocked phoneme engine: real — probe showed encode/decode/score/
  detect are deterministic numpy lookups (no torch), so integration value
  is free. Only `synthesize` stays un-called for real (would build the
  voice engine); its error branch is covered via mock.

## Gates

- [x] route-parity pins for the 3 routers
- [x] behavior tests green (pytest, conda env) — 45/45
- [x] languages bug fixed + regression test
- [x] no regressions (tests/server: baseline captured pre-change, identical
      4 pre-existing failures after; full-suite gate PASSED 19:10 — see
      Evidence → Gate verdict)
- [x] lint clean on changed files (ruff check + ruff format)

## Evidence

### Evidence — log files index (all under `~/.config/dev-notes/evidence/card102/`)

| file | what |
|---|---|
| `chunk_tests.log` | **THE GATE** — full `tests/` (incl. `tests/server/`), launched 19:06 as systemd unit `card102-suite` |
| `chunk_corepy.log` | chunk 2 — died in 1s on the conftest/arg quirk; ruled out of blast radius, not a gate (see Gate verdict) |
| `chunk_apps.log` | belt-and-braces chunk 3 (`apps/cli/tests`); ends with `ALL_CHUNKS_DONE` |
| `full_suite5.log` | attempt 5 — killed by restart at 63% (setsid escaped pgid, NOT the cgroup) |
| `full_suite4.log` | attempt 4 — killed at 19% |
| `full_suite.log` / `2` / `3` | attempts 1–3: collection error / empty (restart kills) |

Runner: `~/.local/state/sloughgpt-gene/run_card102_suite.sh` via
`systemd-run --user --unit=card102-suite` (own cgroup → survives opencode
restarts; card100's lesson). Each chunk writes its own log + `EXIT=` +
`CHUNK_END` lines, so a kill only loses the in-flight chunk. Invocation:
`.wt-rt`, `PYTHONPATH=downcraft:core-py:apps/api/server`, `pytest <path> -q`.

Also: `~/.config/dev-notes/evidence/mergetree-Btip2.txt` (79-conflict
merge-tree output vs B tip `9e7e0d7bc`) + `mergetree-Btip.txt` (77 @ 10:27) —
referenced by the line-reconciliation note.

Known pre-existing failures — **any other failure is suspect**:

- `tests/server`: exactly 4 (`test_system_router` `TestTailOutput` /
  `TestOutputStream`); baseline 4 failed / 1892 passed pre-change.
- `downcraft.download` collection error only when PYTHONPATH prefix missing
  (this run has it → must NOT appear).
- known slonet / quantization sets; ~21 pugqeep debt (same on parent).
- Historic root-suite baseline (09-28): `tests/` sans server = 1145 passed /
  9 failed / 5 errors.

Suspect scope = failures touching `tests/server/*` or `domain/voice`.
Only production delta in this branch: 1-line `phoneme.py` fix (property
access), so anything outside voice/server failing is environmental/pre-existing.

### Gate verdict — PASSED (2026-10-01 19:10, `chunk_tests.log`)

**13 failed / 3079 passed / 5 errors** on full `tests/` — triage:

| set | identity | verdict |
|---|---|---|
| 4 F `tests/server/test_system_router` | `TestTailOutput` ×1 + `TestOutputStream` ×3 | documented pre-existing (exact identities) |
| 9 F root | `test_feedback_domain` ×2, `test_inference_generate` ×6, `test_rag` ×1 | = 09-28 note's documented "9 failed" baseline (counts exact, predates branch); zero file overlap with this diff; absent from main-tree runs → worktree env (missing untracked `.env`/model state) |
| 5 E root | `tests/test_cli_chat.py` setup errors ×5 | = documented "5 errors" baseline, same reasoning |
| suspect scope | `tests/server/*` + `domain/voice` | **zero unexpected failures** ✓ |

Belt-and-braces: `apps/cli` chunk 422 passed / EXIT=0 (`chunk_apps.log`);
`apps/api/server/tests/test_voice_router.py` 2 passed (voice → domain.voice
→ fixed file); `packages/core-py/tests` **out of blast radius** (its phoneme
tests import `domain.multimodal._internal.unified_phoneme_encoder`, never
`domain/voice/_internal/phoneme.py`) — not run as a gate. Invocation lesson
(recorded for reuse): `pytest packages/core-py/tests` alone dies on
`No module named 'tests.conftest'`; it needs `tests/` in the arg list
(run5 shape, probe-verified 45393 collected).

### Progress log

**Worktree/branch**: `.wt-rt`, `test/router-coverage` @ `a1ebb7f2d`
(main had advanced past the triage snapshot).

**Delivered files**

| file | tests | notes |
|---|---|---|
| `tests/server/test_chat_router.py` | 20 | 8 pre-existing regenerate tests kept verbatim + 12 new |
| `tests/server/test_model_stack_router.py` | 9 | new |
| `tests/server/test_phoneme_router.py` | 16 | new, real engine |
| `domain/voice/_internal/phoneme.py` | — | 1-line production bugfix |

- chat new: route-parity pin (7 routes), respond (envelope / forwarded
  options / 422 / manager-exception classified), stream SSE (tokens +
  heartbeat + done parsed as events; mid-stream exception → SSE error event
  with phase STREAM), cancel (True/False), active, health, addons — via
  `patch("domain.chat.get_chat_manager")`.
- model_stack: route-parity pin (5 routes), default base, push/remove
  lifecycle, remove-missing → false, duplicate-name & missing-name →
  classified error, `set_base` preserves layers, clear-all & clear-by-kind.
  Module-global `_stack` reset (autouse) per test.
- phoneme: route-parity pin (9 routes), status, **languages regression**,
  encode→decode roundtrip, visualize, score==1.0 identical input, batch
  (list of per-text dicts), detect→`fr`, 4×422 validation, and 3 engine-
  failure branches → 500 with `E_ENCODE_FAILED`/`E_SYNTHESIZE_FAILED`/
  `E_LANGUAGES_FAILED` (stub engine; real synthesize never invoked — it
  would build the voice engine). Real engine is pure-numpy rule tables
  (probe: encode/decode/score/detect deterministic + fast).

**Production bug found + fixed (1 line)**

`GET /phoneme/languages` returned 500 `'list' object is not callable`:
`UnifiedPhonemeEncoder.supported_languages` is a `@property` returning
`list[str]`, but `domain/voice/_internal/phoneme.py:178` called it with
`()`. Probe confirmed broken pre-fix; fixed caller (dropped parens).
Other consumers (`multimodal` router, encoder CLI) already treat it as an
attribute. Regression test: `test_languages_lists_supported`.

**Check-if-exists miss, caught + recovered**

Initial inventory of `tests/server/*.py` ran against the stale main-tree
checkout; on the newer worktree base `tests/server/test_chat_router.py`
already existed (commits `c2379138e`, `7a30b1c4e` — card 051). First
`write` clobbered it (8 tests), spotted via `git status` showing ` M`
instead of `??` and an 8-test arithmetic gap (1892+39≠1923). Recovered:
`git checkout HEAD --` restored the original, then merged — their 8
regenerate tests kept verbatim (they cover no-context/success/no-provider/
multi-token/sse-error/disconnect/thinking/405 via provider-layer mocks),
my 12 non-regenerate tests appended (my 2 regenerate tests dropped as
redundant). Lesson recorded: inventory against the WORKTREE base.

**Gate results**

- Targeted 3 files: **45/45** (before and after `ruff format`).
- `ruff check` (4 changed py files): clean. `ruff format`: 3 test files
  reformatted (repo convention — pre-existing test files verified
  already-formatted; CI format scope is `packages/core-py` + `apps/api/server`).
- `tests/server/` baseline **captured before any change**: 4 failed /
  1892 passed — the 4 are pre-existing `test_system_router` tail/stream
  failures (`TestTailOutput`, `TestOutputStream`). After changes: same 4,
  passed +net new tests.
- Regenerate behavior learned en route: on new main the handler guards
  `SessionCore.get_messages` (→ SSE error `E_VAL_REQUEST` 400) and
  `get_provider` (→ `E_INFRA_REGISTRY` 503) **before** the thinking event.
- Full-suite run: first attempt aborted at collection —
  `packages/core-py/tests/test_hf_hub.py` → `ModuleNotFoundError:
  downcraft.download`. Root cause: conda `site-packages/downcraft` 0.2.0
  (flat layout: `state.py`, `hf_hub.py` at top level, pip'd from
  `anomalyco/sloughgpt`) **shadows** the repo's `packages/downcraft`
  (which has the `download/` subpackage) — conftest appends repo paths
  AFTER site-packages. Workaround (no env/install change, same pattern as
  card 093): prefix
  `PYTHONPATH="$PWD/packages/downcraft:$PWD/packages/core-py:$PWD/apps/api/server"`.
  Rerun backgrounded → result appended below.

**Repo-history divergence (found during the card, flagged for user)**

`main`/`origin/main` (`a1ebb7f2d`) and the line carrying cards 081–101
(tip `aec2b37ec`) **diverged at `f27654e54`, 2026-09-25** — main 208
commits ahead, aec2b line 165 ahead, truly disjoint since (no merge-base
overlap beyond Sep 25); the aec2b line exists only on local branches
(`feat/*`), no remote branch contains it. main's reflog shows parallel
sessions committing directly to main on its line (card 092 autograd,
planner verify, rng fingerprint, backup). Card-relevant facts:
- the phoneme `supported_languages()` bug exists on **both** lines
  (aec2b line: `domain/voice/_internal/phoneme.py:178`) → fix portable;
- `tests/server/test_chat_router.py` exists **only** on the main line;
- `tests/test_shared_indexes.py` (governance, card 081) exists **only**
  on the aec2b line → its gate is N/A on this base (nothing indexed by
  this card regardless);
- `.wt-rt` stays based on `main` (= origin/main, canonical remote);
  cross-line merge is a user decision, not taken unilaterally.

**Card double-implemented (found 09:50, while this card's full-suite gate
was still running)**

Another session committed equivalent work mid-gate: `789c97f81` (09:13,
"router coverage for chat, model_stack, phoneme — parity gate exits 0") on
`feat/design-system-tables` (line-B based, does NOT contain main) + board
flip `5005ad8a0` (09:42, cards `34b396b7` + `f71f9f9e` → done). Board now
says done; this note (source of truth) stays `in_progress` until the
full-suite gate returns and the deltas below are resolved.

Delta between the two implementations:

- **theirs (committed):** `apps/api/server/tests/{test_chat_router,
  test_model_stack_router,test_phoneme_router}.py` (+495 lines); phoneme
  runs on a fake engine (`PhonemeRouter._get_engine` monkeypatched →
  `_FakeEngine`), plus auth coverage (token on/off) this work lacks.
- **mine (uncommitted, `.wt-rt`):** `tests/server/…` 45 tests with
  route-parity pins + real-engine roundtrip/detect/score, **and the 1-line
  `phoneme.py` production fix**.
- **critical gap in theirs:** their `domain/voice/_internal/phoneme.py` is
  byte-identical to main's — `encoder.supported_languages()` still called
  with `()` on the `@property` → `GET /phoneme/languages` still 500s on
  that branch; the engine-level fake means their `test_languages` passes
  *over* the real bug. The fix exists only in this worktree.
- No git conflict (different test roots — the repo already carries both
  `tests/server/` and `apps/api/server/tests/`), but route coverage is now
  duplicated across roots; reconciliation must pick union vs one side.
No action taken on either copy; dropping/merging either is a user call.
