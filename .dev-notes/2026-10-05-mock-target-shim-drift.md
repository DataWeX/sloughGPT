---
title: Patch targets that never install — shim/`_internal` mock drift
date: 2026-10-05
status: in-progress
branch: feat/create-router-projection
tags: [testing, pytest, mocking, refactor-drift, qa]
---

# Landed

| commit | contents |
|---|---|
| `a0912fa93` | `test-doctor --mock-drift` (confirmed + unresolvable), 18 unit tests, `docs/TESTING.md` § Patch Target Drift, and the 4 retargeted test files |
| `2443a1abe` | `qa-developer.md` Triage/Commands + `test-writer.md` backend rules now point at the tool |
| `64d0862f9` | **suspects** — reader reachable within 2 imports, advisory (never flips exit 1); import graph + bounded BFS; 6 more tests (24 total) |
| `c1308c9c9` | **card b236257b** — the two spellings unified (`monkeypatch.setattr("a.b.c", …)` is now scanned alongside `patch("a.b.c")`, 166 → 203 targets) + `--dead-seam <file>`: boom-vs-baseline probe answering "is this patch ever *executed*?", with `scripts/dead_seam_plugin.py`; 22 more tests (46 total); merged `32005e630` |

Measured before building it: 40 latent pairs, **9 reachable within two hops** —
the old "latent" label read as *all clear* while a second router could be on live
state. Validation case: `domain.infrastructure._internal.model_registry` is
patched, but the readers are shims copying names **by value at import**
(`domains/…/model_registry.py` does `import *` then `setattr` over
`dir(_real)`; `domain/…/model_registry.py` does `import *` again — 0
`__getattr__` across all three), so `controllers/models.py:454` can never see the
patch. The test passes only because that path is not hit: the silent case.
Suspects are not yet triaged into real/not-real — that is the next unit.

Evidence at commit time: lint clean, `tests/test_doctor_mock_drift.py` 18 passed,
all four retargeted files individually green, `tests/server` **2 failed /
1988 passed / 1 skipped / 68 deselected — exact baseline** (both failures
pre-existing), `test-doctor --flake` / `--recent` / `--help` unaffected.
`packages/core-py/tests` gate was still running (0 failures at 12%) at commit
time and remains the merge gate.

Not merged: the branch also carries another session's commits
(`fb7fedd7c`, `140402a4e`), so merging is their call, not mine.

# Patch targets that never install (silent live-state tests)

## Problem

A test does `patch("pkg._internal.mod.attr")`, but production reads the same
symbol through a **different binding** — the `pkg.mod` shim, or the `pkg`
package `__init__`. The fake never installs, so the test runs against
**live state** (real store rows, real server singleton) and passes by
coincidence or fails intermittently.

This is the mechanism behind both root causes fixed in `ff06a8d1b`:

| test | patched | production actually reads |
|---|---|---|
| `test_user_adapters_router` | `domain.feedback._internal.per_user_lora.get_per_user_lora` | `from domain.feedback import get_per_user_lora` (`__init__.py:27`, eager re-export) |
| `test_dashboard_router` | `domain.infrastructure._internal.server_state.get_server_state` | `from domain.infrastructure.server_state import get_server_state` (shim does `from domains... import *`) |

In the first, `len(adapters)` returned **1** (a real `user_id: 'default'` row)
instead of the fake's 2. In the second, `model_loaded` came back `True` from
the **live** server state.

## Detection is not enough — verification is the point

Static analysis flagged 45 suspects out of 168 `._internal.` patch targets, but
it cannot tell a broken patch from a benign dual-path import (relative
`from .config import ...` resolves to the *same* module inside `_internal/`).

The detector therefore **verifies empirically**: import the reader module,
enter the patch, re-resolve the name through the reader's namespace, and check
whether it changed.

```
before = getattr(reader_module, attr)
with patch(target):
    after = getattr(reader_module, attr)
if after is before:  -> BROKEN (fake never installs)
```

**Result: 28 distinct targets whose package binding does not follow the patch
(32 assertions) across ~20 test files.**

**Correction — that number is NOT "28 broken tests."** The empirical check
proves only that `getattr(pkg, attr)` does not move. It is a bug *only if the
code under test actually reads through `pkg`* — when the subject imports
`_internal` directly at call time, the patch lands and the package binding is
irrelevant. Measured example:

| target | non-patched reader | is *that* module the subject? | verdict |
|---|---|---|---|
| `domain.memory._internal.task_memory.list_archive` | `apps/cli/src/commands/memory.py`, `domain/knowledge/__init__.py` | no — `routers/memory.py:351` imports `_internal` at call time | **false positive** |
| `domain.feedback._internal.per_user_lora.get_per_user_lora` | `routers/user_adapters.py:54` | yes — the router *is* the subject | **real** |

So a finding requires **both** conditions:

1. **EMPIRICAL** — patching `_internal.X.attr` does not move `reader.attr`; and
2. **RELEVANT** — some test that patches the target imports the file-module
   doing the non-patched read.

Applying both narrows 28 → **4 confirmed**, and all 4 were in fact failing:

| broken target | code under test | test |
|---|---|---|
| `feedback.model_health.get_health_monitor` | `routers.health:711` | `test_health_router` |
| `feedback.response_tracker.get_response_tracker` | `routers.benchmark:304` | `test_benchmark_router` |
| `infrastructure.lifecycle.get_lifecycle_manager` | `routers.system:319` | `test_system_router` |
| `infrastructure.output_buffer.get_server_buffer` | `routers.system:336,365` | `test_system_router` |

A fifth (`test_system_router` executor) was found by the *failure*, not the
detector: `patch("domain.training.executor._instance")` names a module that
does not exist (`ModuleNotFoundError`) — `domain/training/__init__.py` exposes
only `get_training_executor` in `_LAZY_IMPORTS`. `TrainingEngine._get_executor`
reads `domain.training._internal.executor._instance` at call time
(`engine.py:602`), so that was the right target.

The discriminator is whether the reader package re-exports **lazily or
eagerly** — verified directly:

- `domain/feedback/__init__.py` — **zero** `__getattr__`; lines 19–32 are
  eager `from domain.feedback._internal.X import attr`. The binding freezes at
  import, so `patch("domain.feedback._internal.X.attr")` never reaches
  `domain.feedback.attr` → **BROKEN**.
- `domain/training/__init__.py` — defines `__getattr__` gated on
  `_LAZY_IMPORTS`, which does `importlib.import_module(...)` then
  `getattr(mod, attr)` **on every access**. The lookup re-reads the patched
  module each time → **ok**. (`SloughGPTTrainer` is *not* in the eager import
  block at lines 13–30; it rides the lazy path.)

So the same static shape (`pkg._internal.mod.attr` patched, `pkg.attr` read)
is a bug in one package and harmless in another. Only the empirical check
discriminates — which is why the detector opens the patch context and compares
`getattr` before/after rather than pattern-matching import lines.

## Scope

Empirically-broken bindings (the raw 28 — **pre**-relevance-filter; see the
correction above, only the 4 in that table are confirmed as failures):

- `domain.feedback.*` — `hf_dpo.get_feedback_db`, `hf_dpo.HFDPOTrainer`,
  `model_health.get_health_monitor`, `per_user_lora.get_per_user_lora`,
  `response_tracker.get_response_tracker`, `workflow.get_feedback_workflow`
- `domain.infrastructure.*` — `context_core`, `conversion_tracker`,
  `errors.classify_exception`, `lifecycle`, `model_registry`, `model_resolver`,
  `output_buffer`, `process_guard`, `quantization`, `resource_manager`,
  `server_state`, `truth_labeler`
- `domain.agents.system.get_agent_system`, `domain.models.provider.get_provider`
- `domain.multimodal.manager.get_multimodal_manager`,
  `domain.slolib.gpu.get_accelerator`, `domain.shell.{kernel.Kernel,commands._api_get}`,
  `domain.inference.vector_store.simple_embed`, `domain.memory.task_memory.*`

Affected files include `tests/server/test_{agents,audit_instrumentation,lora_eval,
feedback_controller,kb,startup_routers,session,memory,souls}_router.py`,
`packages/core-py/tests/test_{hf_dpo,devices_more,infer_router,chat_domain,
shell_repl_more,shell_runtime,memory_router,models_router}.py`.

## Fix rule

**Patch where it is read.** The correct target is the namespace the consumer
resolves *at call time* — `patch("<reader_module>.<attr>")` — not the
`_internal` module. Retargeting makes the fake take effect, which can then
expose a second defect (a stale fake signature), exactly as it did for
`update_adapter(user_id, feedback_signal=...)` in `test_user_adapters_router`.

### Four defect classes the retarget sweep surfaced

1. **Dead target** — patched `_internal`, production reads the package re-export.
2. **Phantom target** — patched a path that never existed
   (`domain.training.executor` → `ModuleNotFoundError`).
3. **Wrong package root** — `patch("domains.get_benchmark_domain")` (the old
   plural package) while the router reads `domain.benchmark` at call time
   (`benchmark.py:280,341,357`). 3 of the 4 `test_benchmark_router` failures.
4. **Stale fake surface** — the target was right but the fake was not: the
   engine reads `inst.max_workers` / `inst.job_count` (public properties,
   `executor.py:295`), the fake set only `._max_workers` / `._jobs`. Because
   `MagicMock` auto-creates any missing attribute, the test saw a MagicMock
   that `jsonable_encoder` turned into `[]` (MagicMock `__iter__` defaults to
   empty) — so `assert data["max_workers"] == 4` read `assert [] == 4`. **The
   fake must mirror the public surface the code under test reads**, and a
   MagicMock's willingness to answer any attribute name hides the mismatch.

### Not every BROKEN entry is "the target is wrong" — some are split coverage

`domain.models._internal.provider.get_provider` is read through **both**
paths in production:

| reader | consumers |
|---|---|
| `domain.models._internal.provider` | `domain/chat/_internal/{manager,domain}.py`, `domain/feedback/_internal/lora_eval.py`, `domain/infrastructure/_internal/memory_pressure.py`, `apps/api/server/main.py`, `routers/tools.py` |
| `domain.models` (eager `__init__`) | `apps/api/server/routers/{session,infer,models}.py` |

The same `patch("_internal.provider.get_provider")` therefore mocks the first
group correctly and silently misses the second. A test exercising
`routers/session.py` with that patch is not mocking the provider **at all**,
while an identical-looking test over `domain/chat` is fine.

Consequence: the right target is decided by *the code under test*, not by the
symbol. The detector's job is to tell you when your patch does not reach it —
not to hand you one canonical string.

## Acceptance

- [x] Detector lands in `scripts/` — as `scripts/test-doctor.py --mock-drift`
      (test-doctor already owns health/flaky/`--fix-hint`; no new script).
      Documented in `docs/TESTING.md` § *Patch Target Drift*. Reads its
      `sys.path` from `pytest.ini`'s `pythonpath`, not a hardcoded list.
      Reports both classes: **confirmed** (silent dead re-export) and
      **unresolvable** (patch raises at setup).
- [x] 18 unit tests in `tests/test_doctor_mock_drift.py` — including the three
      bugs found in the detector itself (missing `re.MULTILINE`, `domain/` as a
      pythonpath root, and reporting a module's own re-binding as dead).
- [x] Retarget batch bounded by evidence, not by the raw list of 28:
      4 confirmed + 1 phantom in `test_system_router`/`health`/`benchmark`
      (18F → 34P), plus `test_chat_loop_e2e` 6E → 6P (phantom + split coverage).
- [x] Regression gate `tests/server`: **2 failed, 1988 passed, 1 skipped,
      68 deselected — exact baseline**, both failures pre-existing
      (`test_mobile_training…falls_back_to_logs`,
      `test_souls_router…chat_with_valid_fields`).
- [ ] Regression gate `packages/core-py/tests` (contains the 3 edited files).
- [x] No production code changed — test + docs + tooling only.
- [ ] Coordination: `.kanban/*`, `domain/inference/*`, `domain/training/*`,
      `apps/cli/*` are dirty from a concurrent session and must not be staged
      (verified staged set is exactly the 7 files of this change).
- [x] **Suspects** (2-hop import reach) added as a third class in `64d0862f9`,
      advisory only. 24 unit tests. All 9 triaged empirically — see Round 3.
- [x] Agent docs (`qa-developer`, `test-writer`) point at `--mock-drift` —
      otherwise the next session re-derives the diagnosis from scratch.

## Not done / deferred

- The remaining ~24 empirically-broken bindings are **latent** — no test that
  patches them imports the corresponding reader, so nothing misbehaves today.
  They become real the moment someone writes such a test. `--mock-drift -v`
  lists them; do not mass-retarget blind.
- ~~Correlation is one hop~~ — **closed in `64d0862f9`.** A test importing a
  router that itself imports the reader is now followed out to 2 hops through a
  static module→module import graph, reported as **suspects** (advisory: import
  reach is weaker than runtime use, so it never sets the exit code).
- **Suspect verifier: keep throwaway — decided by user (2026-10-05).** The
  harness that produced the 9 verdicts stays in `/tmp/opencode/`, not shipped.
  The method is documented as a paste-and-run recipe in `docs/TESTING.md`
  § *Verifying a suspect*, so it needs no committed code and the advisory class
  still has a self-service verification path. All 9 suspects came back harmless.
- Background 18-file run timed out at 30 min — that batch is too large to run
  whole; per-file runs are the workable unit.

## Round 2 — phantom + split coverage in `tests/test_chat_loop_e2e.py`

That file is `pytestmark = pytest.mark.slow` (line 23), so **6 tests were
ERRORing invisibly** — `6 deselected` in every default run, never surfaced by
either regression gate. Running it explicitly with `-m "slow or not slow"`
showed all 6 erroring at fixture setup:

```
AttributeError: <module 'domain.learner._internal' (namespace) ...>
does not have the attribute 'get_learner'
```

Two separate defects, fixed in sequence:

1. **Phantom target** — `patch("domain.learner._internal.get_learner")`.
   `domain/learner/_internal/` has *no* `__init__.py` (namespace package), so
   the attribute never existed. `get_learner` is defined in
   `_internal/continual.py:554` and re-export by `domain/learner/__init__.py`.
   The exercised path is the chat stream, which calls it at
   `routers/inference.py:839` via a **module-level** import, so the target is
   `routers.inference.get_learner` (consistent with the
   `routers.inference._enrich_knowledge` patch the same fixture already used).
   → 6 ERROR → 4 passed, 2 failed.

2. **Split coverage** — the remaining 2 returned an *empty* body
   (`assert 'Hello! …' in ''`). The fixture patched only
   `domain.models._internal.provider.get_provider`, but `chat_stream`
   (inference.py:1821–2829) reads `get_provider` at line **2353** through
   `routers/inference.py:45`'s module-level `from domain.models import
   get_provider`, and `domain/models/__init__.py` has **0 `__getattr__`**
   (eager). The `_internal` patch therefore never reached it. Both arms are
   genuinely needed — `_internal` is still read at call time by
   `domain/chat/_internal/{manager,domain}.py` and
   `domain/feedback/_internal/lora_eval.py` — so the fixture now carries both
   patches. → **6 passed**.

### Detector gap closed: phantoms ARE now reported

`--mock-drift` reports two classes:

1. **confirmed** — silent: the re-export is frozen, so the fake never installs,
   and a test that patches the target imports the reader. Requires both
   conditions; (2) alone over-reports (the `domain.memory` case).
2. **unresolvable** — loud: `patch()` raises at setup, so the test ERRORs.
   Still worth reporting because a `slow`-marked file is deselected and its
   ERRORs never reach a normal run — which is exactly how
   `test_chat_loop_e2e` hid 6 of them.

Measured cost of (2): resolving all 166 distinct targets takes **1.8s**. The
one intentional case (`domain.shell._internal.repl.TuiRepl`, patched with
`create=True`) is detected by walking back from the target to its enclosing
`patch(...)` call and checking for `create=True`, so it is skipped rather than
reported. Verified against the real pre-fix target: caught with the exact
`AttributeError`, and skipped when the same target is given `create=True`.

## Round 3 — suspects (`64d0862f9`) and their triage

Condition (2) required a *direct* import of the reader's file-module, so it
under-reported. Measured on the real tree before building anything: **40 latent
pairs, 9 reachable within two imports.** Left as-is, "latent" reads as *all
clear* while a second router could quietly be on live state.

Promoting them to *confirmed* would have been dishonest — import reach is not
runtime use. So they became a third class, **suspects**: printed loudly,
**never** flipping the exit code (confirmed and unresolvable still do).

Then triaged all 9 empirically rather than asserting a verdict:

| group | test file | baseline | probe | verdict |
|---|---|---|---|---|
| `feedback.MogDB` (via `mogdb`, `mogdb.database`) | `test_feedback_training.py` | 24 P | 24 P | harmless |
| `pugqeep.facade` `ModelQueue`/`QueueConfig`/`load_model_to_points` | `test_pugqeep_facade.py` | 62 P | 62 P | harmless |
| `model_registry.get_model_registry`, `server_state.get_server_state` | `tests/test_training_distill.py` | 12 P | 12 P | harmless |
| `knowledge_augmenter.get_knowledge_ingestor`/`get_knowledge_memory` | `tests/test_learner_pipeline.py` | 17 P | 17 P | harmless |

**9 of 9 harmless.** The sentinel was installed (`PROBE-INSTALLED` printed) and
never hit: no reader function is called on these paths, so the dead `_internal`
patch changes nothing today. They stay reported as traps for whoever later
extends a test down that path.

Method: patch `reader_mod.attr` **and** each reader file module's own attribute
(both, because a function-level `from shim import x` re-reads the shim at call
time while an already-imported module-level read only moves via its own
binding), install in `pytest_configure` *before collection*, rerun the file.

**Design vindicated:** had suspects set exit 1, the tool would have cried wolf
on 9 out of 9.

**The probe was validated before trusting it** — otherwise all 9 verdicts could
be false negatives from a sentinel that never fires. Synthetic control with both
binding modes (baseline `2 passed`):

- module-level `from shim import f` → caught via the reader file module's own
  attribute: `PROBE-CALLED md_ctrl_reader.f`
- function-level `from shim import f` → caught via the shim:
  `PROBE-CALLED md_ctrl_pkg.f`

Both failed, so the mechanism fires on both paths — which is exactly why the
recipe patches **both** `reader_mod` and the reader file module.

Validated example, structurally: `domain/infrastructure/model_registry.py` is
`from domains.... import *`, and `packages/core-py/domains/.../model_registry.py`
does `import *` **plus** a `setattr(_mod, _k, getattr(_real, _k))` sweep over
`dir(_real)` — names copied **by value at import**, 0 `__getattr__` on all three
links — so `controllers/models.py:454` can never see a patch to
`domain.infrastructure._internal.model_registry`. The test passes only because
that path is not hit: precisely the silent case this work is about.

Probe harness: `/tmp/opencode/md_probe_plugin.py` + `md_suspect_probe.py` —
throwaway, **not committed**, and the user decided (2026-10-05) it stays that
way. The method ships instead as a recipe in `docs/TESTING.md`.

# Round 4 — gate 1, the slow-test hunt, and 2 production bugs

## Gate 1 acceptance: DONE (798/798 processed) — with a correction

Restart-proof per-file runner (`flock`'d, state in `~/.cache/sloughgpt-gate1/`
because `/tmp/opencode` gets wiped). My three edited router files:

| file | result |
|---|---|
| `packages/core-py/tests/test_benchmark_router.py` | 6 passed |
| `packages/core-py/tests/test_health_router.py` | 9 passed |
| `packages/core-py/tests/test_system_router.py` | 19 passed |

86 files carry failures; **none is one I touched, and this work changed no
production code** (until the approved `feedback_dir` fix below). Buckets,
root-caused by sample: native ext not built 42, LoRA/HF 41+52+18, GPU 62+40+14,
`FakeDevice` 24, `KeyError: 'level'` 10, plus `test_user_journeys` 75
(all-fail-in-5s vs slow-partial-pass locally — the known order-dependence).

**Correction (2026-10-07).** "798/798 processed" overstates coverage. The
`done` ledger records per-file summaries, and **11 files read
`NO-SUMMARY exit=5` — pytest collected zero tests and gate 1 logged them as
processed anyway:**

```
test_buildroot_docker   test_multimodal_coverage   test_multimodal_generation
test_multimodal_v2      test_pdf_vlm               test_process_isolation
test_server_integration test_slo_embedder          test_tools
test_infer_router       test_export
```

All eleven are `slow`-marked, so `-m "not slow"` emptied them. Gate 1 therefore
ran **787 files, not 798** — and the eleven gap files are precisely where the
drift hid. Four of them are now fixed (`test_infer_router` 14P,
`test_server_integration` 57P, `test_tools` 61P, `test_process_isolation`
below). A twelfth, `test_quantization_benchmark_e2e`, recorded
`NO-SUMMARY exit=124` — the per-file timeout killed it, so it too has no result.

**The tool was right and my reading of it was wrong**: exit 5 is a legitimate
pytest answer meaning "nothing ran", and a runner that logs *processed* rather
than *outcome* will report success over an empty set. Any future gate should
treat `exit=5` as "not tested", not "passed".

## The slow-test blind spot, quantified

`pytest.ini` `addopts` carries `-m "not slow"`, so **24 slow-marked files /
715 tests are deselected in every default run** — their failures cannot reach
any regression gate. This is the same class as the `test_chat_loop_e2e` 6 ERRORs.

Recounted 2026-10-07 with correct logic (the first pass counted *all* tests in a
file containing any marker): **693 are module-level `pytestmark = slow`** —
meaning the whole file is empty under a default run — and **22 are individual
`@pytest.mark.slow` decorators across 7 files** (`test_learner_pipeline` 9,
`test_server_api` 4, `test_continual_learner` 3, `test_quantization_benchmark_e2e`
2, `test_pugqeep_facade` 2, `test_native_engine_real` 1, `test_buildroot_docker` 1).

## 11 real silent failures → all fixed (`64820c3aa`)

| count | file | root cause | fix |
|---|---|---|---|
| 9 | `tests/test_e2e_smoke.py` | **envelope drift** — endpoints return `success_response()`'s `{"status","data"}`, tests read top level | unwrap to `resp.json()["data"]` |
| 1 | `tests/test_feedback_system.py` | **test drift** — asserted `"filepath"`; no endpoint in the repo returns that key (convention is `"file"`); also sent `max_pairs`, never a field of `ExportTextRequest` | `"file"`, `target_count` |
| 1 | `tests/test_feedback_system.py` | **production bug** — see below | `feedback_dir` defined |

Payload shapes were pinned *empirically*, not guessed: `/souls`, `/models/hf` and
`GET /chat/sessions` return the **bare list** (no `{"souls": …}` wrapper), and
`POST /chat/sessions` puts `"created"` in the envelope's *message*, not the
payload — so unwrapping alone could not have fixed that one.

**Not a failure:** the 2 `test_continual_learner.py` errors were an artifact of
my own `--timeout=120`. Neither `pytest.ini` configures a timeout key. Re-run
without it: **19 passed in 289s** — `train_now()` genuinely trains, which is
precisely why those tests are `slow`-marked. Corrected total: **11, not 13.**

Final: `30 passed, 4 skipped, 1 xfailed, 0 failed`; ruff clean;
`test-doctor --mock-drift` exit 0.

## Production bug 1 — `feedback_dir` (FIXED, user-approved)

`FeedbackController` was migrated to MogDB ("*instead of raw JSON/JSONL*") and
dropped `feedback_dir`, but `training/jobs_api.py:362,376` still read
`ctrl.feedback_dir` → **`/training/export-text` 500'd with `AttributeError` on
every call.** Added `self.feedback_dir = repo_root / "data"`, matching
`training/feeds.py:34` and `experience_adapter.py:71`.

Hollow-but-correct: it now returns 200 with `pairs_count: 0`, because
`data/feedback.jsonl` does not exist anywhere in the main tree. The endpoint was
**never migrated to MogDB** even though `record_feedback()` stores exactly the
fields it filters on (`user_message`, `assistant_response`, `quality`). Reading
the collection instead would make it export real pairs — **not done, needs a call.**

## Production bug 2 — `/auto-train/*` route DELETED (reported, NOT fixed)

`d63f5cb0c` *"split training service into focused modules"* removed the router.
`a95d6ed83` (its parent) still contains it; `d63f5cb0c` does not. Still-broken
consumers:

| consumer | kind |
|---|---|
| `domain/shell/_internal/commands.py:217` | **production** — shell `checkpoints` cmd |
| `domain/shell/_internal/repl.py:200` | **production** — REPL listing |
| `tests/test_e2e_smoke.py` | test (mine) |
| `tests/server/test_server_api.py:188` | test |
| `tests/server/test_endpoint_registry.py` | test |

Card `20260814_002639` (done, Aug 14) shows the endpoint worked then.

My test is now **`xfail(strict=True)`** naming all of the above: green while
broken, **RED the moment the route is restored** — which is the signal to delete
the marker. User chose this over fixing production (out of my test-side lane).

Collateral found in the same run: **`GET /metrics` also 404s**
(`tests/server/test_endpoint_registry.py:167`).

## Incidents

- **Disk ENOSPC mid-run** (`OSError: [Errno 28]`) — `/` at 96%, 5.1G free.
  `apps` 12G, `models` 5.6G, `node_modules` 887M, `~/.cache` 3.2G. **Deleted
  nothing.** Card already exists for the 16GB `opencode.db`.
- **Shell needs an explicit `workdir`** after a server restart — without it
  every command returns empty output, which reads like a tool outage.
- **Branch moved under me** to `feat/dev-ai-unified`; `feat/create-router-projection`
  is an ancestor, all 10 earlier commits verified IN-HEAD. Committed with a
  **pathspec** (`git commit -- <paths>`) so another session's lint-staged
  `slonet_provider.py` stayed staged and untouched; stash `94d6917d4` intact.

# Round 5 — the slow-test march (27 failures cleared)

Walking the 24 slow files that no gate ever runs. **27 failures → 0** across
three commits; every root cause landed in one of four buckets.

## `3f8f9bbe8` — integration, auto-train, infer router

| file | before | after | root cause |
|---|---|---|---|
| `tests/test_integration.py` | 6 failed | 25 P + 2 xfailed | 3 envelope drift, 1 contract drift, 2 dead routes |
| `tests/test_auto_train_integration.py` | 2 failed | 1 P + 1 S + 2 xfailed | `/auto-train/*` deleted |
| `packages/core-py/tests/test_infer_router.py` | 4 failed | 14 P | 2 test-harness defects + 1 dead patch |

`test_integration.py` hits a **live server on :8000**, so its envelope evidence
is real production behaviour, not a fixture artifact.

**Contract drift:** `test_generate_empty_prompt` asserted 200, but
`GenerateRequest` declares `prompt: Field(..., min_length=1)`. 422 *is* the
graceful answer — the test's "server accepts empty prompt" comment predated
schema validation.

**Three defects in `test_infer_router.py`'s harness**, all found by reading what
the router actually reads rather than what the test *thought* it fed it:

1. The bare `FastAPI()` fixture had **no AppError handler**, so `raise_error()`
   re-raised out of `TestClient` instead of becoming the expected 503. Fixed with
   `register_app_error_handler` — the module's own *"minimal handler for test
   clients"* variant.
2. `_patch_state(model=None, phase="ready")` **accepted `phase` and never applied
   it**, and never touched `_model_ready` at all — which reads **global `state`**
   (`state.model` / `state.provider` / `ServerState`), *not*
   `InferRouter._get_model`. So a `FakeModel` in hand still produced 503
   `E_MODEL_LOADING`.
3. **Dual module binding.** `routers.inference` and
   `apps.api.server.routers.inference` are **two distinct module objects for the
   same file** (verified: `id()` differs). My first fix patched the long-form name
   — a silent no-op, because `_get_model_status.__globals__` resolve
   `_model_ready` in `routers.inference`. This is the exact mechanism this whole
   work is about, reproduced in my own fix.

## `732cb6a5c` — the two dead routes, marked where they fail

| file | failures | root cause |
|---|---|---|
| `tests/server/test_endpoint_registry.py` | 3 | `/metrics` + `/auto-train/*` |
| `tests/server/test_server_api.py` | 1 | `/auto-train/*` |

Second dead router: **root `/metrics`** deleted by `5717e1f5f`
*"delete dead routers"* — but it was not dead. Still consumed by
`apps/cli/src/commands/dev.py:1334`, still allowlisted in
`infrastructure/auth_middleware.py:32`, still documented in
`docs/DEPLOYMENT.md:175,178`. Live routes are only `/system/metrics` and
`/benchmark/metrics`. Same pattern, different commit, as `/auto-train`.

All four assertions are `xfail(strict=True)` referencing card `e205b15c`.

**This is systemic, not three coincidences.** By 2026-10-07 the board also carried
`10a762e7` — *"restore `/collections` router (**killed as dead**, frontend still
calls it)"* — already in `done`, i.e. another session hit the identical failure
mode independently. Three routers now, across three separate commits
(`5717e1f5f`, `d63f5cb0c`, and the collections deletion), each time from the same
reasoning: a router with no *test* caller looks like a router with no caller.

What actually proves a route is dead is a **caller census**, not a test grep —
this repo keeps callers outside the test tree (`apps/cli`, the auth allowlist,
`docs/`, `domain/shell`), which is precisely where all three were still wired up.
Cards: `e205b15c` + `db9c4e70` (this work), `10a762e7` (fixed elsewhere).

## The detector validated itself

Hand-probing found `test_generate_with_provider` patching
`domain.models._internal.provider.get_provider` while `infer.py:224` does
`from domain.models import get_provider` — the flagship case from this note's
Round 1. Reverting the test to the bad target and re-running:

```
✖ domain.models._internal.provider.get_provider
  fix: patch("domain.models.get_provider")
1 confirmed  (17 targets follow the patch, 39 latent, 9 suspect)   exit=1
```

**It named the same fix I had derived manually**, from the same two conditions.
A detector that only agrees with you when you already know the answer is worth
trusting; one that finds it cold is worth keeping.

## Incidents, round 5

- **Shell needs an explicit `workdir`** after every restart — without it every
  command returns empty output, which reads exactly like a tool outage.
- **`pytest --timeout=N` is not a repo setting.** Neither `pytest.ini` declares
  it, so passing one *creates* failures. `test_continual_learner` "failed" at
  120s and passes at 289s with no flag.
- **`pytest.importorskip("torch")` gives exit 5** (0 items collected) —
  `tests/test_optimized_pipeline.py` does this, so it is invisible *and*
  non-zero. Sanctioned gate, ambiguous signal.

# Round 6 — `test_process_isolation.py`: a third defect class

`tools/…/slow_batch3.txt`: `test_tools` 61P, `test_server_integration` 57P,
`test_learner_pipeline` 26P all clean. `test_process_isolation.py` —
**11 failed, 62 passed**, splitting into two independent causes.
**Outcome after the fixes: 73 passed, exit 0** (`82d7c8b53`), and the file's
wall time fell **394 s → 88 s** purely from the sparse-file change.

## Cause A — attributes moved into `ModelConfig` (8 → 0 fixed)

`ModelConfig`'s own docstring says it *"Replaces the 20-parameter mix of
guard/slo/hf config that was previously duplicated across `ProcessGuard.__init__`
and `ModelWorkerProcess.__init__`."* After the move the constructor still accepts
the legacy kwargs, but stores them on `self._config` — and **renames two**:

| test read | actual location | renamed? |
|---|---|---|
| `g._slnc_path` | `g._config.slnc_path` | no |
| `g._model_id` | `g._config.model_id` | no |
| `guard._quantize` / `_quant_bits` | `g._config.quant_*` | no |
| `guard.model_cls_path` | `g._config.hf_model_cls_path` | **yes** |
| `guard.model_kwargs` | `g._config.hf_model_kwargs` | **yes** |

Exactly the `feedback_dir` shape from Round 4: the constructor signature stayed
backward-compatible, so nothing errored at *construction* — only the later
attribute reads did. 8 reads rewritten; direct-attribute reads in the file now 0.

## Cause B — a **dead seam** (8 tests), and why `--mock-drift` cannot see it

`TestProcessGuardSlo`'s autouse fixture does
`monkeypatch.setattr(mw_mod, "_slo_worker_main", _fake_slo_worker_main)`. Every
one of the 8 died with `RuntimeError: ThreadWorker[slo-guard]: model load failed`.

Root cause: `ProcessGuard.__init__`'s default is `mode: ExecutionMode =
ExecutionMode.THREAD` (line 381). In THREAD mode `_launch_worker` takes
`_ThreadWorker` and `_run_slo()` runs a **real**
`SloNetChatProvider.from_slnc("/fake/test.slnc")` → the fake path does not exist →
`("dead", None)` → the error. `_slo_worker_main` is read only by the SUBPROCESS
branch (`model_worker.py:837`), which the fixture never reached.

The default flipped in **`b002cc789` — *"feat: add /self-train to sidebar
navigation + fix benchmark import"***, a commit with nothing to do with process
execution mode. A drive-by default change silently retargeted 8 tests at a
different concurrency model.

Fix was the opt-in the file already uses twice (lines 624 and 841):
`"mode": ExecutionMode.SUBPROCESS`. Not a rebuild of the fake — the tests are
*about* process isolation (`g._worker._process.pid`, `os.kill(SIGKILL)`,
`new_pid != old_pid`), semantics `_ThreadWorker` structurally cannot express.
Building a thread-mode fake would have meant deleting the crash assertions.

**Why this is a third class, distinct from confirmed/unresolvable/latent:**

1. The empirical condition **passes** — `monkeypatch.setattr(mw_mod, …)` patches
   the exact module production reads, so the binding *does* move. Condition (1)
   is about binding, and the binding is sound.
2. A `PROBE-CALLED` boom probe **also fails to catch it** — with the seam booby-
   trapped the tests still fail, just with `model load failed` instead of
   `PROBE-CALLED`. The probe only distinguishes *pass* (harmless) from *fail with
   the boom* (confirmed); a test that fails **either way** falls between.

So: **a patch can install perfectly, be aimed at a symbol that genuinely exists,
and still be dead — because the caller stopped routing through it.** Detecting
that needs the call path, not the binding: either the run's own failures
(which is what this march used), or a probe that compares boom-vs-baseline rather
than pass/fail alone.

### Capability candidates fed to `33773fce-b`

1. **Fail-first triage**: run the `slow` set before reasoning about any of it —
   the failures themselves named every seam worth checking. Static drift analysis
   found 0 of these 11.
2. **Boom-vs-baseline probe** for suspects: a suspect whose tests fail *with and
   without* the boom is not "harmless", it is an unrelated failure sitting next
   to a dead seam. Current probe labels that case by omission.
3. **Scan `monkeypatch.setattr(module, "name", …)` too** — `--mock-drift` reads
   `patch("…")` only, so this fixture's target was never enumerated at all.

# Round 7 — a gate that gates nothing it should (`7d2f1d51f`)

Batch 4 came back green (`test_pdf_vlm` 11P, `test_slo_embedder` 45P+1S,
`test_buildroot_docker` 9P) except `test_export.py`, which reported **exit 5
again even with `-m "slow or not slow"`** — so slow-marking was not its cause.

The cause was line 247:

```python
pytest.importorskip("gguf", reason="gguf not installed")
```

placed at **module level, mid-file**. `importorskip` raises `Skipped` during
collection, so the import aborted *there* and pytest recorded
`collected 0 items / 1 skipped` — discarding the entire module, not just the
section that follows it.

**32 tests had never run anywhere.** Gate 1 logged the file as
`NO-SUMMARY exit=5` and counted it processed; the same runner bug as Round 5's
correction.

The gate was unnecessary on three independent grounds:

| test group | count | why it does not need `gguf` |
|---|---|---|
| everything above line 247 | 24 | never references `gguf` (ModelMetadata, configs, SOU, format listing) |
| `TestGGUFWrappers` | 8 | stubs `gguf_export.export_to_gguf` before it can reach the dependency |

The stub works because `export.py` imports that name **inside the function body**
(line 566) — patch-where-it-is-read holds — and `gguf_export.py` itself imports
`gguf` **lazily, inside functions** (lines 884, 1053), never at module level.
`export.py` never imports it at all. With `gguf` confirmed absent
(`ModuleNotFoundError`), **32 passed, exit 0**.

## The class

**An over-broad capability gate.** Distinct from every earlier bucket: nothing is
drifted, nothing was deleted, no seam is dead — a gate is simply wider than the
capability it protects, and pytest's module-level `importorskip` makes the width
invisible because the report says *1 skipped*, not *32 skipped*.

Worth generalising: `importorskip` mid-file is a footgun precisely because the
skip count it reports understates what it removed. A class- or test-scoped
`skipif` states its own blast radius.

### Sweep 2: was the 4 GB write a one-off? Essentially — one file left

Same discipline as above (run the check, get an answer). Scanned every
`test_*.py` for `write_bytes` / big byte-literal allocations — **59 hits, and
57 of them are bytes-to-KB fixtures** (`.soul`, `.soul`, 8-byte placeholders).
Only two files are material:

| file | per-run write | when |
|---|---|---|
| `test_process_guard.py:24,36` | **150 MB** (100 + 50), in RAM *and* on disk | every default run |
| `test_process_isolation.py:642` | 100 MB | `slow` only |

Left both alone deliberately: 150 MB is wasteful, not dangerous, and the disk
has since recovered from **96% (4.6 G free) → 83% (20 G free)**. The one that
*was* dangerous was the 4 GB — and it is gone. Not filing a card for 150 MB of
noise; if the volume ever trends back toward 100%, `test_process_guard.py:24,36`
is the only other place to look.

**A "skipped"/"exit 5" is an empty result, never a green one.** Three files in
this march (`test_export`, `test_optimized_pipeline`, and gate 1's eleven) were
all "passing" only because nothing ran.

### Sweep: is there another one? No — the class is exhausted

First pass over every `test_*.py` flagged **9 "mid-file" `importorskip` sites**
(`test_vision.py:35`, `test_multimodal_generation.py:378/436/443`,
`test_point_library.py:521/541/558`, `test_shell_repl_more.py:27008/27017`).
**All nine were false positives**: the first scan tested *position* only and did
not check *indentation*. Re-ran it distinguishing module-level from
function-level:

| class | count | verdict |
|---|---|---|
| module-level, top of file | 20 | correct — gates exactly its own file |
| function-level (indented) | 13 | correct — skips only that one test |
| **module-level, mid-file** | **0** | **bug — `test_export.py`, already fixed** |

`pytest.importorskip` at column 0 is only a footgun when it appears *after* the
tests have begun; indented inside a test it is the sanctioned idiom. So
`7d2f1d51f` closed the last instance — same shape of conclusion as the
envelope-drift sweep: **run the check, get 0, and know the bucket is empty
rather than assuming it.**
