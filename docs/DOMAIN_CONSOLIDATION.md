# Domain Consolidation — Design Record

**Status:** decision record. No code has been moved.
**Recorded:** 2026-10-02 · **Measured on:** `HEAD 18c720be5`
**Card:** `d0324e7f` (continues `97260724`, the doc-vs-code truth-up)
**Method:** all counts from `git grep` / `find` over **tracked files only**, pattern
`(from|import) domain.<pkg>([.[:space:]]|$)`. `git grep` excludes sibling worktrees and
`node_modules`; plain `grep -r` does not (it produced a 107,500 figure that was
discarded as `.wt-*` leakage). **`[[:space:]]`, not `[\s]`** — see the method
correction in §6, which cost an 11.5% undercount in the first draft.

---

## 1. The decision — `domain/` has four top-level domains

| Domain                                | Files |   Lines | Import refs | Status                                         |
| ------------------------------------- | ----: | ------: | ----------: | ---------------------------------------------- |
| **cognition**                         |    23 |   9,643 |         140 | exists                                         |
| **core**                              |    15 |   7,594 |          19 | exists (lazy `__getattr__` facade)             |
| **generate**                          |     — |       — |           — | **does not exist yet — rename of `inference`** |
| **training**                          |    68 |  37,201 |       1,288 | exists (lazy `__getattr__` facade)             |
| _(`inference`, source of `generate`)_ |  _29_ | _8,700_ |       _490_ | _rename target_                                |

`domain/` held **37 top-level directories** at the start of this effort (36
excluding `cognitive`, a byte-duplicate retained only for its own `__init__.py`
self-imports). It now holds **33**: `billing`, `mobile`, `auth` and `testing`
left for the `services/` and `testing/` layers (§6), which is why the four
above are the whole story rather than four-of-thirty-seven. Of the remaining
packages, `inference` becomes the fourth target by renaming (§3), 2 are deleted,
1 is pending, and the rest nest under one of the four (§6).

`cognition`, `core` and `training` are already **lazy aggregation facades**
(PEP-562 `def __getattr__` in their `__init__.py`). The four are therefore
categories with real code beneath them, not empty barrels — with one exception:
`generate` must be _built_ by renaming, not by adding a re-export layer.

---

## 2. Gateway is not a domain

```
apps/gateway/   Cargo.toml   name = "slough-gateway"   axum 0.7 + tokio 1
                10 .rs files · 0 .py files
                tracked: Cargo.toml, Cargo.lock, src/{main,filters,compression}.rs
domain/gateway/ does not exist        refs to domain.gateway: 0
                refs to apps.gateway: 0
```

**`gateway` is a Rust service.** Different language, runtime, lifecycle and
deployment boundary from anything `domain/` can express. CCGT's "Gateway" is
therefore an _application_ boundary, not a Python package.

**Consequence:** do not create `domain/gateway`. It has no implementation to
hold, and the Rust service is not a candidate for one. A doc that lists
`gateway` among `domain/`'s domains is wrong and should be corrected on sight.

---

## 3. `generate` — what it absorbs, and what it must not

### It absorbs

- `domain/inference` (29 files, 8,700 lines, 490 refs) → renamed to `domain/generate`.
  The package name is the outlier: the external contract already says _generate_ —
  `POST /generate`, `POST /generate/stream`, `SloEngine.generate()`,
  `generate-stream.test.ts`, `apps/web/lib/generate-controller.ts`. Only the
  internal package still says _inference_.
- `domain/generation` (1 file, 68 lines, **0 refs**) → **deleted, not migrated.**

### Why `generation` dies rather than moves

Its docstring claims it _"handles token generation, model inference, voice
synthesis, multimodal processing, and chat"_, but `__all__` contains **no
generate function of any kind** — only vector stores, model interfaces, TTS and
personality objects. Git history shows the sequence:

```
cb8d3ff29  2026-09-20  fix(routers): point dead domain.generation imports at domain.inference
5bf0d7400  2026-09-23  creation (side effect of a 104-file sweep; never modified since)
cd05755bd  2026-10-02  fix(generation): move barrel to domain/, drop broken core-py copy
```

Its consumers were repointed three days _before_ it was committed, so it was born
already orphaned. It is a module that claims to handle generation, exports
nothing that generates, and is imported by nobody.

### Scope guard — must NOT be folded in

`domain/core/_internal/soul.py` lines **372 / 615 / 1321** define
`generate`, `generate_async`, `generate_with_topk`. That is **soul**, a
distinct concern. Pulling it into `generate` is scope creep and would merge two
unrelated public surfaces under one name.

### Mechanism — atomic, no shim

`inference` → `generate` must be `git mv` plus one codemod over the **490 refs**
in a single change. Creating `domain/generate/__init__.py` that re-exports from
`domain.inference` would add a **47th** shim to the 46 we are driving to zero
(§5), re-creating the very coupling the consolidation exists to remove. The
90-line lazy barrel in `domain/inference/__init__.py` should be **dropped in the
move**, not reproduced: `generate` is a real top-level domain, not a re-export.

---

## 4. The `core-py` decision already exists — and is not in HEAD

A written decision has existed since **2026-09-24**, in the _"Dual Python trees
— finding & decision"_ section of `docs/REPO_STRUCTURE_MIGRATION.md`:

> **Consolidate onto singular `domain/` as the only source of truth.**
>
> 1. **Do not** grow `packages/core-py/domains` — new code goes under
>    `domain/…/_internal/` with a thin public facade.
> 2. **Migrate** the ~27 remaining real plural-only files into `domain/`
>    (pugqeep, shell cmds, quant/slnc, …).
> 3. **Retarget** the last plural imports → `domain.*`.
> 4. **Delete** `packages/core-py/domains` only after (2)+(3) and a green suite;
>    then drop `domains*` from packaging `include`.
> 5. **Interim:** keep plural shims so `import domains…` does not break; mark
>    them deprecated in module docstrings.

### Landmine under step 4 — it deletes the native engine's only source

Step 4 says _delete `packages/core-py/domains`_. That tree contains files with **no
copy anywhere else and no copy in git**:

```
packages/core-py/domains/inference/native/
    transformer_forward.c          11,308 B   UNTRACKED
    transformer_forward.h           1,576 B   UNTRACKED
    libtransformer_forward.dylib   21,384 B   UNTRACKED   (macOS artifact)

domain/inference/_internal/native/
    bindings.py · engine.py · weight_mapper.py · __init__.py   ← no native dir at all
```

`bindings.py:25-26` loads from `Path(__file__).parent`, i.e. the **singular** tree —
so the lookup always fails. That single fact accounts for **42 of the baseline
failures** (`test_native_engine_real`: `RuntimeError: libtransformer_forward.dylib
not found`), which is why it is worth more than its size suggests.

Two independent problems, not one:

1. **Wrong tree.** The engine was built into the plural copy; the code reads the
   singular copy. A literal execution of step 4 **silently destroys the source** —
   `git clean` or `rm -rf` would take `transformer_forward.c/.h` with it, and
   nothing in git history would bring them back.
2. **Wrong platform.** The artifact is a `.dylib`; this is a Linux host. Copying it
   across would not help — a `.so` must be built.

**Gate before step 4 runs:** track and relocate
`native/{transformer_forward.c,transformer_forward.h}` →
`domain/inference/_internal/native/`, build `libtransformer_forward.so` there, and
make `bindings.py`'s failure message name the platform-correct compile command
(it currently prints macOS-only `-dynamiclib -framework Accelerate` flags on Linux).

This is also the likeliest answer to open question 1: step 4 may never have landed
because it deletes live, unversioned source.

### Status: executed on branches, absent from HEAD

| Commit      | What                                                            | In HEAD? |
| ----------- | --------------------------------------------------------------- | -------- |
| `68bd7d922` | dual-tree step 2 complete — "plural reals = 0"                  | **no**   |
| `0850db392` | dual-tree step 2c — shell addons/cmds → forward shims           | **no**   |
| `b00a6ee62` | dual-tree step 3 — retarget `domains.*` → `domain.*` (card 057) | **no**   |
| `68872d00c` | step 4 — delete plural tree (297 files)                         | **no**   |
| `cd05755bd` | today — move generation barrel, drop broken core-py copy        | **no**   |

And HEAD still carries the problem in full:

```
files in packages/core-py/domains      301     (0 would mean done)
`domains.*` import lines               141     (0 would mean done)
decision section in HEAD's REPO_STRUCTURE_MIGRATION.md   ABSENT
that doc still lists `domains` under "Canonical paths"    YES
```

Divergence at measurement time:

```
feat/domain-port-merged        HEAD ahead 225   branch ahead 56   (tip 2026-09-24)
feat/move-generation-to-domain HEAD ahead 0     branch ahead 1    (tip 2026-10-02)
```

The main port branch is a week stale and 225 commits behind. **The design is
sound; the diffs are likely unusable.** That is the open question in §8.

---

## 5. The packaging trap — must land _before_ step 4

Every packaging config publishes **`domains*`** and none publishes **`domain*`**:

```python
# setup.py
_core = "packages/core-py"
packages = find_packages(where=_core, include=("domains*", "utils*"))
package_dir = {"": _core, "apps.cli": "apps/cli"}

# pyproject.toml
package-dir = {"" = "packages/core-py", "apps" = "apps"}
include = ["domains*", "utils*", "apps*", "memory*"]

# packages/core-py/pyproject.toml
include = ["domains*"]
```

Observed at `HEAD 18c720be5`:

```
clean interpreter (repo root not on sys.path):
    domains  OK   -> packages/core-py/domains/__init__.py
    domain   FAIL -> ModuleNotFoundError: No module named 'domain'

with repo root on path (pytest / uvicorn):
    domains  OK
    domain   OK   -> domain/__init__.py

editable finder maps: apps, domains, domains.api, domains.context,
                      domains.training.models, apps.* ...
    occurrences of 'domain': 0
```

**`domain` is importable only because cwd happens to be the repository root.**
The installed package publishes `domains`, not `domain`.

### Why this breaks step 4 as written

Step 4 says _"delete `packages/core-py/domains` … then drop `domains*` from
include."_ Done literally, the package ends up publishing **neither**: `domains*`
removed, `domain*` never added. The suite stays green — it runs from the repo
root, so it is structurally blind to this — and the failure surfaces only when
something runs from another directory.

**Ordering correction:** make `domain` installable _first_, then migrate, then
drop `domains*`. The test suite cannot gate this.

### The fix already has precedent in the same file

`package_dir = {"": _core, "apps.cli": "apps/cli"}` is exactly the mechanism for
publishing a top-level package that lives outside the search root — `domain/`
sits at repo root, outside `packages/core-py/`, which is why
`find_packages(where="packages/core-py")` can never see it. Adding
`"domain": "domain"` plus `domain*` to `include` follows the pattern already in
use.

**Should `packages/core-py` survive?** Yes — as _packaging_, not as a code tree.
It is the packaging root; deleting it entirely (an alternative) means the root
`pyproject.toml` must take over publication, and `packages/core-py/tests/` needs
a new home. Recommended: keep `core-py` as a packaging-only shell now, retire it
once the tree is empty.

---

## 6. Placement map — all 37 top-level packages

Measured on `HEAD 18c720be5`, strict pattern, tracked files:

|                                                   |      Refs |
| ------------------------------------------------- | --------: |
| Total `domain.<pkg>` import refs                  | **6,505** |
| Into the **3** existing CCGT targets              |     1,586 |
| Into **non**-target packages                      | **4,919** |
| …of which `inference` (becomes a target after §3) |       490 |
| **Non-target refs after the `generate` rename**   | **4,429** |

**Totals: training 5 · cognition 9 · generate 5 · core 15 · DELETE 2 · PENDING 1 = 37.** ✓

**Method correction (2026-10-03).** The first draft measured every ref count with
`([.\s]|$)` in **POSIX ERE**, where `[\s]` is the character class `{`\`, `s`}
— _not_ whitespace. `from domain.training import DataImporter` therefore never
matched, because a space follows the package name. Every figure below was
undercounted by that bug (total 5,835 → **6,505**, +11.5%). Corrected with
`([.[:space:]]|$)`. **No placement changed** — the ranking order is preserved —
but `shared` (76→192), `logging` (81→135), `testing` (37→90) and `models`
(80→119) were badly understated, and the real post-rename blast radius is
**4,429**, not 3,934.

| Package          | → Domain              |  Refs |  Lines | Evidence                                                                                                                                             |
| ---------------- | --------------------- | ----: | -----: | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `training`       | training              | 1,384 | 37,201 | "Unified training capabilities (datasets, checkpoints, LoRA)"; `_internal/train_pipeline.py`, `lora.py`; lazy facade                                 |
| `feedback`       | training              |   155 |  5,336 | `_internal/{hf_dpo,lora_eval,online_train,model_health}.py`; deps training(9)                                                                        |
| `collections`    | **training**          |   157 |  4,484 | `__all__` = `TrainingDataAdapter`, `TrainingDataConfig`, `TrainingDatasetBuilder`; `training_bridge.py` emits `TrainingDataConfig(block_size=128,…)` |
| `learner`        | training              |    57 |  1,017 | "Continual learner (ingests data, **fine-tunes incrementally**)"; `_internal/continual.py`                                                           |
| `benchmark`      | training              |    12 |    360 | `_internal/domain.py`: "response quality tracking … coherence scoring, perplexity estimation"                                                        |
| `cognition`      | cognition             |   164 |  9,643 | "Reasoning, thinking, creativity, RAG, and consciousness"; lazy facade                                                                               |
| `memory`         | cognition             |    83 |  3,845 | "chat- and task-agnostic auto-memory facade"                                                                                                         |
| `agents`         | cognition             |    67 |  3,483 | `Agent/ToolRunner/SecurityBoundary`; deps knowledge(2), multimodal(2)                                                                                |
| `knowledge`      | cognition             |   117 |  2,375 | "high-level knowledge API"; deps memory(20)                                                                                                          |
| `multimodal`     | **cognition**         |   109 |  8,806 | `manager.py`: "Unified interface for **speech recognition and image understanding**"; router census 5 perception / 1 generate                        |
| `soul`           | cognition             |    28 |  1,374 | "Cognitive engine, HD memory, quantum processing"                                                                                                    |
| `context`        | cognition             |    25 |    992 | `PersonalityManager/MemoryManager/StyleManager/ConsciousnessManager`; deps shared, infrastructure only                                               |
| `companion`      | cognition             |    11 |    296 | `ResponseStyle/CompanionTraits/ConversationContext`; 0 internal deps                                                                                 |
| `ai_personality` | cognition             |     6 |    165 | `PersonalityType/PERSONALITIES` = identity; 0 internal deps                                                                                          |
| `inference`      | **generate** (rename) |   490 |  8,700 | "Model inference, vector store, soul format"; contract already says _generate_ (§3)                                                                  |
| `models`         | **generate**          |   119 |  1,752 | `_internal/models.py` defines `ModelInterface.generate()` (L53), `SloughGPTModel.generate()` (L260)                                                  |
| `voice`          | generate              |    29 |  5,311 | "Voice layer - TTS, STT, phoneme encoding"; `routers/phoneme.py` calls it "single source of truth"                                                   |
| `chat`           | generate              |    21 |    721 | `_internal/domain.py`: "receive message → **generate response** → log"                                                                               |
| `tools`          | generate              |     5 |    597 | `TOOL_PROFILES` + `ToolsEngine` "Registry + prompt rendering"                                                                                        |
| `core`           | core                  |    38 |  7,594 | "The 'body' — config, database, events, numpy engine, NPU, kernel, VM"                                                                               |
| `infrastructure` | core                  | 1,350 | 33,718 | "Core infrastructure (config, event bus, lifecycle, errors)"                                                                                         |
| `shell`          | core _(see §6.1)_     | 1,524 | 55,561 | "Interactive Dait shell (kernel, processes, VM, TUI)"                                                                                                |
| `logging`        | core                  |   135 |  3,466 | "OOP logger hierarchy"; only `fastapi` hits are log-suppression patterns                                                                             |
| `shared`         | core                  |   192 |  1,010 | "Shared utilities, types, constants across all domains"                                                                                              |
| `slolib`         | core                  |    16 |  1,299 | "Unified tensor library (autograd, GPU, inference)"                                                                                                  |
| `testing`        | core _(forced)_       |    90 |  1,019 | "UI journey testing library" — 83/91 import lines are tests                                                                                          |
| `ops`            | core                  |     9 |    616 | `_internal/ops.py`: "Pure NumPy … Fused attention, Fused layer norm" — **numerics, not DevOps**                                                      |
| `settings`       | core                  |    24 |    322 | "Persistent user configuration"; 0 internal deps                                                                                                     |
| `plugins`        | core                  |     8 |    228 | "Plugin framework"; 0 internal deps                                                                                                                  |
| `errors`         | core                  |     6 |     72 | "no FastAPI imports here so domains stay portable"                                                                                                   |
| `api`            | core _(forced)_       |    26 |    279 | SSE **envelope**, single module `sse_envelope.py`, 0 framework imports; consumers = `infrastructure/{training,task}_queue.py` + 4 routers            |
| `auth`           | core _(forced)_       |    14 |    628 | `User/Tenant/Workspace/RBAC`; production consumers = 4 `apps/api/server/routers/*` (users, tenants, workspaces, search)                              |
| `mobile`         | core _(forced)_       |    15 |    469 | "Push notification service for React Native … Expo Push"; consumers `routers/mobile.py`, `training/helpers.py`                                       |
| `billing`        | core _(forced)_       |    13 |    305 | `TokenAccount/TierManager/CreditManager`; sole production consumer `routers/tokens.py:22`                                                            |
| `cognitive`      | **DELETE**            |     3 |   4876 | 9/10 files byte-identical to `core`; 0 external importers (the "3" are its own `__init__.py`)                                                        |
| `generation`     | **DELETE**            |     0 |     68 | 0 refs; `__all__` has no generate function (§3)                                                                                                      |
| `search`         | **PENDING**           |     3 |  1,121 | 17 untracked files, 0 tracked — in-flight, never classify as dead                                                                                    |

Constraint checks, both verified: **no package in `domain/` imports fastapi/starlette
or uses `APIRouter`/`HTTPException`/`Depends`** — the only textual hits are
`logging/_internal/errors.py:37-38` (a _suppression_ list) and
`ops/_internal/wandb_server.py:70` (a wandb tag string). And `domain/gateway`
still does not exist, `apps/gateway` is still Rust.

### Three placements the evidence overruled

- **`collections` → `training`, not `cognition`.** `perception.py` + `world_bridge.py`
  are 462 of 4,484 lines (10%) and **neither is in `__all__`**. The `baby_*` world
  sim (1,700 lines) is imported by **tests only**. Decisive: **zero `domain/*`
  packages import `collections`** — its only consumers are `routers/collections.py`,
  `apps/cli/src/groups/{collect,world}.py` and tests. It is a leaf that _manufactures
  data for the learning loop_; cognition's perception slot belongs to `multimodal`.
- **`models` → `generate`, not `core`.** It defines `generate()` and answers
  `POST /generate` via `provider/{router,slo_transformer}.py`. The one prior attempt
  at a "generate" package (`domain/generation`) already claimed it. ⚠ `_internal/models.py:22-39`
  has six **eager top-level** `from domain.training._internal.slonet import …` — a
  `generate → training` edge that only exists because the tensor core sits in the
  wrong package (§6.1).
- **`multimodal` → `cognition`, not `generate`.** This record already said
  `cognition` in its draft table; the evidence agrees. Router census: **5 perception,
  9 training, 6 phoneme, 1 generate**. Its speech endpoints delegate to
  `domain.voice`, and its six phoneme encoders are byte-identical stale copies of
  `voice`'s (3,373 lines) while `routers/phoneme.py` declares `voice` the source of
  truth. Follow-up: extract `LatentDiffusionModel`/`SloVAE` (~947 lines) into
  `generate`, delete the duplicate encoders.

### Five packages that do not fit any of the four

Product/business/transport concerns — none contains API-server code, so this is
about _domain_, not framework.

**Executed.** `domain/` now contains `cognition`, `core`, `generate`,
`training`. `billing`, `mobile` and `auth` moved to a new **`services/`** layer
at repo root; `testing` moved to `testing/` at root.

| Package   | Placement                                 | Why it is not one of the four                                                                                                                                                |
| --------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `billing` | **`services/billing`** ✓ done             | Subscription tiers and credits; sole consumer `routers/tokens.py`; **0** tracked intra-domain edges                                                                          |
| `mobile`  | **`services/mobile`** ✓ done              | Expo Push device tokens; sole consumer `routers/mobile.py`; **0** tracked intra-domain edges                                                                                 |
| `auth`    | **`services/auth`** ✓ done                | Tenancy/RBAC: 8 routers, **0** tracked intra-domain edges                                                                                                                    |
| `testing` | **`testing/`** (repo root) ✓ done         | Browser automation; 83/91 import lines are tests, 0 runtime consumers                                                                                                        |
| `api`     | **stays in `domain/` → `core`** ⏸ blocked | Pure wire-shape, but `domain/infrastructure/_internal/{task_queue,training_queue}.py` consume it (**12** cross-package refs). Moving it out would invert `domain → services` |

#### The services layer

`services/` is a **layer**, not a fifth domain — it sits at repo root as a peer
of `domain/`, `apps/` and `infra/`, all of which are root-level:

- `services/*` may import `domain/*`.
- **Nothing under `domain/` may import `services/*`.** That invariant is the
  whole reason the split exists; `domain/api` is the one package that had to
  stay inside `domain/` precisely because of it.
- Packaging publishes `services*` and `testing*` alongside `domain*`
  (`setup.py` `package_dir` + `pyproject.toml` `include`) — the §5 trap applies
  to every root-level package.

**`testing/` is at root, not `packages/testing/`.** `packages/*` is only
importable _after_ `pip install -e .`, so parking it there would break 11 test
files until the reinstall lands. At root it resolves the same way `domain` and
`services` already do. Its import name `testing` was verified free in both the
conda env and `.venv`.

**One known exception, untracked.** `domain/search/adapters/members.py:22`
imports `services.auth` (`UserRepository`, `WorkspaceRepository`) — a
`domain → services` edge. This file is **0 tracked / 17 untracked** (card
`97260724`), so nothing committed is affected, but its owner must clear the
inversion when `search` gets placed: either `search` also leaves `domain/`, or
its user/workspace lookups go through a `core` repository.

Note that `git grep` silently skips untracked files — this edge is invisible to
it, and an earlier intra-domain census run with `git grep` reported **0**
intra-domain consumers for `auth`, which was wrong.

Two structural notes:

- **`domain/core` has 19 refs against 7,594 lines** — because it is lazy: the
  package is reached as `import domain.core` + attribute access, which produces
  no static `from`/`import` line. **Ref counts systematically undercount the four
  lazy packages** (`cognition`, `core`, `inference`, `training`). They are
  useful for _relative_ ranking, not as a merge gate.
- The nesting must be **one atomic codemod** (~4,429 refs) or it reintroduces the
  shim layer. Leaving `domain/<pkg>` as a compat re-export puts us back at the
  46-shim state this whole effort is trying to escape.

### 6.1 `shell` — the one placement that needs a caveat

**`core` is the right label among the four, and it is still the wrong physical
home.** `shell` is the only package that genuinely behaves like a fifth domain.

It _is_ the substrate more literally than `shared` or `errors` are — `kernel.py`,
`kernel_{process,scheduler,syscall,memory,interrupts}.py`, `vm_{engine,devices,permissions}.py`,
`vfs.py`, `ioctl.py`: a kernel, a process model, a syscall surface, an x86 VM.
`domain/core/__init__.py` already says so ("NPU, kernel, VM") and eagerly imports
`DaitRuntime, Kernel, NeuralKernel` at line 41. It has no better home — it is
neither perception/memory/knowledge, nor token output, nor the learning loop.

Why it is still the biggest risk:

1. **It inverts the layering `core` stands for.** Measured direction:
   `shell → training(9), shared(6), infrastructure(6), inference(5), logging(3)`
   — all deferred (e.g. `vm_devices.py:611` imports `SloAdam, Tensor` from
   `training`; `npu_device.py:471` imports `SlonetChatProvider` from `inference`).
   A substrate that depends on the learning loop and the generator is not a
   substrate. Real `core` members do not: `errors → {infrastructure}`;
   `settings/plugins/companion/ai_personality/api/testing → {}`.
2. **Mass.** 63 files / 55,561 lines vs `core`'s 15 / 7,594 — **7.3×**. A blind
   `git mv` yields a 78-file, 63k-line `core` larger than `infrastructure`, hiding
   a TUI, a 3D world renderer and an x86 emulator under "shared utilities."
3. **It is three subsystems stapled together:** an OS (kernel/VM/devices), a UI
   (TUI/markdown/prompt/console), and a world sim (`world_driver/world_render/simulation`)
   — the same world `collections/baby_*` and the staged `routers/world_render.py` touch.
4. **Blast radius:** 1,514 refs across 212 files, and **2 of its own files are
   currently staged** (`domain/shell/__init__.py`, `domain/shell/_internal/vm.py`).

**Ruling:** `shell` is **`core`-domain, but stays a top-level package** — group it
under `core` in the record, do not relocate its bytes. If "four" is ever relaxed,
`shell` is the fifth (`runtime`), and by mass it is a peer of `training`, not a
child of `core`.

**Runner-up: `infrastructure`** (171 files / 33,718 L / 1,333 refs across 413
importing files — the widest fan-out in the repo). It is not pure substrate
either: `model_server.py` (1,761 L), `model_worker.py` (1,321), `inference_engine.py`
(622) and `training_queue.py` (415) are _serving_ code, 13.6% of the package. All
29 cross-domain edges are deferred (0 top-level), which makes the codemod
mechanical but very wide — and 18 of them rewrite inside `core` when `inference`
becomes `generate`.

**Latent, and larger than either:** `domain/training/_internal/slonet.py` is
**8,276 lines** of tensor/autograd/model core (`Tensor`, `SloTransformer`,
`SloAdam`, `export_to_sou`) imported by 9 packages plus `apps/cli`, `apps/api`
and `scripts` — the _numerics substrate for the whole system lives inside the
learning-loop package_, while `slolib` (whose docstring claims autograd/GPU/
inference) has its `_internal/slolib.py` on the callerless list (§7). `slolib → core`
is right, but `core` does not actually own the numerics until `slonet.py` is
extracted out of `training`. Follow-up, not a blocker.

---

## 7. Caller reachability — why code gets deleted

`domain/generation` was not deleted for being wrong; it was **swept for having no
roots**. Routers were repointed away from it, nothing re-established a caller,
and it became unreachable. The same fate awaits anything built, tested, but never
wired into a handler.

**Method, and its failure rate:** a first pass flagging modules with zero static
references returned **32**. Re-resolving relative imports (`from .base import X`)
and facade re-exports cut that to **10** — a **69% false-positive rate**. This is
the fourth time string-matching has misled this analysis (three earlier
static-import passes were discarded outright). Any tooling for this must build an
import graph from **AST the way Python resolves**, not grep dotted paths.

Genuinely callerless at `HEAD 18c720be5` (10):

```
domain/cognitive/_internal/grounding.py            domain/core/_internal/math_rag.py
domain/cognitive/_internal/knowledge_graph_v2.py   domain/infrastructure/safetensors_loader.py
domain/core/_internal/base.py                      domain/slolib/_internal/slolib.py
domain/core/_internal/core.py                      domain/training/_internal/owned_objective.py
domain/core/_internal/grounding.py                 (domain/core/_internal/knowledge_graph_v2.py)
```

Note `math_rag.py` has a 421-line `tests/test_math_rag.py` but no production
caller — built, tested, and still collectable, because a test is a weak root.

**Not dead:** `domain/search/` (17 files) and `apps/api/server/routers/search.py`
are **100% untracked** — in-flight work, not orphans. Any reachability check must
classify untracked code as _pending_, never as dead.

---

## 8. Sequencing

**Landed:** step 1 (this record + the `REPO_STRUCTURE_MIGRATION.md` decision
restored) and step 2 (`domain` published in `setup.py`/`pyproject.toml`) are in
**`feat/domain-consolidation-phase0` @ `c2f1e0369`**. The packaging fix is
inert until `pip install -e .` re-runs the editable finder — that mutates the
shared `.venv`, so it needs an owner's go-ahead.

**Blocking facts, re-measured and corrected:** there are **6 staged files under
`domain/`**, not 1 —

```
domain/inference/_internal/slonet_provider.py      domain/voice/engine.py
domain/infrastructure/_internal/model_config.py    domain/shell/__init__.py
domain/infrastructure/_internal/process_guard.py   domain/shell/_internal/vm.py
```

plus **19 staged `apps/api/server/` files**, whose `domain.*` imports reach
`infrastructure`(8), `training`(7), `shared`(5), `api`/`models`/`feedback`(2 each)
and `inference`/`tools`/`agents`/`cognition`/`core`/`knowledge`/`logging`/`shell`
(1 each). Two distinct conflict types: **(A)** a staged _file_ cannot be `git mv`ed
→ blocks `inference`, `infrastructure`, `shell`, `voice`; **(B)** a staged _file's_
imports cannot be codemodded → additionally blocks `tools`, `models`, `agents`,
`api`, `cognition`, `core`, `knowledge`, `feedback`, `logging`, `training`, `shared`.

1. **Confirm the 221-file stage has landed or been aborted.** Nothing below should
   start while it is open.
2. ~~Land the decision record~~ · ~~Publish `domain`~~ — both in `c2f1e0369`.
3. **Delete first** (zero codemod, verify with AST not grep — §7's 69% false-positive
   rate): `generation`, then `cognitive`.
4. **Leaves with no staged consumers** — `errors`, `plugins`, `settings`,
   `ai_personality`, `companion` → core/cognition. (`tools` has one staged
   consumer, `routers/tools.py` — defer it to 6.)
5. **Small/leaf** — `benchmark`→training, `context`→cognition, `ops`→core _after_
   extracting `_internal/wandb_server.py` to `training` (it would otherwise add a
   `core → training` edge), `slolib`→core (resolve the callerless `_internal/slolib.py`
   in the same change). ~~then the five does-not-fit packages (§6)~~ — **done**:
   `billing`/`mobile`/`auth` → `services/`, `testing` → `testing/` (§6).
   **`api` → `core` is the only remainder and is currently blocked**: its two
   consumers `apps/api/server/infrastructure/sse_fallback.py` and
   `apps/api/server/routers/agents.py` are both staged.
6. **`inference` → `generate`** (§3) — one atomic `git mv` + one codemod over the
   490 refs, no shim. Must also hit the 8 `packages/core-py/domains/*` files that
   reference `domain.inference`, or those shims break.
7. **Mid/large, gated on the stage resolving** — `voice`, `logging`, `shared`,
   `models`, `multimodal`, then `training` and `infrastructure`.
8. **`shell` dead last** (§6.1) — group under `core`, do not move the bytes.
9. **Caller-reachability gate** — as a CI assertion in the existing parity script,
   not an advisory report.

`domain/search` stays **pending** through all of it (§6) — 17 untracked files
invisible to `git grep`, as is `apps/api/server/routers/search.py` and 12 search
test files.

---

## 9. Parity

**This document deliberately carries no `parity-claims:v1` block.**

`scripts/check_docs_api_parity.py` reads claims from exactly one file —
`CLAIMS_MD = docs/DOC_VS_CODE_GAP_AUDIT.md` (`CLAIMS_BLOCK_RE` is searched only
there) — and accepts only eleven measurable keys (`routers_internal_*`,
`controllers_*`, `combined_internal_*`). Every claim key must be present in
`arch_measured`, otherwise the run reports
`"<key>: claimed but not measurable"` and fails. This document's figures are not
in that set, so a block here would be **ignored**, and one in the right file
would **break the build**.

Figures below are for human review. Machine-checking them requires extending
`collect_architecture()` with new measured keys — a separate change, not made
here.

| Claim                                | Value                               | Method                                                     |
| ------------------------------------ | ----------------------------------- | ---------------------------------------------------------- |
| `domain/` top-level dirs             | 37 (36 excl. `cognitive`)           | `ls -d domain/*/`                                          |
| CCGT four for `domain/`              | cognition, core, generate, training | decision, §1                                               |
| `domain/gateway` exists              | no (0 refs)                         | `find` + `git grep`                                        |
| `apps/gateway` language              | Rust, 0 `.py`                       | `find` + `Cargo.toml`                                      |
| `domain/generation` refs             | 0                                   | `git grep`, strict pattern                                 |
| `core-py/domains` files in HEAD      | 301                                 | `git ls-tree -r HEAD`                                      |
| `domains.*` import lines in HEAD     | 141                                 | `git grep HEAD`                                            |
| Non-target import refs               | 4,919 → 4,429 post-rename           | `git grep`, `(from\|import) domain.<pkg>([.[:space:]]\|$)` |
| Path collisions, both trees          | 84                                  | `comm -12` of `find`                                       |
| Reverse shims (`domain` → `core-py`) | 46 files                            | `grep -rln 'from domains\.'`                               |
| Genuinely callerless modules         | 10 (32 raw − 22 false positives)    | §7 method                                                  |

---

## 10. Open questions

1. **Resurrect or redo `core-py` steps 2–4?** All five step merges are absent
   from HEAD, which is 225 commits ahead of the port branch. The design is
   settled; the diffs are stale. _Partially answered:_ step 4 deletes live,
   unversioned native-engine source (§4) — that alone is a plausible reason it
   never landed, and it is now a hard precondition rather than an open question.
   The remaining archaeology is whether it broke anything _else_.
2. ~~Do the five does-not-fit packages actually leave `domain/`?~~ → **Closed
   for four of five.** `billing`, `mobile`, `auth` → `services/`; `testing` →
   `testing/` (§6). `domain/` is now `cognition`, `core`, `generate`,
   `training`. Only `api` stays, because moving it would invert
   `domain → services`. The one live thread is `domain/search`'s untracked
   `domain → services` edge (§6).
3. **Is `packages/mogdb` meant to be installed?** It is a real package
   (`pyproject.toml` + `src/`) that is **not installed**, and it is the single
   largest source of red in the suite: **127 failures and collection errors**
   (75 test failures, 52 collection errors) all reduce to one missing
   prerequisite — `mogdb` resolves as a namespace package in some tests
   (`cannot import name 'MogDB' from 'mogdb' (unknown location)`) and not at all
   in others. One `pip install -e packages/mogdb` would clear it. _Candidate:
   add a packaging preflight to the test entrypoint so a missing prerequisite
   fails loudly instead of diffusing into 127 red lines._
4. **Should `collect_architecture()` learn these keys?** Extension would make §9
   machine-checked, but enlarges the contract the CI job enforces.
5. **Which of the 10 callerless modules get wired and which get deleted?**
   Ownership sits with whoever holds the routers — one-file-per-feature means a
   fresh `routers/<f>.py`, which collides with another session's staged router
   work.

**Closed since the first draft:**

- ~~Where do `billing` / `mobile` / `auth` go?~~ → **`services/` layer** (§6).
  Not `core` — _core is for database and the like_ — and not `apps/api/`,
  because `billing` contains no HTTP at all and nesting it there would invert
  domain ↔ transport. `core` now means something specific instead of being the
  fallback bucket.
- ~~Does `testing` belong in `packages/`?~~ → **No: repo root.** `packages/*`
  only resolves after `pip install -e .`, so `packages/testing/` would break 11
  test files until that reinstall lands (§6).

- ~~`shell` has no CCGT home~~ → **`core`-domain, top-level package; do not
  relocate the bytes** (§6.1).
- ~~`collections` is open~~ → **`training`** (§6): 0 `domain/*` consumers, and its
  public surface is `TrainingDataAdapter`/`TrainingDatasetBuilder`, not perception.
- ~~`multimodal` vs `models`~~ → **`cognition`** and **`generate`** respectively
  (§6), both settled by router census and `ModelInterface.generate()`.
