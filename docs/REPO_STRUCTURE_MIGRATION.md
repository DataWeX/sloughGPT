## Repository Structure Migration

This repository has been moved to a layered monorepo layout:

- `apps/` runnable applications (`api`, `web`, `cli`)
- `packages/` shared libraries (`core-py`, `sdk-py`, `sdk-ts`, `standards`)
- `infra/` docker and kubernetes assets

### Root compatibility shims (removed)

Some earlier layouts used root symlinks (for example `server` → `apps/api/server`, `web` → `apps/web`). **Those links are not present in the current repository.** Use the canonical paths below and update any old scripts or docs that still assume root aliases.

Deployment assets live under **`infra/`** (Docker, Kubernetes, Helm, etc.).

### Canonical paths

- API server: `apps/api/server/main.py`
- Web app: `apps/web`
- CLI (also reachable as repo-root `cli.py`): `apps/cli/cli.py`
- Python “domains” package: `domain` (import name **`domains`** after **`python3 -m pip install -e .`** from repo root)
- Python SDK: `packages/sdk-py/sloughgpt_sdk`
- TypeScript SDK: `packages/sdk-ts/typescript-sdk`
- Standards / schemas: `packages/standards/`

## Dual Python trees — finding & decision (2026-09-24)

### Current state

Two importable trees coexist:

| Tree | Path | Import name | Role |
|------|------|-------------|------|
| Singular (migrated) | `domain/` (repo root) | `domain.*` | **Canonical** for almost all application code |
| Plural (legacy install) | `domain/` | `domains.*` | Mostly backward-compat shims; pip-installable via `packages/core-py` |

**Import census (this repo):**

- `packages/core-py` → `domain.*`: **1390** / → `domains.*`: **1** (`test_slolib_gpu`)
- `apps/` + `tests/` → `domain.*`: **173** / → `domains.*`: **0**
- `scripts/` → `domain.*`: **51** / → `domains.*`: **2** (`benchmark_block_quantization` → pugqeep)

**File counts:** `domain/` **574** `.py` (≈226 real, 77 shims still pointing at `domains.*`, 231 internal facades).  
`domain` **268** `.py` (**232 shims → `domain.*`**, only **~27 real** modules left).

### Remaining real code only in `domain` (~27)

- `infrastructure/pugqeep/*` (engine, compressor, point, …)
- `infrastructure/quant_core/wrapper.py`, `infrastructure/slnc/spec.py`
- `infrastructure/cache/`, `infrastructure/deployment/`, `infrastructure/gpu/compile_shaders.py`
- `shell/addons/*`, `shell/cmds/*` (linux, dashboard, …)
- `collections/training_bridge.py`

Everything else in the plural tree is a `Backward-compatibility shim` re-exporting `domain.*`.

### Packaging note

Root `setup.py` / `packages/core-py/pyproject.toml` still **`include = ["domains*"]`** — editable installs expose **`domains`**, while the monorepo path also puts **`domain`** on `sys.path`. That is why both import names work today.

### Decision

**Consolidate onto singular `domain/` as the only source of truth.**

1. **Do not** grow `domain` with new code — all new modules go under `domain/…/_internal/` with a thin public facade.
2. **Migrate** the ~27 remaining real plural-only files into `domain/` (pugqeep, shell cmds, quant/slnc, …), leaving shims behind until callers are retargeted.
3. **Retarget** the last plural imports (`test_slolib_gpu`, `benchmark_block_quantization`) to `domain.*`.
4. **Delete** `domain` only after (2)+(3) and a green suite; then drop `domains*` from packaging `include`.
5. **Interim:** keep plural shims so `import domain…` does not break external scripts; mark them deprecated in module docstrings.

**Why not keep dual trees long-term:** bidirectional shims already cause attribute errors (e.g. `KNOWLEDGE_DIR` on a re-export module); two packaging roots invite split-brain; the import graph is already ~99% singular.

### Related pre-existing test breakage (this card)

- `test_learner_continual` fixture patched `domain.learner._internal.knowledge` (shim without path attrs) → **fixed** to patch `domain.memory._internal.knowledge_store`.
- `test_slonet_wave_a` accelerator tests mocked old `domain.slolib.gpu` / `domain.training.gpu` paths → **fixed** to `…_internal.gpu` matching production imports.
- `test_slo_manager`, `test_truth_maintainer`, `test_kg_pipeline`, `test_console_logger` were already green on main.

### Follow-ups (not in this card)

- `.wt-merge/` still holds pre-cutover copies (separate cleanup).
- Quantization `make_torch_forward` gap → card `20260924_037`.
