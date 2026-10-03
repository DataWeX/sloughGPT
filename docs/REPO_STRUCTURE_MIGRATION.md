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
- Python **domain** package: `domain/` (import name **`domain`**) — **canonical**. The plural `packages/core-py/domains` (import name `domains`) is legacy and scheduled for deletion; see the decision below.
  > **Packaging caveat.** `setup.py` and `pyproject.toml` now publish `domain*`, `services*` and `testing*` explicitly (`package_dir` + `include`), but that fix is **inert until `pip install -e .` is re-run**. Until then `import domain` resolves _only_ because the repo root is on `sys.path`, while `import domains` succeeds from the installed editable finder. Run the reinstall **before** deleting the plural tree, or the installed package ends up publishing neither. See _Domain Consolidation_ §5.
- Python **services** layer: `services/` (import name **`services`**) — `auth`, `billing`, `mobile`. A _layer_, not a fifth domain: `services/*` may import `domain/*`, and nothing under `domain/` may import `services/*`.
- Python **testing** library: `testing/` (import name **`testing`**) — UI-journey/browser automation. Deliberately at repo root rather than `packages/testing/`, because `packages/*` resolves only after an editable install.
- Python SDK: `packages/sdk-py/sloughgpt_sdk`
- TypeScript SDK: `packages/sdk-ts/typescript-sdk`
- Standards / schemas: `packages/standards/`

---

## Dual Python trees — finding & decision (2026-09-24)

> Restored 2026-10-03. This section was written in `6dfd4f857` but never reached
> `main`, so this document asserted the opposite of the decision for eight days.
> Full current analysis: **`DOMAIN_CONSOLIDATION.md`**.

### Current state

Two importable trees coexist:

| Tree                    | Path                        | Import name | Role                                                                 |
| ----------------------- | --------------------------- | ----------- | -------------------------------------------------------------------- |
| Singular (migrated)     | `domain/` (repo root)       | `domain.*`  | **Canonical** for almost all application code                        |
| Plural (legacy install) | `packages/core-py/domains/` | `domains.*` | Mostly backward-compat shims; pip-installable via `packages/core-py` |

### Decision

**Consolidate onto singular `domain/` as the only source of truth.**

1. **Do not** grow `packages/core-py/domains` with new code — all new modules go under `domain/…/_internal/` with a thin public facade.
2. **Migrate** the ~27 remaining real plural-only files into `domain/` (pugqeep, shell cmds, quant/slnc, …), leaving shims behind until callers are retargeted.
3. **Retarget** the last plural imports (`test_slolib_gpu`, `benchmark_block_quantization`) to `domain.*`.
4. **Delete** `packages/core-py/domains` only after (2)+(3) and a green suite; then drop `domains*` from packaging `include`.
5. **Interim:** keep plural shims so `import domains…` does not break external scripts; mark them deprecated in module docstrings.

**Why not keep dual trees long-term:** bidirectional shims already cause attribute errors (e.g. `KNOWLEDGE_DIR` on a re-export module); two packaging roots invite split-brain; the import graph is already ~99% singular.

### Refresh — what changed since 2026-09-24

Measured `HEAD 18c720be5`, 2026-10-02, tracked files only.

| Metric                                   | 2026-09-24 | 2026-10-02 | Direction |
| ---------------------------------------- | ---------: | ---------: | --------- |
| `domain/` tracked `.py`                  |        574 |        589 | +15       |
| `domain/` reverse shims → `domains.*`    |         77 |     **46** | −31 ✓     |
| `packages/core-py/domains` tracked `.py` |        268 |    **301** | **+33 ✗** |

**The plural tree grew by 33 files**, in breach of decision 1 ("do not grow
`packages/core-py/domains`"). Steps 2–4 of this plan were executed and merged on
branches (`68bd7d922`, `b00a6ee62`, `68872d00c`) but **none are ancestors of
`HEAD`** — the port branch is 225 commits behind. The design stands; the diffs
are stale. See `DOMAIN_CONSOLIDATION.md` §4 and §8 for the resurrect-vs-redo
decision that is still open.

### Packaging blocker (must precede step 4)

Step 4 says to drop `domains*` from packaging `include`. Today nothing publishes
`domain*`, so doing that first leaves a package that imports **neither** name.
`domain` is importable only because the repo root is on `sys.path`. Add the
`domain` mapping to `package_dir` before deleting the plural tree —
`DOMAIN_CONSOLIDATION.md` §5.
