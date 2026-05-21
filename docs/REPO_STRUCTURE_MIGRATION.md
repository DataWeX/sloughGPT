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
- Python “domains” package: `packages/core-py/domains` (import name **`domains`** after **`python3 -m pip install -e .`** from repo root)
- Python SDK: `packages/sdk-py/sloughgpt_sdk`
- TypeScript SDK: `packages/sdk-ts/typescript-sdk`
- Standards / schemas: `packages/standards/`
