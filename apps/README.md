## Apps

Runnable services and application entrypoints.

- `api/` — FastAPI server and routers (`api/server/main.py`; see **`api/README.md`**). HTTP training uses **`SloughGPTTrainer`** **`step_*.pt`** charset semantics — **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*).
- `web/` — Next.js frontend (**`app/(app)/`** routes under **`app/`**). Talks to the API over HTTP only (`NEXT_PUBLIC_API_URL`); no Python in the bundle — see **`web/README.md`** (*UI vs core engine*).
- `cli/` — CLI (`sloughgpt` entrypoint via `pyproject.toml`; see **`cli/README.md`** for **`cli.py train`** export naming (**`--save-stem`**) and **`cli.py generate`** local `.sou` resolution).
- `tui/` — Phase 1 Python shell: session + read-only HTTP/local probes (`python3 -m apps.tui`); see **`tui/README.md`**. TypeScript Ink TUI: **`packages/tui-ts/`** (`sloughgpt-tui`). Roadmap: **`docs/plans/tui-cli-port.md`**.

### Quick start (repo root)

Use **Node 20** locally if you can: **`nvm use`** / **`fnm use`** reads **`.nvmrc`** at the repo root (same as **`test-web`** / **`test-sdk-ts`** in **`ci_cd.yml`**).

1. **`./verify.sh`** — checks core paths; runs the same ruff smoke as CI when `ruff` is installed (use `python3 -m pip install -e ".[dev]"` if needed); runs **`apps/web`** **`npm run ci`** when **`node_modules`** exists. Also prints **`ci_cd.yml`** parity commands (**`test-web`**, **`test-sdk-ts`**, **`sdk-test-py`**, **`standards-schemas`**).
2. **API:** `python3 apps/api/server/main.py`, or `cd apps/api/server && python3 -m uvicorn main:app --reload --port 8000`.
3. **Web:** `cd apps/web && npm run dev`. Before pushing web changes, **`npm ci && npm run ci`** (matches CI job **`test-web`** in **`.github/workflows/ci_cd.yml`**). From the **repo root**, **`npm install && npm run dev:stack`** starts API + Next dev together (same as **`./scripts/dev-stack.sh`**). **`npm run test:repo-root`**, **`make test-repo-root`**, or **`python3 -m pytest tests/test_repo_root_package_json.py -q`** runs the root **`package.json`** contract tests.

**Docker:** `docker compose -f infra/docker/docker-compose.yml up -d api` (see **QUICKSTART.md** / **docs/DEPLOYMENT.md** for profiles).
