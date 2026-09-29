## Apps

Runnable services and application entrypoints.

- `api/` — FastAPI server and routers (`api/server/main.py`; see **`api/README.md`**). HTTP training uses **`SloughGPTTrainer`** **`.soul`** charset semantics — **`docs/policies/CONTRIBUTING.md`** (_Checkpoint vocabulary_).
- `web/` — Vite frontend (**`app/(app)/`** routes under **`app/`**). Talks to the API over HTTP only (`NEXT_PUBLIC_API_URL`); no Python in the bundle — see **`web/README.md`** (_UI vs core engine_).
- `cli/` — CLI (`sloughgpt` entrypoint via `pyproject.toml`; see **`cli/README.md`** for **`sloughgpt train`** export naming (**`--save-stem`**) and **`sloughgpt generate`** local `.soul` resolution). `sloughgpt shell` is the interactive TUI (split-pane curses; `sloughgpt tui` is an alias) and the line-mode REPL when run without `--tui`.
- `gateway/` — Rust edge gateway (`gateway/src/main.rs`, release binary): owns the public socket (:8080) — health-contract passthrough, CORS, compression, rate limits — then proxies to the API. Built once, shared by every session (`cargo build --release`); self-supervising (`MAN_GATEWAY_SUPERVISE=1`). See **`docs/engineering-overview.md` §5**.
- `mobile/` — mobile app. Its `node_modules` (~7.5G) is regenerable build state — do NOT rebuild it while work can use the shared root install (shared-not-rebuilt rule in **AGENTS.md**).
- `data/` — user data stores (chat sessions, conversations, experiments, feedback, training exports, voice messages, API keys). **Never delete** — see **AGENTS.md** _File Safety_.

### Quick start (repo root)

Use **Node 20** locally if you can: **`nvm use`** / **`fnm use`** reads **`.nvmrc`** at the repo root (same as **`test-web`** / **`test-sdk-ts`** in **`ci_cd.yml`**).

1. **`./verify.sh`** — checks core paths; runs the same ruff smoke as CI when `ruff` is installed (use `python3 -m pip install -e ".[dev]"` if needed); runs **`apps/web`** **`npm run ci`** when **`node_modules`** exists. Also prints **`ci_cd.yml`** parity commands (**`test-web`**, **`test-sdk-ts`**, **`sdk-test-py`**, **`standards-schemas`**).
2. **API:** `python3 apps/api/server/main.py`, or `cd apps/api/server && python3 -m uvicorn main:app --reload --port 8000`.
3. **Web:** `cd apps/web && npm run dev`. Before pushing web changes, **`npm ci && npm run ci`** (matches CI job **`test-web`** in **`.github/workflows/ci_cd.yml`**). From the **repo root**, **`npm install && npm run dev:stack`** starts API + Next dev together (same as **`./scripts/dev-stack.sh`**). **`npm run test:repo-root`**, **`make test-repo-root`**, or **`python3 -m pytest tests/test_repo_root_package_json.py -q`** runs the root **`package.json`** contract tests.

**Docker:** `docker compose -f infra/docker/docker-compose.yml up -d api` (see **QUICKSTART.md** / **docs/DEPLOYMENT.md** for profiles).
