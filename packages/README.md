## Packages

Shared libraries consumed by apps.

- `core-py/` — Python shared domain logic (`domains` package on the import path when the repo is installed); see **`core-py/README.md`**.
- `app-planner/` — kanban board + dev-notes store (`app_planner`; canonical `PlannerStore` in `src/app_planner/store.py`, hash-chained board ordering + `verify_chain()`).
- `avion/` — computer-use agent + journey library: perception-action `Agent`, 7 backends (Playwright/CDP/Selenium/API/CLI/Appium/Desktop), skill library, vision, event replay, and the **`avion`** CDP CLI (`python -m avion`). `arken/` and `voyager/` are back-compat shims that re-export it. Sync callers use **`SyncRunner`** (`avion.sync`, one persistent loop — not a duplicate API); page controls beyond the core `Backend` (`wait_for_function`, `set_viewport_size`, `focus`, `press_element`) are the optional **`PageControls`** protocol, and `click`/`fill` accept `force=`. The journey suite (`packages/core-py/tests/test_user_journeys.py`, 103 tests) and `apps/web/scripts/screenshot_headers.py` drive avion directly instead of rolling their own Playwright. Events run through avion's **own stdlib-only logger**: an append-only JSONL journal (`avion.events.EventJournal`) is the source of truth, `EventRecorder`/`StructuredLogger` are views over it, and delivery is pluggable via the **`EventSink`** API (`StdlibSink`/`CallbackSink`/`BusSink`) — pass `Arken(journal_path=...)` to make a session durable (`tests/test_stdlib_only.py` proves the logger never imports `domain/`). Run **`python3 -m pytest packages/avion/tests`**; CLI smoke needs `pip install "avion[cdp]"`.
- `eslint-config/` — shared ESLint configuration: the legacy `.eslintrc` chain (`index.js` → `react.js` → `next.js`, consumed by `apps/web` + `apps/lib` via `@sloughgpt/eslint-config[/next]`) and the ESLint 9 **flat-config factories in `flat.mjs`** (pure, zero `require()`s — `packages/strui` and `packages/sdk-ts/typescript-sdk` each ship an `eslint.config.js` that imports the factories and injects its own plugins from the consumer tree; `apps/web` deliberately stays on its legacy `eslint ^8.57.0` pin).
- `downcraft/` — generic HTTP(S) downloader: cross-session Range resume, SLZ4 (LZ4) compression, multipart, SHA-256 verify; see **`downcraft/README.md`**.
- `sdk-py/` — Python SDK (`sloughgpt_sdk`); see **`sdk-py/sloughgpt_sdk/README.md`**.
- `sdk-ts/typescript-sdk/` — TypeScript SDK (npm package); see **`sdk-ts/typescript-sdk/README.md`**.
- `standards/` — contracts/schemas; see **`standards/README.md`**.

Install from the **repository root** so imports resolve: `python3 -m pip install -e ".[dev]"` (see **pyproject.toml**). Layout overview: **docs/STRUCTURE.md**.

For **npm** work (`sdk-ts/typescript-sdk/` and **`apps/web/`**), use the **Node** version in **`.nvmrc`** at the repo root (`nvm use` / `fnm use`; matches CI).

For **`sdk-py/`** changes, run **`python3 -m pytest tests/test_sdk.py`** (CI job **`sdk-test-py`**).

For **`standards/`** changes, run **`python3 scripts/validate_standards_schemas.py`** (**`jsonschema`** is included in **`python3 -m pip install -e ".[dev]"`**; otherwise **`python3 -m pip install jsonschema`**). CI job **`standards-schemas`**.
