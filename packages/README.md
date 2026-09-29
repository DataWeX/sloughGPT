## Packages

Shared libraries consumed by apps. **One shared copy for every session
and worktree** (AGENTS.md _shared-not-rebuilt_): reuse, never reinstall
or fork — when you fix a package, fix it here so all sessions benefit.

| Package          | What it is                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| `aml/`           | AML — Automatic Markup Language                                                                                    |
| `app-planner/`   | Unified notes + kanban CLI (`app-planner`) — in-repo planner entry point                                           |
| `arken/`         | Back-compat shim: arken was renamed to avion — install `avion`                                                     |
| `avion/`         | Arken — minimal web autoclicker: find, click, type via Playwright                                                  |
| `bawl/`          | Tiny, zero-dependency crawler — fetch, parse, crawl, sitemap, store, GUI                                           |
| `chargectl/`     | Zero-dependency Linux battery charge reader, overcharge control, longevity band                                    |
| `core-py/`       | Python shared domain logic (`domains` on the import path when the repo is installed); see **`core-py/README.md`**  |
| `data/`          | Runtime data stores (logged responses, mobile notifications/training, RAG store) — not a library; **never delete** |
| `downcraft/`     | Generic HTTP/HTTPS downloader with cross-session resume (`Range` headers)                                          |
| `eslint-config/` | `@sloughgpt/eslint-config` — shared ESLint rules for web + packages                                                |
| `infra-lib/`     | Type-agnostic infrastructure primitives: cancellation, task queues, server state                                   |
| `mogdb/`         | Embedded document database — JSONL journaling with compaction, zero dependencies                                   |
| `planner/`       | Kanban/notes planner library — `board.jsonl` ↔ dev-notes sync; powers `app-planner` and `/api/planner`             |
| `sdk-py/`        | Python SDK (`sloughgpt_sdk`); see **`sdk-py/sloughgpt_sdk/README.md`**                                             |
| `sdk-ts/`        | TypeScript SDK (npm package, `sdk-ts/typescript-sdk/`); see **`sdk-ts/typescript-sdk/README.md`**                  |
| `standards/`     | Contracts/schemas; see **`standards/README.md`**                                                                   |
| `strui/`         | SloughGPT web design system — pastel lattice tokens, `sl-*` utilities, Radix + CVA (`@sloughgpt/strui`)            |
| `voyager/`       | Back-compat shim: voyager was renamed to arken — install `arken`                                                   |

Install from the **repository root** so imports resolve: `python3 -m pip install -e ".[dev]"` (see **pyproject.toml**). Layout overview: **docs/STRUCTURE.md**.

For **npm** work (`sdk-ts/typescript-sdk/` and **`apps/web/`**), use the **Node** version in **`.nvmrc`** at the repo root (`nvm use` / `fnm use`; matches CI).

For **`sdk-py/`** changes, run **`python3 -m pytest tests/test_sdk.py`** (CI job **`sdk-test-py`**).

For **`standards/`** changes, run **`python3 scripts/validate_standards_schemas.py`** (**`jsonschema`** is included in **`python3 -m pip install -e ".[dev]"`**; otherwise **`python3 -m pip install jsonschema`**). CI job **`standards-schemas`**.

New package? Add its row above **in the same change** — the contract
test `tests/test_shared_indexes.py` fails CI until you do.
