## Packages

Shared libraries consumed by apps.

- `core-py/` — Python shared domain logic (`domains` package on the import path when the repo is installed); see **`core-py/README.md`**.
- `sdk-py/` — Python SDK (`sloughgpt_sdk`); see **`sdk-py/sloughgpt_sdk/README.md`**.
- `sdk-ts/typescript-sdk/` — TypeScript SDK (npm package); see **`sdk-ts/typescript-sdk/README.md`**.
- `standards/` — contracts/schemas; see **`standards/README.md`**.

Install from the **repository root** so imports resolve: `python3 -m pip install -e ".[dev]"` (see **pyproject.toml**). Layout overview: **docs/STRUCTURE.md**.

For **npm** work (`sdk-ts/typescript-sdk/` and **`apps/web/`**), use the **Node** version in **`.nvmrc`** at the repo root (`nvm use` / `fnm use`; matches CI).

For **`sdk-py/`** changes, run **`python3 -m pytest tests/test_sdk.py`** (CI job **`sdk-test-py`**).

For **`standards/`** changes, run **`python3 scripts/validate_standards_schemas.py`** (**`jsonschema`** is included in **`python3 -m pip install -e ".[dev]"`**; otherwise **`python3 -m pip install jsonschema`**). CI job **`standards-schemas`**.
