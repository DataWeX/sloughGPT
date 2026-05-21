# Monorepo structure

**Repository root** keeps packaging and primary entrypoints (`pyproject.toml`, `README.md`, `config.yaml`, `cli.py`, `train_sloughgpt.py`, `sloughgpt_colab.ipynb`, `package.json` (optional npm: **`dev:stack`**, **`test:repo-root`**), `Makefile`, `verify.sh`, `install.sh`, `run.sh`, …). Secondary docs live under **`docs/`** (e.g. `docs/TODO.md`, `docs/INSTALL.md`, `docs/misc/`). Sample / auxiliary config files live under **`config/`** (see **`config/README.md`**). Runtime experiment, feature-store, tuning, and vector DB files live under **`data/`** (see **`data/README.md`**). Operational shell scripts live under **`scripts/deploy/`**; full local setup also uses **`scripts/setup.sh`**. Standalone Python utilities are in **`scripts/tools/`**; one-off legacy snippets are in **`scripts/legacy/`**.

Lightweight layout:

- `apps/` — runnable apps (see **[apps/README.md](../apps/README.md)**): API (`apps/api/server/`), CLI (`apps/cli/`), web (`apps/web/`), TUI placeholder (`apps/tui/` — see **[docs/plans/tui-cli-port.md](plans/tui-cli-port.md)**)
- `packages/core-py/` — domains and shared Python core (`domains/`, `utils/`); see **[packages/core-py/README.md](../packages/core-py/README.md)**
- `packages/sdk-py/sloughgpt_sdk/` — Python API client SDK (**[README.md](../packages/sdk-py/sloughgpt_sdk/README.md)**)
- `packages/sdk-ts/typescript-sdk/` — TypeScript SDK (npm root; **[README.md](../packages/sdk-ts/typescript-sdk/README.md)**)
- `packages/tui-ts/` — Ink terminal UI (`sloughgpt-tui`); depends on built SDK — **[README.md](../packages/tui-ts/README.md)**
- `packages/standards/` — standards docs and JSON schemas; see **[packages/standards/README.md](../packages/standards/README.md)**
- `tests/` — pytest suite (root `pyproject.toml` `testpaths`)
- `infra/docker/` — Dockerfiles and Compose file for local runs
- `run.sh` — optional wrapper: prepends `.venv/bin` to `PATH`, then runs the rest (e.g. `./run.sh python3 -m pytest tests/ -q`)
- `Makefile` — optional: **`make help`** (**`dev-stack`**, **`test-repo-root`**, **`colab-smoke`** / **`colab-test`**); see **README.md** / **QUICKSTART.md**
- `sloughgpt_colab.ipynb` — Colab-oriented walkthrough; local full execute via **`scripts/run_colab_notebook_smoke.sh`** (optional deps: `python3 -m pip install -e ".[notebook]"` — see **README.md** *Google Colab*). Native **`step_*.pt`** / §13 saves embed char vocab for **`cli.py eval`** — **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*).
- `.cursor/rules/`, `.agents/skills/` — editor/agent guidance (not part of the installable Python package). Entry **`AGENTS.md`** at the repo root links here.

Install the package from the repository root (dev extras include **ruff**, **pytest**, **jsonschema**, and the **`sloughgpt`** CLI entry):

```sh
python3 -m pip install -e ".[dev]"
```

Minimal install without dev tools: `python3 -m pip install -e .`

See `docs/REPO_STRUCTURE_MIGRATION.md` for the full move map.

Contributor and security policies: **[CONTRIBUTING.md](../CONTRIBUTING.md)** and **[SECURITY.md](../SECURITY.md)** at the repo root (symlinks to `docs/policies/`).

CI: **`.github/workflows/reusable-ci-core.yml`** (Python ruff smoke + core pytest), **`ci_cd.yml`** (Python SDK **`sdk-test-py`**, TypeScript SDK **`test-sdk-ts`**, TUI **`test-tui-ts`** (builds SDK then **`packages/tui-ts`**), Next.js **`test-web`**, **`standards-schemas`**, Docker images, benchmarks), **`publish.yml`** (PyPI). Local parity commands are in **CONTRIBUTING.md**.
