# Monorepo structure

**Repository root** keeps packaging and primary entrypoints (`pyproject.toml`, `README.md`, `config.yaml`, `cli.py`, `sloughgpt_colab.ipynb`, `package.json` (optional npm: **`dev:stack`**, **`test:repo-root`**), `Makefile`, `verify.sh`, `install.sh`, `run.sh`, …). Secondary docs live under **`docs/`** (e.g. `docs/INSTALL.md`, `docs/misc/`). Sample / auxiliary config files live under **`config/`** (see **`config/README.md`**). Runtime experiment, feature-store, tuning, and vector DB files live under **`data/`**. Operational shell scripts live under **`scripts/deploy/`**; full local setup also uses **`scripts/setup.sh`**. Standalone Python utilities are in **`scripts/tools/`**; one-off legacy snippets are in **`scripts/legacy/`**.

### Where new code goes (settled — do not re-litigate)

| Put it here | When it is |
| ----------- | ---------- |
| `domain/` | **Product logic** — the body of the system. Anything the running app reasons over. |
| `packages/<name>/` | A **shared library with a real import name**, consumed by more than one thing. `src/<name>/` layout; add `<name>/src` to `pytest.ini` `pythonpath`. |
| `apps/` | A runnable surface (API, CLI, web) rather than a library. |
| `tests/` | Test suites **and the machinery they drive** (fixtures, harnesses, flow registries) — e.g. `tests/ux_journeys/`. Product code must never *import* from here; it may *invoke* a harness across a process boundary and read the artifact it writes. |
| `scripts/` | Ops and one-off entrypoints. A script may *forward* to a library, never reimplement it. |
| `docs/` | Specification — what a thing is meant to do, not how it is wired. |

Two tie-breakers that settle most arguments:

1. **Does the product need it at runtime?** If yes → `domain/`. If it only
   *produces evidence* the product reads → give the harness a home under
   `tests/`, write a report artifact, and have the product read that file.
   Cross a process boundary rather than importing test machinery — see
   Mole → `tests/ux_journeys`.
2. **Is it generic or ours?** Generic → the package that already owns that
   domain (`avion` = computer-use, deliberately free of `domain/` imports).
   Ours — routes, copy, spec links, thresholds → **a folder, not a new
   package.** Creating `packages/<name>` needs a *second* consumer beyond the
   one thing using it; a single-use home lives in `tests/` or `scripts/`.

Precedent: `domain/journeys` was a parking spot, not a design decision (it was
re-homed out of `domain/testing` when that tree was deleted), so it became
`tests/ux_journeys` — no new package was minted for a one-consumer harness.

Lightweight layout:

- `apps/` — runnable apps (see **[apps/README.md](../apps/README.md)**): API (`apps/api/server/`), CLI (`apps/cli/`), web (`apps/web/`)
- `packages/core-py/` — shared Python core (`utils/`); domain code lives at repo-root `domain/`; see **[packages/core-py/README.md](../packages/core-py/README.md)**
- `packages/sdk-py/sloughgpt_sdk/` — Python API client SDK (**[README.md](../packages/sdk-py/sloughgpt_sdk/README.md)**)
- `packages/sdk-ts/typescript-sdk/` — TypeScript SDK (npm root; **[README.md](../packages/sdk-ts/typescript-sdk/README.md)**)
- `packages/standards/` — standards docs and JSON schemas; see **[packages/standards/README.md](../packages/standards/README.md)**
- `tests/` — pytest suite (root `pyproject.toml` `testpaths`)
- `infra/docker/` — Dockerfiles and Compose file for local runs
- `run.sh` — optional wrapper: activates the project env (conda `sloughgpt`, else `.venv`) then runs the rest (e.g. `./run.sh python3 -m pytest tests/ -q`); `scripts/python` is the same resolution without activating anything
- `Makefile` — optional: **`make help`** (**`dev-stack`**, **`test-repo-root`**, **`colab-smoke`** / **`colab-test`**); see **README.md** / **QUICKSTART.md**
- `sloughgpt_colab.ipynb` — Colab-oriented walkthrough; local full execute via **`scripts/run_colab_notebook_smoke.sh`** (optional deps: `python3 -m pip install -e ".[notebook]"` — see **README.md** _Google Colab_). Native **`.soul`** checkpoints / §13 saves embed char vocab for **`cli.py eval`** — **`docs/policies/CONTRIBUTING.md`** (_Checkpoint vocabulary_).
- `.cursor/rules/`, `.agents/skills/` — editor/agent guidance (not part of the installable Python package). Entry **`AGENTS.md`** at the repo root links here.

Install the package from the repository root (dev extras include **ruff**, **pytest**, **jsonschema**, and the **`sloughgpt`** CLI entry):

```sh
python3 -m pip install -e ".[dev]"
```

Minimal install without dev tools: `python3 -m pip install -e .`

See `docs/REPO_STRUCTURE_MIGRATION.md` for the full move map.

Contributor and security policies: **[CONTRIBUTING.md](../CONTRIBUTING.md)** and **[SECURITY.md](../SECURITY.md)** at the repo root (symlinks to `docs/policies/`).

CI: **`.github/workflows/reusable-ci-core.yml`** (Python ruff smoke + core pytest), **`ci_cd.yml`** (Python SDK **`sdk-test-py`**, TypeScript SDK **`test-sdk-ts`**, Next.js **`test-web`**, **`standards-schemas`**, Docker images, benchmarks). Local parity commands are in **CONTRIBUTING.md**.

### Infrastructure docs

| Doc                                                          | Covers                                                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| **[PRODUCER_CONSUMER_QUEUE.md](PRODUCER_CONSUMER_QUEUE.md)** | `ProducerConsumerQueue` — bounded work queue, priority, backpressure, consumer pools |
| **[PUGQEEP.md](PUGQEEP.md)**                                 | `pugqeep` — Point protocol, PointLibrary, ModelTree, TaskQueue, Engine, parallel ops |
