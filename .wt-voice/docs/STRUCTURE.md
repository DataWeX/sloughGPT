# Monorepo structure

**Repository root** keeps packaging and primary entrypoints (`pyproject.toml`, `README.md`, `config.yaml`, `cli.py`, `sloughgpt_colab.ipynb`, `package.json` (optional npm: **`dev:stack`**, **`test:repo-root`**), `Makefile`, `verify.sh`, `install.sh`, `run.sh`, …). Secondary docs live under **`docs/`** (e.g. `docs/INSTALL.md`, `docs/misc/`). Sample / auxiliary config files live under **`config/`** (see **`config/README.md`**). Runtime experiment, feature-store, tuning, and vector DB files live under **`data/`**. Operational shell scripts live under **`scripts/deploy/`**; full local setup also uses **`scripts/setup.sh`**. Standalone Python utilities are in **`scripts/tools/`**; one-off legacy snippets are in **`scripts/legacy/`**.

Lightweight layout:

- `apps/` — runnable apps (see **[apps/README.md](../apps/README.md)**): API (`apps/api/server/`), CLI (`apps/cli/`), web (`apps/web/`)
- `packages/core-py/` — shared Python core (`utils/`); domain code lives at repo-root `domain/`; see **[packages/core-py/README.md](../packages/core-py/README.md)**
- `packages/sdk-py/sloughgpt_sdk/` — Python API client SDK (**[README.md](../packages/sdk-py/sloughgpt_sdk/README.md)**)
- `packages/sdk-ts/typescript-sdk/` — TypeScript SDK (npm root; **[README.md](../packages/sdk-ts/typescript-sdk/README.md)**)
- `packages/standards/` — standards docs and JSON schemas; see **[packages/standards/README.md](../packages/standards/README.md)**
- `domain/` — product core subsystems (34 packages — **map below**, check before building)
- `tests/` — pytest suite (root `pyproject.toml` `testpaths`)
- `infra/docker/` — Dockerfiles and Compose file for local runs
- `run.sh` — optional wrapper: activates the project env (conda `sloughgpt`, else `.venv`) then runs the rest (e.g. `./run.sh python3 -m pytest tests/ -q`); `scripts/python` is the same resolution without activating anything
- `Makefile` — optional: **`make help`** (**`dev-stack`**, **`test-repo-root`**, **`colab-smoke`** / **`colab-test`**); see **README.md** / **QUICKSTART.md**
- `sloughgpt_colab.ipynb` — Colab-oriented walkthrough; local full execute via **`scripts/run_colab_notebook_smoke.sh`** (optional deps: `python3 -m pip install -e ".[notebook]"` — see **README.md** _Google Colab_). Native **`.soul`** checkpoints / §13 saves embed char vocab for **`cli.py eval`** — **`docs/policies/CONTRIBUTING.md`** (_Checkpoint vocabulary_).
- `.cursor/rules/`, `.agents/skills/` — editor/agent guidance (not part of the installable Python package). Entry **`AGENTS.md`** at the repo root links here.

## `domain/` — product core subsystems (current tree)

**Check this table before building anything under `domain/`.** Every subsystem's
entry point is its package `__init__` **public API / lazy map** (name → module) —
never a new module beside it. Descriptions below are each package's own
`__init__` docstring first line (mechanical, not hand-written).

| Subsystem               | Owns                                                                          |
| ----------------------- | ----------------------------------------------------------------------------- |
| `domain/agents`         | agents — Agentic infrastructure for safe internal tool execution.             |
| `domain/ai_personality` | ai_personality — Personality system for model outputs.                        |
| `domain/api`            | api — SSE envelope and streaming utilities.                                   |
| `domain/benchmark`      | benchmark — Benchmark domain for performance testing.                         |
| `domain/chat`           | chat — Chat domain for message handling.                                      |
| `domain/cognition`      | cognition — Reasoning, thinking, creativity, RAG, and consciousness.          |
| `domain/cognitive`      | cognitive — Cognitive architecture domain.                                    |
| `domain/collections`    | collections — Data collection pipeline: sources, stores, filters, collectors. |
| `domain/companion`      | companion — AI companion system for natural conversation.                     |
| `domain/context`        | context — Context managers for model behavior steering.                       |
| `domain/core`           | core — Infrastructure, hardware, system-level operations.                     |
| `domain/errors`         | errors — Domain-layer exceptions for SloughGPT.                               |
| `domain/feedback`       | feedback — Feedback storage, meta-weight learning, response tracking.         |
| `domain/generation`     | generation — Inference, output, voice, multimodal.                            |
| `domain/inference`      | inference — Model inference, vector store, soul format.                       |
| `domain/infrastructure` | infrastructure — Core infrastructure (config, event bus, lifecycle, errors).  |
| `domain/knowledge`      | knowledge — high-level knowledge API.                                         |
| `domain/learner`        | learner — Continual learner (ingests data, fine-tunes incrementally).         |
| `domain/logging`        | logging — OOP logger hierarchy for all interfaces.                            |
| `domain/memory`         | Memory layer - chat- and task-agnostic auto-memory facade.                    |
| `domain/models`         | models — Pluggable model backends.                                            |
| `domain/multimodal`     | multimodal — Vision, speech, image captioning, diffusion, TTS.                |
| `domain/ops`            | ops — Optimized NumPy operations (fused ops, attention, softmax).             |
| `domain/plugins`        | plugins — Plugin framework for loading and managing external plugins.         |
| `domain/search`         | System-wide search — one query, every registered store.                       |
| `domain/settings`       | settings — Persistent user configuration.                                     |
| `domain/shared`         | shared — Shared utilities, types, constants across all domains.               |
| `domain/shell`          | shell — Interactive Dait shell (kernel, processes, VM, TUI).                  |
| `domain/slolib`         | slolib — Unified tensor library (autograd, GPU acceleration, inference).      |
| `domain/soul`           | soul — Cognitive engine, HD memory, quantum processing.                       |
| `domain/tools`          | tools — Declarative everyday tools (writing, translate, rewrite).             |
| `domain/training`       | training — Unified training capabilities (datasets, checkpoints, LoRA).       |
| `domain/voice`          | Voice layer - TTS, STT, phoneme encoding, audio processing.                   |

## Branch lanes — structure moves across branches

Layout on the checked-out branch is **not** the whole truth: code migrates
_between branches_, so a working-tree search alone can miss code that exists.
Known lane: **`feat/generate-consolidation`** owns structure moves —
`domain/generate/{models,voice,_internal}` with a reduced `domain/` top level
(plus its own active fix history). Before building **any** new module, package,
or restructure, run the all-refs check:

```sh
# exists on any branch?
for ref in $(git for-each-ref --format='%(refname:short)' refs/heads); do
  git ls-tree -d "$ref" -- domain/<thing> | grep -q . && echo "EXISTS on $ref"
done
# renamed/moved anywhere in history?
git log --all --diff-filter=R --name-status | grep -i <thing>
```

Standing rules this encodes: **check-if-exists across ALL refs** (AGENTS.md
core rule + coordination rule "port, don't rebuild") — and when in doubt, the
kanban card owning the lane decides, not a new build.

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
