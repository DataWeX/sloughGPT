# Plan: Interactive TUI for SloughGPT CLI

This document maps **today’s argparse CLI** to a future **terminal UI (TUI)** without duplicating core training or inference logic. Status: **Phase 1** — minimal **`apps/tui/`** (session, read-only HTTP/local adapters, stub CLI); **Textual** and optional install extras are later. Normal **`pip install -e .`** unchanged.

## Goals

- **One session**: persistent context (repo root, device, API base URL, last checkpoint / `.sou`, optional job id).
- **Same backends**: reuse **`SloughGPTTrainer`**, **`train_sloughgpt`**, **`domains.training.export`**, **`SoulEngine`**, **`apps/api/server`** HTTP contracts — not a second trainer.
- **Stable non-interactive CLI**: keep **`sloughgpt`** / **`python3 cli.py`** for scripts and CI; add **`sloughgpt-tui`** or **`sloughgpt interactive`** as an optional entry.

## Current codebase inventory

### Entrypoints

| Piece | Location | Notes |
|--------|-----------|--------|
| Console script | **`pyproject.toml`** → `sloughgpt = "apps.cli.cli:main"` | Depends include **`rich`**; no TUI framework today. |
| Root wrapper | **`cli.py`** | Imports **`apps.cli.cli.main`**. |
| Monolith | **`apps/cli/cli.py`** | **argparse** + many **`cmd_*`**; nested **`config`**, **`data`**, **`docker`** subparsers. |
| SDK CLI | **`packages/sdk-py/sloughgpt_sdk/cli.py`** | HTTP/client-focused; separate from **`apps/cli`**. Optional reuse for API-attached TUI “rooms”. |

### Config / CWD assumptions

- **`config_loader.py`** (repo root): **`load_config`**, **`merge_args_with_config`** — local **`cmd_train`** expects to run from repo root with **`config.yaml`**.
- **`cmd_train`**: **local** path = config + **`SloughGPTTrainer`**; **`--api`** path = **`requests.post(.../training/start, json=TrainingRequest)`** matching **`apps/api/server/training/router.py`**.

### Core Python (`packages/core-py/domains/`)

TUI actions eventually call (directly or via thin wrappers):

- **`domains.training.train_pipeline`**: **`SloughGPTTrainer`**, **`prepare_data`**, progress callback on **`train()`**.
- **`domains.training.checkpoint_utils`**: resume / weights normalization.
- **`domains.training.export`**: **`export_model`**, formats (used by CLI export).
- **`domains.core.soul`** / **`domains.inference.sou_format`**: Soul / `.sou`.
- **`domains.training.huggingface`**: **`cmd_hf_download`**, **`cmd_hf_serve`**, benchmarks.

### Parallel training surface

- **`train_sloughgpt.py`**: script-level char pipeline; Colab and docs reference it. TUI should either subprocess it or call **`train_sloughgpt()`** explicitly — pick one primary story per “mode” to avoid drift.

### Command handlers (`apps/cli/cli.py`)

Representative **`cmd_*`** (non-exhaustive): **`cmd_chat`**, **`cmd_generate`**, **`cmd_train`**, **`cmd_export_cli`** (local export only), **`cmd_soul`**, **`cmd_quick`**, **`cmd_datasets`**, **`cmd_models`**, **`cmd_personalities`**, **`cmd_stats`**, **`cmd_data_tool`**, **`cmd_info`**, **`cmd_serve`**, **`cmd_health`**, **`cmd_status`**, **`cmd_monitor`**, **`cmd_api_*`**, **`cmd_hf_*`**, **`cmd_benchmark`**, **`cmd_compare`**, **`cmd_config_*`**, **`cmd_system`**, **`cmd_optimize`**, **`cmd_setup`**, **`cmd_docker_*`**, **`cmd_demo`**, **`cmd_rlhf_demo`**, **`cmd_cloud_setup`**.

**API export:** use **`POST /model/export`** on **`apps/api/server`** — there is no separate **`cmd_export`**; the old unused **`GET /export/model/...`** helper was removed to avoid confusion with **`cmd_export_cli`**.

### Tests

- CLI tests are **sparse** (e.g. chat hint, export stem, soul candidates). TUI needs **adapter-level** tests (mock HTTP, temp dirs) and optional **headless** smoke if using Textual.

## Target architecture (TUI package)

Layout (Phase 1 started: **`apps/tui/`** Python probes; **`packages/tui-ts/`** Ink + TS SDK; **`screens/`** and full train/export adapters later):

- **`apps/tui/`** — **`session`**, **`adapters/http_api`**, **`adapters/local_status`** (`python3 -m apps.tui`).
- **`packages/tui-ts/`** — **`@sloughgpt/tui`**, **`sloughgpt-tui`** (Ink; uses **`@sloughgpt/typescript-sdk`**).
- **`session.py`**: repo root, **`--host`/`--port`**, device, paths, last error.
- **`adapters/`**: `LocalTrainAdapter`, `HttpTrainAdapter` (canonical **`TrainingRequest`**), `ExportAdapter`, `InferenceAdapter`, `DockerAdapter` — each thin; **no** flag reimplementation beyond building a **`Namespace`** or dataclass.
- **`screens/`**: Home, Train, Play (chat/generate), Ship (export), Data, API, Docker (later).
- **Optional dependency**: e.g. **`[project.optional-dependencies] tui = ["textual>=0.47"]`** (version pin TBD).

**Output strategy:** Phase 1 can **stream subprocess stdout** into a log pane; Phase 2 can add **`logging.Handler`** or **`on_progress`** hooks for **`SloughGPTTrainer.train`**.

## Phased rollout

| Phase | Scope | Outcome |
|-------|--------|---------|
| **0 — Hygiene** | Done: **`models`** / **`personalities`** wired; dead **`cmd_export`** removed; **`cli.py train --api`** sends **`TrainingRequest`** JSON to **`POST /training/start`**. | Cleaner surface for TUI to mirror. |
| **1 — Shell** | Home + read-only rooms (**`config check`**, **`health`**, **`stats`**, **`datasets`**); global command palette stub; log viewer for one short command. | Proves layout + session, low risk. **TDD:** `tests/test_tui_phase1.py` — **`TuiSession`**, **`discover_repo_root`**, **`fetch_health` / `fetch_metrics` / `fetch_health_detailed`**, **`scan_local_repo`**; CLI: **`python3 -m apps.tui --local-status`** and **`--api-*`**. |
| **2 — Core local** | Train room (build trainer kwargs / **`Namespace`**), Generate/Soul play room, Export wizard; use **`on_progress`** where available. | “Heavenly” daily driver for local work. |
| **3 — API-attached** | Training jobs, model load, auth using **`apps/api/server`** (and optionally **`sloughgpt_sdk`**). | Parity with web console patterns. |
| **4 — Ops** | Docker compose panes, setup; demos behind confirmations. | Power users. |
| **5 — Hardening** | Headless CI smoke, keyboard map, docs. | Shippable optional extra. |

## Risks and decisions

| Risk | Mitigation |
|------|------------|
| **`apps/cli/cli.py` size** | Incrementally split **`commands/`** modules; TUI imports those, not a larger god file. |
| **Blocking I/O** | Workers in threads / subprocess; Textual **worker** pattern; never block UI thread on long **`train()`** without feedback. |
| **CLI / HTTP parity** | **`cli.py train --api`** and TUI should both use **`TrainingRequest`** JSON on **`apps/api/server`** (the legacy demo in **`domains/ui/api_server.py`** is a different contract). |
| **Flag drift** | Prefer generating forms from **argparse** introspection or a small **YAML manifest** synced with **`train --help`**. |

## Documentation touchpoints when implementing

- **`apps/README.md`**: link TUI app and optional extra.
- **`QUICKSTART.md`**: optional “interactive mode” section.
- **`.agents/skills/SKILL.md`**: one line pointing here for agents.

## Related docs

- **[docs/STRUCTURE.md](../STRUCTURE.md)** — repo layout.
- **[apps/cli/README.md](../../apps/cli/README.md)** — current CLI behavior (train stem, generate `.sou` order).
- **[docs/policies/CONTRIBUTING.md](../policies/CONTRIBUTING.md)** — training entry points and HTTP fields.
