# SloughGPT - Enterprise AI Framework

## Overview

SloughGPT is an enterprise-grade AI framework with production-ready ML infrastructure for training, fine-tuning, and deploying large language models.

## Project Structure

```
/Users/mac/sloughGPT/
├── apps/
│   ├── api/server/          # FastAPI app (main.py), API requirements.txt
│   ├── web/                 # Next.js 14 UI — app/(app)/ (port 3000)
│   ├── cli/                 # cli.py (repo-root launcher may wrap this)
│   └── tui/                 # Phase 1: python3 -m apps.tui (session + probes; plan: docs/plans/tui-cli-port.md)
├── packages/
│   ├── core-py/domains/     # Python domains (training, inference, models, …)
│   ├── sdk-py/              # Python SDK
│   ├── sdk-ts/typescript-sdk/  # TypeScript SDK (npm package root)
│   └── standards/           # SloughGPT Standard v1 docs
├── infra/docker/            # docker-compose and deployment assets
├── tests/                   # pytest suites
├── requirements.txt         # Root Python deps (install first)
└── docs/TODO.md             # Roadmap notes
```

## Key Features

### Training
Two training drivers share the same default **`ModelInterface`** implementation (**`SloughGPTModel`**); checkpoints differ:
- **`train_sloughgpt.py`** (repo root): **`train_sloughgpt()`** — char-level file dataset, exports, **`--resume`**; Colab-aligned.
- **`SloughGPTTrainer`** (`domains.training.train_pipeline`): **`cli.py train`** and **`python3 -m domains.training.train_pipeline`** (**`--resume`**, **`--resume-latest`**, **`--max-checkpoints`**; module **`main`** also **`--dropout`**, **`--lora-alpha`**), **`cli.py eval`** / **`python3 -m domains.training.lm_eval_char`** (char-LM eval via **`evaluate_sloughgpt_char_lm`**), **`POST /training/start`** on **`apps/api/server`** (**`training.router`**; JSON **`TrainingRequest`** in **`training/schemas.py`**, optional **`log_interval`** / **`eval_interval`**, live fields on **`GET /training/jobs`**), **`examples/quick_train.py`**. A **legacy** query-param training demo lives in **`packages/core-py/domains/ui/api_server.py`** (different contract — see file header). Vocabulary size defaults from the corpus unless **`vocab_size`** is set. **`train(resume=True)`** uses **`checkpoint_utils.normalize_raw_checkpoint` / `extract_state_dict`** for weights-only bundles; full optimizer/scheduler load is best-effort. **`cli.py train`** (local) merges **`config.yaml`** with CLI via **`config_loader.merge_args_with_config`** then feeds **`SloughGPTTrainer`**: hyperparameters from **`training`**, LoRA from **`lora`**, step checkpoints from **`checkpoint.trainer_*`**, **`model.dropout`**, **`model.soul_name`** / **`checkpoint.export_format`**, **`get_device(config.device)`** ( **`--train-device`** overrides **`device.type`**), **`--optimized`** (fp16 mixed precision after merge), **`--api`** posts merged dimensions + intervals + **`max_steps`**. Default export stem is **`{model}-{dataset}-{YYYY-MM-DD-HHMMSS}`** under **`checkpoint.save_dir`** (override **`--save-stem`**).

CI: **`tests/test_checkpoint_utils.py`**, **`tests/test_wandb_helpers.py`**, **`tests/test_trainer_protocol.py`**, **`tests/test_train_sloughgpt_*.py`**, **`tests/test_sloughgpt_trainer_smoke.py`**, **`tests/test_sloughgpt_trainer_resume.py`**, **`tests/test_sloughgpt_trainer_progress_callback.py`**, **`tests/test_cli_train_export_stem.py`**, **`tests/test_cli_train_api_payload.py`**, **`tests/test_training_router_kwds.py`**, **`tests/test_training_schemas.py`**, **`tests/test_lm_eval_char.py`**, **`tests/test_cli_local_soul_candidates.py`**, **`tests/test_soul_engine_conversation.py`**, **`tests/test_repo_root_package_json.py`**, **`tests/test_sloughgpt_colab_notebook.py`** (notebook **`sloughgpt_colab.ipynb`**: JSON + smoke hooks), **`tests/test_config.py`** ( **`merge_args_with_config`** / **`get_device`**), **`train_sloughgpt.py --help`**.

Shared checkpoint I/O: **`packages/core-py/domains/training/checkpoint_utils.py`** (`normalize_raw_checkpoint`, `load_sloughgpt_from_checkpoint`, …) used by **`train_sloughgpt.py`**, **`SloughGPTTrainer`** resume + loads, **`cli.py generate`** (local `.sou` order: **`models/sloughgpt.sou`**, then newest **`models/*.sou`** — see **`_local_soul_candidate_paths`** in **`apps/cli/cli.py`**), **`ModelLoader._load_pt`**, and **`scripts/export_to_gguf.py`** for consistent `.pt` parsing. **Char vocab on disk:** **`stoi` / `itos` / `chars`** on full **`step_*.pt`** and typical notebook §13 saves; **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*).

- **LR Schedulers**: Cosine, warmup, OneCycle, cyclic, polynomial
- **Mixed Precision**: FP32, FP16, BF16 with GradScaler
- **Distributed**: DDP + FSDP
- **Memory**: Activation/gradient checkpointing, flash attention
- **Fine-tuning**: LoRA, QLoRA, LoRA+, IA3
- **RLHF**: PPO training with reward model

### Efficiency (Low-End Devices)
- **Quantization**: INT4, INT8, FP16
- **Pruning**: Magnitude, gradient, structured, lottery ticket
- **Distillation**: Temperature-based, feature-based, progressive
- **Optimizations**: KV cache, CPU optimizations, dynamic batching

### Deployment
- **Web UI**: Next.js 14 with chat, training, models, experiments; shared **“Ochre & ink”** tokens in `app/globals.css` + Tailwind `theme.extend.colors` (CSS variables), light mode via `html.light`, mode/accent classes on `<html>` (`lib/theme-storage.ts`, `lib/sync-html-theme.ts`, inline bootstrap in `app/layout.tsx`), accent presets `theme-*`. App routes use **`components/AppRouteHeader.tsx`** with **`AppRouteHeaderLead`** for title/subtitle; **`InferenceRuntimeToolbar`** / **`InferenceStatusBar`** cover API/runtime in the header (chat toolbar is a custom row).
- **strui** (`packages/strui`, `@sloughgpt/strui`): shared primitives / composed / AI for the web app. Storybook: **Design principles** (a11y addon, Tailwind viewports), **Foundations**, **Component gallery** — **`packages/strui/README.md`**; **`npm run storybook`** / **`npm run ci`** (job **`test-strui`**).
- **Colab**: Root **`sloughgpt_colab.ipynb`** — **§2** dataset, **§3–§6** setup, then exactly one training path: manual **§7**, **`RUN_TRAIN_PIPELINE`** / **`train_sloughgpt()`** (**§7b**), or optional **`SloughGPTTrainer`**. **§11** cognitive (SM-2 + SCAMPER) must stay **one** code cell (single **`_asyncio_run`** — see **`tests/test_sloughgpt_colab_notebook.py`**). Optional smoke execute: **`scripts/run_colab_notebook_smoke.sh`** / **`make colab-smoke`** (`--help`, **`make help`**) → **`sloughgpt_colab.executed.ipynb`** (gitignored); **`make colab-test`** for the regression module only. **README.md** (*Google Colab*) summarizes order and env vars.
- **API**: FastAPI with auth, streaming, WebSocket
- **CLI**: Full command-line interface (`apps/cli/cli.py`); optional TUI: Python **`apps/tui/`** (Phase 1 probes) or TS **`packages/tui-ts/`** (Ink + **`@sloughgpt/typescript-sdk`**); roadmap **`docs/plans/tui-cli-port.md`**.
- **Models**: 14 custom architectures + HuggingFace integration

## Commands

```bash
# Editable install (domains + apps.cli; dev extras: ruff, pytest, sloughgpt CLI)
python3 -m pip install -e ".[dev]"
# Optional: notebook execute deps (nbconvert smoke script) — jupyter, nbclient, nbformat
# python3 -m pip install -e ".[notebook]"

# Start API server
cd apps/api/server && python3 main.py
# From repo root (pin :8000 so Next.js proxy matches): SLOUGHGPT_API_PORT=8000 python3 apps/api/server/main.py

# Start web UI
cd apps/web && npm run dev

# API + web in one terminal (from repo root; Ctrl+C stops both)
./scripts/dev-stack.sh
# make dev-stack
# npm install && npm run dev:stack   # same two processes via concurrently (optional)
# npm run test:repo-root             # tests/test_repo_root_package_json.py (optional)
# make test-repo-root                # same (Makefile)

# Run tests
python3 -m pytest tests/ -q
# or (uses .venv when present): ./run.sh python3 -m pytest tests/ -q

# CLI (repo root)
python3 cli.py --help
python3 apps/cli/cli.py --help
# `cli.py train` / `cli.py generate` saves & local .sou resolution: apps/cli/README.md

# Char-level trainer script (repo root)
python3 train_sloughgpt.py --help

# Colab notebook full execute (smoke defaults; needs jupyter / nbconvert)
./scripts/run_colab_notebook_smoke.sh   # add --help for SLOUGH_* env defaults
# make help   # lists colab-smoke / colab-test (README → Google Colab)
# make colab-smoke / make colab-test
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Server health |
| `/auth/login` | Login (query params) |
| `/auth/register` | Register (query params) |
| `/models` | List models |
| `/inference/generate` | Run inference |
| `/training/start` | Start training job |
| `/training/jobs` | List jobs |
| `/experiments` | List experiments |

## Technology Stack

- **Backend**: FastAPI, SQLite, WebSocket
- **Frontend**: Next.js 14, Tailwind CSS, Zustand
- **ML**: PyTorch, Transformers
- **Testing**: pytest (`tests/`)

## Important Notes

1. **No downloads required** - Everything runs locally
2. **Auth uses query params** - Not JSON body (for /auth/login, /auth/register)
3. **Run `python3 -m pytest tests/`** - full suite is the source of truth for pass count
4. **Quantization reduces memory**: 7B model = 14GB (FP16) → 3.3GB (INT4)
5. **Multi-modal is disabled** - Focus on core LLM training

## Testing

```bash
# Full suite (from repo root)
python3 -m pytest tests/ -q

# Optional: path checks + ruff smoke + web `npm run ci` when node_modules exists (prints ci_cd.yml parity)
./verify.sh
```

- **Python CI subset:** `.github/workflows/reusable-ci-core.yml` (`workflow_call`): ruff smoke includes **`train_sloughgpt.py`**; pytest includes **`tests/test_domains_errors.py`** (`domains.errors`), training smoke tests (see **CONTRIBUTING.md**).
- **Also in `ci_cd.yml`:** `test-web` (lint, typecheck, Vitest, **`build:clean`** — same as local **`npm run ci`** in **`apps/web`**: ends with **`rm -rf .next`** + **`next build`**, not a leading **`clean`**), `test-strui` (**`npm run ci`** in **`packages/strui`** — typecheck, Vitest, Storybook build), `test-sdk-ts` (**`npm run ci`** in **`packages/sdk-ts/typescript-sdk`**), `sdk-test-py`, `standards-schemas` (run `python3 scripts/validate_standards_schemas.py`; `jsonschema` is in `python3 -m pip install -e ".[dev]"`).

## Environment

- Python 3.9+ (use `python3` command)
- Node.js **20** for web / TS SDK (repo root **`.nvmrc`**; matches CI `setup-node`)
- SQLite for data (server/database.sqlite)

## Gaps Filled (Recent)

- RLHF/PPO training
- Model pruning
- Knowledge distillation  
- INT4/INT8 quantization
- AWQ/GPTQ support
- KV cache optimization
- CPU-specific optimizations
