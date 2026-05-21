# Contributing

Thanks for helping improve SloughGPT.

## Quick start

1. **Branch** from `main` with a descriptive name.
2. **Install** from the repo root: `python3 -m pip install -e ".[dev]"` (see **QUICKSTART.md**). Optional — Colab smoke / `nbconvert`: `python3 -m pip install -e ".[notebook]"` (**`pyproject.toml`**), then **`scripts/run_colab_notebook_smoke.sh`** or **`make colab-smoke`** (**`make help`**; **README.md** *Google Colab*).
3. **Validate**:
   - `./verify.sh` and `python3 -m pytest tests/` (Python; CI subset in **`.github/workflows/reusable-ci-core.yml`**, including **`tests/test_checkpoint_utils.py`**, **`tests/test_config.py`** (**`merge_args_with_config`**, **`get_device`**), **`tests/test_domains_errors.py`**, **`tests/test_wandb_helpers.py`**, **`tests/test_trainer_protocol.py`** (**`TrainerProtocol`**, **`run_trainer_async`**), **`tests/test_train_sloughgpt_*.py`**, **`tests/test_sloughgpt_trainer_smoke.py`**, **`tests/test_sloughgpt_trainer_resume.py`**, **`tests/test_sloughgpt_trainer_progress_callback.py`**, **`tests/test_cli_train_export_stem.py`**, **`tests/test_cli_train_api_payload.py`**, **`tests/test_training_router_kwds.py`**, **`tests/test_training_schemas.py`**, **`tests/test_lm_eval_char.py`**, **`tests/test_cli_local_soul_candidates.py`**, **`tests/test_soul_engine_conversation.py`**, **`tests/test_repo_root_package_json.py`** (root **`package.json`**, **`dev:stack`**, **`verify.sh`**, **`make test-repo-root`**), **`tests/test_sloughgpt_colab_notebook.py`**, and **`python3 train_sloughgpt.py --help`**). With a **`.venv`**, you can use **`./run.sh python3 -m pytest tests/`** so the same interpreter is on **`PATH`**.
   - **`tests/test_sloughgpt_colab_notebook.py`** runs **`bash`** for **`scripts/run_colab_notebook_smoke.sh --help`**; if **`bash`** is not on **`PATH`**, that case is **skipped** (on Windows, use Git Bash, WSL, or rely on Linux/macOS CI).
   - Optional — **`tests/test_integration.py`** calls a **live** API on **`localhost:8000`** by default (see **`tests/README.md`** for env vars and **`websocket-client`**). Start the server first (e.g. **`SLOUGHGPT_API_PORT=8000 python3 apps/api/server/main.py`** or **`./scripts/dev-stack.sh`**); if nothing is listening, that module **skips**.
   - If you touch **repo root `package.json`** (e.g. **`dev:stack`** / **`concurrently`**): **`npm run test:repo-root`** (after **`npm install`** at repo root), **`make test-repo-root`**, or **`python3 -m pytest tests/test_repo_root_package_json.py -q`**.
   - If you touch **`apps/web/`**: `cd apps/web && npm ci && npm run ci` (lint + typecheck + Vitest + **`npm run build:clean`** — removes **`.next`** then **`next build`**; avoids flaky trace/cache issues; same as job **`test-web`**). To run **API + web** locally in one terminal: **`./scripts/dev-stack.sh`**, **`make dev-stack`**, or **`npm install` at repo root** then **`npm run dev:stack`** (see **QUICKSTART.md**).
   - If you touch **`packages/sdk-ts/typescript-sdk/`**: `cd packages/sdk-ts/typescript-sdk && npm ci && npm run ci` (job **`test-sdk-ts`**).
   - If you touch **`packages/strui/`**: `cd packages/strui && npm ci && npm run ci` (job **`test-strui`** — typecheck, Vitest, Storybook build).
   - If you touch **`packages/sdk-py/sloughgpt_sdk/`**: `python3 -m pytest tests/test_sdk.py -q` (job **`sdk-test-py`**).
   - If you change **`packages/standards/`** or schemas: `python3 scripts/validate_standards_schemas.py` (**`jsonschema`** is in **`python3 -m pip install -e ".[dev]"`**; otherwise `python3 -m pip install jsonschema`). Job **`standards-schemas`**.
4. **Open a PR** with a clear description of intent and scope (see **`.github/pull_request_template.md`**).

## Issues

Use **`.github/ISSUE_TEMPLATE/`** when filing bugs or feature requests (blank issues stay enabled).

## Training entry points

- **`train_sloughgpt.py`** (repo root): char-level dataset file → **`train_sloughgpt()`**; rich export/resume flags; Colab **`sloughgpt_colab.ipynb`** exposes the same stack in **§7b** (`RUN_TRAIN_PIPELINE`), alongside a manual **§7** loop and an optional **`SloughGPTTrainer`** cell (use one path per run). Optional full-notebook smoke: **`scripts/run_colab_notebook_smoke.sh`** or **`make colab-smoke`** (writes gitignored **`sloughgpt_colab.executed.ipynb`**; **`README.md`** documents env overrides; **`make help`** lists Makefile shortcuts). Colab regression only: **`make colab-test`**.
- **`SloughGPTTrainer`** (`packages/core-py/domains/training/train_pipeline.py`): **`cli.py train`** and **`python3 -m domains.training.train_pipeline`** (local), API **`/training/start`**, and **`examples/quick_train.py`** / **`lora_train.py`**.
- **`TrainerProtocol`** / **`run_trainer_async`** (`packages/core-py/domains/training/trainer_protocol.py`): minimal typing contract for class trainers whose **`train(...)`** returns a **`dict`**; **`run_trainer_async`** runs **`SloughGPTTrainer.train`** in a worker thread so async callers do not block the event loop. **`train_sloughgpt()`** is not protocol-shaped (it returns **`(model, stoi, itos)`**); wrap it with **`asyncio.to_thread`** (or similar) if you need non-blocking script execution.

Both default to the **`ModelInterface`** implementation **`domains.models.SloughGPTModel`** (other backends implement **`ModelInterface`** via **`ModelLoader`**); checkpoints are not interchangeable between drivers without aligning keys/format. **`SloughGPTTrainer.train(resume=True)`** accepts weights-only bundles (via **`checkpoint_utils`**) and full trainer **`step_*.pt`** files; **`cli.py train`** and **`python -m domains.training.train_pipeline`** (module **`main`**) support **`--resume PATH`** and **`--resume-latest`**, and the module entrypoint also exposes **`--dropout`** and **`--lora-alpha`**. Local **`cli.py train`** loads **`config.yaml`** and merges CLI flags (**`merge_args_with_config`**); **`--optimized`** applies an fp16 mixed-precision preset after merge (ignored for **`--api`**). When **`vocab_size`** is omitted, the trainer sets vocabulary size from the corpus (CLI/config may still pass an explicit size).

### Checkpoint vocabulary (char LM)

Char-level training uses **`stoi`** (character → int), **`itos`** (int → character), and optionally **`chars`** (ordered vocabulary list). Full trainer **`step_*.pt`** files written by **`CheckpointManager.save`** include these so **`domains.training.lm_eval_char.evaluate_sloughgpt_char_lm`** and **`cli.py eval`** score text with the **same** charset as training (**`tokenizer_maps_from_bundle`** in **`checkpoint_utils.py`**). **`train_sloughgpt.py`** periodic **`checkpoint_step_*.pt`** files (when **`checkpoint_interval` > 0**) also embed **`stoi`** / **`itos`**. Bundles with only **`model_state_dict`** + **`training_info`** / **`config`** but **no** **`stoi`** make eval fall back to building the alphabet from the **eval file**; **`cli.py eval`** then warns when that may skew perplexity. Other deployment artifacts (HF **`tokenizer.save_pretrained`**, GGUF vocab chunks, **`.sou`**) encode vocabulary differently; they are not drop-in replacements for char **`stoi`**/**`itos`** without an explicit conversion step.

**Regression:** **`tests/test_sloughgpt_trainer_resume.py`** asserts **`stoi`** / **`itos`** / **`chars`** on trainer-native **`step_*.pt`**.

**`cli.py train` exports:** default Soul (and other format) basename under **`checkpoint.save_dir`** is **`{model}-{dataset}-{YYYY-MM-DD-HHMMSS}`**; use **`--save-stem`** for a fixed name. **`cli.py generate`** (local, no API) loads **`models/sloughgpt.sou`** first, then the newest **`models/*.sou`** by mtime, then **`models/sloughgpt_finetuned.pt`** (see **`_local_soul_candidate_paths`** in **`apps/cli/cli.py`**).

**HTTP training (Console / SDK):** **`POST /training/start`** in **`apps/api/server/training/`** accepts **`TrainingRequest`** JSON (including **`log_interval`** and **`eval_interval`**). Tracked jobs expose live fields via **`GET /training/jobs`** / **`GET /training/jobs/{job_id}`** (**`progress`**, **`current_epoch`**, **`global_step`**, **`train_loss`**, **`eval_loss`**). Trainer **`step_*.pt`** files on the API host use the same char-vocab embedding rules as local **`cli.py train`** (*Checkpoint vocabulary* above). The legacy demo server in **`packages/core-py/domains/ui/api_server.py`** uses a different **`/training/start`** contract (query params, toy loop); prefer **`apps/api/server`** for parity with the web app and SDKs.

## Structure

See **docs/STRUCTURE.md** and **docs/MIGRATION.md** for layout and migration history.
