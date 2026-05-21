## CLI

`apps/cli/` hosts the **`sloughgpt`** console entrypoint (**`pyproject.toml`** → **`apps.cli.cli:main`**). The repo root **`cli.py`** forwards to the same **`main`**.

```bash
python3 -m pip install -e ".[dev]"   # or python3 -m pip install -e .
sloughgpt --help
# or: python3 cli.py --help
```

**`python3 cli.py --help`** groups commands by how people use them (train/chat first, then server and config). **`gen`** is an alias for **`generate`**. Top-level **`stats`** reports **`models/`** + **`datasets/`** sizes; **`data stats PATH`** inspects one path.

**Inventory:** **`python3 cli.py models`** lists **`models/*.pt`**, **`.safetensors`**, **`.sou`**; **`personalities`** prints built-in **`PersonalityType`** presets from **`domains.ai_personality`**.

See **QUICKSTART.md** for common commands and **CONTRIBUTING.md** for validation.

### Training over HTTP (`cli.py train --api`)

With the FastAPI app in **`apps/api/server`**, **`--api`** sends a **`TrainingRequest`** JSON body to **`POST /training/start`** (same contract as the web console): merged hyperparameters (dropout, weight decay, grad clip/accum, mixed precision, scheduler/LoRA, trainer checkpoint dir, non-auto **`device.type`**). Uses global **`--host`** / **`--port`** and **`--config`**; poll **`GET /training/jobs/{id}`**. Server-side **`step_*.pt`** carries **`stoi` / `itos` / `chars`** for char-LM eval (**CONTRIBUTING.md**, *Checkpoint vocabulary*).

### Training saves (`cli.py train`)

Beginner path: set **`--dataset`**, **`--max-steps`** (or **`--epochs`**), **`--checkpoint-dir`**, then **`--resume-latest`** when continuing. Run **`python3 cli.py train --help`** for grouped flags, defaults, and copy-paste examples. For the legacy argparse-only driver (same **`SloughGPTTrainer`**), use **`python3 -m domains.training.train_pipeline --help`**.

**Log / eval cadence:** defaults live under **`training.log_interval`** and **`training.eval_interval`** in **`config.yaml`** (10 / 100). **`--log-interval`** and **`--eval-interval`** override for the local trainer and for **`train --api`** (**`TrainingRequest`** JSON).

**Step cap:** optional **`training.max_steps`** in **`config.yaml`** is honored when **`--max-steps`** is omitted (local trainer and **`train --api`** body).

**Job / Soul name:** optional **`model.soul_name`** sets the trainer Soul label and the HTTP job **`name`** when **`--soul-name`** is omitted.

**Post-training format:** **`checkpoint.export_format`** (default **`sou`**) matches **`--save-format`** after merge for local runs.

**Dropout:** **`model.dropout`** in **`config.yaml`** is passed into **`SloughGPTTrainer`**; **`--dropout`** overrides after merge.

**`--optimized`:** after merge, forces mixed precision **on** and **`fp16`** for local **`SloughGPTTrainer`** and for the **`--api`** JSON body.

**Device:** local **`train`** uses **`get_device(config.device)`** when **`device.type`** is **`auto`**. For **`--api`**, only an explicit **`device.type`** (e.g. from **`--train-device`**) is sent so the server does not inherit the client’s resolved **`auto`**. **`--train-device`** overrides **`device.type`** from YAML before merge.

Final exports use **`checkpoint.save_dir`** from **`config.yaml`**. Default basename is **`{model}-{dataset}-{YYYY-MM-DD-HHMMSS}`** (no overwrite on each run). Use **`--save-stem NAME`** for a fixed filename stem.

### Char-LM eval (`cli.py eval`)

**`python3 cli.py eval --checkpoint PATH --data PATH`** reports mean cross-entropy and **character-token perplexity** on a UTF-8 file (non-overlapping **`block_size`** windows). Uses **`stoi`** / **`itos`** / **`chars`** from the bundle when saved (including **`cli.py train`** `step_*.pt`); otherwise warns if the eval charset was inferred from the file. Same logic as **`python3 -m domains.training.lm_eval_char`** (**`--json`** for machine-readable output). Flags: **`--device`**, **`--no-strict`**. Background: **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*).

### Local generate (`cli.py generate`)

Without the API, loads weights in order: **`models/sloughgpt.sou`** (if present), else the **newest** **`models/*.sou`** by modification time, else **`models/sloughgpt_finetuned.pt`**. Implemented via **`_local_soul_candidate_paths`** in **`cli.py`**.
