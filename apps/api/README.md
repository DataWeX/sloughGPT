## API

`apps/api/server/` contains the FastAPI app (**`main.py`**) and API-specific **`requirements.txt`**.

```bash
# From repo root
python3 apps/api/server/main.py

# Or with reload
cd apps/api/server && python3 -m uvicorn main:app --reload --port 8000
```

Training routes and shared schemas live alongside **`main.py`** (e.g. **`training/`**). Native trainer **`step_*.pt`** on the API host embeds **`stoi` / `itos` / `chars`** for **`cli.py eval`**; see **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*). See **QUICKSTART.md** and **docs/AI_SOFTWARE_ENGINEERING.md** for boundaries.

**`GET /info`** — Returns PyTorch/CUDA model metadata and, when **`psutil`** is installed (**`apps/api/server/requirements.txt`**), a **`host`** object with real CPU % and RAM usage for the machine running the API process, plus **`process_rss_bytes`** (API process RSS when available). The web **Monitoring** page reads these fields instead of placeholder values.

### Weights & Biases (optional)

Install: `pip install -e ".[wandb]"` from the repo root. Common env vars: `WANDB_API_KEY`, `WANDB_PROJECT` (default `sloughgpt`), `WANDB_ENTITY`, `WANDB_MODE` (`online` / `offline` / `disabled`).

- **`SLOUGHGPT_WANDB_TRAINING=1`** — log **`POST /training/start`** jobs (hyperparameters + trainer metrics via `SloughGPTTrainer`).
- **`SLOUGHGPT_WANDB_SERVER=1`** — long-lived run logging HTTP totals (`server/http_requests_total`, `server/http_latency_mean_ms`), batched inference stats from **`POST /chat`** / **`POST /inference/generate`**, and host snapshots (`host/cpu_percent`, `host/memory_percent`, `server/process_rss_bytes`) aligned with **`GET /info`**. Interval: **`SLOUGHGPT_WANDB_SERVER_INTERVAL_SEC`** (default `60`). Optional run name: **`WANDB_SERVER_RUN_NAME`**.
