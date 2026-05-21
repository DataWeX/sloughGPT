# SloughGPT

Self-hosted LLM infrastructure with local model training, inference, and experimentation.

## Features

- **Local Model Training** - Train SloughGPTModel from scratch or fine-tune HuggingFace models
- **Soul Engine** - Every model IS a soul (.sou format) with personality and cognitive capabilities
- **Production Inference** - High-performance inference engine with streaming
- **GGUF Export** - Mobile deployment via llama.rn (iOS/Android)
- **ONNX Export** - Cross-platform deployment (server, web, mobile)
- **Quantization** - FP16, INT8, INT4, Q4_K_M, Q5_K_M support
- **Experiment Tracking** - MLflow-style metrics and parameter logging
- **Federated Learning** - Privacy-preserving distributed training
- **Benchmarking** - Performance metrics and model comparison
- **Optimizations** - Mixed precision, gradient checkpointing, torch.compile, Flash Attention
- **API Security** - JWT auth, rate limiting, input validation, audit logging
- **Batch Processing** - Process up to 50 prompts in one request
- **Response Caching** - TTL-based caching with hit/miss stats
- **Kubernetes Ready** - Helm charts, health probes, Prometheus metrics
- **Docker Ready** - Docker Compose with API, GPU, monitoring stacks
- **TypeScript SDK** - Full-featured SDK with webhooks, billing, caching
- **Feedback System** - User feedback collection with per-user LoRA adapters for personalized model adaptation

## Quick Start

### CLI Commands
```bash
# Quick train + generate (auto-optimized)
python3 cli.py quick --steps 100 --prompt "Hello world"

# Char-level training: merges config.yaml with flags (see apps/cli/README.md)
python3 cli.py train --dataset shakespeare --epochs 3 --checkpoint-dir ckpts

# Char-LM perplexity (fair when stoi/itos/chars are on the checkpoint, e.g. cli.py train step_*.pt)
python3 cli.py eval --checkpoint models/sloughgpt.pt --data datasets/shakespeare/input.txt
# python3 -m domains.training.lm_eval_char --checkpoint PATH --data PATH [--json]
# docs/policies/CONTRIBUTING.md — Checkpoint vocabulary (char LM)

# Benchmark inference
python3 cli.py benchmark -m gpt2 -d mps

# Check optimizations
python3 cli.py optimize

# System info
python3 cli.py system

# Generate text (local: models/sloughgpt.sou, else newest models/*.sou, else sloughgpt_finetuned.pt)
python3 cli.py generate "Hello world"
python3 cli.py gen "Hello world"   # alias

# Interactive chat (starts the API with uvicorn if it is not already running)
python3 cli.py chat

# Interactive chat + auto-load a model first
python3 cli.py chat --auto-model gpt2

# Same, but pin HF local load device (forwarded to POST /models/load)
python3 cli.py chat --auto-model gpt2 --device mps --load-mode local
```

### Start Server
```bash
# CPU mode (default for Intel Macs)
python3 apps/api/server/main.py

# With uvicorn (from repo root)
python3 -m uvicorn main:app --app-dir apps/api/server --host 0.0.0.0 --port 8000
```

**Web UI** (another terminal): `cd apps/web && npm install && npm run dev` → http://localhost:3000

**API + web in one terminal** (from repo root): `./scripts/dev-stack.sh`, `make dev-stack`, or `npm install && npm run dev:stack` — see **QUICKSTART.md**. Root **`package.json`** checks: **`npm run test:repo-root`**, **`make test-repo-root`**, or **`python3 -m pytest tests/test_repo_root_package_json.py -q`**.

### Docker
```bash
# Start with Docker (run from repo root)
./scripts/deploy/docker-manage.sh start

# Development mode
./scripts/deploy/docker-manage.sh dev

# GPU mode
./scripts/deploy/docker-manage.sh gpu
```

### Feedback & Adaptation System
```bash
# The feedback system runs automatically when the server starts
# It collects user feedback and creates per-user LoRA adapters

# View feedback stats
curl http://localhost:8000/meta-weights/stats

# View user adapters
curl http://localhost:8000/user-adapters

# View workflow status
curl http://localhost:8000/workflow/status

# Export training data (DPO format)
curl -X POST http://localhost:8000/feedback/export-training \
  -H "Content-Type: application/json" \
  -d '{"format": "dpo"}'
```

See **Training page** in the web UI for admin controls (aggregation, pruning, export).

### Verify install (optional)
From the repo root, `verify.sh` checks core paths and (if **`ruff`** is available) runs the same lint smoke as CI; if **`node`** is available and **`apps/web/node_modules`** exists, it runs **`npm run ci`** there (lint + typecheck + Vitest + **`npm run build:clean`** — clean **`.next`** then **`next build`**). It prints **root `package.json`** contract commands (**`npm run test:repo-root`**, **`make test-repo-root`**, etc.) and other commands that mirror CI (**`test-web`**, **`test-sdk-ts`**, **`sdk-test-py`**, **`standards-schemas`** in **`.github/workflows/ci_cd.yml`**). See **QUICKSTART.md** for the full install flow.

```bash
python3 -m pip install -e ".[dev]"   # optional; includes ruff + pytest among dev tools
./verify.sh
# ./run.sh puts .venv/bin on PATH when present — e.g. ./run.sh python3 -m pytest tests/ -q
```

### Google Colab
Use `sloughgpt_colab.ipynb` in the repo root. After the runtime can see the repo (clone or upload), run the dependency cell: it installs base packages plus **`python3 -m pip install -e .`** so **`cli.py`** and **`domains`** imports resolve.

Recommended order: **§2** (dataset; default Shakespeare → `datasets/shakespeare.txt` via Karpathy URL), **§3** (device + imports), **§4–§6** (CONFIG, model, exploration). Train with exactly one path: manual **§7** loop, **`RUN_TRAIN_PIPELINE`** in **§7b** (`train_sloughgpt()`), or the optional **`SloughGPTTrainer`** cell (skip the others). Then **§8+** (generation / Soul).

Then a typical one-shot chat + model load is:

```bash
python3 cli.py chat --auto-model gpt2
```

If your editor reformats the notebook JSON on save, prefer small cell edits or restore with `git checkout -- sloughgpt_colab.ipynb` to avoid large diffs.

For a **fast local execute** (CPU, capped §7 batches), run [`scripts/run_colab_notebook_smoke.sh`](scripts/run_colab_notebook_smoke.sh) or **`make colab-smoke`** (`./scripts/run_colab_notebook_smoke.sh --help` lists env defaults; **`make help`** lists Makefile shortcuts); it writes `sloughgpt_colab.executed.ipynb` (ignored by git). **`make colab-test`** runs only the Colab regression pytest module. Override `SLOUGH_NOTEBOOK_TRAIN_CAP` or `SLOUGH_NOTEBOOK_FORCE_CPU` as needed. The script calls **`jupyter nbconvert`** when available, otherwise **`python3 -m nbconvert`**. Install tools with **`python3 -m pip install jupyter nbclient nbformat`** or **`python3 -m pip install -e ".[notebook]"`** (see **`pyproject.toml`** optional-deps **`notebook`**).

### Documentation map

- **Structure & conventions:** [docs/STRUCTURE.md](docs/STRUCTURE.md)
- **Roadmap:** [docs/TODO.md](docs/TODO.md)
- **Installation details:** [docs/INSTALL.md](docs/INSTALL.md)
- **Extra reference notes** (stack comparisons, API drafts, reports): [docs/misc/](docs/misc/)
- **Standalone Python tools** (benchmarks, GGUF helpers, …): [scripts/tools/README.md](scripts/tools/README.md)
- **Deploy / Docker shell scripts**: [scripts/deploy/README.md](scripts/deploy/README.md)
- **Extra config samples** (`datasets.yaml` sketch, conda env, prod env example): [config/README.md](config/README.md)
- **Runtime data** (experiments, feature store, tuning, vector DB): [data/README.md](data/README.md)
- **Full local setup** (conda/venv/Docker): [scripts/setup.sh](scripts/setup.sh)

## GPU Support

| Hardware | Support | Speed |
|----------|---------|-------|
| NVIDIA GPU (CUDA) | ✅ Full | Fast |
| Apple Silicon (MPS) | ✅ Full | Good |
| AMD GPU (ROCm) | ✅ Linux only | Good |
| Intel Mac + AMD | ❌ CPU only | Slow |

## Industry Standard Optimizations

```python
from domains.training.optimized_trainer import Presets

# Auto-detect best settings
config = Presets.auto()

# Or specific hardware
config = Presets.high_end_gpu()   # A100, H100, RTX 4090
config = Presets.apple_silicon()  # M1/M2/M3
config = Presets.cpu_only()        # CPU training
```

Optimizations:
- Mixed Precision (FP16/BF16) - 2-3x speedup
- Gradient Checkpointing - 50% memory savings
- torch.compile - 1.5-2x speedup
- Flash Attention (NVIDIA/AMD) - 2-4x speedup

## API Endpoints

### Health & Status
```bash
# Basic health
curl http://localhost:8000/health

# Liveness probe (Kubernetes)
curl http://localhost:8000/health/live

# Readiness probe (Kubernetes)
curl http://localhost:8000/health/ready

# Detailed health
curl http://localhost:8000/health/detailed
```

### Authentication
```bash
# Create JWT token from API key
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-api-key"}'

# Verify token
curl -X POST http://localhost:8000/auth/verify \
  -H "Authorization: Bearer <token>"

# Refresh token
curl -X POST http://localhost:8000/auth/refresh \
  -H "Authorization: Bearer <token>"
```

### Rate Limiting
```bash
# Check rate limit status
curl http://localhost:8000/rate-limit/status

# Check your current usage
curl http://localhost:8000/rate-limit/check
```

### Caching
```bash
# Cache statistics
curl http://localhost:8000/cache/stats

# Clear cache
curl -X DELETE http://localhost:8000/cache
```

### Metrics
```bash
# JSON metrics
curl http://localhost:8000/metrics

# Prometheus format
curl http://localhost:8000/metrics/prometheus

# Security audit logs
curl http://localhost:8000/security/audit
```

### Models (Hugging Face local load)
```bash
curl -s -X POST http://localhost:8000/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id":"gpt2","mode":"local","device":"cpu"}'
# Response includes effective_device when weights attach to inference globals.
```

### Inference
```bash
# Generate text
curl -X POST http://localhost:8000/inference/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_new_tokens": 50}'

# Streaming (SSE)
curl -X POST http://localhost:8000/inference/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_new_tokens": 100}'

# Batch processing (up to 50 prompts)
curl -X POST http://localhost:8000/inference/batch \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello", "Hi there", "Good morning"], "max_new_tokens": 50}'

# Inference stats
curl http://localhost:8000/inference/stats
```

### Training

Native trainer `step_*.pt` on the API host includes `stoi` / `itos` / `chars` for fair `cli.py eval`; see **docs/policies/CONTRIBUTING.md** (*Checkpoint vocabulary*).

```bash
# Start training
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"dataset": "shakespeare", "epochs": 5, "batch_size": 32}'

# List training jobs
curl http://localhost:8000/training/jobs
```

### Experiments
```bash
# Create experiment
curl -X POST "http://localhost:8000/experiments?name=test&description=Testing"

# Log metrics
curl -X POST "http://localhost:8000/experiments/{id}/log_metric?metric_name=loss&value=2.5&step=0"
```

### Benchmarking
```bash
# Run benchmark
curl -X POST http://localhost:8000/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_new_tokens": 20}'

# Compare quantization levels
curl http://localhost:8000/benchmark/compare
```

### Model Export

Export artifacts (API or `cli.py export`) are for deployment; char-LM perplexity parity uses native trainer `step_*.pt` — **docs/policies/CONTRIBUTING.md** (*Checkpoint vocabulary*).

```bash
# Export model (SafeTensors, Torch, ONNX, GGUF, .sou)
curl -X POST http://localhost:8000/model/export \
  -H "Content-Type: application/json" \
  -d '{"output_path": "models/exported", "format": "safetensors"}'

# List formats
curl http://localhost:8000/model/export/formats

# CLI export examples (see cli.py export --help; -f is an alias for --format)
python3 cli.py export models/sloughgpt.pt -f safetensors
python3 cli.py export models/sloughgpt.pt -f gguf_q4_k_m --quantize Q4_K_M  # mobile
python3 cli.py export models/sloughgpt.pt -f onnx --seq-len 128            # cross-platform
```

### Soul Engine
```bash
# Load a soul (.sou model)
curl -X POST http://localhost:8000/load-soul \
  -H "Content-Type: application/json" \
  -d '{"soul_path": "models/slough.sou"}'

# Get current soul profile
curl http://localhost:8000/soul

# Generate with soul (uses personality + reasoning)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "use_soul": true}'
```

### Datasets

Multi-source dataset import, export, and management:

```bash
# List datasets
curl http://localhost:8000/datasets

# Import from various sources
curl -X POST http://localhost:8000/datasets/import/github \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/user/repo", "name": "my-dataset"}'

curl -X POST http://localhost:8000/datasets/import/huggingface \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "username/dataset"}'

curl -X POST http://localhost:8000/datasets/import/kaggle \
  -H "Content-Type: application/json" \
  -d '{"dataset": "username/dataset-name"}'

curl -X POST http://localhost:8000/datasets/import/csv \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/data.csv", "name": "my-csv-data"}'

# Export datasets
curl -X POST http://localhost:8000/datasets/my-dataset/export \
  -H "Content-Type: application/json" \
  -d '{"format": "json"}'

# Combine datasets
curl -X POST http://localhost:8000/datasets/combine \
  -H "Content-Type: application/json" \
  -d '{"dataset_ids": ["ds1", "ds2"], "name": "combined"}'

# Preview dataset
curl http://localhost:8000/datasets/my-dataset/preview
```

**Web UI**: http://localhost:3000/datasets - Import, combine, export datasets with UI.

## Architecture

```
SloughGPT/
├── apps/
│   ├── api/server/main.py   # FastAPI app (primary API)
│   ├── cli/                 # CLI implementation
│   └── web/                 # Next.js UI — app/(app)/
├── packages/
│   ├── core-py/domains/     # Soul engine, training, inference, etc.
│   ├── sdk-py/sloughgpt_sdk/
│   ├── sdk-ts/typescript-sdk/
│   └── standards/
├── infra/docker/            # Dockerfiles & docker-compose.yml
├── infra/k8s/               # Kubernetes & Helm
├── tests/
├── cli.py                   # CLI entry (wrapper)
└── pyproject.toml
```

## Supported Models

| Model | Size | Context | Recommended |
|-------|------|---------|-------------|
| GPT-2 | 124M | 1K | FP16 |
| GPT-2 Medium | 355M | 1K | FP16 |
| GPT-2 Large | 774M | 1K | FP16 |
| Phi-2 | 2.7B | 2K | Q4_K |
| Mistral 7B | 7.3B | 32K | Q4_K |
| LLaMA-2 7B | 7B | 4K | Q4_K |
| Qwen2 1.5B | 1.5B | 32K | Q4_K |
| Gemma 2B | 2B | 8K | Q4_K |

## Installation

```bash
python3 -m pip install torch transformers fastapi uvicorn
```

## Development

```bash
# Run server
python3 apps/api/server/main.py

# Train a model (after `pip install -e .`; same as packages/core-py `domains.training.train_pipeline`)
python3 -m domains.training.train_pipeline --data datasets/shakespeare/input.txt --epochs 5
# Periodic step_*.pt embeds stoi/itos/chars for cli.py eval — docs/policies/CONTRIBUTING.md (Checkpoint vocabulary)
```

## Contributing & security

- **Contributing** — [CONTRIBUTING.md](CONTRIBUTING.md)
- **Security** — [SECURITY.md](SECURITY.md)
- **Agents / automation** — [AGENTS.md](AGENTS.md) (links [`.agents/skills/SKILL.md`](.agents/skills/SKILL.md) and structure docs)

## License

MIT
