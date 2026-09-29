# SloughGPT Quick Start Guide

## Get Started in 5 Minutes

### 1. Install
```bash
git clone git@github.com:DataWeX/sloughGPT.git
cd sloughGPT
python3 -m pip install torch transformers fastapi uvicorn pydantic pytest
# Editable install + dev tools (ruff, pytest, …) and the ``sloughgpt`` console script
python3 -m pip install -e ".[dev]"
./verify.sh
# With a .venv, prefix commands so they use that interpreter: ./run.sh python3 -m pytest tests/ -q
# Minimal editable install only: python3 -m pip install -e .  (add dev extras or python3 -m pip install ruff to use ./verify.sh lint)
# JS workspaces (Turborepo): npm install at repo root — installs all packages, then:
#   npx turbo run lint typecheck test   # run checks across all JS packages
#   npx turbo run dev                   # start all dev servers
# Web app: cd apps/web && npm ci && npm run ci
# strui: cd packages/strui && npm run lint && npm run typecheck && npm test
# TypeScript SDK: cd packages/sdk-ts/typescript-sdk && npm run ci
# Python SDK: python3 -m pytest tests/test_sdk.py
# Standards: python3 scripts/validate_standards_schemas.py
```

### 2. Quick Training (CLI)
```bash
./sloughgpt train quick --steps 100 --prompt "Hello world"
```

### 3. Start API Server
```bash
python3 apps/api/server/main.py
# Access at http://localhost:8000/docs
```

**Web UI** (another terminal): `cd apps/web && npm install && npm run dev` → http://localhost:3000

**API + web together** (one terminal; Ctrl+C stops both): `./scripts/dev-stack.sh`, `make stack`, or **`npm install` at repo root once then `npm run dev:stack`** (auto-restarts on crash; same processes as the shell script).

**Root `package.json` contract test** (optional): **`npm run test:repo-root`** (after `npm install` at repo root), **`make test-repo-root`**, or **`python3 -m pytest tests/test_repo_root_package_json.py -q`**.

Load a Hugging Face model for real generation (optional):

```bash
curl -s -X POST http://localhost:8000/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id":"gpt2","mode":"local","device":"cpu"}'
# Response includes "effective_device" when weights are attached to globals.
```

---

## CLI Commands

### Training
```bash
# Quick train + generate (auto-optimized)
./sloughgpt train quick --steps 100 --prompt "The future is"

# Custom model config
./sloughgpt train quick --epochs 3 --batch 64 --embed 256 --layers 6

# CPU only (no optimizations)
./sloughgpt train quick --no-optimize

# Trainers: start (API-backed job), native (in-process), eval (perplexity)
./sloughgpt train start --dataset shakespeare --epochs 3
# Module entrypoint (no config.yaml merge; --dropout / --lora-alpha on main): python3 -m domain.training._internal.train_pipeline --data datasets/shakespeare/input.txt --epochs 3
# API job: ./sloughgpt train start --api --dataset shakespeare --epochs 2
# Char-LM perplexity on held-out text (fair when checkpoint embeds stoi/itos/chars — e.g. sloughgpt train step_*.soul):
#   ./sloughgpt train eval --checkpoint models/sloughgpt.soul --data datasets/shakespeare/input.txt
#   python3 -m domain.training._internal.lm_eval_char --checkpoint PATH --data PATH [--json]
# Weights-only bundles without stoi: eval rebuilds vocab from --data (see eval warning). See docs/policies/CONTRIBUTING.md (Checkpoint vocabulary).
# Details: apps/cli/README.md
```

### Inference
```bash
# Generate text (local: uses models/sloughgpt.soul if present, else newest models/*.soul)
./sloughgpt generate "Hello world" --max-tokens 100
# Short alias:
./sloughgpt gen "Hello world" --max-tokens 100

# Interactive chat (auto-start API if needed)
./sloughgpt chat

# Interactive chat with model preload (recommended)
./sloughgpt chat --auto-model gpt2

# HuggingFace model
./sloughgpt hf-serve gpt2

# Download model (interactive picker)
./sloughgpt model download
```

### Benchmarking
```bash
# Benchmark inference
./sloughgpt model benchmark -m gpt2 -d mps -t latency

# Full benchmark suite
./sloughgpt model benchmark -m gpt2 -d mps -t all

# Check GPU optimizations
./sloughgpt system optimize
```

### Model Export

Export targets (ONNX, GGUF, `.soul`, …) do not preserve native char `stoi` / `itos` the same way as trainer `step_*.soul`; for perplexity parity with training, score the native bundle — **docs/policies/CONTRIBUTING.md** (*Checkpoint vocabulary*).

```bash
# Export a checkpoint on disk (-f / --format; see sloughgpt model export --help)
./sloughgpt model export models/sloughgpt.soul -f onnx --seq-len 128
./sloughgpt model export models/sloughgpt.soul -f safetensors

# GGUF-style exports support --quantize (Q4_K_M, Q5_K_M, Q8_0, F16, F32)
./sloughgpt model export models/sloughgpt.soul -f gguf_q4_k_m --quantize Q4_K_M
```

### System
```bash
# System info
./sloughgpt system

# Health check
./sloughgpt health

# Docker management
./sloughgpt docker status
./sloughgpt docker logs

# Environment check
./sloughgpt system config

# Configuration validation
./sloughgpt system config --validate

# Generate secrets
./sloughgpt system config --generate --type all

# Disk summary (models/, datasets/, checkpoints/, data/experiments/, …)
./sloughgpt stats

# One path: dataset stats or validate
./sloughgpt dataset stats shakespeare
./sloughgpt dataset validate

# List datasets
./sloughgpt dataset list

# List model artifacts under models/ (.soul, .gguf, .safetensors) + HF hints
./sloughgpt model list

# Built-in personality presets
./sloughgpt personality list

# Inspect a checkpoint file (tensor layout)
./sloughgpt model info models/sloughgpt.soul
```

### API Management
```bash
# Check status
./sloughgpt system status

# Test API endpoints or authentication
./sloughgpt system api

# Compare models
./sloughgpt model compare
```

---

## API Endpoints

### Health Check
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
# Create JWT token
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
# Cache usage per model
curl http://localhost:8000/models/cache-usage
```

### Metrics
```bash
# JSON metrics
curl http://localhost:8000/system/metrics

# Security audit logs
curl http://localhost:8000/security/audit
```

### Generate Text
```bash
curl -X POST http://localhost:8000/inference/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_new_tokens": 50}'
```

### Streaming Generation
```bash
curl -X POST http://localhost:8000/inference/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_new_tokens": 100}'
```

### Training

Native trainer `step_*.soul` on the API host includes `stoi` / `itos` / `chars` for fair `sloughgpt train eval`; see **docs/policies/CONTRIBUTING.md** (*Checkpoint vocabulary*).

```bash
curl -X POST http://localhost:8000/train \
  -d "dataset=shakespeare&epochs=5&batch_size=32"

# Tracked jobs (JSON TrainingRequest); optional log_interval / eval_interval control
# how often train_loss / eval_loss update on GET /training/jobs
curl -s -X POST http://localhost:8000/training/start \
  -H "Content-Type: application/json" \
  -d '{"name":"demo","model":"slough-base","dataset":"shakespeare","epochs":1,"batch_size":8,"learning_rate":0.001,"log_interval":10,"eval_interval":100}'
```

### Benchmarking
```bash
curl -X POST http://localhost:8000/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2", "num_runs": 10}'
```

---

## GPU Support

| Hardware | Speed | Command |
|----------|-------|---------|
| NVIDIA GPU | Fast | `--device cuda` |
| Apple Silicon (M1/M2/M3) | Good | `--device mps` |
| AMD GPU (Linux + ROCm) | Good | `--device cuda` |
| Intel Mac AMD GPU | ❌ | Use CPU |
| CPU | Slow | `--device cpu` |

### Verify GPU
```bash
./sloughgpt system optimize
```

---

## Docker Deployment

From the **repository root**, pass the stack file explicitly:

```bash
# Start API server
docker compose -f infra/docker/docker-compose.yml up -d api

# Development mode
docker compose -f infra/docker/docker-compose.yml --profile dev up -d dev

# GPU mode (NVIDIA) — api-gpu only (stop the CPU api service first to avoid port 8000 conflicts)
docker compose -f infra/docker/docker-compose.yml --profile gpu up -d api-gpu

# Stop
docker compose -f infra/docker/docker-compose.yml down
```

## Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace sloughgpt

# Apply manifests (from repository root)
kubectl apply -f infra/k8s/k8s/

# Check status
kubectl get pods -n sloughgpt

# View logs
kubectl logs -n sloughgpt -l app=sloughgpt-api
```

---

## Optimization Presets

```bash
# Model size presets (tiny / small / medium / large)
./sloughgpt train quick --preset small --steps 100

# Hardware flags on all trainers
./sloughgpt train native --device cpu --tokenizer char ...
```

### Speedup Estimates

| Optimization | Speedup | Memory |
|-------------|---------|--------|
| FP16 | 2-3x | -50% |
| torch.compile | 1.5-2x | +10% |
| Flash Attention | 2-4x | -20% |
| **Combined** | **3-6x** | **-60%** |

---

## Troubleshooting

### macOS PyTorch hangs?
```bash
# Add to ~/.zshrc or ~/.bashrc (Intel Mac + some GPU stacks)
export DYLD_INSERT_LIBRARIES=""
```
- **API server**: MPS status is reported from health checks via `domain.infrastructure.mps_monitor`.
- **Compute**: availability detection lives in `domain/training/_internal/gpu/accelerator.py`; compute runs through the numpy-based torch shim, so Macs train on CPU.

### Docker not running?
```bash
open -a Docker
```

### Out of memory?
```bash
# Smaller batch size
./sloughgpt train quick --batch 8

# Or export a smaller artifact (example: GGUF with 4-bit)
./sloughgpt model export models/sloughgpt.soul -f gguf_q4_k_m --quantize Q4_K_M
```

---

## File Structure

```
SloughGPT/
├── domain/                 # Core Python (40+ packages)
│   ├── training/
│   │   ├── engine.py
│   │   ├── tokenizer_engine.py
│   │   └── _internal/
│   │       └── train_pipeline.py   # Training loop (module entrypoint)
│   └── inference/
│       └── _internal/
│           └── slo_format.py        # .slo/.sou model format
├── apps/api/server/main.py  # FastAPI server
├── cli.py                    # CLI commands
├── infra/k8s/               # Kubernetes + Grafana assets
├── infra/docker/docker-compose.yml  # Docker deployment
├── tests/                   # Test suites (docs/TESTING.md)
├── datasets/                # Training data (runtime — created by imports/downloads)
├── data/                    # Runtime state (experiments, feature store, tuning, vector DB)
└── sloughgpt_colab.ipynb   # Colab notebook
```

---

## Next Steps

1. **Run the notebook**: `jupyter notebook sloughgpt_colab.ipynb` (in Colab: install → **§2** dataset → **§3–§6** → pick one of **§7** manual loop or **`SloughGPTTrainer`**; then e.g. `./sloughgpt chat --auto-model gpt2`). For a **fast local full execute**, use `./scripts/run_colab_notebook_smoke.sh` or **`make colab-smoke`** (**README.md** → *Google Colab*; install **`jupyter`** / **`python3 -m nbconvert`** as documented there).
2. **Try different training corpora**: the default `datasets/shakespeare/input.txt` (downloaded/imported on first use) or a path to your own `.txt`
3. **Explore model architecture**: Section 5 in the notebook
4. **Deploy with Docker**: See Docker section above
5. **Read the docs**: `README.md`, `docs/API.md`, `docs/DEVELOPER_GUIDE.md`

---

## Links

- **GitHub**: https://github.com/DataWeX/sloughGPT
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Security**: [SECURITY.md](SECURITY.md)
- **Agents**: [AGENTS.md](AGENTS.md)
- **API Docs**: http://localhost:8000/docs
- **Tests**: `python3 -m pytest tests/`
