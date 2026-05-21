# SloughGPT Quick Start Guide

## Get Started in 5 Minutes

### 1. Install
```bash
git clone https://github.com/iamtowbee/sloughGPT.git
cd sloughGPT
python3 -m pip install torch transformers fastapi uvicorn pydantic pytest
# Editable install + dev tools (ruff, pytest, …) and the ``sloughgpt`` console script
python3 -m pip install -e ".[dev]"
./verify.sh
# With a .venv, prefix commands so they use that interpreter: ./run.sh python3 -m pytest tests/ -q
# Minimal editable install only: python3 -m pip install -e .  (add dev extras or python3 -m pip install ruff to use ./verify.sh lint)
# Next.js (apps/web): npm ci && npm run ci — same as CI job test-web (lint, typecheck, Vitest, build:clean / next build)
# TypeScript SDK (packages/sdk-ts/typescript-sdk): npm ci && npm run ci — job test-sdk-ts (lint + build + test)
# Python SDK: python3 -m pytest tests/test_sdk.py — job sdk-test-py
# Standards: python3 scripts/validate_standards_schemas.py (jsonschema in .[dev]) — job standards-schemas
# Colab notebook full execute locally: python3 -m pip install -e ".[notebook]" — ./scripts/run_colab_notebook_smoke.sh or make colab-smoke; make help lists colab targets (README → Google Colab). make colab-test runs tests/test_sloughgpt_colab_notebook.py only. Colab pytest module shells out to bash for --help; without bash that subtest skips (Windows: Git Bash / WSL).
```

### 2. Quick Training (CLI)
```bash
python3 cli.py quick --steps 100 --prompt "Hello world"
```

### 3. Start API Server
```bash
python3 apps/api/server/main.py
# Access at http://localhost:8000/docs
```

**Web UI** (another terminal): `cd apps/web && npm install && npm run dev` → http://localhost:3000

**API + web together** (one terminal; Ctrl+C stops both): `./scripts/dev-stack.sh`, `make dev-stack`, or **`npm install` at repo root once then `npm run dev:stack`** (uses `concurrently`; same processes as the shell script).

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
python3 cli.py quick --steps 100 --prompt "The future is"

# Custom model config
python3 cli.py quick --epochs 3 --batch 64 --embed 256 --layers 6

# CPU only (no optimizations)
python3 cli.py quick --no-optimize

# Full char-level trainer: merges config.yaml with CLI flags (intervals, device, dropout, LoRA, checkpoints, …)
python3 cli.py train --dataset shakespeare --epochs 3 --checkpoint-dir ckpts
# Module entrypoint (no config.yaml merge; --dropout / --lora-alpha on main): python3 -m domains.training.train_pipeline --data datasets/shakespeare/input.txt --epochs 3
# FP16 mixed-precision preset (after merge): add --optimized
# API job: python3 cli.py train --api --dataset shakespeare --epochs 2
# Char-LM perplexity on held-out text (fair when checkpoint embeds stoi/itos/chars — e.g. cli.py train step_*.pt):
#   python3 cli.py eval --checkpoint models/sloughgpt.pt --data datasets/shakespeare/input.txt
#   python3 -m domains.training.lm_eval_char --checkpoint PATH --data PATH [--json]
# Weights-only bundles without stoi: eval rebuilds vocab from --data (see eval warning). See docs/policies/CONTRIBUTING.md (Checkpoint vocabulary).
# Details: apps/cli/README.md
```

### Inference
```bash
# Generate text (local: uses models/sloughgpt.sou if present, else newest models/*.sou, else models/sloughgpt_finetuned.pt)
python3 cli.py generate "Hello world" --max-tokens 100
# Short alias:
python3 cli.py gen "Hello world" --max-tokens 100

# Interactive chat (auto-start API if needed)
python3 cli.py chat

# Interactive chat with model preload (recommended)
python3 cli.py chat --auto-model gpt2

# HuggingFace model
python3 cli.py hf-serve gpt2

# Download model
python3 cli.py hf-download gpt2
```

### Benchmarking
```bash
# Benchmark inference
python3 cli.py benchmark -m gpt2 -d mps -t latency

# Full benchmark suite
python3 cli.py benchmark -m gpt2 -d mps -t all

# Check GPU optimizations
python3 cli.py optimize
```

### Model Export

Export targets (ONNX, GGUF, `.sou`, …) do not preserve native char `stoi` / `itos` the same way as trainer `step_*.pt`; for perplexity parity with training, score the native bundle — **docs/policies/CONTRIBUTING.md** (*Checkpoint vocabulary*).

```bash
# Export a checkpoint on disk (-f / --format; see cli.py export --help)
python3 cli.py export models/sloughgpt.pt -f onnx --seq-len 128
python3 cli.py export models/sloughgpt.pt -f safetensors

# GGUF-style exports support --quantize (Q4_K_M, Q5_K_M, Q8_0, F16, F32)
python3 cli.py export models/sloughgpt.pt -f gguf_q4_k_m --quantize Q4_K_M
```

### System
```bash
# System info
python3 cli.py system

# Health check
python3 cli.py health

# Docker management
python3 cli.py docker status
python3 cli.py docker logs

# Environment check
python3 cli.py config check

# Configuration validation
python3 cli.py config validate

# Generate secrets
python3 cli.py config generate --type all

# Disk summary (models/, datasets/, checkpoints/, data/experiments/, …)
python3 cli.py stats

# One path: line/char stats or validate
python3 cli.py data stats datasets/shakespeare/input.txt
python3 cli.py data validate datasets/shakespeare

# List datasets
python3 cli.py datasets

# List model artifacts under models/ (.pt, .safetensors, .sou) + HF hints
python3 cli.py models

# Built-in personality presets
python3 cli.py personalities

# Inspect a checkpoint file (tensor layout)
python3 cli.py info models/sloughgpt.pt
```

### API Management
```bash
# Check API status
python3 cli.py api-status

# Test API endpoints
python3 cli.py api-test

# Test authentication
python3 cli.py api-auth

# Compare models
python3 cli.py compare
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

### Batch Processing
```bash
# Batch generation (up to 50 prompts)
curl -X POST http://localhost:8000/inference/batch \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello", "Hi"], "max_new_tokens": 20}'
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

Native trainer `step_*.pt` on the API host includes `stoi` / `itos` / `chars` for fair `cli.py eval`; see **docs/policies/CONTRIBUTING.md** (*Checkpoint vocabulary*).

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
python3 cli.py optimize
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

# Helm chart (bundled in repo)
helm install sloughgpt ./infra/k8s/helm/sloughgpt/ -n sloughgpt --create-namespace
```

---

## Optimization Presets

```python
from domains.training.optimized_trainer import Presets

# Auto-detect best settings
config = Presets.auto()

# Specific hardware
Presets.high_end_gpu()   # A100, H100, RTX 4090
Presets.mid_range_gpu()  # RTX 3060, V100
Presets.apple_silicon()  # M1/M2/M3
Presets.cpu_only()       # CPU training
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
- **API server** (`apps/api/server/main.py`): disables MPS by default on macOS via `domains.torch_runtime.apply_api_process_torch_env`. To experiment with MPS inference: `SLOUGHGPT_API_ENABLE_MPS=1` (before `import torch`). Linux/CUDA is no longer forced to CPU.
- **Training DataLoaders**: `domains.torch_runtime.effective_dataloader_num_workers` clamps workers to `0` on macOS (fork + MPS deadlocks). See `packages/core-py/domains/torch_runtime.py` for env vars (`SLOUGHGPT_SKIP_TORCH_ENV`, etc.).

### Docker not running?
```bash
open -a Docker
```

### Out of memory?
```bash
# Smaller batch size
python3 cli.py quick --batch 8

# Or export a smaller artifact (example: GGUF with 4-bit)
python3 cli.py export models/sloughgpt.pt -f gguf_q4_k_m --quantize Q4_K_M
```

---

## File Structure

```
SloughGPT/
├── domains/
│   ├── inference/          # Inference engine
│   │   ├── engine.py       # Production inference
│   │   ├── quantization.py  # FP16/INT8/INT4
│   │   ├── optimizations.py # KV cache, batching
│   │   └── sou_format.py   # .sou model format
│   ├── training/            # Training
│   │   ├── train_pipeline.py
│   │   ├── optimized_trainer.py  # Optimized training
│   │   ├── models/nanogpt.py
│   │   ├── huggingface/     # HuggingFace integration
│   │   └── scripts/tools/lora.py  # LoRA fine-tuning (utility)
│   └── ml_infrastructure/  # Infrastructure
│       ├── benchmarking.py
│       └── experiment_tracker.py
├── apps/api/server/main.py  # FastAPI server
├── cli.py                    # CLI commands
├── infra/k8s/               # Kubernetes, Helm, Grafana assets
├── tests/                   # Unit tests (100+ tests)
├── datasets/                # Training data
├── data/                    # Runtime state (experiments, feature store, tuning, vector DB)
├── infra/docker/docker-compose.yml  # Docker deployment
└── sloughgpt_colab.ipynb   # Colab notebook
```

---

## Next Steps

1. **Run the notebook**: `jupyter notebook sloughgpt_colab.ipynb` (in Colab: install → **§2** dataset → **§3–§6** → pick one of **§7** manual loop, **§7b** `train_sloughgpt()`, or optional **`SloughGPTTrainer`**; then e.g. `python3 cli.py chat --auto-model gpt2`). For a **fast local full execute**, use `./scripts/run_colab_notebook_smoke.sh` or **`make colab-smoke`** (**`make help`**, **README.md** → *Google Colab*; install **`jupyter`** / **`python3 -m nbconvert`** as documented there).
2. **Try different datasets**: Shakespeare, **`tiny`** (small on-disk slice), or a path to your own `.txt`
3. **Explore model architecture**: Section 5 in the notebook
4. **Deploy with Docker**: See Docker section above
5. **Read the docs**: `README.md`, `docs/API.md`, `docs/TODO.md`

---

## Links

- **GitHub**: https://github.com/iamtowbee/sloughGPT
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Security**: [SECURITY.md](SECURITY.md)
- **Agents**: [AGENTS.md](AGENTS.md)
- **API Docs**: http://localhost:8000/docs
- **Tests**: `python3 -m pytest tests/`
