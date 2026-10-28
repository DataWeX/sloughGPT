# SloughGPT

Self-hosted LLM infrastructure with local model training, inference, and a web UI. Training and inference run on SloNet, an in-repository pure-NumPy autograd engine (reverse-mode and forward-mode differentiation) — PyTorch and HuggingFace Transformers are not required.

## Features

- **SloNet engine** — pure-NumPy training and inference with KV-cache streaming; removes approximately 4 GB of framework dependencies
- **Soul checkpoints** — `.sou` files carry personality metadata (traits, system prompt) alongside weights; system prompts hot-reload without a model restart
- **`.slnc` weight format** — memory-mapped zero-copy loading, measured 2.2× faster than safetensors on GPT-2
- **Interactive shell** — 40+ commands, pipelines, aliases, tab completion (`sloughgpt shell`; see [docs/SHELL.md](docs/SHELL.md))
- **Rust gateway** — Axum HTTP/SSE gateway in `apps/gateway`
- **Export and quantization** — GGUF (llama.rn mobile), ONNX, safetensors; FP16, INT8, INT4, Q4_K_M, Q5_K_M
- **Dataset pipelines** — import from GitHub, HuggingFace, Kaggle, URL, or CSV; combine, preview, and version datasets
- **Feedback-driven adaptation** — thumbs feedback trains per-user LoRA adapters, aggregated and evaluated automatically
- **Interfaces** — Next.js web UI (`@sloughgpt/strui` component library), React Native mobile client, Python and TypeScript SDKs

## Prerequisites

- Python >= 3.9
- Node.js >= 20

## Quick Start

```bash
git clone git@github.com:DataWeX/sloughGPT.git
cd sloughGPT
python3 -m pip install -e ".[dev]"
```

Start the API server and web UI:

```bash
# Terminal 1 — API
python3 apps/api/server/main.py

# Terminal 2 — Web UI
cd apps/web && npm install && npm run dev
# → http://localhost:5173

# Or both in one command
./scripts/dev-stack.sh

# Or as a systemd --user unit — survives session/opencode restarts
./scripts/systemd/install.sh           # install + enable + start slough-api
journalctl --user -u slough-api -f     # logs

# Or the PGQ-supervised Python launcher — one entry point, waits for both
# ports, prints the one URL (this is what the notebook's Serve section uses)
python3 scripts/serve_ui.py
# → http://localhost:5173  (web UI)   http://localhost:8000  (API)
```

## CLI

The `sloughgpt` CLI provides training, inference, chat, and model management:

```bash
# Train from scratch (SloNet)
sloughgpt train start --dataset shakespeare --epochs 3

# Train a native transformer
sloughgpt train native --dataset shakespeare --epochs 10 --embed 128

# Quick train + generate (auto-optimized)
sloughgpt train quick --steps 100 --prompt "Hello world"

# Distill from a teacher model
sloughgpt train distill datasets/shakespeare/input.txt --epochs 10

# Interactive chat
sloughgpt chat

# Generate text
sloughgpt generate "The meaning of life is" --model gpt2

# Evaluate perplexity
sloughgpt train eval --checkpoint models/sloughgpt.soul --data datasets/shakespeare/input.txt

# Export a model
sloughgpt model export models/sloughgpt.soul -f gguf_q4_k_m

# Run the interactive shell
sloughgpt shell

# System status
sloughgpt system status
```

## Project Structure

```
sloughGPT/
├── apps/
│   ├── api/server/            # FastAPI backend
│   ├── web/                   # Next.js frontend
│   ├── cli/                   # CLI implementation
│   ├── mobile/                # React Native app
│   ├── gateway/               # Rust Axum gateway
│   └── lib/                   # Shared frontend/backend helpers
├── domain/                   # Core Python logic (training, inference, feedback, shell, infrastructure)
├── packages/
│   ├── core-py/               # Python core (tests, utils)
│   ├── strui/                 # @sloughgpt/strui component library
│   ├── mogdb/                 # Document database
│   ├── sdk-py/                # Python SDK
│   ├── sdk-ts/                # TypeScript SDK
│   └── standards/             # Shared schemas
├── datasets/                  # Training data
├── models/                    # Saved checkpoints
├── tests/                     # Test suite
└── scripts/                   # Build, deploy, benchmarks
```

## Consciousness System

SloughGPT includes a consciousness system with 26 pages for monitoring and managing AI self-awareness:

| Feature | Description |
|---------|-------------|
| Dashboard | Main consciousness overview with health metrics |
| Training | Consciousness training controls and progress |
| Playground | Interactive experimentation with qualia and beliefs |
| Analytics | Trend analysis and statistics |
| Benchmark | Performance benchmarking tools |
| API Explorer | Interactive API testing interface |

Access the consciousness UI at `http://localhost:5173/consciousness/dashboard` after starting the web UI.

## API Endpoints

Start the server and visit `http://localhost:8000/docs` for the full interactive API reference.

Core endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Chat with a loaded model (SSE streaming — the endpoint's only state) |
| `POST` | `/inference/generate` | Text generation |
| `POST` | `/inference/generate/stream` | Streaming generation (SSE) |
| `POST` | `/training/start` | Start training (SSE progress) |
| `GET` | `/health` | Server and model health |
| `GET` | `/models` | List available models |
| `GET` | `/souls` | List available souls |
| `POST` | `/souls/switch` | Switch active soul |
| `GET` | `/datasets` | List datasets |
| `GET` | `/consciousness/status` | Get consciousness status |
| `POST` | `/consciousness/reflect` | Trigger self-reflection |
| `GET` | `/consciousness/health` | Consciousness health check |

## Development

```bash
# Python tests (parallel)
cd packages/core-py && python -m pytest -n auto -x -q

# Frontend tests
cd apps/web && npm run test

# TypeScript check
cd apps/web && npm run typecheck

# Lint
cd apps/web && npm run lint
```

## GPU Support

SloNet dispatches large tensor operations to the first available backend in `domain/slolib/_internal/gpu.py`; everything else falls back to NumPy.

| Backend | Hardware | Notes |
|---------|----------|-------|
| Metal | Apple Silicon | Disabled on Intel Macs (`x86_64` — false-positive MPS detection crashed the process); those machines run CPU |
| CUDA | NVIDIA | Requires `pip install cupy-cuda12x` |
| OpenCL | Various | Requires an OpenCL runtime |
| CPU (NumPy) | Any | Always available |

Accelerator use is forward-only — the backward pass stays in NumPy — and small operations (under 4096 elements) stay on CPU to avoid dispatch overhead.

## Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Get started in 5 minutes |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [SECURITY.md](SECURITY.md) | Security policy |
| [AGENTS.md](AGENTS.md) | AI agent workflow and conventions |
| [INFRASTRUCTURE.md](INFRASTRUCTURE.md) | Pre-LLM infrastructure layers |
| [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | Developer reference |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment guide |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | Environment configuration |
| [docs/STRUCTURE.md](docs/STRUCTURE.md) | Project structure and conventions |
| [docs/SHELL.md](docs/SHELL.md) | Shell REPL documentation |
| [docs/mogdb-guide.md](docs/mogdb-guide.md) | MogDB usage guide |
| [docs/engineering-overview.md](docs/engineering-overview.md) | System architecture overview |

## Data Infrastructure

SloughGPT uses **MogDB** as the primary embedded document database with optional JSON sync for human-readable backups.

```python
# Quick example
from mogdb import MogDB

db = MogDB("data/my_collection")
col = db.collection("items")
col.insert_one({"name": "item", "value": 42})
doc = col.find_one({"name": "item"})
```

All data stores (knowledge, RAG, model catalog, response logs, experiments, etc.) use MogDB with automatic JSON sync and TTL expiration. See [docs/mogdb-guide.md](docs/mogdb-guide.md) for details.

## Docker

```bash
./scripts/deploy/docker-manage.sh start    # API + web
./scripts/deploy/docker-manage.sh gpu      # with GPU support
```

## License

MIT
