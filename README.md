# SloughGPT

Self-hosted LLM infrastructure with local model training, inference, and a web UI.

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
# → http://localhost:3000

# Or both in one command
./scripts/dev-stack.sh
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
│   ├── gateway/               # API gateway
│   └── data/                  # Data utilities
├── packages/
│   ├── core-py/domains/       # Core Python logic
│   │   ├── training/          # SloNet, training pipelines, distillation
│   │   ├── inference/         # Vector store, context, model loading
│   │   ├── feedback/          # LoRA adapters, DPO, workflow manager
│   │   ├── multimodal/        # Vision encoder, cross-attention
│   │   ├── shell/             # Interactive REPL
│   │   └── infrastructure/    # Config, errors, rate-limiter, lifecycle
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

## API Endpoints

Start the server and visit `http://localhost:8000/docs` for the full interactive API reference.

Core endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Chat with a loaded model |
| `POST` | `/chat/stream` | Streaming chat (SSE) |
| `POST` | `/inference/generate` | Text generation |
| `POST` | `/inference/generate/stream` | Streaming generation (SSE) |
| `POST` | `/auto-train/start` | Start training (SSE progress) |
| `POST` | `/training/start` | HuggingFace fine-tuning |
| `GET` | `/health` | Server and model health |
| `GET` | `/models` | List available models |
| `GET` | `/souls` | List available souls |
| `POST` | `/souls/switch` | Switch active soul |
| `GET` | `/datasets` | List datasets |

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

| Hardware | Status | Notes |
|----------|--------|-------|
| NVIDIA (CUDA) | Supported | Fastest |
| Apple Silicon (MPS) | Supported | M1/M2/M3/M4 |
| AMD (ROCm) | Supported | Linux only |
| Intel Mac | CPU only | Stable, slower |

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

## Docker

```bash
./scripts/deploy/docker-manage.sh start    # API + web
./scripts/deploy/docker-manage.sh gpu      # with GPU support
```

## License

MIT
