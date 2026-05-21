# SloughGPT Formal Specification

## Overview

SloughGPT is a self-hosted LLM infrastructure platform with character-level training from scratch.

## Package Architecture

```
sloughGPT/
├── packages/
│   ├── core-py/          # Python domain logic (importable as `domains`)
│   ├── sdk-py/           # Python API client
│   ├── sdk-ts/           # TypeScript SDK
│   ├── strui/            # React UI components
│   ├── standards/         # JSON schemas
│   └── tui-ts/           # Terminal UI
├── apps/
│   ├── api/server/       # FastAPI server
│   ├── cli/              # CLI app
│   ├── web/              # Next.js web UI
│   └── tui/              # Terminal UI
└── tests/                # pytest suite
```

## Core Packages

### `packages/core-py/domains/` - Domain Logic

| Module | Purpose |
|--------|---------|
| `models/` | Model interfaces, SloughGPTModel (RoPE + SwiGLU + RMSNorm) |
| `training/` | Model training, `SloughGPTTrainer` (canonical), `TrainerProtocol` |
| `inference/` | Production inference engine, KV cache, batching |
| `cognitive/` | Memory, reasoning, learning, creativity |
| `soul/` | SLO (Soul) evolution stages |
| `infrastructure/` | Database, cache, RAG, deployment |
| `enterprise/` | Auth, monitoring, billing |

### `packages/sdk-py/` - Python SDK

API client library. **Does NOT train locally.**

```python
from sloughgpt_sdk import SloughGPTClient

client = SloughGPTClient()
client.start_training(dataset_id="my-data")
```

### `packages/sdk-ts/` - TypeScript SDK

Same as Python SDK, for JS/TS applications.

## Canonical Implementations

### Trainer

**Canonical:** `domains.training.train_pipeline.SloughGPTTrainer`
**Protocol:** `domains.training.trainer_protocol.TrainerProtocol`

```python
from domains.training import SloughGPTTrainer

trainer = SloughGPTTrainer(data_path="data.txt")
trainer.train()
```

### Training Config

**Canonical:** `domains.training.train_pipeline.TrainerConfig`

All other `TrainingConfig` classes are deprecated.

### Model

**Canonical:** `domains.models.SloughGPTModel`

Uses:
- Rotary Position Embeddings (RoPE)
- SwiGLU activation
- RMSNorm
- SDPA attention

### TextDataset

**Canonical:** `domains.training.train_pipeline.TextDataset`

## Deprecated/Unified Classes

| Deprecated | Use Instead |
|------------|-------------|
| `Trainer` (unified_training) | `SloughGPTTrainer` |
| `OptimizedTrainer` | `SloughGPTTrainer` |
| `TrainingConfig` (optimized_trainer) | `TrainerConfig` |
| `TrainingConfig` (unified_training) | `TrainerConfig` |
| `InferenceEngine` (training/) | `InferenceEngine` (inference/) |
| `InferenceEngine` (ml_infrastructure/) | `Predictor` |

## Duplicate Consolidations

### TextDataset (4 copies → 1)

Consolidate into `domains/training/train_pipeline.py`:
- `domains/training/unified_training.py`
- `domains/training/train_pipeline.py`
- `train_sloughgpt.py`
- `scripts/tools/export_for_cloud.py`

### RAG System

Choose canonical: `domains/infrastructure/rag.RAGSystem`

Deprecate:
- `domains/cognitive/rag.ProductionRAG`
- `domains/inference/streaming.StreamingRAG`
- `domains/cognitive/grounding.RAGGrounder`

### KnowledgeGraph

Choose canonical: `domains/cognitive/knowledge_graph_v2.KnowledgeGraph`

Deprecate:
- `domains/cognitive/knowledge_graph.KnowledgeGraph` (v1)
- `domains/infrastructure/rag.SLOKnowledgeGraph`

### BM25

Choose canonical: `domains/infrastructure/rag.BM25`

Deprecate:
- `domains/cognitive/rag.BM25Indexer`

### InferenceEngine

**Canonical:** `domains/inference/engine.InferenceEngine`

**Deprecated:**
- `domains/training/inference_engine.InferenceEngine` → use canonical

**Removed:**
- `domains/ml_infrastructure/model_serving.py` → entirely unused (contained duplicate InferenceEngine + generic ModelServer)

## Public API Contracts

### TrainerProtocol

```python
class TrainerProtocol(Protocol):
    def train(
        self,
        resume: bool = False,
        resume_path: Optional[str] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        ...
```

### Model Interface

```python
class ModelInterface(Protocol):
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ...

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        ...
```

## Entry Points

| Entry Point | Command |
|-------------|---------|
| Training | `python train_sloughgpt.py` |
| CLI | `python -m apps.cli` |
| API Server | `python -m apps.api.server.main` |
| TUI | `python -m apps.tui` |

## Development

### Testing
```bash
python3 -m pytest tests/ -q
```

### Dev Stack
```bash
./scripts/dev-stack.sh
```

## File Naming Conventions

- `snake_case.py` for modules
- `PascalCase.py` only for main entry points if needed
- `__init__.py` required for all packages
- `__all__` required in all `__init__.py`

## Import Conventions

```python
# Absolute imports within package
from domains.training import SloughGPTTrainer

# Relative imports within same package
from .models import SloughGPTModel

# Type hints
from typing import Optional, Dict, Any
```

## TODO Items

- [x] Consolidate trainer classes to `SloughGPTTrainer`
- [x] Consolidate `TextDataset` classes
- [x] Unify `TrainingConfig` classes
- [x] Consolidate RAG implementations
- [x] Consolidate KnowledgeGraph implementations
- [x] Remove `recovered.py` files
- [x] Simplify SDK imports
- [x] Fix duplicate ConnectionManager bug
- [x] Consolidate DatasetQualityScorer
- [x] Add __all__ exports to key modules
- [x] Consolidate InferenceEngine classes (3 implementations)
- [x] Add __all__ exports to infrastructure submodules
- [x] Fix string literal bug in inference/__init__.py
- [x] Deprecate duplicate device functions (get_optimal_device, get_device_name)
- [x] Deprecate duplicate RoPE functions (rotate_half, apply_rotary_pos_emb)
- [x] Remove unused deployment_utils.py
- [x] Fix empty apps/cli/__init__.py
- [x] Consolidate find_available_port (3 copies → shared utils)
- [x] Remove unused files: benchmark.py, model_loader.py, wandb_integration.py, mlflow_integration.py, external_integrations.py, knowledge_graph_engine.py
- [x] Remove unused deployment_utils.py (331 lines)
- [x] Add smart GPU auto-detection (Metal/CUDA)
- [x] Use llama.cpp with CPU/GPU selection (no Ollama dependency)

## Performance Notes (2026-04-11)

### Hardware Analysis

| Hardware | Configuration | Prompt | Generation |
|----------|---------------|--------|------------|
| Intel MacBook Pro 15,1 | AMD Radeon 555X (CPU faster!) | ~31 tok/s | ~16 tok/s |
| llama.cpp CPU | llama3.2-1b Q8_0 | ~31 tok/s | ~16 tok/s |
| llama.cpp Metal (AMD) | llama3.2-1b Q8_0 | ~25 tok/s | ~5 tok/s |

### Benchmark (llama3.2-1b Q8_0, 1.22GB)
```
llama-bench -m ~/models/llama3.2-1b-q8_0.gguf -t 8 -ngl 0
| model    | pp32 | tg32 |
|----------|-------|------|
| Q8_0     | 33 t/s| 12 t/s|
```

### Inference Engine Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  API Layer (simple_server.py)                     │
│                  POST /generate → backend.generate()              │
└─────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────────┐   ┌───────────────────┐   ┌─────────────────────┐
│   llama.cpp-CPU   │   │  llama.cpp-GPU   │   │   PyTorch GPT-2     │
│  (auto-detected)  │   │ (M3/M4 with Metal │   │    (fallback)       │
│   ~16 tok/s       │   │    tensor ops)    │   │                     │
└───────────────────┘   └───────────────────┘   └─────────────────────┘
```

### llama.cpp Inference Engine

GGUF models handled by llama.cpp:
- **llama-cpp-python**: Primary (if installed)
- **llama-cli**: Subprocess fallback

Located at `packages/core-py/domains/inference/llama_engine.py`:
- `LlamaInferenceEngine`: Main engine
- `LlamaCLIInferenceEngine`: CLI subprocess
- `detect_gpu()`: Detect Metal/CUDA GPU
- `auto_select_backend()`: Smart CPU/GPU selection

### GPU Auto-Detection

| GPU | Detection | Result |
|-----|-----------|--------|
| Apple M3/M4/M5 | Metal tensor ops | ✅ GPU |
| AMD Radeon (Intel Mac) | No tensor ops | ❌ CPU (faster) |
| Intel integrated | No GPU | ❌ CPU |
| NVIDIA (Linux) | CUDA tensor cores | ✅ GPU |

**Environment Variables:**
```bash
SLOUGHGPT_MODEL_PATH=~/models/model.gguf  # Required
SLOUGHGPT_FORCE_GPU=1                      # Force GPU
SLOUGHGPT_FORCE_CPU=1                      # Force CPU
```

### Server Usage

```bash
# Start server
cd apps/api/server
SLOUGHGPT_MODEL_PATH=~/models/llama3.2-1b-q8_0.gguf python simple_server.py

# Generate
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_new_tokens":50}'
```

### Status
- ✅ GGUF model loading
- ✅ Smart CPU/GPU selection
- ✅ llama-cli fallback
- ✅ PyTorch GPT-2 fallback
- ✅ No external dependencies (Ollama removed)
- ✅ llama-cli fallback when llama-cpp-python unavailable
- ✅ GPU auto-detection and smart backend selection
