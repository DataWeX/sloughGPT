# SloughGPT Roadmap

## Vision
A fast, lightweight local AI inference engine that runs on any device, with a universal weight import system for any HuggingFace model architecture.

---

## Phase 1: Core Inference Engine ✅ DONE
| Item | Status | Notes |
|------|--------|-------|
| SloNet autograd (Tensor, ops, layers) | ✅ | Pure NumPy, no PyTorch dependency |
| SloTransformer (attention, FFN, blocks) | ✅ | GPT-2 + LLaMA archs |
| GPT-2 weight import (HF→SloNet converter) | ✅ | Universal arch detection via `build_arch()` |
| `forward_fast()` + `pre_extract_weights()` | ✅ | 96x faster than naive path |
| KV cache for greedy generation | ✅ | Per-layer K/V caching |
| `generate_numpy()` (inlined autoregressive) | ✅ | 8.2ms/tok on GPT-2 |
| Bidirectional DAG (forward-mode AD) | ✅ | JVP/tangent propagation |
| `.slnc` memory-mapped format | ✅ | 2.2x faster load, demand paging |
| `.sou` checkpoint format (v3 binary) | ✅ | 1960x faster than JSON |

## Phase 2: Training Pipeline ✅ DONE
| Item | Status | Notes |
|------|--------|-------|
| Char-level LSTM trainer (SloughGPTTrainer) | ✅ | SloNet native |
| GPT-2 → SloTransformer distillation | ✅ | `distill_gpt2.py` module + CLI + API endpoint |
| HuggingFace fine-tune (Trainer + LoRA) | ✅ | `hf_finetune.py` with peft |
| Auto-train SSE streaming | ✅ | TrainingSequence phases |
| Online LoRA adapter updates | ✅ | Per-user feedback-driven |
| Activity classifier (CNN) | ✅ | Sensor data, 87.5% accuracy |
| Gradient clipping + LR scheduler | ✅ | SloReduceLROnPlateau |

## Phase 3: Model Serving ✅ DONE
| Item | Status | Notes |
|------|--------|-------|
| ModelServer (semaphore, timeout, circuit breaker) | ✅ | Crash-isolated |
| ModelRegistry (TTL cache, health summary) | ✅ | Composable backends |
| ProcessGuard (subprocess isolation) | ✅ | Memory tracking, crash recovery |
| Backend priority: Guard > Numpy > Local | ✅ | Pluggable backends |
| torch.inference_mode() optimization | ✅ | 5-15% speedup |
| CPU thread optimization | ✅ | torch.set_num_threads based on cores |
| torch.compile after warmup | ✅ | Skips <10M param models |
| MPS OOM recovery | ✅ | Auto CPU fallback |
| Health endpoint + detailed metrics | ✅ | /health, /health/detailed |

## Phase 4: API & CLI ✅ DONE
| Item | Status | Notes |
|------|--------|-------|
| FastAPI routers (20+ endpoints) | ✅ | Chat, generate, models, training, etc. |
| SSE streaming (standard envelope) | ✅ | {stream, phase, status, data, meta} |
| CLI (31+ commands) | ✅ | Click-based: model, train, checkpoint, etc. |
| Shell REPL (40+ commands) | ✅ | Pipelines, tab completion, background jobs |
| Distill CLI command | ✅ | Local + API modes, presets |
| Checkpoint list/load/delete | ✅ | CLI + shell |
| Progress bars (flicker-free) | ✅ | TTY-aware, space-padding |
| Training progress streaming | ✅ | Shell auto-follow, manual `train follow <id>` |

## Phase 5: Frontend ✅ DONE
| Item | Status | Notes |
|------|--------|-------|
| Next.js app (25+ pages) | ✅ | Chat, models, training, settings, etc. |
| Strui component library | ✅ | Card, Button, Input, etc. |
| Chat (markdown, code blocks, streaming) | ✅ | Message actions, regeneration |
| Training page (distill + fine-tune) | ✅ | Loss chart, checkpoint catalog |
| Model catalog (soul switcher) | ✅ | Thumbnails, model details dialog |
| Controllers (axios-based, all migrated) | ✅ | 15+ controllers, zero legacy api.ts |
| Knowledge management | ✅ | CRUD, batch, search, categories |
| System health monitoring | ✅ | CPU/memory chart, real-time |
| E2E test suite (Cypress) | ✅ | 6 specs, 12 assertions |
| 2000+ vitest tests | ✅ | All lib files covered |

---

## Phase 6: Remaining Work (Active Roadmap)

### 6A. Inference Quality
| Item | Priority | Est. | Description |
|------|----------|------|-------------|
| Beam search with KV cache | P2 | 2h | Current beam search doesn't use KV cache |
| Repetition penalty fix | P2 | 1h | Apply penalty to generated tokens only |
| top-k sampling in SloNet numpy | P2 | 3h | Currently greedy-only for numpy engine |
| Flash attention for HF models | P3 | 4h | torch SDPA when available |

### 6B. Training
| Item | Priority | Est. | Description |
|------|----------|------|-------------|
| Distill eval metrics (perplexity, BLEU) | P1 | 2h | Auto-evaluate after distillation |
| Checkpoint resume from .soul | P1 | 3h | Load checkpoint, continue training |
| LoRA merge export (.sou) | P2 | 2h | Export merged weights as .sou checkpoint |
| Distributed training | P3 | 8h | Multi-GPU data parallel |

### 6C. Model Management
| Item | Priority | Est. | Description |
|------|----------|------|-------------|
| Model quantization UI | P1 | 3h | INT8/INT4 via bitsandbytes (endpoint exists, needs UI) |
| Auto-download from HF Hub | P1 | 2h | `POST /models/download` endpoint |
| Model comparison benchmark | P2 | 4h | Side-by-side quality metrics |
| SLNC auto-conversion on load | P2 | 2h | Convert to .slnc on first load |

### 6D. Knowledge & RAG
| Item | Priority | Est. | Description |
|------|----------|------|-------------|
| RAG pipeline integration | P1 | 4h | Knowledge search → context injection |
| Document chunking strategies | P2 | 3h | Split by paragraph/heading/semantic |
| Embedding model auto-download | P2 | 2h | Download all-MiniLM on first use |
| Knowledge auto-ingest from chat | P3 | 2h | Extract facts from conversations |

### 6E. CLI & Shell
| Item | Priority | Est. | Description |
|------|----------|------|-------------|
| CLI unit tests | P1 | 4h | Zero test coverage for CLI commands |
| Shell auto-train pipeline | P2 | 2h | `train --auto` in shell |
| CLI completion (bash/zsh/fish) | P2 | 1h | Generated from Click |
| Interactive model selector | P3 | 2h | `model select` with fuzzy search |

### 6F. Infrastructure
| Item | Priority | Est. | Description |
|------|----------|------|-------------|
| ONNX backend (proper integration) | P2 | 6h | Wire existing ONNXEngine into ModelServer |
| Docker deployment | P2 | 3h | Dockerfile + docker-compose |
| Rate limiting | P3 | 2h | Per-user request throttling |
| Prometheus metrics | P3 | 3h | /metrics endpoint |

---

## Phase 7: Mobile (Future)
| Item | Priority | Est. | Description |
|------|----------|------|-------------|
| React Native app shell | P1 | 8h | Navigation, auth, chat UI |
| On-device inference (CoreML/Metal) | P2 | 16h | SloNet → CoreML converter |
| Background sensor collection | P2 | 6h | Activity classifier training data |
| Offline mode | P3 | 4h | Local inference without server |

---

## How to Pick the Next Task

1. **Look at Phase 6** — pick highest priority item not yet done
2. **Check blockers** — some items depend on others (e.g., RAG needs embedding model)
3. **Estimate time** — pick items that fit available time
4. **Write tests** — every task should include test coverage
5. **Update this file** — mark done, add new items as they come up

## Velocity Tracking

| Session | Items Done | Tests Added | Net LOC |
|---------|-----------|-------------|---------|
| 2026-07-11 (distill CLI) | 8 | 0 | +500 |
| 2026-07-12 (fixes + tests) | 7 | 36 | +400 |
