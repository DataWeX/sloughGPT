# API Routers Documentation

All routes are served by the FastAPI application under the base URL (e.g. `http://localhost:8000`). Every public endpoint uses `classify_and_raise(e, source="router.method")` for structured error responses.

## Health Router (`/health`)

Registered pre-lifespan in `main.py` for startup/load balancer probes.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Basic health check with 5 s fallback timeout. |
| `GET` | `/health/live` | Liveness probe. |
| `GET` | `/health/ready` | Readiness probe (model loaded). |
| `GET` | `/health/detailed` | Full health snapshot. |
| `GET` | `/health/startup-progress` | Startup phase progress. |
| `GET` | `/health/debug` | Debug health info. |
| `GET` | `/health/model` | Model-specific health. |
| `GET` | `/health/summary` | Aggregated health summary. |
| `GET` | `/health/stream` | SSE health stream. |

## Status Router (`/`)

Registered pre-lifespan in `main.py`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/status` | Server status. |
| `GET` | `/ready` | Readiness check. |
| `GET` | `/live` | Liveness check. |

## System Router (`/system`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/system/metrics` | Real-time CPU, memory, disk, GPU usage. |
| `GET` | `/system/info` | OS, CPU count, memory total. |
| `GET` | `/system/disk` | Disk usage per mount point. |
| `GET` | `/system/lifecycle` | Full health snapshot (model status, inference counts, metrics). |
| `GET` | `/system/stream` | SSE system metrics stream. |
| `GET` | `/system/output` | Tail recent log output. |
| `GET` | `/system/executor` | Training executor status. |
| `GET` | `/system/executor/{job_id}` | Training executor job detail. |
| `GET` | `/system/executor/{job_id}/result` | Training executor job result. |
| `POST` | `/system/executor/purge` | Purge completed executor jobs. |
| `POST` | `/system/executor/{job_id}/cancel` | Cancel a running executor job. |
| `GET` | `/system/inference-pool` | Inference thread-pool status. |

## Inference Router (`/inference`, `/chat`, `/context`, `/session`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/inference/generate` | Non-streaming text generation. |
| `POST` | `/inference/generate/stream` | Streaming text generation (SSE). |
| `POST` | `/chat` | Chat (non-streaming). |
| `POST` | `/chat/stream` | Chat (SSE streaming). |
| `GET` | `/context/inspect` | Inspect context layers. |
| `POST` | `/context/store-fact` | Store a fact in context. |
| `GET` | `/context/facts` | List context facts. |
| `POST` | `/context/reset` | Reset context state. |
| `GET` | `/session/list` | List all sessions. |
| `POST` | `/session/create` | Create a new session. |
| `GET` | `/session/{session_id}` | Get session details. |
| `DELETE` | `/session/{session_id}` | Delete a session. |
| `POST` | `/session/{session_id}/regenerate` | Regenerate the last response. |
| `GET` | `/session/{session_id}/suggestions` | Chat suggestions for a session. |
| `GET` | `/providers` | List model providers. |
| `GET` | `/operations` | List active operations. |
| `POST` | `/operations/{op_id}/cancel` | Cancel an operation. |
| `POST` | `/operations/cancel-all` | Cancel all operations. |
| `POST` | `/operations/purge` | Purge completed operations. |

## Infer Router (`/infer`)

Separate inference endpoint backed by the direct model server.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/infer` | Generate text from a prompt. |
| `POST` | `/infer/stream` | Streaming generation (SSE). |
| `POST` | `/infer/embed` | Compute text embeddings. |
| `POST` | `/infer/tokenize` | Tokenize text to token IDs. |
| `POST` | `/infer/detokenize` | Convert token IDs to text. |
| `GET` | `/infer/health` | Inference engine health. |
| `GET` | `/infer/info` | Loaded model information. |

## Models Router (`/models`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/models` | List available models. |
| `POST` | `/models/load` | Load a model into memory. |
| `POST` | `/models/unload` | Unload the current model. |
| `GET` | `/models/current` | Get the currently loaded model. |
| `GET` | `/models/huggingface` | List HuggingFace cached models. |
| `GET` | `/models/logs` | Get model server logs. |
| `GET` | `/models/export/formats` | Get available export formats. |
| `POST` | `/models/download` | Start downloading a model. |
| `GET` | `/models/download/status` | Get download status. |
| `GET` | `/models/downloads` | List all active downloads. |
| `POST` | `/models/download/cancel` | Cancel a download. |
| `POST` | `/models/download/retry` | Retry a failed download. |
| `GET` | `/models/cache/usage` | Get HuggingFace cache disk usage. |
| `POST` | `/models/download/gguf` | Download Qwen GGUF for mobile. |
| `POST` | `/models/visual-model/load` | Load a vision model. |
| `POST` | `/models/quantize` | Quantize a model. |
| `POST` | `/models/dequantize` | Dequantize a model. |
| `POST` | `/models/precision` | Set model precision. |
| `GET` | `/models/catalog` | Get model catalog. |
| `GET` | `/models/catalog/stats` | Get catalog statistics. |
| `GET` | `/models/conversion/{task_id}` | Get conversion task status. |
| `GET` | `/models/process-guard` | Get process guard status. |
| `POST` | `/models/process-guard` | Configure process guard. |
| `GET` | `/models/engine/status` | Get engine status. |
| `POST` | `/models/engine/reload` | Reload the model engine. |

## Souls Router (`/souls`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/souls` | List available souls. |
| `GET` | `/souls/current` | Get the currently active soul. |
| `GET` | `/souls/{soul_name}` | Get details for a specific soul. |
| `POST` | `/souls/switch` | Switch to a different soul. |
| `POST` | `/souls/chat` | Chat with a soul. |
| `GET` | `/souls/stats` | Soul system statistics. |
| `GET` | `/souls/weights` | Get current trait weights. |
| `POST` | `/souls/weights` | Save trait weights. |
| `GET` | `/souls/weights/modes` | List available weight modes. |
| `GET` | `/souls/weights/snapshots` | List weight snapshots. |
| `POST` | `/souls/weights/snapshot/{name}` | Create a weight snapshot. |
| `POST` | `/souls/weights/snapshot/{name}/load` | Load a weight snapshot. |
| `DELETE` | `/souls/weights/snapshot/{name}` | Delete a weight snapshot. |

## Config Router (`/config`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/config/generation` | Get generation config. |
| `PUT` | `/config/generation` | Replace generation config. |
| `PATCH` | `/config/generation` | Partially update generation config. |

## Auth Router (`/auth`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/auth/whoami` | Get current user info. |

## Session Router (`/session`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/session/{session_id}/context` | Build context for a session. |
| `GET` | `/session/{session_id}/messages` | List session messages. |
| `GET` | `/session/{session_id}/inspector` | Inspect session state. |
| `POST` | `/session/{session_id}/regenerate` | Regenerate last response. |

## Feedback Router (`/feedback`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/feedback` | Record user feedback. |
| `POST` | `/feedback/workflow-record` | Record workflow feedback. |
| `GET` | `/feedback/stats/summary` | Feedback statistics summary. |
| `POST` | `/feedback/conversations` | Create a conversation record. |
| `GET` | `/feedback/conversations` | List conversations. |
| `GET` | `/feedback/conversations/{conv_id}` | Get a conversation. |
| `PATCH` | `/feedback/conversations/{conv_id}` | Update a conversation. |
| `DELETE` | `/feedback/conversations/{conv_id}` | Delete a conversation. |
| `GET` | `/feedback/{message_id}` | Get feedback for a message. |

## Knowledge Base Router (`/knowledge`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/knowledge` | List knowledge items. |
| `POST` | `/knowledge` | Create a knowledge item. |
| `PATCH` | `/knowledge/{item_id}` | Update a knowledge item. |
| `DELETE` | `/knowledge/{item_id}` | Delete a knowledge item. |
| `POST` | `/knowledge/batch` | Batch create items. |
| `GET` | `/knowledge/search` | Search knowledge items. |
| `GET` | `/knowledge/stats` | Knowledge base statistics. |
| `GET` | `/knowledge/topics` | List topics. |
| `POST` | `/knowledge/ingest-url` | Ingest from a URL. |
| `POST` | `/knowledge/ingest-file` | Ingest from a file upload. |
| `POST` | `/knowledge/bulk-ingest` | Bulk ingest items. |
| `POST` | `/knowledge/batch-delete` | Batch delete items. |
| `POST` | `/knowledge/suggest-topic` | Suggest a topic. |
| `POST` | `/knowledge/check-duplicate` | Check for duplicates. |
| `POST` | `/knowledge/categorize` | Auto-categorize items. |
| `GET` | `/knowledge/gaps` | Identify knowledge gaps. |
| `POST` | `/knowledge/search-files` | Search across ingested files. |
| `GET` | `/knowledge/context` | Get knowledge context. |
| `GET` | `/knowledge/{item_id}/related` | Get related items. |
| `POST` | `/knowledge/train-adapter` | Train a LoRA adapter on knowledge. |
| `GET` | `/knowledge/adapter-status` | Adapter training status. |
| `POST` | `/knowledge/train-embedder` | Train an embedder model. |
| `GET` | `/knowledge/embedder-status` | Embedder training status. |
| `GET` | `/knowledge/reviews/due` | Get items due for review. |
| `POST` | `/knowledge/reviews/{item_id}/schedule` | Schedule a review. |
| `GET` | `/knowledge/label` | Get label info. |
| `POST` | `/knowledge/rag/ingest` | Ingest documents for RAG. |
| `POST` | `/knowledge/rag/query` | Query the RAG pipeline. |
| `POST` | `/knowledge/rag/verify` | Verify RAG results. |
| `GET` | `/knowledge/rag/documents` | List RAG documents. |
| `POST` | `/knowledge/rag/clear` | Clear RAG store. |
| `GET` | `/knowledge/rag/stats` | RAG pipeline statistics. |
| `POST` | `/knowledge/kg/sync` | Sync knowledge graph. |
| `GET` | `/knowledge/kg/pipeline-stats` | Knowledge graph pipeline stats. |

## Memory Router (`/memory`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/memory/stats` | Memory service stats (enabled, total facts, topic buckets). |
| `GET` | `/memory/list` | List stored facts (limit 1..1000, default 100). |
| `GET` | `/memory/search` | Search facts by relevance (requires `q`). |
| `POST` | `/memory/store` | Store a fact directly. Body: `{content, source?}`. |
| `POST` | `/memory/remember` | Extract and store facts from a message. Body: `{message, source?}`. |
| `POST` | `/memory/update` | Update a fact. |
| `DELETE` | `/memory/delete` | Delete a fact. |
| `POST` | `/memory/set-config` | Update runtime memory settings. |
| `GET` | `/memory/get-config` | Get current memory settings. |
| `POST` | `/memory/clear` | Clear all stored facts. |
| `POST` | `/memory/consolidate` | Trigger memory consolidation. |
| `POST` | `/memory/archive` | Archive old facts. |
| `GET` | `/memory/archive/stats` | Archive statistics. |
| `POST` | `/memory/archive/prune` | Prune archived facts. |

Fail-closed when `SLO_MEMORY_ENABLED=false`.

## Datasets Router (`/datasets`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/datasets` | List all registered datasets. |
| `POST` | `/datasets` | Create a new dataset. |
| `GET` | `/datasets/search` | Search datasets. |
| `GET` | `/datasets/search/books` | Search for books. |
| `GET` | `/datasets/search/github` | Search GitHub for datasets. |
| `GET` | `/datasets/{dataset_id}` | Get a dataset. |
| `GET` | `/datasets/{dataset_id}/stats` | Dataset statistics. |
| `PATCH` | `/datasets/{dataset_id}` | Update dataset metadata. |
| `DELETE` | `/datasets/{dataset_id}` | Delete a dataset. |
| `POST` | `/datasets/{dataset_id}/data` | Append rows. |
| `GET` | `/datasets/{dataset_id}/preview` | Preview first N rows. |
| `POST` | `/datasets/{dataset_id}/export` | Export to file. |
| `POST` | `/datasets/{dataset_id}/versions` | Create a version snapshot. |
| `GET` | `/datasets/{dataset_id}/versions` | List versions. |
| `POST` | `/datasets/{dataset_id}/versions/{timestamp}` | Restore a version. |
| `POST` | `/datasets/import/local` | Import from local path. |
| `POST` | `/datasets/import/github` | Import from GitHub. |
| `POST` | `/datasets/import/huggingface` | Import from HuggingFace. |
| `POST` | `/datasets/import/url` | Import from URL. |
| `POST` | `/datasets/import/kaggle` | Import from Kaggle. |
| `POST` | `/datasets/import/csv` | Import from CSV. |
| `POST` | `/datasets/import/batch` | Batch import. |
| `POST` | `/datasets/import/isbn` | Import by ISBN. |
| `POST` | `/datasets/from-chat` | Create dataset from chat messages. |
| `POST` | `/datasets/convert-to-messages` | Convert dataset to message format. |

## Training Router (`/training`)

Unified training control plane. All training operations go through `/training/*`.

Routes are split across focused sub-modules:

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `training/router.py` | ~640 | Recovery, finetuned models, checkpoints, stream, stop |
| `training/execution.py` | ~330 | Core start_training + includes (lora, distill, visual, feedback, builds) |
| `training/lora.py` | ~336 | lora-finetune, load-adapter, unload-adapter |
| `training/distill.py` | ~261 | Knowledge distillation |
| `training/from_feedback.py` | ~189 | Feedback-based training |
| `training/visual.py` | ~164 | VLM fine-tune |
| `training/builds.py` | ~86 | Build listing |
| `training/legacy.py` | ~142 | /train, /train/resolve (backward compat) |
| `training/jobs_api.py` | ~295 | Job CRUD, export, purge |
| `training/control.py` | ~156 | Status, start/pause/resume/stop/reset |
| `training/helpers.py` | ~111 | Shared `_finish_job`, `_sloughgpt_trainer_kwds`, `_run_async` |
| `training/turbo_endpoints.py` | ~122 | Turbo start + from-sessions start + turbo status |
| `training/sse_stream.py` | ~282 | Shared SSE stream helper, stop_all_training, cancel_from_sessions |

### Execution Routes (`execution.py`, `legacy.py`)

| Method | Path | Module | Description |
|--------|------|--------|-------------|
| `POST` | `/train` | legacy.py | Start a training job (legacy). |
| `POST` | `/train/resolve` | legacy.py | Resolve data path (dry run). |
| `POST` | `/training/start` | execution.py | Start a tracked training job (web UI). |
| `POST` | `/training/visual-start` | execution.py | Start a VLM fine-tune. |
| `POST` | `/training/distill` | execution.py | Knowledge distillation (teacher→student). |
| `POST` | `/training/lora-finetune` | execution.py | LoRA fine-tuning on .slnc models. |
| `POST` | `/training/load-adapter` | execution.py | Load a LoRA adapter for inference. |
| `POST` | `/training/unload-adapter` | Unload the active LoRA adapter. |
| `POST` | `/training/from-feedback` | Train from collected feedback data. |
| `GET` | `/training/builds` | List all training builds. |

### Control Routes (`control.py`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/training/status` | Training status. |
| `POST` | `/training/control/start` | Start training (control plane). |
| `POST` | `/training/control/pause` | Pause the active job. |
| `POST` | `/training/control/resume` | Resume the paused job. |
| `POST` | `/training/control/stop` | Stop the active job. |
| `POST` | `/training/control/reset` | Reset training controller. |
| `GET` | `/training/is-running` | Check if training is running. |

### Job Routes (`jobs_api.py`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/training/jobs` | List all training jobs. |
| `GET` | `/training/jobs/{job_id}` | Get a specific job. |
| `POST` | `/training/jobs/{job_id}/stop` | Stop a specific job. |
| `GET` | `/training/jobs/{job_id}/summary` | Get job summary. |
| `DELETE` | `/training/jobs/{job_id}` | Delete a job. |
| `POST` | `/training/jobs/purge` | Purge old jobs. |
| `GET` | `/training/export/{job_id}` | Export job data (JSON). |
| `POST` | `/training/export-text` | Export job data (text). |

### Stream & Utility Routes (`router.py`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/training/stop` | Stop the active job. |
| `POST` | `/training/turbo-start` | Start turbo training. |
| `GET` | `/training/turbo/status` | Turbo training status. |
| `GET` | `/training/log` | Training log. |
| `GET` | `/training/stream` | SSE training stream. |
| `GET` | `/training/from-sessions/cancel` | Cancel session-based training. |
| `GET` | `/training/from-sessions-stream` | SSE stream from sessions. |
| `GET` | `/training/checkpoints` | List checkpoints. |
| `DELETE` | `/training/checkpoints/{name}` | Delete a checkpoint. |
| `POST` | `/training/checkpoints/{name}/load` | Load a checkpoint. |
| `GET` | `/training/checkpoints/{name}/download` | Download a checkpoint. |
| `GET` | `/training/checkpoints/{name}/info` | Checkpoint info. |
| `GET` | `/training/metrics/export` | Export training metrics. |

### Recovery Routes (`router.py`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/recovery/check` | Check for crashed jobs. |
| `GET` | `/recovery/recoverable` | Get recoverable jobs. |
| `POST` | `/recovery/recover/{job_id}` | Recover an interrupted job. |
| `DELETE` | `/recovery/abandon/{job_id}` | Abandon a crashed job. |
| `GET` | `/recovery/stats` | Recovery statistics. |

### Finetuned Model Routes (`router.py`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/training/finetuned-models` | List HF fine-tuned models. |
| `POST` | `/training/finetuned-models/{name}/load` | Load a fine-tuned model. |
| `DELETE` | `/training/finetuned-models/{name}` | Delete a fine-tuned model. |

> **Note:** The legacy `/auto-train/*` endpoints (in `routers/auto_train.py`) are deprecated. They are a parallel implementation, not shims — new clients should use `/training/*` instead. The `/training/stop`, `/training/turbo-start`, and `/training/stream` routes now use `training/sse_stream.py` and `training/turbo_endpoints.py` which delegate to `domains.training.service` (core layer).

## Self-Train Router (`/self-train`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/self-train/start` | Start self-training. |
| `POST` | `/self-train/stop` | Stop self-training. |
| `GET` | `/self-train/status` | Self-training status. |

## Learner Router (`/learn`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/learn/search` | Search learning content. |
| `GET` | `/learn/feed` | Get learning feed. |
| `POST` | `/learn/feed` | Add to learning feed. |
| `POST` | `/learn/ingest-url` | Ingest from URL. |
| `GET` | `/learn/knowledge` | List learned knowledge. |
| `POST` | `/learn/ingest` | Ingest training data. |
| `POST` | `/learn/train` | Train the learner. |
| `POST` | `/learn/deploy` | Deploy trained model. |
| `POST` | `/learn/evaluate` | Evaluate trained model. |
| `GET` | `/learn/status` | Learner status. |

## Tokenizer Router (`/tokenizer`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/tokenizer/stats` | Vocabulary size, merge count, token-to-id map stats. |
| `POST` | `/tokenizer/pretokenize` | Pre-tokenize text. |
| `POST` | `/tokenizer/decompose` | Decompose text into tokens. |
| `POST` | `/tokenizer/analyze` | Analyze token distribution. |
| `POST` | `/tokenizer/tokenize` | Tokenize string to token IDs. |
| `POST` | `/tokenizer/detokenize` | Convert token IDs to string. |
| `GET` | `/tokenizer/vocab` | Full vocabulary list. |
| `GET` | `/tokenizer/merges` | BPE merge operations. |
| `POST` | `/tokenizer/train` | Train a fresh BPE tokenizer. |
| `GET` | `/tokenizer/sample` | Get sample tokens. |
| `GET` | `/tokenizer/samples` | Get sample tokens/words for UI. |

## Token Tree Router (`/token-tree`)

Tree-based BPE tokenizer — training, encoding, embedding/semantic explorer.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/token-tree/stats` | Tree summary (vocab, merges, embeddings, compression ratio). |
| `GET` | `/token-tree/vocab` | Paged vocabulary (limit 1..500, default 50). |
| `GET` | `/token-tree/merges` | Ranked merge rules (optional `query` filter). |
| `GET` | `/token-tree/saved` | List saved trees. |
| `POST` | `/token-tree/save` | Save current tree. Body: `{name}`. |
| `POST` | `/token-tree/load` | Load a saved tree. Body: `{name}`. |
| `DELETE` | `/token-tree/saved/{name}` | Delete a saved tree. |
| `POST` | `/token-tree/train` | Train on `{texts?, vocab_size, embed_dim, min_frequency}`. |
| `POST` | `/token-tree/similar` | Nearest-neighbor tokens. Body: `{token, top_k}`. |
| `POST` | `/token-tree/embedding` | Inspect token embedding. Body: `{token, top_k}`. |
| `POST` | `/token-tree/encode` | Encode text. Body: `{text}` → `{tokens, ids}`. |
| `POST` | `/token-tree/path` | Trace greedy trie walk. Body: `{text}` → `{steps, ids}`. |
| `POST` | `/token-tree/decode` | Decode token IDs. Body: `{ids}` → `{text}`. |
| `POST` | `/token-tree/lineage` | Render merge lineage. Body: `{token}`. |
| `GET` | `/token-tree/matrix` | Embedding matrix overview (top_k). |
| `POST` | `/token-tree/compare` | Diff two saved trees. Body: `{a, b, top_k}`. |

Semantic endpoints return `404` for unknown tokens; `embedding` returns `422` when tree has no embeddings.

## Agent Router (`/agents`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/agents` | List all agents. |
| `POST` | `/agents` | Create a new agent. |
| `GET` | `/agents/{agent_id}` | Get agent details. |
| `PUT` | `/agents/{agent_id}` | Update an agent. |
| `DELETE` | `/agents/{agent_id}` | Delete an agent. |
| `POST` | `/agents/{agent_id}/execute` | Execute an agent. |
| `GET` | `/agents/runs` | List agent runs. |
| `GET` | `/agents/runs/{run_id}` | Get a specific run. |
| `POST` | `/agents/orchestrate` | Orchestrate multi-agent task. |

## Multimodal Router (`/multimodal`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/multimodal/status` | Multimodal engine status. |
| `POST` | `/multimodal/train` | Train on a single image. |
| `POST` | `/multimodal/train-batch` | Train on a batch of images. |
| `POST` | `/multimodal/train-video` | Train video captioning. |
| `POST` | `/multimodal/video-infer` | Infer video captions. |
| `POST` | `/multimodal/dpo` | Run DPO training. |
| `POST` | `/multimodal/analyze` | Analyze an image. |
| `POST` | `/multimodal/pdf/upload` | Upload and analyze a PDF. |
| `POST` | `/multimodal/process-video` | Process a video file. |
| `POST` | `/multimodal/transcribe` | Transcribe audio. |
| `POST` | `/multimodal/synthesize-speech` | Text-to-speech synthesis. |
| `POST` | `/multimodal/generate-image` | Generate an image. |
| `POST` | `/multimodal/visual-dataset` | Create visual dataset. |
| `GET` | `/multimodal/checkpoints` | List checkpoints. |
| `POST` | `/multimodal/checkpoints/{name}/load` | Load a checkpoint. |
| `DELETE` | `/multimodal/checkpoints/{name}` | Delete a checkpoint. |
| `POST` | `/multimodal/reset` | Reset the multimodal engine. |

## Benchmark Router (`/benchmark`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/benchmark/run` | Run a benchmark. |
| `GET` | `/benchmark/metrics` | Get model metrics. |
| `GET` | `/benchmark/{model_id}` | Benchmark results for a model. |
| `POST` | `/benchmark/perplexity` | Calculate perplexity. |
| `GET` | `/benchmark/quality` | Quality metrics. |
| `GET` | `/benchmark/responses` | Logged responses. |
| `GET` | `/benchmark/stats` | Tracker statistics. |
| `POST` | `/benchmark/history/clear` | Clear benchmark history. |

## Companion Router (`/companion`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/companion/` | Get current companion. |
| `DELETE` | `/companion/` | Delete companion. |
| `POST` | `/companion/personality` | Set companion personality. |
| `PATCH` | `/companion/personality` | Update companion personality. |
| `POST` | `/companion/preset` | Apply a preset. |
| `GET` | `/companion/prompt` | Get companion system prompt. |
| `POST` | `/companion/chat` | Chat with companion. |
| `GET` | `/companion/presets` | List available presets. |

## Collections Router (`/collections`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/collections` | List all pipelines. |
| `POST` | `/collections/create` | Create a pipeline. |
| `POST` | `/collections/run` | Run a pipeline. |
| `POST` | `/collections/collect` | Direct collect (no pipeline). |
| `GET` | `/collections/stats` | Pipeline statistics. |
| `GET` | `/collections/{pipeline_id}` | Get a pipeline. |
| `DELETE` | `/collections/{pipeline_id}` | Delete a pipeline. |
| `POST` | `/collections/{pipeline_id}/collect` | Collect for a pipeline. |
| `GET` | `/collections/{pipeline_id}/records` | List pipeline records. |

## Docstore Router (`/docstore`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/docstore/{collection}` | List documents in a collection. |
| `GET` | `/docstore/{collection}/{doc_id}` | Get a document. |
| `PUT` | `/docstore/{collection}/{doc_id}` | Create/replace a document. |
| `PATCH` | `/docstore/{collection}/{doc_id}` | Partially update a document. |
| `DELETE` | `/docstore/{collection}/{doc_id}` | Delete a document. |
| `DELETE` | `/docstore/{collection}` | Clear a collection. |
| `POST` | `/docstore/{collection}/bulk` | Bulk upsert documents. |

## Errors Router (`/errors`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/errors/log` | Log an error. |
| `POST` | `/errors/logs/ingest` | Ingest error logs. |
| `GET` | `/errors/recent` | Get recent errors. |
| `GET` | `/errors/grouped` | Get grouped errors. |
| `GET` | `/errors/trends` | Error trend analysis. |
| `GET` | `/errors/export` | Export error logs. |
| `DELETE` | `/errors/clear` | Clear error logs. |
| `GET` | `/errors/unread` | Get unread error count. |
| `GET` | `/errors/log` | Get OpenCode error log. |

## Experiments Router (`/experiments`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/experiments` | Create an experiment. |
| `GET` | `/experiments` | List all experiments. |
| `GET` | `/experiments/{experiment_id}` | Get an experiment. |
| `DELETE` | `/experiments/{experiment_id}` | Delete an experiment. |
| `GET` | `/experiments/{experiment_id}/runs` | List experiment runs. |
| `GET` | `/experiments/{experiment_id}/data` | Get experiment data. |
| `POST` | `/experiments/{experiment_id}/complete` | Mark experiment complete. |
| `POST` | `/experiments/{experiment_id}/log_metric` | Log a metric. |
| `POST` | `/experiments/{experiment_id}/log_param` | Log a parameter. |

## Vector Router (`/vector`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/vector/init` | Initialize vector store. |
| `GET` | `/vector/stats` | Vector store statistics. |
| `POST` | `/vector/upsert` | Upsert vectors. |
| `POST` | `/vector/search` | Search vectors. |
| `GET` | `/vector/ingest/status` | Ingestion status. |

## User Adapters Router (`/user-adapters`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/user-adapters` | List all user adapters. |
| `GET` | `/user-adapters/quality` | Adapter quality metrics. |
| `GET` | `/user-adapters/{user_id}` | Get a user adapter. |
| `POST` | `/user-adapters/{user_id}/update` | Update a user adapter. |
| `POST` | `/user-adapters/{user_id}/reset` | Reset a user adapter. |
| `DELETE` | `/user-adapters/{user_id}` | Delete a user adapter. |
| `POST` | `/user-adapters/merge` | Merge adapters. |
| `POST` | `/user-adapters/aggregate-best` | Aggregate best adapters. |
| `POST` | `/user-adapters/prune` | Prune adapters. |

## LoRA Eval Router (`/lora-eval`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/lora-eval/run` | Run LoRA evaluation. |
| `GET` | `/lora-eval/history` | Evaluation history. |
| `POST` | `/lora-eval/aggregate` | Aggregate evaluation results. |

## Registry Router (`/registry`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/registry` | List registered models. |
| `GET` | `/registry/{model_id}` | Get a registered model. |
| `GET` | `/registry/best` | Get best model for a task. |
| `GET` | `/registry/stats` | Registry statistics. |

## Meta-Weights Router (`/meta-weights`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/meta-weights/ping` | Health probe. |
| `POST` | `/meta-weights/get` | Get meta-weights. |
| `GET` | `/meta-weights/stats` | Meta-weights statistics. |

## VM Router (`/vm`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/vm/run` | Run x86 assembly in sandboxed VM. |
| `GET` | `/vm/training/jobs/{job_id}` | Training job status. |
| `POST` | `/vm/training/jobs/{job_id}/stop` | Stop a VM training job. |
| `GET` | `/vm/builtins` | List built-in assembly programs. |
| `GET` | `/vm/info` | VM capabilities and limits. |

## Workflow Router (`/workflow`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/workflow/status` | Workflow status. |
| `POST` | `/workflow/start` | Start a workflow. |
| `POST` | `/workflow/stop` | Stop a workflow. |
| `POST` | `/workflow/trigger/{action}` | Trigger a workflow action. |

## Shell Router (`/shell`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/shell/exec` | Execute a shell command. |
| `POST` | `/shell/exec/stream` | Execute with SSE streaming. |

## Images Router (`/images`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/images/gallery` | List generated images. |
| `GET` | `/images/styles` | List available styles. |

## Voice Router (`/voice`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/voice/tts` | Convert text to speech. |
| `GET` | `/voice/status` | Voice synthesis status. |

## Files Router (`/files`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/files` | List files. |
| `GET` | `/files/{file_id}` | Get a file. |
| `POST` | `/files` | Upload a file. |

## Security Router (`/security`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/security/audit-logs` | Get audit logs. |
| `GET` | `/security/keys` | List API keys. |

## Rate Limit Router (`/ratelimit`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ratelimit/status` | Rate limit status. |
| `GET` | `/ratelimit/check` | Check rate limit. |
| `GET` | `/ratelimit/policy` | Get rate limit policy. |
| `POST` | `/ratelimit/config` | Update rate limit config. |

## World Render Router (`/world`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/world/render` | Render the world state. |
| `POST` | `/world/render/image` | Render as PNG image. |
| `POST` | `/world/neural` | Process through neural pipeline. |
| `POST` | `/world/tick` | Run a simulation tick. |
| `GET` | `/world/stats` | World rendering statistics. |

## OpenAPI Specification

FastAPI automatically provides the OpenAPI spec at `/openapi.json`:
```
curl http://localhost:8000/openapi.json
```
The spec includes all routes described above, request/response schemas, and example payloads.
