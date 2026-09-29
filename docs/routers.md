# API Routers Documentation

All routes are served by the FastAPI application under the base URL (e.g. `http://localhost:8000`). Every public endpoint uses `classify_and_raise(e, source="router.method")` for structured error responses.

## Health Router (`/health`)

Registered pre-lifespan in `main.py` for startup/load balancer probes.

| Method | Path                       | Description                                   |
| ------ | -------------------------- | --------------------------------------------- |
| `GET`  | `/health`                  | Basic health check with 5 s fallback timeout. |
| `GET`  | `/health/live`             | Liveness probe.                               |
| `GET`  | `/health/ready`            | Readiness probe (model loaded).               |
| `GET`  | `/health/detailed`         | Full health snapshot.                         |
| `GET`  | `/health/startup-progress` | Startup phase progress.                       |
| `GET`  | `/health/debug`            | Debug health info.                            |
| `GET`  | `/health/model`            | Model-specific health.                        |
| `GET`  | `/health/summary`          | Aggregated health summary.                    |
| `GET`  | `/health/stream`           | SSE health stream.                            |
| `GET` | `/health/services` | Services Health |
| `GET` | `/health/startup-compare` | Startup Compare |
| `GET` | `/health/startup-config` | Startup Config |
| `GET` | `/health/startup-diagnostics` | Startup Diagnostics |
| `GET` | `/health/startup-health` | Startup Health |
| `GET` | `/health/startup-history` | Startup History |
| `GET` | `/health/startup-rollback` | Startup Rollback |
| `GET` | `/health/startup-status` | Startup Status |
| `GET` | `/health/startup-stream` | Startup Stream |

## Status Router (`/`)

Registered pre-lifespan in `main.py`.

| Method | Path      | Description      |
| ------ | --------- | ---------------- |
| `GET`  | `/status` | Server status.   |
| `GET`  | `/ready`  | Readiness check. |
| `GET`  | `/live`   | Liveness check.  |
| `GET` | `/` | Root |

## System Router (`/system`)

| Method | Path                               | Description                                                     |
| ------ | ---------------------------------- | --------------------------------------------------------------- |
| `GET`  | `/system/metrics`                  | Real-time CPU, memory, disk, GPU usage.                         |
| `GET`  | `/system/info`                     | OS, CPU count, memory total.                                    |
| `GET`  | `/system/disk`                     | Disk usage per mount point.                                     |
| `GET`  | `/system/lifecycle`                | Full health snapshot (model status, inference counts, metrics). |
| `GET`  | `/system/stream`                   | SSE system metrics stream.                                      |
| `GET`  | `/system/output`                   | Tail recent log output.                                         |
| `GET`  | `/system/executor`                 | Training executor status.                                       |
| `GET`  | `/system/executor/{job_id}`        | Training executor job detail.                                   |
| `GET`  | `/system/executor/{job_id}/result` | Training executor job result.                                   |
| `POST` | `/system/executor/purge`           | Purge completed executor jobs.                                  |
| `POST` | `/system/executor/{job_id}/cancel` | Cancel a running executor job.                                  |
| `GET`  | `/system/inference-pool`           | Inference thread-pool status.                                   |
| `GET` | `/system/battery` | Get Battery |
| `POST` | `/system/battery/limit` | Set Battery Limit |
| `PUT` | `/system/battery/policy` | Set Battery Policy |

## Inference Router (`/inference`, `/chat`, `/context`, `/session`)

| Method   | Path                                | Description                      |
| -------- | ----------------------------------- | -------------------------------- |
| `POST`   | `/inference/generate`               | Non-streaming text generation.   |
| `POST`   | `/inference/generate/stream`        | Streaming text generation (SSE). |
| `POST`   | `/chat`                             | Chat (non-streaming).            |
| `POST`   | `/chat/stream`                      | Chat (SSE streaming).            |
| `GET`    | `/context/inspect`                  | Inspect context layers.          |
| `GET`    | `/context/facts`                    | List context facts.              |
| `POST`   | `/context/reset`                    | Reset context state.             |
| `GET`    | `/providers`                        | List model providers.            |
| `GET`    | `/operations`                       | List active operations.          |
| `POST`   | `/operations/purge`                 | Purge completed operations.      |
| `DELETE` | `/chat/sessions/{session_id}` | Delete Session |
| `GET` | `/chat/active` | Active Sessions |
| `GET` | `/chat/addons` | Addons |
| `GET` | `/chat/audio/{session_id}/{message_id}` | Get Voice Audio |
| `GET` | `/chat/health` | Health |
| `GET` | `/chat/sessions` | List Sessions |
| `GET` | `/chat/sessions/current` | Get Current Session |
| `GET` | `/chat/sessions/search` | Search Sessions |
| `GET` | `/chat/sessions/{session_id}` | Get Session |
| `GET` | `/chat/suggestions` | Chat Suggestions |
| `GET` | `/chat/tools` | List Chat Tools |
| `POST` | `/chat/control` | Chat Control |
| `POST` | `/chat/sessions` | Create Session |
| `POST` | `/chat/voice/{session_id}` | Send Voice Message |
| `POST` | `/chat/{session_id}/cancel` | Cancel |
| `POST` | `/chat/{session_id}/regenerate` | Regenerate |
| `POST` | `/context/fact` | Store Fact |
| `PUT` | `/chat/sessions/{session_id}` | Upsert Session |

## Infer Router (`/infer`)

Separate inference endpoint backed by the direct model server.

| Method | Path                | Description                  |
| ------ | ------------------- | ---------------------------- |
| `POST` | `/infer`            | Generate text from a prompt. |
| `POST` | `/infer/stream`     | Streaming generation (SSE).  |
| `POST` | `/infer/embed`      | Compute text embeddings.     |
| `POST` | `/infer/tokenize`   | Tokenize text to token IDs.  |
| `POST` | `/infer/detokenize` | Convert token IDs to text.   |
| `GET`  | `/infer/health`     | Inference engine health.     |
| `GET`  | `/infer/info`       | Loaded model information.    |

## Models Router (`/models`)

| Method | Path                           | Description                       |
| ------ | ------------------------------ | --------------------------------- |
| `GET`  | `/models`                      | List available models.            |
| `POST` | `/models/load`                 | Load a model into memory.         |
| `POST` | `/models/unload`               | Unload the current model.         |
| `GET`  | `/models/current`              | Get the currently loaded model.   |
| `GET`  | `/models/logs`                 | Get model server logs.            |
| `GET`  | `/models/export/formats`       | Get available export formats.     |
| `POST` | `/models/download`             | Start downloading a model.        |
| `GET`  | `/models/downloads`            | List all active downloads.        |
| `POST` | `/models/quantize`             | Quantize a model.                 |
| `POST` | `/models/dequantize`           | Dequantize a model.               |
| `POST` | `/models/precision`            | Set model precision.              |
| `GET`  | `/models/catalog`              | Get model catalog.                |
| `GET`  | `/models/catalog/stats`        | Get catalog statistics.           |
| `GET`  | `/models/process-guard`        | Get process guard status.         |
| `POST` | `/models/process-guard`        | Configure process guard.          |
| `GET`  | `/models/engine/status`        | Get engine status.                |
| `POST` | `/models/engine/reload`        | Reload the model engine.          |
| `DELETE` | `/models/external/servers/{name}` | Remove External Server |
| `GET` | `/models/backends` | List Backends |
| `GET` | `/models/backends/active` | Get Active Backend |
| `GET` | `/models/cache-usage` | Cache Usage |
| `GET` | `/models/conversion-status` | Get Conversion Status |
| `GET` | `/models/debug/providers` | Debug Providers |
| `GET` | `/models/download/qwen-gguf` | Download Qwen Gguf |
| `GET` | `/models/download/{model_id}` | Get Download Status |
| `GET` | `/models/external/models` | List External Models |
| `GET` | `/models/external/servers` | List External Servers |
| `GET` | `/models/file/{model_id}/{file_path}` | Serve Model File |
| `GET` | `/models/hf` | List Hf Models |
| `GET` | `/models/history` | Download History |
| `GET` | `/models/memory-pressure` | Memory Pressure |
| `POST` | `/models/backends/active` | Set Active Backend |
| `POST` | `/models/download/{model_id}/cancel` | Cancel Download |
| `POST` | `/models/download/{model_id}/pause` | Pause Download |
| `POST` | `/models/download/{model_id}/resume` | Resume Download |
| `POST` | `/models/download/{model_id}/retry` | Retry Download |
| `POST` | `/models/download/{model_id}/verify` | Verify Download |
| `POST` | `/models/export` | Export Model |
| `POST` | `/models/external/download` | Download External Model |
| `POST` | `/models/external/servers` | Register External Server |
| `POST` | `/models/memory-cleanup` | Memory Cleanup |
| `POST` | `/models/visual-load` | Visual Model Load |

## Souls Router (`/souls`)

| Method   | Path                                  | Description                      |
| -------- | ------------------------------------- | -------------------------------- |
| `GET`    | `/souls`                              | List available souls.            |
| `GET`    | `/souls/current`                      | Get the currently active soul.   |
| `GET`    | `/souls/{soul_name}`                  | Get details for a specific soul. |
| `POST`   | `/souls/switch`                       | Switch to a different soul.      |
| `POST`   | `/souls/chat`                         | Chat with a soul.                |
| `GET`    | `/souls/stats`                        | Soul system statistics.          |
| `GET`    | `/souls/weights`                      | Get current trait weights.       |
| `POST`   | `/souls/weights`                      | Save trait weights.              |
| `GET`    | `/souls/weights/modes`                | List available weight modes.     |
| `GET`    | `/souls/weights/snapshots`            | List weight snapshots.           |
| `POST`   | `/souls/weights/snapshot/{name}`      | Create a weight snapshot.        |
| `POST`   | `/souls/weights/snapshot/{name}/load` | Load a weight snapshot.          |
| `DELETE` | `/souls/weights/snapshot/{name}`      | Delete a weight snapshot.        |

## Config Router (`/config`)

| Method  | Path                 | Description                         |
| ------- | -------------------- | ----------------------------------- |
| `GET`   | `/config/generation` | Get generation config.              |
| `PUT`   | `/config/generation` | Replace generation config.          |
| `PATCH` | `/config/generation` | Partially update generation config. |

## Auth Router (`/auth`)

| Method | Path           | Description            |
| ------ | -------------- | ---------------------- |
| `GET` | `/auth/me` | Get Me |
| `POST` | `/auth/login` | Login |
| `POST` | `/auth/refresh` | Refresh Token |
| `POST` | `/auth/register` | Register |
| `POST` | `/auth/token` | Create Token |
| `POST` | `/auth/verify` | Verify Token |

## Session Router (`/session`)

| Method | Path                               | Description                  |
| ------ | ---------------------------------- | ---------------------------- |
| `POST` | `/session/{session_id}/context`    | Build context for a session. |
| `GET`  | `/session/{session_id}/messages`   | List session messages.       |
| `GET`  | `/session/{session_id}/inspector`  | Inspect session state.       |

## Feedback Router (`/feedback`)

| Method   | Path                                | Description                   |
| -------- | ----------------------------------- | ----------------------------- |
| `POST`   | `/feedback`                         | Record user feedback.         |
| `POST`   | `/feedback/workflow-record`         | Record workflow feedback.     |
| `GET`    | `/feedback/stats/summary`           | Feedback statistics summary.  |
| `POST`   | `/feedback/conversations`           | Create a conversation record. |
| `GET`    | `/feedback/conversations`           | List conversations.           |
| `GET`    | `/feedback/conversations/{conv_id}` | Get a conversation.           |
| `PATCH`  | `/feedback/conversations/{conv_id}` | Update a conversation.        |
| `DELETE` | `/feedback/conversations/{conv_id}` | Delete a conversation.        |
| `GET`    | `/feedback/{message_id}`            | Get feedback for a message.   |

## Knowledge Base Router (`/knowledge`)

| Method   | Path                                    | Description                        |
| -------- | --------------------------------------- | ---------------------------------- |
| `GET`    | `/knowledge`                            | List knowledge items.              |
| `POST`   | `/knowledge`                            | Create a knowledge item.           |
| `PATCH`  | `/knowledge/{item_id}`                  | Update a knowledge item.           |
| `DELETE` | `/knowledge/{item_id}`                  | Delete a knowledge item.           |
| `POST`   | `/knowledge/batch`                      | Batch create items.                |
| `GET`    | `/knowledge/search`                     | Search knowledge items.            |
| `GET`    | `/knowledge/stats`                      | Knowledge base statistics.         |
| `GET`    | `/knowledge/topics`                     | List topics.                       |
| `POST`   | `/knowledge/ingest-url`                 | Ingest from a URL.                 |
| `POST`   | `/knowledge/ingest-file`                | Ingest from a file upload.         |
| `POST`   | `/knowledge/bulk-ingest`                | Bulk ingest items.                 |
| `POST`   | `/knowledge/batch-delete`               | Batch delete items.                |
| `POST`   | `/knowledge/suggest-topic`              | Suggest a topic.                   |
| `POST`   | `/knowledge/check-duplicate`            | Check for duplicates.              |
| `POST`   | `/knowledge/categorize`                 | Auto-categorize items.             |
| `GET`    | `/knowledge/gaps`                       | Identify knowledge gaps.           |
| `POST`   | `/knowledge/search-files`               | Search across ingested files.      |
| `GET`    | `/knowledge/context`                    | Get knowledge context.             |
| `GET`    | `/knowledge/{item_id}/related`          | Get related items.                 |
| `POST`   | `/knowledge/train-adapter`              | Train a LoRA adapter on knowledge. |
| `GET`    | `/knowledge/adapter-status`             | Adapter training status.           |
| `POST`   | `/knowledge/train-embedder`             | Train an embedder model.           |
| `GET`    | `/knowledge/embedder-status`            | Embedder training status.          |
| `GET`    | `/knowledge/reviews/due`                | Get items due for review.          |
| `POST`   | `/knowledge/reviews/{item_id}/schedule` | Schedule a review.                 |
| `GET`    | `/knowledge/label`                      | Get label info.                    |
| `POST`   | `/knowledge/rag/ingest`                 | Ingest documents for RAG.          |
| `POST`   | `/knowledge/rag/query`                  | Query the RAG pipeline.            |
| `POST`   | `/knowledge/rag/verify`                 | Verify RAG results.                |
| `GET`    | `/knowledge/rag/documents`              | List RAG documents.                |
| `POST`   | `/knowledge/rag/clear`                  | Clear RAG store.                   |
| `GET`    | `/knowledge/rag/stats`                  | RAG pipeline statistics.           |
| `POST`   | `/knowledge/kg/sync`                    | Sync knowledge graph.              |
| `GET`    | `/knowledge/kg/pipeline-stats`          | Knowledge graph pipeline stats.    |

## Memory Router (`/memory`)

| Method   | Path                    | Description                                                         |
| -------- | ----------------------- | ------------------------------------------------------------------- |
| `GET`    | `/memory/stats`         | Memory service stats (enabled, total facts, topic buckets).         |
| `GET`    | `/memory/list`          | List stored facts (limit 1..1000, default 100).                     |
| `GET`    | `/memory/search`        | Search facts by relevance (requires `q`).                           |
| `POST`   | `/memory/store`         | Store a fact directly. Body: `{content, source?}`.                  |
| `POST`   | `/memory/remember`      | Extract and store facts from a message. Body: `{message, source?}`. |
| `POST`   | `/memory/clear`         | Clear all stored facts.                                             |
| `POST`   | `/memory/consolidate`   | Trigger memory consolidation.                                       |
| `GET`    | `/memory/archive/stats` | Archive statistics.                                                 |
| `POST`   | `/memory/archive/prune` | Prune archived facts.                                               |
| `DELETE` | `/memory/{item_id}` | Delete Item |
| `GET` | `/memory/archive` | Archive |
| `GET` | `/memory/config` | Get Config |
| `PATCH` | `/memory/{item_id}` | Update Item |
| `POST` | `/memory/config` | Set Config |

Fail-closed when `SLO_MEMORY_ENABLED=false`.

## Datasets Router (`/datasets`)

| Method   | Path                                          | Description                        |
| -------- | --------------------------------------------- | ---------------------------------- |
| `GET`    | `/datasets`                                   | List all registered datasets.      |
| `POST`   | `/datasets`                                   | Create a new dataset.              |
| `GET`    | `/datasets/search`                            | Search datasets.                   |
| `GET`    | `/datasets/search/books`                      | Search for books.                  |
| `GET`    | `/datasets/search/github`                     | Search GitHub for datasets.        |
| `GET`    | `/datasets/{dataset_id}`                      | Get a dataset.                     |
| `GET`    | `/datasets/{dataset_id}/stats`                | Dataset statistics.                |
| `PATCH`  | `/datasets/{dataset_id}`                      | Update dataset metadata.           |
| `DELETE` | `/datasets/{dataset_id}`                      | Delete a dataset.                  |
| `POST`   | `/datasets/{dataset_id}/data`                 | Append rows.                       |
| `GET`    | `/datasets/{dataset_id}/preview`              | Preview first N rows.              |
| `POST`   | `/datasets/{dataset_id}/export`               | Export to file.                    |
| `POST`   | `/datasets/{dataset_id}/versions`             | Create a version snapshot.         |
| `GET`    | `/datasets/{dataset_id}/versions`             | List versions.                     |
| `POST`   | `/datasets/{dataset_id}/versions/{timestamp}` | Restore a version.                 |
| `POST`   | `/datasets/import/local`                      | Import from local path.            |
| `POST`   | `/datasets/import/github`                     | Import from GitHub.                |
| `POST`   | `/datasets/import/huggingface`                | Import from HuggingFace.           |
| `POST`   | `/datasets/import/url`                        | Import from URL.                   |
| `POST`   | `/datasets/import/kaggle`                     | Import from Kaggle.                |
| `POST`   | `/datasets/import/csv`                        | Import from CSV.                   |
| `POST`   | `/datasets/import/batch`                      | Batch import.                      |
| `POST`   | `/datasets/import/isbn`                       | Import by ISBN.                    |
| `POST`   | `/datasets/from-chat`                         | Create dataset from chat messages. |
| `POST`   | `/datasets/convert-to-messages`               | Convert dataset to message format. |
| `GET` | `/datasets/{dataset_id}/quality` | Quality Dataset |

## Training Router (`/training`)

Unified training control plane. All training operations go through `/training/*`.

Routes are split across focused sub-modules:

| Module                        | Lines | Responsibility                                                           |
| ----------------------------- | ----- | ------------------------------------------------------------------------ |
| `training/router.py`          | ~640  | Recovery, finetuned models, checkpoints, stream, stop                    |
| `training/execution.py`       | ~330  | Core start_training + includes (lora, distill, visual, feedback, builds) |
| `training/lora.py`            | ~336  | lora-finetune, load-adapter, unload-adapter                              |
| `training/distill.py`         | ~261  | Knowledge distillation                                                   |
| `training/from_feedback.py`   | ~189  | Feedback-based training                                                  |
| `training/visual.py`          | ~164  | VLM fine-tune                                                            |
| `training/builds.py`          | ~86   | Build listing                                                            |
| `training/legacy.py`          | ~142  | /train, /train/resolve (backward compat)                                 |
| `training/jobs_api.py`        | ~295  | Job CRUD, export, purge                                                  |
| `training/control.py`         | ~156  | Status, start/pause/resume/stop/reset                                    |
| `training/helpers.py`         | ~111  | Shared `_finish_job`, `_sloughgpt_trainer_kwds`, `_run_async`            |
| `training/turbo_endpoints.py` | ~122  | Turbo start + from-sessions start + turbo status                         |
| `training/sse_stream.py`      | ~282  | Shared SSE stream helper, stop_all_training, cancel_from_sessions        |

### Execution Routes (`execution.py`, `legacy.py`)

| Method | Path                       | Module                              | Description                               |
| ------ | -------------------------- | ----------------------------------- | ----------------------------------------- |
| `POST` | `/train`                   | legacy.py                           | Start a training job (legacy).            |
| `POST` | `/train/resolve`           | legacy.py                           | Resolve data path (dry run).              |
| `POST` | `/training/start`          | execution.py                        | Start a tracked training job (web UI).    |
| `POST` | `/training/visual-start`   | execution.py                        | Start a VLM fine-tune.                    |
| `POST` | `/training/distill`        | execution.py                        | Knowledge distillation (teacher→student). |
| `POST` | `/training/lora-finetune`  | execution.py                        | LoRA fine-tuning on .slnc models.         |
| `POST` | `/training/load-adapter`   | execution.py                        | Load a LoRA adapter for inference.        |
| `POST` | `/training/unload-adapter` | Unload the active LoRA adapter.     |
| `POST` | `/training/from-feedback`  | Train from collected feedback data. |
| `GET`  | `/training/builds`         | List all training builds.           |

### Control Routes (`control.py`)

| Method | Path                       | Description                     |
| ------ | -------------------------- | ------------------------------- |
| `GET`  | `/training/status`         | Training status.                |
| `POST` | `/training/control/start`  | Start training (control plane). |
| `POST` | `/training/control/pause`  | Pause the active job.           |
| `POST` | `/training/control/resume` | Resume the paused job.          |
| `POST` | `/training/control/stop`   | Stop the active job.            |
| `POST` | `/training/control/reset`  | Reset training controller.      |
| `GET`  | `/training/is-running`     | Check if training is running.   |

### Job Routes (`jobs_api.py`)

| Method   | Path                              | Description             |
| -------- | --------------------------------- | ----------------------- |
| `GET`    | `/training/jobs`                  | List all training jobs. |
| `GET`    | `/training/jobs/{job_id}`         | Get a specific job.     |
| `POST`   | `/training/jobs/{job_id}/stop`    | Stop a specific job.    |
| `GET`    | `/training/jobs/{job_id}/summary` | Get job summary.        |
| `DELETE` | `/training/jobs/{job_id}`         | Delete a job.           |
| `POST`   | `/training/jobs/purge`            | Purge old jobs.         |
| `GET`    | `/training/export/{job_id}`       | Export job data (JSON). |
| `POST`   | `/training/export-text`           | Export job data (text). |

### Stream & Utility Routes (`router.py`)

| Method   | Path                                    | Description                    |
| -------- | --------------------------------------- | ------------------------------ |
| `POST`   | `/training/stop`                        | Stop the active job.           |
| `POST`   | `/training/turbo-start`                 | Start turbo training.          |
| `GET`    | `/training/turbo/status`                | Turbo training status.         |
| `GET`    | `/training/log`                         | Training log.                  |
| `GET`    | `/training/stream`                      | SSE training stream.           |
| `GET`    | `/training/from-sessions/cancel`        | Cancel session-based training. |
| `GET`    | `/training/from-sessions-stream`        | SSE stream from sessions.      |
| `GET`    | `/training/checkpoints`                 | List checkpoints.              |
| `DELETE` | `/training/checkpoints/{name}`          | Delete a checkpoint.           |
| `POST`   | `/training/checkpoints/{name}/load`     | Load a checkpoint.             |
| `GET`    | `/training/checkpoints/{name}/download` | Download a checkpoint.         |
| `GET`    | `/training/checkpoints/{name}/info`     | Checkpoint info.               |
| `GET`    | `/training/metrics/export`              | Export training metrics.       |

### Recovery Routes (`router.py`)

| Method   | Path                         | Description                 |
| -------- | ---------------------------- | --------------------------- |
| `GET`    | `/recovery/check`            | Check for crashed jobs.     |
| `GET`    | `/recovery/recoverable`      | Get recoverable jobs.       |
| `POST`   | `/recovery/recover/{job_id}` | Recover an interrupted job. |
| `DELETE` | `/recovery/abandon/{job_id}` | Abandon a crashed job.      |
| `GET`    | `/recovery/stats`            | Recovery statistics.        |

> **Note:** `POST /recovery/recover/{job_id}` runs a preflight before it writes any recovery state and answers **422** `{"error": ...}` when a resume cannot work: no `data_path` recorded on the job, the dataset file is gone, or the job ran a trainer this path cannot restart (distill / LoRA / visual). `GET /recovery/recoverable` applies the same rule, so a job that would 422 is never listed as recoverable. Checkpoint fallback scans only the job's own stem — `checkpoint_dir` is shared (`models/auto-training`), so an unscoped "newest that loads" would adopt another job's weights.

### Finetuned Model Routes (`router.py`)

| Method   | Path                                     | Description                |
| -------- | ---------------------------------------- | -------------------------- |
| `GET`    | `/training/finetuned-models`             | List HF fine-tuned models. |
| `POST`   | `/training/finetuned-models/{name}/load` | Load a fine-tuned model.   |
| `DELETE` | `/training/finetuned-models/{name}`      | Delete a fine-tuned model. |
| `DELETE` | `/training/webhooks/{webhook_id}` | Unregister Webhook |
| `GET` | `/training/feed` | Training Feed |
| `GET` | `/training/feeds` | Training Feeds |
| `GET` | `/training/recommend` | Get Training Recommendation |
| `GET` | `/training/trends` | Get Training Trends |
| `GET` | `/training/webhooks` | List Webhooks |
| `GET` | `/training/webhooks/dead-letters` | Get Webhook Dead Letters |
| `GET` | `/training/webhooks/retry-queue` | Get Webhook Retry Queue |
| `GET` | `/training/webhooks/stats` | Get Webhook Stats |
| `GET` | `/training/webhooks/{webhook_id}` | Get Webhook |
| `GET` | `/training/webhooks/{webhook_id}/deliveries` | Get Webhook Deliveries |
| `POST` | `/training/from-sessions-start` | Start From Sessions Unified |
| `POST` | `/training/webhooks` | Register Webhook |
| `POST` | `/training/webhooks/test` | Test Webhook |

> **Note:** The legacy `/auto-train/*` endpoints (in `routers/auto_train.py`) are deprecated. They are a parallel implementation, not shims — new clients should use `/training/*` instead. The `/training/stop`, `/training/turbo-start`, and `/training/stream` routes now use `training/sse_stream.py` and `training/turbo_endpoints.py` which delegate to `domain.training._internal.service` (core layer).

## Self-Train Router (`/self-train`)

| Method | Path                 | Description           |
| ------ | -------------------- | --------------------- |
| `POST` | `/self-train/start`  | Start self-training.  |
| `POST` | `/self-train/stop`   | Stop self-training.   |
| `GET`  | `/self-train/status` | Self-training status. |

## Learner Router (`/learn`)

| Method | Path                | Description              |
| ------ | ------------------- | ------------------------ |
| `POST` | `/learn/search`     | Search learning content. |
| `GET`  | `/learn/feed`       | Get learning feed.       |
| `POST` | `/learn/feed`       | Add to learning feed.    |
| `POST` | `/learn/ingest-url` | Ingest from URL.         |
| `GET`  | `/learn/knowledge`  | List learned knowledge.  |
| `POST` | `/learn/ingest`     | Ingest training data.    |
| `POST` | `/learn/train`      | Train the learner.       |
| `POST` | `/learn/deploy`     | Deploy trained model.    |
| `POST` | `/learn/evaluate`   | Evaluate trained model.  |
| `GET`  | `/learn/status`     | Learner status.          |

## Tokenizer Router (`/tokenizer`)

| Method | Path                     | Description                                          |
| ------ | ------------------------ | ---------------------------------------------------- |
| `GET`  | `/tokenizer/stats`       | Vocabulary size, merge count, token-to-id map stats. |
| `POST` | `/tokenizer/pretokenize` | Pre-tokenize text.                                   |
| `POST` | `/tokenizer/decompose`   | Decompose text into tokens.                          |
| `POST` | `/tokenizer/analyze`     | Analyze token distribution.                          |
| `POST` | `/tokenizer/tokenize`    | Tokenize string to token IDs.                        |
| `POST` | `/tokenizer/detokenize`  | Convert token IDs to string.                         |
| `GET`  | `/tokenizer/vocab`       | Full vocabulary list.                                |
| `GET`  | `/tokenizer/merges`      | BPE merge operations.                                |
| `POST` | `/tokenizer/train`       | Train a fresh BPE tokenizer.                         |
| `GET`  | `/tokenizer/sample`      | Get sample tokens.                                   |
| `GET`  | `/tokenizer/samples`     | Get sample tokens/words for UI.                      |

## Token Tree Router (`/token-tree`)

Tree-based BPE tokenizer — training, encoding, embedding/semantic explorer.

| Method   | Path                       | Description                                                  |
| -------- | -------------------------- | ------------------------------------------------------------ |
| `GET`    | `/token-tree/stats`        | Tree summary (vocab, merges, embeddings, compression ratio). |
| `GET`    | `/token-tree/vocab`        | Paged vocabulary (limit 1..500, default 50).                 |
| `GET`    | `/token-tree/merges`       | Ranked merge rules (optional `query` filter).                |
| `GET`    | `/token-tree/saved`        | List saved trees.                                            |
| `POST`   | `/token-tree/save`         | Save current tree. Body: `{name}`.                           |
| `POST`   | `/token-tree/load`         | Load a saved tree. Body: `{name}`.                           |
| `DELETE` | `/token-tree/saved/{name}` | Delete a saved tree.                                         |
| `POST`   | `/token-tree/train`        | Train on `{texts?, vocab_size, embed_dim, min_frequency}`.   |
| `POST`   | `/token-tree/similar`      | Nearest-neighbor tokens. Body: `{token, top_k}`.             |
| `POST`   | `/token-tree/embedding`    | Inspect token embedding. Body: `{token, top_k}`.             |
| `POST`   | `/token-tree/encode`       | Encode text. Body: `{text}` → `{tokens, ids}`.               |
| `POST`   | `/token-tree/path`         | Trace greedy trie walk. Body: `{text}` → `{steps, ids}`.     |
| `POST`   | `/token-tree/decode`       | Decode token IDs. Body: `{ids}` → `{text}`.                  |
| `POST`   | `/token-tree/lineage`      | Render merge lineage. Body: `{token}`.                       |
| `GET`    | `/token-tree/matrix`       | Embedding matrix overview (top_k).                           |
| `POST`   | `/token-tree/compare`      | Diff two saved trees. Body: `{a, b, top_k}`.                 |

Semantic endpoints return `404` for unknown tokens; `embedding` returns `422` when tree has no embeddings.

## Agent Router (`/agents`)

| Method   | Path                         | Description                   |
| -------- | ---------------------------- | ----------------------------- |
| `GET`    | `/agents`                    | List all agents.              |
| `POST`   | `/agents`                    | Create a new agent.           |
| `GET`    | `/agents/{agent_id}`         | Get agent details.            |
| `PUT`    | `/agents/{agent_id}`         | Update an agent.              |
| `DELETE` | `/agents/{agent_id}`         | Delete an agent.              |
| `POST`   | `/agents/{agent_id}/execute` | Execute an agent.             |
| `GET`    | `/agents/runs`               | List agent runs.              |
| `GET`    | `/agents/runs/{run_id}`      | Get a specific run.           |
| `POST`   | `/agents/orchestrate`        | Orchestrate multi-agent task. |

## Multimodal Router (`/multimodal`)

| Method   | Path                                  | Description                  |
| -------- | ------------------------------------- | ---------------------------- |
| `GET`    | `/multimodal/status`                  | Multimodal engine status.    |
| `POST`   | `/multimodal/train`                   | Train on a single image.     |
| `POST`   | `/multimodal/train-batch`             | Train on a batch of images.  |
| `POST`   | `/multimodal/train-video`             | Train video captioning.      |
| `POST`   | `/multimodal/video-infer`             | Infer video captions.        |
| `POST`   | `/multimodal/dpo`                     | Run DPO training.            |
| `POST`   | `/multimodal/analyze`                 | Analyze an image.            |
| `POST`   | `/multimodal/pdf/upload`              | Upload and analyze a PDF.    |
| `POST`   | `/multimodal/process-video`           | Process a video file.        |
| `POST`   | `/multimodal/transcribe`              | Transcribe audio.            |
| `POST`   | `/multimodal/synthesize-speech`       | Text-to-speech synthesis.    |
| `POST`   | `/multimodal/generate-image`          | Generate an image.           |
| `POST`   | `/multimodal/visual-dataset`          | Create visual dataset.       |
| `GET`    | `/multimodal/checkpoints`             | List checkpoints.            |
| `POST`   | `/multimodal/checkpoints/{name}/load` | Load a checkpoint.           |
| `DELETE` | `/multimodal/checkpoints/{name}`      | Delete a checkpoint.         |
| `POST`   | `/multimodal/reset`                   | Reset the multimodal engine. |
| `POST` | `/multimodal/ask` | Ask Question |
| `POST` | `/multimodal/batch-encode-phonemes` | Batch Encode Phonemes |
| `POST` | `/multimodal/batch-score-pronunciation` | Batch Score Pronunciation |
| `POST` | `/multimodal/decode-phonemes` | Decode Phonemes |
| `POST` | `/multimodal/detect` | Detect Objects |
| `POST` | `/multimodal/detect-language` | Detect Language |
| `POST` | `/multimodal/encode-phonemes` | Encode Phonemes |
| `POST` | `/multimodal/score-pronunciation` | Score Pronunciation |

## Benchmark Router (`/benchmark`)

| Method | Path                       | Description                    |
| ------ | -------------------------- | ------------------------------ |
| `POST` | `/benchmark/run`           | Run a benchmark.               |
| `GET`  | `/benchmark/metrics`       | Get model metrics.             |
| `GET`  | `/benchmark/{model_id}`    | Benchmark results for a model. |
| `POST` | `/benchmark/perplexity`    | Calculate perplexity.          |
| `GET`  | `/benchmark/quality`       | Quality metrics.               |
| `GET`  | `/benchmark/responses`     | Logged responses.              |
| `GET`  | `/benchmark/stats`         | Tracker statistics.            |
| `POST` | `/benchmark/history/clear` | Clear benchmark history.       |
| `GET` | `/benchmark/program` | Get Program |
| `POST` | `/benchmark/score` | Score Results |

## Companion Router (`/companion`)

| Method   | Path                     | Description                   |
| -------- | ------------------------ | ----------------------------- |
| `GET`    | `/companion/`            | Get current companion.        |
| `DELETE` | `/companion/`            | Delete companion.             |
| `POST`   | `/companion/personality` | Set companion personality.    |
| `PATCH`  | `/companion/personality` | Update companion personality. |
| `POST`   | `/companion/preset`      | Apply a preset.               |
| `GET`    | `/companion/prompt`      | Get companion system prompt.  |
| `POST`   | `/companion/chat`        | Chat with companion.          |
| `GET`    | `/companion/presets`     | List available presets.       |
| `DELETE` | `/companion/presets/{preset_id}` | Delete Preset |
| `POST` | `/companion/presets` | Create Preset |

## Docstore Router (`/docstore`)

| Method   | Path                              | Description                     |
| -------- | --------------------------------- | ------------------------------- |
| `GET`    | `/docstore/{collection}`          | List documents in a collection. |
| `GET`    | `/docstore/{collection}/{doc_id}` | Get a document.                 |
| `PUT`    | `/docstore/{collection}/{doc_id}` | Create/replace a document.      |
| `PATCH`  | `/docstore/{collection}/{doc_id}` | Partially update a document.    |
| `DELETE` | `/docstore/{collection}/{doc_id}` | Delete a document.              |
| `DELETE` | `/docstore/{collection}`          | Clear a collection.             |
| `POST`   | `/docstore/{collection}/bulk`     | Bulk upsert documents.          |
| `DELETE` | `/docstore/message-notes/{session_id}/{message_id}` | Delete Message Note |
| `GET` | `/docstore/message-notes` | List Message Notes |
| `GET` | `/docstore/message-notes/search` | Search Message Notes |
| `POST` | `/docstore/message-notes` | Put Message Note |

## Errors Router (`/errors`)

| Method   | Path                  | Description             |
| -------- | --------------------- | ----------------------- |
| `POST`   | `/errors/log`         | Log an error.           |
| `POST`   | `/errors/logs/ingest` | Ingest error logs.      |
| `GET`    | `/errors/recent`      | Get recent errors.      |
| `GET`    | `/errors/grouped`     | Get grouped errors.     |
| `GET`    | `/errors/trends`      | Error trend analysis.   |
| `GET`    | `/errors/export`      | Export error logs.      |
| `DELETE` | `/errors/clear`       | Clear error logs.       |
| `GET`    | `/errors/unread`      | Get unread error count. |
| `GET`    | `/errors/log`         | Get OpenCode error log. |
| `GET` | `/errors/stream` | Error Stream |

## Experiments Router (`/experiments`)

| Method   | Path                                      | Description               |
| -------- | ----------------------------------------- | ------------------------- |
| `POST`   | `/experiments`                            | Create an experiment.     |
| `GET`    | `/experiments`                            | List all experiments.     |
| `GET`    | `/experiments/{experiment_id}`            | Get an experiment.        |
| `DELETE` | `/experiments/{experiment_id}`            | Delete an experiment.     |
| `GET`    | `/experiments/{experiment_id}/runs`       | List experiment runs.     |
| `GET`    | `/experiments/{experiment_id}/data`       | Get experiment data.      |
| `POST`   | `/experiments/{experiment_id}/complete`   | Mark experiment complete. |
| `POST`   | `/experiments/{experiment_id}/log_metric` | Log a metric.             |
| `POST`   | `/experiments/{experiment_id}/log_param`  | Log a parameter.          |
| `GET` | `/experiments/compare` | Compare Experiments |

## Vector Router (`/vector`)

| Method | Path                    | Description              |
| ------ | ----------------------- | ------------------------ |
| `POST` | `/vector/init`          | Initialize vector store. |
| `GET`  | `/vector/stats`         | Vector store statistics. |
| `POST` | `/vector/upsert`        | Upsert vectors.          |
| `POST` | `/vector/search`        | Search vectors.          |
| `GET`  | `/vector/ingest/status` | Ingestion status.        |

## User Adapters Router (`/user-adapters`)

| Method   | Path                              | Description              |
| -------- | --------------------------------- | ------------------------ |
| `GET`    | `/user-adapters`                  | List all user adapters.  |
| `GET`    | `/user-adapters/quality`          | Adapter quality metrics. |
| `GET`    | `/user-adapters/{user_id}`        | Get a user adapter.      |
| `POST`   | `/user-adapters/{user_id}/update` | Update a user adapter.   |
| `POST`   | `/user-adapters/{user_id}/reset`  | Reset a user adapter.    |
| `DELETE` | `/user-adapters/{user_id}`        | Delete a user adapter.   |
| `POST`   | `/user-adapters/merge`            | Merge adapters.          |
| `POST`   | `/user-adapters/aggregate-best`   | Aggregate best adapters. |
| `POST`   | `/user-adapters/prune`            | Prune adapters.          |

## LoRA Eval Router (`/lora-eval`)

| Method | Path                   | Description                   |
| ------ | ---------------------- | ----------------------------- |
| `GET`  | `/lora-eval/history`   | Evaluation history.           |
| `POST` | `/lora-eval/aggregate` | Aggregate evaluation results. |
| `POST` | `/lora-eval/run` | Run Eval |

## Registry Router (`/registry`)

| Method | Path                   | Description                |
| ------ | ---------------------- | -------------------------- |
| `GET`  | `/registry/best`       | Get best model for a task. |
| `GET`  | `/registry/stats`      | Registry statistics.       |
| `GET` | `/registry/artifacts` | List Artifacts |
| `GET` | `/registry/models` | List Models |
| `GET` | `/registry/models/{model_id}` | Get Model |
| `GET` | `/registry/verify` | Verify Artifacts |

## Meta-Weights Router (`/meta-weights`)

| Method | Path                  | Description              |
| ------ | --------------------- | ------------------------ |
| `GET`  | `/meta-weights/ping`  | Health probe.            |
| `POST` | `/meta-weights/get`   | Get meta-weights.        |
| `GET`  | `/meta-weights/stats` | Meta-weights statistics. |

## VM Router (`/vm`)

| Method | Path                              | Description                       |
| ------ | --------------------------------- | --------------------------------- |
| `POST` | `/vm/run`                         | Run x86 assembly in sandboxed VM. |
| `GET`  | `/vm/training/jobs/{job_id}`      | Training job status.              |
| `POST` | `/vm/training/jobs/{job_id}/stop` | Stop a VM training job.           |
| `GET`  | `/vm/builtins`                    | List built-in assembly programs.  |
| `GET`  | `/vm/info`                        | VM capabilities and limits.       |

## Workflow Router (`/workflow`)

| Method | Path                         | Description                |
| ------ | ---------------------------- | -------------------------- |
| `GET`  | `/workflow/status`           | Workflow status.           |
| `POST` | `/workflow/start`            | Start a workflow.          |
| `POST` | `/workflow/stop`             | Stop a workflow.           |
| `POST` | `/workflow/trigger/{action}` | Trigger a workflow action. |

## Shell Router (`/shell`)

| Method | Path                 | Description                 |
| ------ | -------------------- | --------------------------- |
| `POST` | `/shell/exec`        | Execute a shell command.    |
| `POST` | `/shell/exec/stream` | Execute with SSE streaming. |

## Images Router (`/images`)

| Method | Path              | Description            |
| ------ | ----------------- | ---------------------- |
| `GET`  | `/images/gallery` | List generated images. |
| `GET`  | `/images/styles`  | List available styles. |
| `POST` | `/images/generate` | Generate Image |

## Voice Router (`/voice`)

| Method | Path            | Description             |
| ------ | --------------- | ----------------------- |
| `POST` | `/voice/tts`    | Convert text to speech. |
| `GET`  | `/voice/status` | Voice synthesis status. |

## Files Router (`/files`)

| Method | Path               | Description    |
| ------ | ------------------ | -------------- |
| `GET`  | `/files`           | List files.    |
| `GET`  | `/files/{file_id}` | Get a file.    |
| `DELETE` | `/files/{file_id}` | Delete File |
| `GET` | `/files/search` | Search Files |
| `POST` | `/files/upload` | Upload File |
| `POST` | `/files/{file_id}/ingest` | Ingest File |

## Security Router (`/security`)

| Method | Path                   | Description     |
| ------ | ---------------------- | --------------- |
| `GET`  | `/security/keys`       | List API keys.  |
| `DELETE` | `/security/keys/{key_id}` | Delete Key |
| `GET` | `/security/audit` | Get Audit Logs |
| `GET` | `/security/keys/{key_id}` | Get Key |
| `POST` | `/security/keys` | Create Key |
| `POST` | `/security/keys/validate` | Validate Key |
| `POST` | `/security/keys/{key_id}/rotate` | Rotate Key |

## World Render Router (`/world`)

| Method | Path                  | Description                      |
| ------ | --------------------- | -------------------------------- |
| `POST` | `/world/render`       | Render the world state.          |
| `POST` | `/world/render/image` | Render as PNG image.             |
| `POST` | `/world/neural`       | Process through neural pipeline. |
| `POST` | `/world/tick`         | Run a simulation tick.           |
| `GET`  | `/world/stats`        | World rendering statistics.      |

## Consciousness Router (`/consciousness`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `DELETE` | `/consciousness/personas/{persona_id}` | Delete Persona |
| `GET` | `/consciousness/backup/download` | Backup Download |
| `GET` | `/consciousness/evaluate` | Evaluate |
| `GET` | `/consciousness/health` | Health Check |
| `GET` | `/consciousness/history/beliefs` | Get Beliefs History |
| `GET` | `/consciousness/history/episodes` | Get Episode History |
| `GET` | `/consciousness/history/qualia` | Get Qualia History |
| `GET` | `/consciousness/personality` | Get Personality |
| `GET` | `/consciousness/personality/conflicts` | Get Conflicts |
| `GET` | `/consciousness/personality/history` | Get Personality History |
| `GET` | `/consciousness/personality/presets` | Get Presets |
| `GET` | `/consciousness/personas` | List Personas |
| `GET` | `/consciousness/personas/{persona_id}` | Get Persona |
| `GET` | `/consciousness/qualia` | Get Qualia |
| `GET` | `/consciousness/self-model` | Get Self Model |
| `GET` | `/consciousness/stats` | Get Stats |
| `GET` | `/consciousness/status` | Get Status |
| `GET` | `/consciousness/stream` | Stream Status |
| `GET` | `/consciousness/train/status` | Train Status |
| `PATCH` | `/consciousness/config` | Update Config |
| `PATCH` | `/consciousness/personality` | Update Personality |
| `POST` | `/consciousness/backup` | Backup |
| `POST` | `/consciousness/backup/import` | Backup Import |
| `POST` | `/consciousness/batch` | Batch Operations |
| `POST` | `/consciousness/clear/beliefs` | Clear Beliefs |
| `POST` | `/consciousness/clear/episodes` | Clear Episodes |
| `POST` | `/consciousness/feedback` | Submit Feedback |
| `POST` | `/consciousness/personality/presets/apply` | Apply Preset |
| `POST` | `/consciousness/personality/reset` | Reset Personality |
| `POST` | `/consciousness/personas/save` | Save Persona |
| `POST` | `/consciousness/personas/{persona_id}/activate` | Activate Persona |
| `POST` | `/consciousness/reflect` | Reflect |
| `POST` | `/consciousness/restore` | Restore |
| `POST` | `/consciousness/seed` | Seed Data |
| `POST` | `/consciousness/train/start` | Train Start |
## Mobile Router (`/mobile`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `DELETE` | `/mobile/knowledge/{item_id}` | Delete Knowledge |
| `DELETE` | `/mobile/train/pair/{pair_id}` | Delete Pair |
| `DELETE` | `/mobile/train/pairs/bulk` | Delete Pairs Bulk |
| `DELETE` | `/mobile/train/synced` | Delete Synced Pairs |
| `GET` | `/mobile/conversations` | List Conversations |
| `GET` | `/mobile/conversations/{session_id}` | Get Conversation |
| `GET` | `/mobile/dashboard` | Get Dashboard |
| `GET` | `/mobile/health` | Get Health |
| `GET` | `/mobile/knowledge` | List Knowledge |
| `GET` | `/mobile/models` | Get Models |
| `GET` | `/mobile/notifications/devices` | List Devices |
| `GET` | `/mobile/notifications/history` | Notification History |
| `GET` | `/mobile/sync/status` | Sync Status |
| `GET` | `/mobile/train/auto-status` | Get Auto Train Status |
| `GET` | `/mobile/train/export` | Export Training Pairs |
| `GET` | `/mobile/train/pairs` | List Training Pairs |
| `GET` | `/mobile/train/pending` | Get Pending Pairs |
| `GET` | `/mobile/train/session/{session_id}` | Get Session Pairs |
| `GET` | `/mobile/train/stats` | Get Training Stats |
| `PATCH` | `/mobile/knowledge/{item_id}` | Update Knowledge |
| `PATCH` | `/mobile/train/auto-config` | Update Auto Train Config |
| `PATCH` | `/mobile/train/pair/{pair_id}` | Update Pair Quality |
| `POST` | `/mobile/knowledge` | Create Knowledge |
| `POST` | `/mobile/models/switch` | Switch Model |
| `POST` | `/mobile/notifications/cleanup` | Cleanup Devices |
| `POST` | `/mobile/notifications/register` | Register Device |
| `POST` | `/mobile/notifications/send` | Send Notification |
| `POST` | `/mobile/notifications/unregister` | Unregister Device |
| `POST` | `/mobile/notify/training-complete` | Notify Training Complete |
| `POST` | `/mobile/sync` | Sync Offline |
| `POST` | `/mobile/train` | Mobile Train |
| `POST` | `/mobile/train/compact` | Compact Training Store |
| `POST` | `/mobile/train/from-sessions` | Train From Sessions |
## Model Stack Router (`/model-stack`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `DELETE` | `/model-stack/{name}` | Remove Layer |
| `GET` | `/model-stack` | Get Stack |
| `POST` | `/model-stack/base` | Set Base |
| `POST` | `/model-stack/clear` | Clear Layers |
| `POST` | `/model-stack/push` | Push Layer |
## Settings Router (`/settings`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `DELETE` | `/settings/training/runs/{run_id}` | Delete Training Run |
| `DELETE` | `/settings/training/runs/{run_id}/tags/{tag}` | Remove Run Tag |
| `GET` | `/settings` | Get Settings |
| `GET` | `/settings/adaptive` | Get Adaptive |
| `GET` | `/settings/adaptive/insights` | Get Adaptive Insights |
| `GET` | `/settings/generation` | Get Generation |
| `GET` | `/settings/providers/api` | Get Providers Api |
| `GET` | `/settings/training` | Get Training |
| `GET` | `/settings/training/analytics` | Get Training Analytics |
| `GET` | `/settings/training/auto-train/status` | Auto Train Status |
| `GET` | `/settings/training/batch-status` | Get Batch Training Status |
| `GET` | `/settings/training/bookmarks` | Get Bookmarked Runs |
| `GET` | `/settings/training/compare` | Compare Training Runs |
| `GET` | `/settings/training/history/export` | Export Training History |
| `GET` | `/settings/training/presets` | List Training Presets |
| `GET` | `/settings/training/presets/{preset_name}` | Get Training Preset |
| `GET` | `/settings/training/runs` | Filter Training Runs |
| `GET` | `/settings/training/runs/{run_id}` | Get Training Run |
| `GET` | `/settings/training/runs/{run_id}/export` | Export Training Run |
| `GET` | `/settings/training/tags` | Get All Tags |
| `GET` | `/settings/training/tags/{tag}` | Get Runs By Tag |
| `GET` | `/settings/ui` | Get Ui |
| `GET` | `/settings/voice` | Get Voice |
| `PATCH` | `/settings/adaptive` | Update Adaptive |
| `PATCH` | `/settings/generation` | Update Generation |
| `PATCH` | `/settings/providers/api` | Update Providers Api |
| `PATCH` | `/settings/training` | Update Training |
| `PATCH` | `/settings/training/auto-train/config` | Auto Train Config |
| `PATCH` | `/settings/ui` | Update Ui |
| `PATCH` | `/settings/voice` | Update Voice |
| `POST` | `/settings/model-card` | Generate Model Card |
| `POST` | `/settings/reset` | Reset Settings |
| `POST` | `/settings/training/history/clear` | Clear Training History |
| `POST` | `/settings/training/presets/{preset_name}/apply` | Apply Training Preset |
| `POST` | `/settings/training/runs/bulk/bookmark` | Bulk Bookmark |
| `POST` | `/settings/training/runs/bulk/delete` | Bulk Delete Runs |
| `POST` | `/settings/training/runs/bulk/tag` | Bulk Add Tag |
| `POST` | `/settings/training/runs/{run_id}/bookmark` | Toggle Bookmark |
| `POST` | `/settings/training/runs/{run_id}/duplicate` | Duplicate Training Run |
| `POST` | `/settings/training/runs/{run_id}/tags` | Add Run Tag |
| `PUT` | `/settings/training/runs/{run_id}/notes` | Set Run Notes |
## Tenants Router (`/tenants`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `DELETE` | `/tenants/{tenant_id}` | Delete Tenant |
| `GET` | `/tenants` | List Tenants |
| `GET` | `/tenants/{tenant_id}` | Get Tenant |
| `GET` | `/tenants/{tenant_id}/stats` | Get Tenant Stats |
| `POST` | `/tenants` | Create Tenant |
| `PUT` | `/tenants/{tenant_id}` | Update Tenant |
## Users Router (`/users`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `DELETE` | `/users/{user_id}` | Delete User |
| `GET` | `/users` | List Users |
| `GET` | `/users/me/profile` | Get Profile |
| `GET` | `/users/{user_id}` | Get User |
| `POST` | `/users` | Create User |
| `POST` | `/users/me/password` | Change Password |
| `PUT` | `/users/me/profile` | Update Profile |
| `PUT` | `/users/{user_id}` | Update User |
## Workspaces Router (`/workspaces`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `DELETE` | `/workspaces/{workspace_id}` | Delete Workspace |
| `DELETE` | `/workspaces/{workspace_id}/members/{user_id}` | Remove Member |
| `DELETE` | `/workspaces/{workspace_id}/share/{share_id}` | Revoke Share |
| `GET` | `/workspaces` | List Workspaces |
| `GET` | `/workspaces/{workspace_id}` | Get Workspace |
| `GET` | `/workspaces/{workspace_id}/activity` | Get Workspace Activity |
| `GET` | `/workspaces/{workspace_id}/export` | Export Workspace Data |
| `GET` | `/workspaces/{workspace_id}/health` | Workspace Health Check |
| `GET` | `/workspaces/{workspace_id}/members` | List Members |
| `GET` | `/workspaces/{workspace_id}/notifications` | Get Workspace Notifications |
| `GET` | `/workspaces/{workspace_id}/permissions` | Get Workspace Permissions |
| `GET` | `/workspaces/{workspace_id}/search` | Search Workspace |
| `GET` | `/workspaces/{workspace_id}/settings` | Get Workspace Settings |
| `GET` | `/workspaces/{workspace_id}/shared` | List Shared Data |
| `GET` | `/workspaces/{workspace_id}/shared/api-keys` | Get Shared Api Keys |
| `GET` | `/workspaces/{workspace_id}/shared/datasets` | Get Shared Datasets |
| `GET` | `/workspaces/{workspace_id}/shared/knowledge` | Get Shared Knowledge |
| `GET` | `/workspaces/{workspace_id}/stats` | Get Workspace Stats |
| `GET` | `/workspaces/{workspace_id}/usage` | Get Workspace Usage |
| `POST` | `/workspaces` | Create Workspace |
| `POST` | `/workspaces/import` | Import Workspace Data |
| `POST` | `/workspaces/{workspace_id}/cleanup` | Cleanup Workspace Data |
| `POST` | `/workspaces/{workspace_id}/clone` | Clone Workspace |
| `POST` | `/workspaces/{workspace_id}/invite` | Invite Member |
| `POST` | `/workspaces/{workspace_id}/members` | Add Member |
| `POST` | `/workspaces/{workspace_id}/members/bulk` | Bulk Import Members |
| `POST` | `/workspaces/{workspace_id}/share` | Share Data |
| `PUT` | `/workspaces/{workspace_id}` | Update Workspace |
| `PUT` | `/workspaces/{workspace_id}/settings` | Update Workspace Settings |
## Cloud Training Router (`/cloud-training`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/cloud-training/jobs` | List Jobs |
| `GET` | `/cloud-training/{job_id}/status` | Job Status |
| `POST` | `/cloud-training/submit` | Submit Job |
| `POST` | `/cloud-training/{job_id}/cancel` | Cancel Job |
## Dashboard Router (`/dashboard`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/dashboard/events` | Dashboard Events |
| `GET` | `/dashboard/stream` | Dashboard Stream |
| `GET` | `/dashboard/summary` | Dashboard Summary |
## Info Router (`/info`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/info` | Get Info |
| `GET` | `/info/soul` | Get Info Soul |
## Monitor Router (`/monitor`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/monitor` | Training Monitor Status |
| `GET` | `/monitor/alerts` | Training Monitor Alerts |
| `GET` | `/monitor/metrics` | Training Monitor Metrics |
| `GET` | `/monitor/resources` | Training Monitor Resources |
| `POST` | `/monitor/reset` | Training Monitor Reset |
## Openwebui Router (`/openwebui`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/openwebui/checkpoints` | List Checkpoints |
| `GET` | `/openwebui/datasets` | List Datasets |
| `GET` | `/openwebui/training/status` | Training Status |
| `POST` | `/openwebui/checkpoint/reload` | Reload Checkpoint |
| `POST` | `/openwebui/training/start` | Start Training |
| `POST` | `/openwebui/training/stop` | Stop Training |
## Phoneme Router (`/phoneme`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/phoneme/languages` | Languages |
| `GET` | `/phoneme/status` | Status |
| `POST` | `/phoneme/batch-encode` | Batch Encode |
| `POST` | `/phoneme/decode` | Decode |
| `POST` | `/phoneme/detect-language` | Detect Language |
| `POST` | `/phoneme/encode` | Encode |
| `POST` | `/phoneme/score` | Score |
| `POST` | `/phoneme/synthesize` | Synthesize |
| `POST` | `/phoneme/visualize` | Visualize |
## Plugins Router (`/plugins`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/plugins` | List Plugins |
| `POST` | `/plugins/reload` | Reload Plugins |
| `POST` | `/plugins/{plugin_name}/disable` | Disable Plugin |
| `POST` | `/plugins/{plugin_name}/enable` | Enable Plugin |
## Profiles Router (`/profiles`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/profiles` | List Profiles |
| `GET` | `/profiles/active` | Get Active Profile |
| `GET` | `/profiles/recommend` | Recommend Profile |
| `GET` | `/profiles/{profile_id}` | Get Profile |
| `POST` | `/profiles/apply` | Apply Profile |
## Rate Limit Router (`/rate-limit`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/rate-limit/check` | Check Rate Limit |
| `GET` | `/rate-limit/status` | Get Rate Limit Status |
## Tokens Router (`/tokens`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/tokens/balance` | Get Balance |
| `GET` | `/tokens/usage/history` | Get Usage History |
| `GET` | `/tokens/usage/summary` | Get Usage Summary |
| `POST` | `/tokens/check` | Check Tokens |
| `POST` | `/tokens/topup` | Topup Credits |
| `POST` | `/tokens/upgrade` | Upgrade Tier |
## Tools Router (`/tools`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/tools` | List Tools |
| `POST` | `/tools/{tool_id}/generate` | Generate Tool |
## Cancel All Router (`/cancel-all`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `POST` | `/cancel-all` | Cancel All Operations |
## Cancel Router (`/cancel`)

| Method | Path | Description |
| ------ | ---- | ----------- |
| `POST` | `/cancel/{op_id}` | Cancel Operation |
## OpenAPI Specification

FastAPI automatically provides the OpenAPI spec at `/openapi.json`:

```
curl http://localhost:8000/openapi.json
```

The spec includes all routes described above, request/response schemas, and example payloads.
