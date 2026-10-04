# API Routers Documentation

> Building or changing a router? Follow the **Router Playbook** in
> [PRODUCT_ENGINEERING.md](PRODUCT_ENGINEERING.md#router-playbook). This file
> is the endpoint map only. Adding a capability rather than a handler? Read
> [TRANSPORT_PROJECTIONS.md](TRANSPORT_PROJECTIONS.md) — routes are projections
> of a declared contract, not endpoint projects.

All routes are served by the FastAPI application under the base URL (e.g. `http://localhost:8000`). Every public endpoint uses `classify_and_raise(e, source="router.method")` for structured error responses.

## Health Router (`/health`)

Registered pre-lifespan in `main.py` for startup/load balancer probes.

| Method | Path                          | Description                                                          |
| ------ | ----------------------------- | -------------------------------------------------------------------- |
| `GET`  | `/health`                     | Basic health check with 5 s fallback timeout.                        |
| `GET`  | `/health/debug`               | Debug health info.                                                   |
| `GET`  | `/health/detailed`            | Full health snapshot.                                                |
| `GET`  | `/health/live`                | Liveness probe.                                                      |
| `GET`  | `/health/model`               | Model-specific health.                                               |
| `GET`  | `/health/ready`               | Readiness probe (model loaded).                                      |
| `GET`  | `/health/services`            | Health check for all subsystems: training, settings, plugins, cloud. |
| `GET`  | `/health/startup-compare`     | Compare two startup runs.                                            |
| `GET`  | `/health/startup-config`      | Startup configuration.                                               |
| `GET`  | `/health/startup-diagnostics` | Startup diagnostics for debugging.                                   |
| `GET`  | `/health/startup-health`      | Quick startup health check.                                          |
| `GET`  | `/health/startup-history`     | Startup performance history.                                         |
| `GET`  | `/health/startup-progress`    | Startup phase progress.                                              |
| `GET`  | `/health/startup-rollback`    | Startup rollback status and controls.                                |
| `GET`  | `/health/startup-status`      | Detailed startup diagnostics.                                        |
| `GET`  | `/health/startup-stream`      | SSE stream for real-time startup progress updates.                   |
| `GET`  | `/health/stream`              | SSE health stream.                                                   |
| `GET`  | `/health/summary`             | Aggregated health summary.                                           |

## Status Router (`/`)

Registered pre-lifespan in `main.py`.

| Method | Path      | Description      |
| ------ | --------- | ---------------- |
| `GET`  | `/live`   | Liveness check.  |
| `GET`  | `/ready`  | Readiness check. |
| `GET`  | `/status` | Server status.   |

## System Router (`/system`)

| Method | Path                               | Description                                                             |
| ------ | ---------------------------------- | ----------------------------------------------------------------------- |
| `GET`  | `/system/battery`                  | Read charge state, charge-cap capability, longevity advice, and policy. |
| `POST` | `/system/battery/limit`            |                                                                         |
| `PUT`  | `/system/battery/policy`           |                                                                         |
| `GET`  | `/system/disk`                     | Disk usage per mount point.                                             |
| `GET`  | `/system/executor`                 | Training executor status.                                               |
| `POST` | `/system/executor/purge`           | Purge completed executor jobs.                                          |
| `GET`  | `/system/executor/{job_id}`        | Training executor job detail.                                           |
| `POST` | `/system/executor/{job_id}/cancel` | Cancel a running executor job.                                          |
| `GET`  | `/system/executor/{job_id}/result` | Training executor job result.                                           |
| `GET`  | `/system/inference-pool`           | Inference thread-pool status.                                           |
| `GET`  | `/system/info`                     | OS, CPU count, memory total.                                            |
| `GET`  | `/system/lifecycle`                | Full health snapshot (model status, inference counts, metrics).         |
| `GET`  | `/system/metrics`                  | Real-time CPU, memory, disk, GPU usage.                                 |
| `GET`  | `/system/output`                   | Tail recent log output.                                                 |
| `GET`  | `/system/stream`                   | SSE system metrics stream.                                              |

## Inference Router (`/inference`, `/chat`, `/context`, `/session`)

| Method   | Path                                    | Description                                         |
| -------- | --------------------------------------- | --------------------------------------------------- |
| `GET`    | `/`                                     |                                                     |
| `POST`   | `/cancel-all`                           |                                                     |
| `POST`   | `/cancel/{op_id}`                       |                                                     |
| `GET`    | `/chat/audio/{session_id}/{message_id}` |                                                     |
| `POST`   | `/chat/control`                         |                                                     |
| `GET`    | `/chat/sessions`                        |                                                     |
| `POST`   | `/chat/sessions`                        |                                                     |
| `GET`    | `/chat/sessions/current`                |                                                     |
| `GET`    | `/chat/sessions/search`                 |                                                     |
| `DELETE` | `/chat/sessions/{session_id}`           |                                                     |
| `GET`    | `/chat/sessions/{session_id}`           |                                                     |
| `PUT`    | `/chat/sessions/{session_id}`           |                                                     |
| `GET`    | `/chat/suggestions`                     |                                                     |
| `GET`    | `/chat/tools`                           |                                                     |
| `POST`   | `/chat/voice/{session_id}`              |                                                     |
| `POST`   | `/context/fact`                         |                                                     |
| `GET`    | `/context/facts`                        | List context facts.                                 |
| `GET`    | `/context/inspect`                      | Inspect context layers.                             |
| `POST`   | `/context/reset`                        | Reset context state.                                |
| `POST`   | `/inference/generate`                   | Non-streaming text generation.                      |
| `POST`   | `/inference/generate/stream`            | Streaming text generation (SSE).                    |
| `GET`    | `/info`                                 |                                                     |
| `GET`    | `/info/soul`                            |                                                     |
| `GET`    | `/operations`                           | List active operations.                             |
| `POST`   | `/operations/purge`                     | Purge completed operations.                         |
| `GET`    | `/providers`                            | List model providers.                               |
| `GET`    | `/v1/models`                            | OpenAI-compatible model discovery (GET /v1/models). |

## Infer Router (`/infer`)

Separate inference endpoint backed by the direct model server.

| Method | Path                | Description                  |
| ------ | ------------------- | ---------------------------- |
| `POST` | `/infer`            | Generate text from a prompt. |
| `POST` | `/infer/detokenize` | Convert token IDs to text.   |
| `POST` | `/infer/embed`      | Compute text embeddings.     |
| `GET`  | `/infer/health`     | Inference engine health.     |
| `GET`  | `/infer/info`       | Loaded model information.    |
| `POST` | `/infer/stream`     | Streaming generation (SSE).  |
| `POST` | `/infer/tokenize`   | Tokenize text to token IDs.  |

## Models Router (`/models`)

| Method   | Path                                            | Description                                                           |
| -------- | ----------------------------------------------- | --------------------------------------------------------------------- |
| `GET`    | `/models`                                       | List available models.                                                |
| `GET`    | `/models/backends`                              |                                                                       |
| `GET`    | `/models/backends/active`                       |                                                                       |
| `POST`   | `/models/backends/active`                       |                                                                       |
| `GET`    | `/models/cache-usage`                           |                                                                       |
| `GET`    | `/models/catalog`                               | Get model catalog.                                                    |
| `GET`    | `/models/catalog/stats`                         | Get catalog statistics.                                               |
| `GET`    | `/models/conversion-status`                     | Get model conversion/download status.                                 |
| `GET`    | `/models/current`                               | Get the currently loaded model.                                       |
| `GET`    | `/models/debug/providers`                       | Diagnostic endpoint: full provider chain status.                      |
| `POST`   | `/models/dequantize`                            | Dequantize a model.                                                   |
| `POST`   | `/models/download`                              | Start downloading a model.                                            |
| `GET`    | `/models/download/qwen-gguf`                    | Download Qwen2.5-0.5B-Instruct GGUF (Q4_K_M) from HuggingFace Hub.    |
| `GET`    | `/models/download/{model_id:path}`              | Get download progress for a specific model.                           |
| `POST`   | `/models/download/{model_id:path}/cancel`       |                                                                       |
| `POST`   | `/models/download/{model_id:path}/pause`        |                                                                       |
| `POST`   | `/models/download/{model_id:path}/resume`       |                                                                       |
| `POST`   | `/models/download/{model_id:path}/retry`        |                                                                       |
| `POST`   | `/models/download/{model_id:path}/verify`       |                                                                       |
| `GET`    | `/models/downloads`                             | List all active downloads.                                            |
| `POST`   | `/models/engine/reload`                         | Reload the model engine.                                              |
| `GET`    | `/models/engine/status`                         | Get engine status.                                                    |
| `POST`   | `/models/export`                                |                                                                       |
| `GET`    | `/models/export/formats`                        | Get available export formats.                                         |
| `POST`   | `/models/external/download`                     |                                                                       |
| `GET`    | `/models/external/models`                       |                                                                       |
| `GET`    | `/models/external/servers`                      |                                                                       |
| `POST`   | `/models/external/servers`                      |                                                                       |
| `DELETE` | `/models/external/servers/{name}`               |                                                                       |
| `GET`    | `/models/file/{model_id:path}/{file_path:path}` |                                                                       |
| `GET`    | `/models/hf`                                    | List HuggingFace available models with actual sizes and cache status. |
| `GET`    | `/models/history`                               |                                                                       |
| `POST`   | `/models/load`                                  | Load a model into memory.                                             |
| `GET`    | `/models/logs`                                  | Get model server logs.                                                |
| `POST`   | `/models/memory-cleanup`                        | Force an immediate memory cleanup cycle.                              |
| `GET`    | `/models/memory-pressure`                       | Return current memory pressure stats (level, thresholds, counters).   |
| `POST`   | `/models/precision`                             | Set model precision.                                                  |
| `GET`    | `/models/process-guard`                         | Get process guard status.                                             |
| `POST`   | `/models/process-guard`                         | Configure process guard.                                              |
| `POST`   | `/models/quantize`                              | Quantize a model.                                                     |
| `POST`   | `/models/unload`                                | Unload the current model.                                             |
| `POST`   | `/models/visual-load`                           |                                                                       |

## Souls Router (`/souls`)

| Method   | Path                                  | Description                      |
| -------- | ------------------------------------- | -------------------------------- |
| `GET`    | `/souls`                              | List available souls.            |
| `POST`   | `/souls/chat`                         | Chat with a soul.                |
| `GET`    | `/souls/current`                      | Get the currently active soul.   |
| `GET`    | `/souls/stats`                        | Soul system statistics.          |
| `POST`   | `/souls/switch`                       | Switch to a different soul.      |
| `GET`    | `/souls/weights`                      | Get current trait weights.       |
| `POST`   | `/souls/weights`                      | Save trait weights.              |
| `GET`    | `/souls/weights/modes`                | List available weight modes.     |
| `DELETE` | `/souls/weights/snapshot/{name}`      | Delete a weight snapshot.        |
| `POST`   | `/souls/weights/snapshot/{name}`      | Create a weight snapshot.        |
| `POST`   | `/souls/weights/snapshot/{name}/load` | Load a weight snapshot.          |
| `GET`    | `/souls/weights/snapshots`            | List weight snapshots.           |
| `GET`    | `/souls/{soul_name}`                  | Get details for a specific soul. |

## Config Router (`/config`)

| Method  | Path                 | Description                         |
| ------- | -------------------- | ----------------------------------- |
| `GET`   | `/config/generation` | Get generation config.              |
| `PATCH` | `/config/generation` | Partially update generation config. |
| `PUT`   | `/config/generation` | Replace generation config.          |

## Auth Router (`/auth`)

| Method | Path             | Description                                                     |
| ------ | ---------------- | --------------------------------------------------------------- |
| `POST` | `/auth/login`    | Authenticate a user with username and password.                 |
| `GET`  | `/auth/me`       | Return the current authenticated user's profile.                |
| `POST` | `/auth/refresh`  | Issue a new JWT token from an existing valid token.             |
| `POST` | `/auth/register` | Register a new user account with username, email, and password. |
| `POST` | `/auth/token`    | Create a JWT access token from a valid API key.                 |
| `POST` | `/auth/verify`   | Verify whether a JWT bearer token is valid and not expired.     |

## Session Router (`/session`)

| Method | Path                               | Description                  |
| ------ | ---------------------------------- | ---------------------------- |
| `POST` | `/session/{session_id}/context`    | Build context for a session. |
| `GET`  | `/session/{session_id}/inspector`  | Inspect session state.       |
| `GET`  | `/session/{session_id}/messages`   | List session messages.       |
| `POST` | `/session/{session_id}/regenerate` | Regenerate last response.    |

## Feedback Router (`/feedback`)

| Method   | Path                                | Description                   |
| -------- | ----------------------------------- | ----------------------------- |
| `POST`   | `/feedback`                         | Record user feedback.         |
| `GET`    | `/feedback/conversations`           | List conversations.           |
| `POST`   | `/feedback/conversations`           | Create a conversation record. |
| `DELETE` | `/feedback/conversations/{conv_id}` | Delete a conversation.        |
| `GET`    | `/feedback/conversations/{conv_id}` | Get a conversation.           |
| `PATCH`  | `/feedback/conversations/{conv_id}` | Update a conversation.        |
| `GET`    | `/feedback/stats/summary`           | Feedback statistics summary.  |
| `POST`   | `/feedback/workflow-record`         | Record workflow feedback.     |
| `GET`    | `/feedback/{message_id}`            | Get feedback for a message.   |

## Knowledge Base Router (`/knowledge`)

| Method   | Path                                    | Description                        |
| -------- | --------------------------------------- | ---------------------------------- |
| `GET`    | `/knowledge`                            | List knowledge items.              |
| `POST`   | `/knowledge`                            | Create a knowledge item.           |
| `GET`    | `/knowledge/adapter-status`             | Adapter training status.           |
| `POST`   | `/knowledge/batch`                      | Batch create items.                |
| `POST`   | `/knowledge/batch-delete`               | Batch delete items.                |
| `POST`   | `/knowledge/bulk-ingest`                | Bulk ingest items.                 |
| `POST`   | `/knowledge/categorize`                 | Auto-categorize items.             |
| `POST`   | `/knowledge/check-duplicate`            | Check for duplicates.              |
| `GET`    | `/knowledge/context`                    | Get knowledge context.             |
| `GET`    | `/knowledge/embedder-status`            | Embedder training status.          |
| `GET`    | `/knowledge/gaps`                       | Identify knowledge gaps.           |
| `POST`   | `/knowledge/ingest-file`                | Ingest from a file upload.         |
| `POST`   | `/knowledge/ingest-url`                 | Ingest from a URL.                 |
| `GET`    | `/knowledge/kg/pipeline-stats`          | Knowledge graph pipeline stats.    |
| `POST`   | `/knowledge/kg/sync`                    | Sync knowledge graph.              |
| `GET`    | `/knowledge/label`                      | Get label info.                    |
| `POST`   | `/knowledge/rag/clear`                  | Clear RAG store.                   |
| `GET`    | `/knowledge/rag/documents`              | List RAG documents.                |
| `POST`   | `/knowledge/rag/ingest`                 | Ingest documents for RAG.          |
| `POST`   | `/knowledge/rag/query`                  | Query the RAG pipeline.            |
| `GET`    | `/knowledge/rag/stats`                  | RAG pipeline statistics.           |
| `POST`   | `/knowledge/rag/verify`                 | Verify RAG results.                |
| `GET`    | `/knowledge/reviews/due`                | Get items due for review.          |
| `POST`   | `/knowledge/reviews/{item_id}/schedule` | Schedule a review.                 |
| `GET`    | `/knowledge/search`                     | Search knowledge items.            |
| `POST`   | `/knowledge/search-files`               | Search across ingested files.      |
| `GET`    | `/knowledge/stats`                      | Knowledge base statistics.         |
| `POST`   | `/knowledge/suggest-topic`              | Suggest a topic.                   |
| `GET`    | `/knowledge/topics`                     | List topics.                       |
| `POST`   | `/knowledge/train-adapter`              | Train a LoRA adapter on knowledge. |
| `POST`   | `/knowledge/train-embedder`             | Train an embedder model.           |
| `DELETE` | `/knowledge/{item_id}`                  | Delete a knowledge item.           |
| `PATCH`  | `/knowledge/{item_id}`                  | Update a knowledge item.           |
| `GET`    | `/knowledge/{item_id}/related`          | Get related items.                 |

## Memory Router (`/memory`)

| Method   | Path                        | Description                                                                                                                           |
| -------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`    | `/memory/archive`           | Return recent task-backed provenance archive records, newest first.                                                                   |
| `POST`   | `/memory/archive/prune`     | Prune archived facts.                                                                                                                 |
| `GET`    | `/memory/archive/stats`     | Archive statistics.                                                                                                                   |
| `GET`    | `/memory/cards`             | List saved memory cards (newest first; `valid` = checksum verified).                                                                  |
| `POST`   | `/memory/cards/load`        | Load a card. Body: `{name, mode?}` — `replace` (default, autosaves current memory first) or `merge` (idempotent, content-hash dedup). |
| `POST`   | `/memory/cards/save`        | Save memory to a portable card. Body: `{name?, overwrite?}`.                                                                          |
| `DELETE` | `/memory/cards/{card_name}` |                                                                                                                                       |
| `POST`   | `/memory/clear`             | Clear all stored facts.                                                                                                               |
| `GET`    | `/memory/config`            | Return the current runtime memory settings.                                                                                           |
| `POST`   | `/memory/config`            |                                                                                                                                       |
| `POST`   | `/memory/consolidate`       | Trigger memory consolidation.                                                                                                         |
| `GET`    | `/memory/list`              | List stored facts (limit 1..1000, default 100).                                                                                       |
| `POST`   | `/memory/remember`          | Extract and store facts from a message. Body: `{message, source?}`.                                                                   |
| `GET`    | `/memory/search`            | Search facts by relevance (requires `q`).                                                                                             |
| `GET`    | `/memory/stats`             | Memory service stats (enabled, total facts, topic buckets).                                                                           |
| `POST`   | `/memory/store`             | Store a fact directly. Body: `{content, source?}`.                                                                                    |
| `DELETE` | `/memory/{item_id}`         | Remove one stored memory item by entry id.                                                                                            |
| `PATCH`  | `/memory/{item_id}`         |                                                                                                                                       |

Fail-closed when `SLO_MEMORY_ENABLED=false`. Cards live in
`data/memory_cards/` (override: `SLO_MEMORY_CARDS_DIR`); setting
`SLO_MEMORY_ACTIVE_CARD=<name>` merge-loads that card automatically after a
successful model load (`.card.json` files are checksummed — corrupt cards are
rejected before any store mutation).

## Datasets Router (`/datasets`)

| Method   | Path                                          | Description                        |
| -------- | --------------------------------------------- | ---------------------------------- |
| `GET`    | `/datasets`                                   | List all registered datasets.      |
| `POST`   | `/datasets`                                   | Create a new dataset.              |
| `POST`   | `/datasets/convert-to-messages`               | Convert dataset to message format. |
| `POST`   | `/datasets/from-chat`                         | Create dataset from chat messages. |
| `POST`   | `/datasets/import/batch`                      | Batch import.                      |
| `POST`   | `/datasets/import/csv`                        | Import from CSV.                   |
| `POST`   | `/datasets/import/github`                     | Import from GitHub.                |
| `POST`   | `/datasets/import/huggingface`                | Import from HuggingFace.           |
| `POST`   | `/datasets/import/isbn`                       | Import by ISBN.                    |
| `POST`   | `/datasets/import/kaggle`                     | Import from Kaggle.                |
| `POST`   | `/datasets/import/local`                      | Import from local path.            |
| `POST`   | `/datasets/import/url`                        | Import from URL.                   |
| `GET`    | `/datasets/search`                            | Search datasets.                   |
| `GET`    | `/datasets/search/books`                      | Search for books.                  |
| `GET`    | `/datasets/search/github`                     | Search GitHub for datasets.        |
| `DELETE` | `/datasets/{dataset_id}`                      | Delete a dataset.                  |
| `GET`    | `/datasets/{dataset_id}`                      | Get a dataset.                     |
| `PATCH`  | `/datasets/{dataset_id}`                      | Update dataset metadata.           |
| `POST`   | `/datasets/{dataset_id}/data`                 | Append rows.                       |
| `POST`   | `/datasets/{dataset_id}/export`               | Export to file.                    |
| `GET`    | `/datasets/{dataset_id}/preview`              | Preview first N rows.              |
| `GET`    | `/datasets/{dataset_id}/quality`              |                                    |
| `GET`    | `/datasets/{dataset_id}/stats`                | Dataset statistics.                |
| `GET`    | `/datasets/{dataset_id}/versions`             | List versions.                     |
| `POST`   | `/datasets/{dataset_id}/versions`             | Create a version snapshot.         |
| `POST`   | `/datasets/{dataset_id}/versions/{timestamp}` | Restore a version.                 |

## Training Router (`/training`)

Unified training control plane. All training operations go through `/training/*`.

Routes are split across focused sub-modules:

| Method   | Path                                         | Description                                                                  |
| -------- | -------------------------------------------- | ---------------------------------------------------------------------------- |
| `GET`    | `/monitor`                                   | Training monitoring status and alerts.                                       |
| `GET`    | `/monitor/alerts`                            | Get training alerts.                                                         |
| `GET`    | `/monitor/metrics`                           | Get training metrics history.                                                |
| `POST`   | `/monitor/reset`                             | Reset training monitor state.                                                |
| `GET`    | `/monitor/resources`                         | Check system resources.                                                      |
| `DELETE` | `/recovery/abandon/{job_id}`                 | Abandon a crashed job.                                                       |
| `GET`    | `/recovery/check`                            | Check for crashed jobs.                                                      |
| `POST`   | `/recovery/recover/{job_id}`                 | Recover an interrupted job.                                                  |
| `GET`    | `/recovery/recoverable`                      | Get recoverable jobs.                                                        |
| `GET`    | `/recovery/stats`                            | Recovery statistics.                                                         |
| `POST`   | `/train`                                     | legacy.py \| Start a training job (legacy).                                  |
| `POST`   | `/train/resolve`                             | legacy.py \| Resolve data path (dry run).                                    |
| `GET`    | `/training/builds`                           | List all training builds.                                                    |
| `GET`    | `/training/checkpoints`                      | List checkpoints.                                                            |
| `POST`   | `/training/checkpoints/compare`              | Same prompt against two checkpoints — neither is served.                     |
| `DELETE` | `/training/checkpoints/{name}`               | Delete a checkpoint.                                                         |
| `GET`    | `/training/checkpoints/{name}/download`      | Download a checkpoint.                                                       |
| `GET`    | `/training/checkpoints/{name}/info`          | Checkpoint info.                                                             |
| `POST`   | `/training/checkpoints/{name}/load`          | Load a checkpoint.                                                           |
| `POST`   | `/training/control/pause`                    | Pause the active job.                                                        |
| `POST`   | `/training/control/reset`                    | Reset training controller.                                                   |
| `POST`   | `/training/control/resume`                   | Resume the paused job.                                                       |
| `POST`   | `/training/control/start`                    | Start training (control plane).                                              |
| `POST`   | `/training/control/stop`                     | Stop the active job.                                                         |
| `POST`   | `/training/distill`                          | execution.py \| Knowledge distillation (teacher→student).                    |
| `POST`   | `/training/export-text`                      | Export job data (text).                                                      |
| `GET`    | `/training/export/{job_id}`                  | Export job data (JSON).                                                      |
| `GET`    | `/training/feed`                             |                                                                              |
| `GET`    | `/training/feeds`                            |                                                                              |
| `GET`    | `/training/finetuned-models`                 | List HF fine-tuned models.                                                   |
| `DELETE` | `/training/finetuned-models/{name}`          | Delete a fine-tuned model.                                                   |
| `POST`   | `/training/finetuned-models/{name}/load`     | Load a fine-tuned model.                                                     |
| `POST`   | `/training/from-feedback`                    | Train from collected feedback data.                                          |
| `POST`   | `/training/from-sessions-start`              | Start from-sessions training.                                                |
| `GET`    | `/training/from-sessions-stream`             | SSE stream from sessions.                                                    |
| `GET`    | `/training/from-sessions/cancel`             | Cancel session-based training.                                               |
| `GET`    | `/training/is-running`                       | Check if training is running.                                                |
| `GET`    | `/training/jobs`                             | List all training jobs.                                                      |
| `POST`   | `/training/jobs/purge`                       | Purge old jobs.                                                              |
| `DELETE` | `/training/jobs/{job_id}`                    | Delete a job.                                                                |
| `GET`    | `/training/jobs/{job_id}`                    | Get a specific job.                                                          |
| `POST`   | `/training/jobs/{job_id}/stop`               | Stop a specific job.                                                         |
| `GET`    | `/training/jobs/{job_id}/summary`            | Get job summary.                                                             |
| `POST`   | `/training/load-adapter`                     | execution.py \| Load a LoRA adapter for inference.                           |
| `GET`    | `/training/log`                              | Training log.                                                                |
| `POST`   | `/training/lora-finetune`                    | execution.py \| LoRA fine-tuning on .slnc models.                            |
| `GET`    | `/training/metrics/export`                   | Export training metrics.                                                     |
| `GET`    | `/training/recommend`                        | Get training configuration recommendations based on dataset characteristics. |
| `POST`   | `/training/start`                            | execution.py \| Start a tracked training job (web UI).                       |
| `GET`    | `/training/status`                           | Training status.                                                             |
| `POST`   | `/training/stop`                             | Stop the active job.                                                         |
| `GET`    | `/training/stream`                           | SSE training stream.                                                         |
| `GET`    | `/training/trends`                           | Get training history trends for the version tracker page.                    |
| `POST`   | `/training/turbo-start`                      | Start turbo training.                                                        |
| `GET`    | `/training/turbo/status`                     | Turbo training status.                                                       |
| `POST`   | `/training/unload-adapter`                   | Unload the active LoRA adapter.                                              |
| `POST`   | `/training/visual-start`                     | execution.py \| Start a VLM fine-tune.                                       |
| `GET`    | `/training/webhooks`                         | List all registered webhooks.                                                |
| `POST`   | `/training/webhooks`                         |                                                                              |
| `GET`    | `/training/webhooks/dead-letters`            | Get dead-lettered webhook deliveries.                                        |
| `GET`    | `/training/webhooks/retry-queue`             | Get pending webhook retries.                                                 |
| `GET`    | `/training/webhooks/stats`                   | Get webhook statistics.                                                      |
| `POST`   | `/training/webhooks/test`                    | Send a test notification to a URL.                                           |
| `DELETE` | `/training/webhooks/{webhook_id}`            | Unregister a webhook.                                                        |
| `GET`    | `/training/webhooks/{webhook_id}`            | Get webhook details (without secret).                                        |
| `GET`    | `/training/webhooks/{webhook_id}/deliveries` | Get delivery log for a webhook.                                              |

### Execution Routes (`execution.py`, `legacy.py`)

### Control Routes (`control.py`)

### Job Routes (`jobs_api.py`)

### Stream & Utility Routes (`router.py`)

### Recovery Routes (`router.py`)

> **Note:** `POST /recovery/recover/{job_id}` runs a preflight before it writes any recovery state and answers **422** `{"error": ...}` when a resume cannot work: no `data_path` recorded on the job, the dataset file is gone, or the job ran a trainer this path cannot restart (distill / LoRA / visual). `GET /recovery/recoverable` applies the same rule, so a job that would 422 is never listed as recoverable. Checkpoint fallback scans only the job's own stem — `checkpoint_dir` is shared (`models/auto-training`), so an unscoped "newest that loads" would adopt another job's weights.

### Finetuned Model Routes (`router.py`)

> **Note:** The legacy `/auto-train/*` endpoints (in `routers/auto_train.py`) are deprecated. They are a parallel implementation, not shims — new clients should use `/training/*` instead. The `/training/stop`, `/training/turbo-start`, and `/training/stream` routes now use `training/sse_stream.py` and `training/turbo_endpoints.py` which delegate to `domains.training.service` (core layer).

## Self-Train Router (`/self-train`)

| Method | Path                 | Description           |
| ------ | -------------------- | --------------------- |
| `POST` | `/self-train/start`  | Start self-training.  |
| `GET`  | `/self-train/status` | Self-training status. |
| `POST` | `/self-train/stop`   | Stop self-training.   |

## Learner Router (`/learn`)

| Method | Path                | Description              |
| ------ | ------------------- | ------------------------ |
| `POST` | `/learn/deploy`     | Deploy trained model.    |
| `POST` | `/learn/evaluate`   | Evaluate trained model.  |
| `GET`  | `/learn/feed`       | Get learning feed.       |
| `POST` | `/learn/feed`       | Add to learning feed.    |
| `POST` | `/learn/ingest`     | Ingest training data.    |
| `POST` | `/learn/ingest-url` | Ingest from URL.         |
| `GET`  | `/learn/knowledge`  | List learned knowledge.  |
| `POST` | `/learn/search`     | Search learning content. |
| `GET`  | `/learn/status`     | Learner status.          |
| `POST` | `/learn/train`      | Train the learner.       |

## Tokenizer Router (`/tokenizer`)

| Method | Path                     | Description                                          |
| ------ | ------------------------ | ---------------------------------------------------- |
| `POST` | `/tokenizer/analyze`     | Analyze token distribution.                          |
| `POST` | `/tokenizer/decompose`   | Decompose text into tokens.                          |
| `POST` | `/tokenizer/detokenize`  | Convert token IDs to string.                         |
| `GET`  | `/tokenizer/merges`      | BPE merge operations.                                |
| `POST` | `/tokenizer/pretokenize` | Pre-tokenize text.                                   |
| `GET`  | `/tokenizer/sample`      | Get sample tokens.                                   |
| `GET`  | `/tokenizer/samples`     | Get sample tokens/words for UI.                      |
| `GET`  | `/tokenizer/stats`       | Vocabulary size, merge count, token-to-id map stats. |
| `POST` | `/tokenizer/tokenize`    | Tokenize string to token IDs.                        |
| `POST` | `/tokenizer/train`       | Train a fresh BPE tokenizer.                         |
| `GET`  | `/tokenizer/vocab`       | Full vocabulary list.                                |

## Token Tree Router (`/token-tree`)

Tree-based BPE tokenizer — training, encoding, embedding/semantic explorer.

| Method   | Path                       | Description                                                  |
| -------- | -------------------------- | ------------------------------------------------------------ |
| `POST`   | `/token-tree/compare`      | Diff two saved trees. Body: `{a, b, top_k}`.                 |
| `POST`   | `/token-tree/decode`       | Decode token IDs. Body: `{ids}` → `{text}`.                  |
| `POST`   | `/token-tree/embedding`    | Inspect token embedding. Body: `{token, top_k}`.             |
| `POST`   | `/token-tree/encode`       | Encode text. Body: `{text}` → `{tokens, ids}`.               |
| `POST`   | `/token-tree/lineage`      | Render merge lineage. Body: `{token}`.                       |
| `POST`   | `/token-tree/load`         | Load a saved tree. Body: `{name}`.                           |
| `GET`    | `/token-tree/matrix`       | Embedding matrix overview (top_k).                           |
| `GET`    | `/token-tree/merges`       | Ranked merge rules (optional `query` filter).                |
| `POST`   | `/token-tree/path`         | Trace greedy trie walk. Body: `{text}` → `{steps, ids}`.     |
| `POST`   | `/token-tree/save`         | Save current tree. Body: `{name}`.                           |
| `GET`    | `/token-tree/saved`        | List saved trees.                                            |
| `DELETE` | `/token-tree/saved/{name}` | Delete a saved tree.                                         |
| `POST`   | `/token-tree/similar`      | Nearest-neighbor tokens. Body: `{token, top_k}`.             |
| `GET`    | `/token-tree/stats`        | Tree summary (vocab, merges, embeddings, compression ratio). |
| `POST`   | `/token-tree/train`        | Train on `{texts?, vocab_size, embed_dim, min_frequency}`.   |
| `GET`    | `/token-tree/vocab`        | Paged vocabulary (limit 1..500, default 50).                 |

Semantic endpoints return `404` for unknown tokens; `embedding` returns `422` when tree has no embeddings.

## Agent Router (`/agents`)

| Method   | Path                         | Description                   |
| -------- | ---------------------------- | ----------------------------- |
| `GET`    | `/agents`                    | List all agents.              |
| `POST`   | `/agents`                    | Create a new agent.           |
| `POST`   | `/agents/orchestrate`        | Orchestrate multi-agent task. |
| `GET`    | `/agents/runs`               | List agent runs.              |
| `GET`    | `/agents/runs/{run_id}`      | Get a specific run.           |
| `DELETE` | `/agents/{agent_id}`         | Delete an agent.              |
| `GET`    | `/agents/{agent_id}`         | Get agent details.            |
| `PUT`    | `/agents/{agent_id}`         | Update an agent.              |
| `POST`   | `/agents/{agent_id}/execute` | Execute an agent.             |

## Multimodal Router (`/multimodal`)

| Method   | Path                                    | Description                                 |
| -------- | --------------------------------------- | ------------------------------------------- |
| `POST`   | `/multimodal/analyze`                   | Analyze an image.                           |
| `POST`   | `/multimodal/ask`                       |                                             |
| `POST`   | `/multimodal/batch-encode-phonemes`     | Batch encode multiple texts to phoneme IDs. |
| `POST`   | `/multimodal/batch-score-pronunciation` | Batch score pronunciation accuracy.         |
| `GET`    | `/multimodal/checkpoints`               | List checkpoints.                           |
| `DELETE` | `/multimodal/checkpoints/{name}`        | Delete a checkpoint.                        |
| `POST`   | `/multimodal/checkpoints/{name}/load`   | Load a checkpoint.                          |
| `POST`   | `/multimodal/decode-phonemes`           | Decode phoneme IDs back to text.            |
| `POST`   | `/multimodal/detect`                    | detect_objects.                             |
| `POST`   | `/multimodal/detect-language`           | Detect the language of input text.          |
| `POST`   | `/multimodal/dpo`                       | Run DPO training.                           |
| `POST`   | `/multimodal/encode-phonemes`           | Encode text to phoneme IDs.                 |
| `POST`   | `/multimodal/generate-image`            | Generate an image.                          |
| `POST`   | `/multimodal/pdf/upload`                | Upload and analyze a PDF.                   |
| `POST`   | `/multimodal/process-video`             | Process a video file.                       |
| `POST`   | `/multimodal/reset`                     | Reset the multimodal engine.                |
| `POST`   | `/multimodal/score-pronunciation`       | Score pronunciation accuracy.               |
| `GET`    | `/multimodal/status`                    | Multimodal engine status.                   |
| `POST`   | `/multimodal/synthesize-speech`         | Text-to-speech synthesis.                   |
| `POST`   | `/multimodal/train`                     | Train on a single image.                    |
| `POST`   | `/multimodal/train-batch`               | Train on a batch of images.                 |
| `POST`   | `/multimodal/train-video`               | Train video captioning.                     |
| `POST`   | `/multimodal/transcribe`                | Transcribe audio.                           |
| `POST`   | `/multimodal/video-infer`               | Infer video captions.                       |
| `POST`   | `/multimodal/visual-dataset`            | Create visual dataset.                      |

## Benchmark Router (`/benchmark`)

| Method | Path                       | Description                                                  |
| ------ | -------------------------- | ------------------------------------------------------------ |
| `POST` | `/benchmark/history/clear` | Clear benchmark history.                                     |
| `GET`  | `/benchmark/metrics`       | Get model metrics.                                           |
| `POST` | `/benchmark/perplexity`    | Calculate perplexity.                                        |
| `GET`  | `/benchmark/program`       | Return current weighted program (config/bench_weights.yaml). |
| `GET`  | `/benchmark/quality`       | Quality metrics.                                             |
| `GET`  | `/benchmark/responses`     | Logged responses.                                            |
| `POST` | `/benchmark/run`           | Run a benchmark.                                             |
| `POST` | `/benchmark/score`         | Score raw results with weighted programmable logic.          |
| `GET`  | `/benchmark/stats`         | Tracker statistics.                                          |
| `GET`  | `/benchmark/{model_id}`    | Benchmark results for a model.                               |

## Companion Router (`/companion`)

| Method   | Path                             | Description                   |
| -------- | -------------------------------- | ----------------------------- |
| `DELETE` | `/companion/`                    | Delete companion.             |
| `GET`    | `/companion/`                    | Get current companion.        |
| `POST`   | `/companion/chat`                | Chat with companion.          |
| `PATCH`  | `/companion/personality`         | Update companion personality. |
| `POST`   | `/companion/personality`         | Set companion personality.    |
| `POST`   | `/companion/preset`              | Apply a preset.               |
| `GET`    | `/companion/presets`             | List available presets.       |
| `POST`   | `/companion/presets`             |                               |
| `DELETE` | `/companion/presets/{preset_id}` |                               |
| `GET`    | `/companion/prompt`              | Get companion system prompt.  |

## Collections Router (`/collections`)

| Method   | Path                                 | Description                   |
| -------- | ------------------------------------ | ----------------------------- |
| `GET`    | `/collections`                       | List all pipelines.           |
| `POST`   | `/collections/collect`               | Direct collect (no pipeline). |
| `POST`   | `/collections/create`                | Create a pipeline.            |
| `POST`   | `/collections/run`                   | Run a pipeline.               |
| `GET`    | `/collections/stats`                 | Pipeline statistics.          |
| `DELETE` | `/collections/{pipeline_id}`         | Delete a pipeline.            |
| `GET`    | `/collections/{pipeline_id}`         | Get a pipeline.               |
| `POST`   | `/collections/{pipeline_id}/collect` | Collect for a pipeline.       |
| `GET`    | `/collections/{pipeline_id}/records` | List pipeline records.        |

## Docstore Router (`/docstore`)

| Method   | Path                                                | Description                     |
| -------- | --------------------------------------------------- | ------------------------------- |
| `GET`    | `/docstore/message-notes`                           |                                 |
| `POST`   | `/docstore/message-notes`                           |                                 |
| `GET`    | `/docstore/message-notes/search`                    |                                 |
| `DELETE` | `/docstore/message-notes/{session_id}/{message_id}` |                                 |
| `DELETE` | `/docstore/{collection}`                            | Clear a collection.             |
| `GET`    | `/docstore/{collection}`                            | List documents in a collection. |
| `POST`   | `/docstore/{collection}/bulk`                       | Bulk upsert documents.          |
| `DELETE` | `/docstore/{collection}/{doc_id}`                   | Delete a document.              |
| `GET`    | `/docstore/{collection}/{doc_id}`                   | Get a document.                 |
| `PATCH`  | `/docstore/{collection}/{doc_id}`                   | Partially update a document.    |
| `PUT`    | `/docstore/{collection}/{doc_id}`                   | Create/replace a document.      |

## Errors Router (`/errors`)

| Method   | Path                  | Description                                                  |
| -------- | --------------------- | ------------------------------------------------------------ |
| `DELETE` | `/errors/clear`       | Clear error logs.                                            |
| `GET`    | `/errors/export`      | Export error logs.                                           |
| `GET`    | `/errors/grouped`     | Get grouped errors.                                          |
| `GET`    | `/errors/log`         | Get OpenCode error log.                                      |
| `POST`   | `/errors/log`         | Log an error.                                                |
| `POST`   | `/errors/logs/ingest` | Ingest error logs.                                           |
| `GET`    | `/errors/recent`      | Get recent errors.                                           |
| `GET`    | `/errors/stream`      | SSE endpoint pushing real-time error events to the frontend. |
| `GET`    | `/errors/trends`      | Error trend analysis.                                        |
| `GET`    | `/errors/unread`      | Get unread error count.                                      |

## Experiments Router (`/experiments`)

| Method   | Path                                      | Description               |
| -------- | ----------------------------------------- | ------------------------- |
| `GET`    | `/experiments`                            | List all experiments.     |
| `POST`   | `/experiments`                            | Create an experiment.     |
| `GET`    | `/experiments/compare`                    |                           |
| `DELETE` | `/experiments/{experiment_id}`            | Delete an experiment.     |
| `GET`    | `/experiments/{experiment_id}`            | Get an experiment.        |
| `POST`   | `/experiments/{experiment_id}/complete`   | Mark experiment complete. |
| `GET`    | `/experiments/{experiment_id}/data`       | Get experiment data.      |
| `POST`   | `/experiments/{experiment_id}/log_metric` | Log a metric.             |
| `POST`   | `/experiments/{experiment_id}/log_param`  | Log a parameter.          |
| `GET`    | `/experiments/{experiment_id}/runs`       | List experiment runs.     |

## Vector Router (`/vector`)

| Method | Path                    | Description              |
| ------ | ----------------------- | ------------------------ |
| `GET`  | `/vector/ingest/status` | Ingestion status.        |
| `POST` | `/vector/init`          | Initialize vector store. |
| `POST` | `/vector/search`        | Search vectors.          |
| `GET`  | `/vector/stats`         | Vector store statistics. |
| `POST` | `/vector/upsert`        | Upsert vectors.          |

## User Adapters Router (`/user-adapters`)

| Method   | Path                              | Description              |
| -------- | --------------------------------- | ------------------------ |
| `GET`    | `/user-adapters`                  | List all user adapters.  |
| `POST`   | `/user-adapters/aggregate-best`   | Aggregate best adapters. |
| `POST`   | `/user-adapters/merge`            | Merge adapters.          |
| `POST`   | `/user-adapters/prune`            | Prune adapters.          |
| `GET`    | `/user-adapters/quality`          | Adapter quality metrics. |
| `DELETE` | `/user-adapters/{user_id}`        | Delete a user adapter.   |
| `GET`    | `/user-adapters/{user_id}`        | Get a user adapter.      |
| `POST`   | `/user-adapters/{user_id}/reset`  | Reset a user adapter.    |
| `POST`   | `/user-adapters/{user_id}/update` | Update a user adapter.   |

## LoRA Eval Router (`/lora-eval`)

| Method | Path                   | Description                   |
| ------ | ---------------------- | ----------------------------- |
| `POST` | `/lora-eval/aggregate` | Aggregate evaluation results. |
| `GET`  | `/lora-eval/history`   | Evaluation history.           |
| `POST` | `/lora-eval/run`       |                               |

## Registry Router (`/registry`)

| Method | Path                          | Description                                    |
| ------ | ----------------------------- | ---------------------------------------------- |
| `GET`  | `/registry/artifacts`         |                                                |
| `GET`  | `/registry/best`              | Get best model for a task.                     |
| `GET`  | `/registry/models`            | List registered models from the live registry. |
| `GET`  | `/registry/models/{model_id}` | Get model details from the live registry.      |
| `GET`  | `/registry/stats`             | Registry statistics.                           |
| `GET`  | `/registry/verify`            |                                                |

## Meta-Weights Router (`/meta-weights`)

| Method | Path                  | Description              |
| ------ | --------------------- | ------------------------ |
| `POST` | `/meta-weights/get`   | Get meta-weights.        |
| `GET`  | `/meta-weights/ping`  | Health probe.            |
| `GET`  | `/meta-weights/stats` | Meta-weights statistics. |

## VM Router (`/vm`)

| Method   | Path                              | Description                       |
| -------- | --------------------------------- | --------------------------------- |
| `GET`    | `/vm/builtins`                    | List built-in assembly programs.  |
| `GET`    | `/vm/info`                        | VM capabilities and limits.       |
| `POST`   | `/vm/run`                         | Run x86 assembly in sandboxed VM. |
| `POST`   | `/vm/session`                     |                                   |
| `DELETE` | `/vm/session/{session_id}`        |                                   |
| `POST`   | `/vm/session/{session_id}/input`  |                                   |
| `GET`    | `/vm/session/{session_id}/stream` |                                   |
| `GET`    | `/vm/training/jobs/{job_id}`      | Training job status.              |
| `POST`   | `/vm/training/jobs/{job_id}/stop` | Stop a VM training job.           |

## Workflow Router (`/workflow`)

| Method | Path                         | Description                |
| ------ | ---------------------------- | -------------------------- |
| `POST` | `/workflow/start`            | Start a workflow.          |
| `GET`  | `/workflow/status`           | Workflow status.           |
| `POST` | `/workflow/stop`             | Stop a workflow.           |
| `POST` | `/workflow/trigger/{action}` | Trigger a workflow action. |

## Shell Router (`/shell`)

| Method | Path                 | Description                 |
| ------ | -------------------- | --------------------------- |
| `POST` | `/shell/exec`        | Execute a shell command.    |
| `POST` | `/shell/exec/stream` | Execute with SSE streaming. |

## Images Router (`/images`)

| Method | Path               | Description            |
| ------ | ------------------ | ---------------------- |
| `GET`  | `/images/gallery`  | List generated images. |
| `POST` | `/images/generate` |                        |
| `GET`  | `/images/styles`   | List available styles. |

## Voice Router (`/voice`)

| Method | Path            | Description             |
| ------ | --------------- | ----------------------- |
| `GET`  | `/voice/status` | Voice synthesis status. |
| `POST` | `/voice/tts`    | Convert text to speech. |

## Files Router (`/files`)

| Method   | Path                      | Description |
| -------- | ------------------------- | ----------- |
| `GET`    | `/files`                  | List files. |
| `GET`    | `/files/search`           |             |
| `POST`   | `/files/upload`           |             |
| `DELETE` | `/files/{file_id}`        |             |
| `GET`    | `/files/{file_id}`        | Get a file. |
| `POST`   | `/files/{file_id}/ingest` |             |

## Security Router (`/security`)

| Method   | Path                             | Description    |
| -------- | -------------------------------- | -------------- |
| `GET`    | `/security/audit`                |                |
| `GET`    | `/security/keys`                 | List API keys. |
| `POST`   | `/security/keys`                 |                |
| `POST`   | `/security/keys/validate`        |                |
| `DELETE` | `/security/keys/{key_id}`        |                |
| `GET`    | `/security/keys/{key_id}`        |                |
| `POST`   | `/security/keys/{key_id}/rotate` |                |

## Rate Limit Router (`/ratelimit`)

| Method | Path                 | Description                            |
| ------ | -------------------- | -------------------------------------- |
| `GET`  | `/rate-limit/check`  | Check if request would be rate limited |
| `GET`  | `/rate-limit/status` | Get current rate limit configuration   |

## World Render Router (`/world`)

| Method | Path                  | Description                      |
| ------ | --------------------- | -------------------------------- |
| `POST` | `/world/neural`       | Process through neural pipeline. |
| `POST` | `/world/render`       | Render the world state.          |
| `POST` | `/world/render/image` | Render as PNG image.             |
| `GET`  | `/world/stats`        | World rendering statistics.      |
| `POST` | `/world/tick`         | Run a simulation tick.           |

## Chat Router (`/chat`)

Defined in `apps/api/server/routers/chat.py` — 7 endpoints.

| Method | Path                            | Description                            |
| ------ | ------------------------------- | -------------------------------------- |
| `POST` | `/chat`                         | Chat (non-streaming).                  |
| `GET`  | `/chat/active`                  |                                        |
| `GET`  | `/chat/addons`                  | List chat processors and capabilities. |
| `GET`  | `/chat/health`                  | Check chat provider availability.      |
| `POST` | `/chat/stream`                  | Chat (SSE streaming).                  |
| `POST` | `/chat/{session_id}/cancel`     |                                        |
| `POST` | `/chat/{session_id}/regenerate` |                                        |

## Cloud Training Router (`/cloud-training`)

Defined in `apps/api/server/routers/cloud_training.py` — 4 endpoints.

| Method | Path                              | Description                         |
| ------ | --------------------------------- | ----------------------------------- |
| `GET`  | `/cloud-training/jobs`            | List recent cloud training jobs.    |
| `POST` | `/cloud-training/submit`          |                                     |
| `POST` | `/cloud-training/{job_id}/cancel` |                                     |
| `GET`  | `/cloud-training/{job_id}/status` | Get status of a cloud training job. |

## Consciousness Router (`/consciousness`)

Defined in `apps/api/server/routers/consciousness.py` — 35 endpoints.

| Method   | Path                                            | Description                                                   |
| -------- | ----------------------------------------------- | ------------------------------------------------------------- |
| `POST`   | `/consciousness/backup`                         | Create a full backup of consciousness state.                  |
| `GET`    | `/consciousness/backup/download`                | Download consciousness backup as a JSON file.                 |
| `POST`   | `/consciousness/backup/import`                  |                                                               |
| `POST`   | `/consciousness/batch`                          |                                                               |
| `POST`   | `/consciousness/clear/beliefs`                  | Reset self-beliefs to defaults.                               |
| `POST`   | `/consciousness/clear/episodes`                 | Clear all episodes from the self-model.                       |
| `PATCH`  | `/consciousness/config`                         |                                                               |
| `GET`    | `/consciousness/evaluate`                       | Evaluate consciousness system quality.                        |
| `POST`   | `/consciousness/feedback`                       |                                                               |
| `GET`    | `/consciousness/health`                         | Quick health check for consciousness system.                  |
| `GET`    | `/consciousness/history/beliefs`                |                                                               |
| `GET`    | `/consciousness/history/episodes`               |                                                               |
| `GET`    | `/consciousness/history/qualia`                 |                                                               |
| `GET`    | `/consciousness/personality`                    | Get the current personality profile.                          |
| `PATCH`  | `/consciousness/personality`                    |                                                               |
| `GET`    | `/consciousness/personality/conflicts`          | Detect personality conflicts and warnings.                    |
| `GET`    | `/consciousness/personality/history`            |                                                               |
| `GET`    | `/consciousness/personality/presets`            | Get available personality presets.                            |
| `POST`   | `/consciousness/personality/presets/apply`      |                                                               |
| `POST`   | `/consciousness/personality/reset`              | Reset personality to defaults.                                |
| `GET`    | `/consciousness/personas`                       | List all saved personas.                                      |
| `POST`   | `/consciousness/personas/save`                  |                                                               |
| `DELETE` | `/consciousness/personas/{persona_id}`          |                                                               |
| `GET`    | `/consciousness/personas/{persona_id}`          |                                                               |
| `POST`   | `/consciousness/personas/{persona_id}/activate` |                                                               |
| `GET`    | `/consciousness/qualia`                         | Get current qualia state and narrative.                       |
| `POST`   | `/consciousness/reflect`                        | Trigger a structured self-reflection and apply belief deltas. |
| `POST`   | `/consciousness/restore`                        |                                                               |
| `POST`   | `/consciousness/seed`                           |                                                               |
| `GET`    | `/consciousness/self-model`                     | Get the self-model state.                                     |
| `GET`    | `/consciousness/stats`                          |                                                               |
| `GET`    | `/consciousness/status`                         | Get current consciousness system status.                      |
| `GET`    | `/consciousness/stream`                         |                                                               |
| `POST`   | `/consciousness/train/start`                    |                                                               |
| `GET`    | `/consciousness/train/status`                   | Get consciousness training status.                            |

## Dashboard Router (`/dashboard`)

Defined in `apps/api/server/routers/dashboard.py` — 3 endpoints.

| Method | Path                 | Description                                               |
| ------ | -------------------- | --------------------------------------------------------- |
| `GET`  | `/dashboard/events`  |                                                           |
| `GET`  | `/dashboard/stream`  |                                                           |
| `GET`  | `/dashboard/summary` | Quick system summary: health, services, active processes. |

## Mobile Router (`/mobile`)

Defined in `apps/api/server/routers/mobile.py` — 33 endpoints.

| Method   | Path                                 | Description |
| -------- | ------------------------------------ | ----------- |
| `GET`    | `/mobile/conversations`              |             |
| `GET`    | `/mobile/conversations/{session_id}` |             |
| `GET`    | `/mobile/dashboard`                  |             |
| `GET`    | `/mobile/health`                     |             |
| `GET`    | `/mobile/knowledge`                  |             |
| `POST`   | `/mobile/knowledge`                  |             |
| `DELETE` | `/mobile/knowledge/{item_id}`        |             |
| `PATCH`  | `/mobile/knowledge/{item_id}`        |             |
| `GET`    | `/mobile/models`                     |             |
| `POST`   | `/mobile/models/switch`              |             |
| `POST`   | `/mobile/notifications/cleanup`      |             |
| `GET`    | `/mobile/notifications/devices`      |             |
| `GET`    | `/mobile/notifications/history`      |             |
| `POST`   | `/mobile/notifications/register`     |             |
| `POST`   | `/mobile/notifications/send`         |             |
| `POST`   | `/mobile/notifications/unregister`   |             |
| `POST`   | `/mobile/notify/training-complete`   |             |
| `POST`   | `/mobile/sync`                       |             |
| `GET`    | `/mobile/sync/status`                |             |
| `POST`   | `/mobile/train`                      |             |
| `PATCH`  | `/mobile/train/auto-config`          |             |
| `GET`    | `/mobile/train/auto-status`          |             |
| `POST`   | `/mobile/train/compact`              |             |
| `GET`    | `/mobile/train/export`               |             |
| `POST`   | `/mobile/train/from-sessions`        |             |
| `DELETE` | `/mobile/train/pair/{pair_id}`       |             |
| `PATCH`  | `/mobile/train/pair/{pair_id}`       |             |
| `GET`    | `/mobile/train/pairs`                |             |
| `DELETE` | `/mobile/train/pairs/bulk`           |             |
| `GET`    | `/mobile/train/pending`              |             |
| `GET`    | `/mobile/train/session/{session_id}` |             |
| `GET`    | `/mobile/train/stats`                |             |
| `DELETE` | `/mobile/train/synced`               |             |

## Model Stack Router (`/model-stack`)

Defined in `apps/api/server/routers/model_stack.py` — 5 endpoints.

| Method   | Path                  | Description                                                                   |
| -------- | --------------------- | ----------------------------------------------------------------------------- |
| `GET`    | `/model-stack`        |                                                                               |
| `POST`   | `/model-stack/base`   |                                                                               |
| `POST`   | `/model-stack/clear`  |                                                                               |
| `POST`   | `/model-stack/push`   | Push a knowledge/dataset/cache layer. Body: {kind, name, path, workspace_id}. |
| `DELETE` | `/model-stack/{name}` |                                                                               |

## Openwebui Router (`/openwebui`)

Defined in `apps/api/server/routers/openwebui.py` — 6 endpoints.

| Method | Path                           | Description                                           |
| ------ | ------------------------------ | ----------------------------------------------------- |
| `POST` | `/openwebui/checkpoint/reload` |                                                       |
| `GET`  | `/openwebui/checkpoints`       | List available model checkpoints for OpenWebUI panel. |
| `GET`  | `/openwebui/datasets`          | List available datasets for OpenWebUI panel.          |
| `POST` | `/openwebui/training/start`    |                                                       |
| `GET`  | `/openwebui/training/status`   | Get current training status.                          |
| `POST` | `/openwebui/training/stop`     | Stop the current training run.                        |

## Phoneme Router (`/phoneme`)

Defined in `apps/api/server/routers/phoneme.py` — 9 endpoints.

| Method | Path                       | Description                |
| ------ | -------------------------- | -------------------------- |
| `POST` | `/phoneme/batch-encode`    |                            |
| `POST` | `/phoneme/decode`          |                            |
| `POST` | `/phoneme/detect-language` |                            |
| `POST` | `/phoneme/encode`          |                            |
| `GET`  | `/phoneme/languages`       | List supported languages.  |
| `POST` | `/phoneme/score`           |                            |
| `GET`  | `/phoneme/status`          | Get phoneme engine status. |
| `POST` | `/phoneme/synthesize`      |                            |
| `POST` | `/phoneme/visualize`       |                            |

## Plugins Router (`/plugins`)

Defined in `apps/api/server/routers/plugins.py` — 4 endpoints.

| Method | Path                             | Description              |
| ------ | -------------------------------- | ------------------------ |
| `GET`  | `/plugins`                       | List all loaded plugins. |
| `POST` | `/plugins/reload`                |                          |
| `POST` | `/plugins/{plugin_name}/disable` |                          |
| `POST` | `/plugins/{plugin_name}/enable`  |                          |

## Profiles Router (`/profiles`)

Defined in `apps/api/server/routers/profiles.py` — 5 endpoints.

| Method | Path                     | Description                                                   |
| ------ | ------------------------ | ------------------------------------------------------------- |
| `GET`  | `/profiles`              | List all available serving profiles.                          |
| `GET`  | `/profiles/active`       | Get the currently active profile ID.                          |
| `POST` | `/profiles/apply`        | Apply a serving profile to the running process.               |
| `GET`  | `/profiles/recommend`    | Auto-detect and recommend the best profile for this hardware. |
| `GET`  | `/profiles/{profile_id}` | Get a single profile by ID.                                   |

## Settings Router (`/settings`)

Defined in `apps/api/server/routers/settings.py` — 41 endpoints.

| Method   | Path                                             | Description                                                           |
| -------- | ------------------------------------------------ | --------------------------------------------------------------------- |
| `GET`    | `/settings`                                      | Get all user settings.                                                |
| `GET`    | `/settings/adaptive`                             | Get adaptive settings.                                                |
| `PATCH`  | `/settings/adaptive`                             |                                                                       |
| `GET`    | `/settings/adaptive/insights`                    | Get adaptive training insights from history.                          |
| `GET`    | `/settings/generation`                           | Get generation settings.                                              |
| `PATCH`  | `/settings/generation`                           |                                                                       |
| `POST`   | `/settings/model-card`                           |                                                                       |
| `GET`    | `/settings/providers/api`                        | Get external API provider settings (key never returned).              |
| `PATCH`  | `/settings/providers/api`                        |                                                                       |
| `POST`   | `/settings/reset`                                | Reset all settings to defaults.                                       |
| `GET`    | `/settings/training`                             | Get training settings.                                                |
| `PATCH`  | `/settings/training`                             |                                                                       |
| `GET`    | `/settings/training/analytics`                   | Get aggregated training analytics for charts and summaries.           |
| `PATCH`  | `/settings/training/auto-train/config`           |                                                                       |
| `GET`    | `/settings/training/auto-train/status`           | Get auto-trainer status and configuration.                            |
| `GET`    | `/settings/training/batch-status`                | Get status of all training jobs (running, queued, completed, failed). |
| `GET`    | `/settings/training/bookmarks`                   | Get all bookmarked training runs.                                     |
| `GET`    | `/settings/training/compare`                     |                                                                       |
| `POST`   | `/settings/training/history/clear`               |                                                                       |
| `GET`    | `/settings/training/history/export`              |                                                                       |
| `GET`    | `/settings/training/presets`                     | List all available training presets.                                  |
| `GET`    | `/settings/training/presets/{preset_name}`       | Get a specific training preset.                                       |
| `POST`   | `/settings/training/presets/{preset_name}/apply` |                                                                       |
| `GET`    | `/settings/training/runs`                        | Filter training runs by model, method, convergence, or quality.       |
| `POST`   | `/settings/training/runs/bulk/bookmark`          |                                                                       |
| `POST`   | `/settings/training/runs/bulk/delete`            |                                                                       |
| `POST`   | `/settings/training/runs/bulk/tag`               |                                                                       |
| `DELETE` | `/settings/training/runs/{run_id}`               |                                                                       |
| `GET`    | `/settings/training/runs/{run_id}`               | Get a single training run by ID.                                      |
| `POST`   | `/settings/training/runs/{run_id}/bookmark`      |                                                                       |
| `POST`   | `/settings/training/runs/{run_id}/duplicate`     |                                                                       |
| `GET`    | `/settings/training/runs/{run_id}/export`        | Export a single training run as JSON or YAML.                         |
| `PUT`    | `/settings/training/runs/{run_id}/notes`         |                                                                       |
| `POST`   | `/settings/training/runs/{run_id}/tags`          |                                                                       |
| `DELETE` | `/settings/training/runs/{run_id}/tags/{tag}`    |                                                                       |
| `GET`    | `/settings/training/tags`                        | Get all unique tags across all training runs.                         |
| `GET`    | `/settings/training/tags/{tag}`                  | Get all training runs with a specific tag.                            |
| `GET`    | `/settings/ui`                                   | Get UI settings.                                                      |
| `PATCH`  | `/settings/ui`                                   |                                                                       |
| `GET`    | `/settings/voice`                                | Get voice settings.                                                   |
| `PATCH`  | `/settings/voice`                                |                                                                       |

## Tenants Router (`/tenants`)

Defined in `apps/api/server/routers/tenants.py` — 6 endpoints.

| Method   | Path                         | Description |
| -------- | ---------------------------- | ----------- |
| `GET`    | `/tenants`                   |             |
| `POST`   | `/tenants`                   |             |
| `DELETE` | `/tenants/{tenant_id}`       |             |
| `GET`    | `/tenants/{tenant_id}`       |             |
| `PUT`    | `/tenants/{tenant_id}`       |             |
| `GET`    | `/tenants/{tenant_id}/stats` |             |

## Tokens Router (`/tokens`)

Defined in `apps/api/server/routers/tokens.py` — 6 endpoints.

| Method | Path                    | Description |
| ------ | ----------------------- | ----------- |
| `GET`  | `/tokens/balance`       |             |
| `POST` | `/tokens/check`         |             |
| `POST` | `/tokens/topup`         |             |
| `POST` | `/tokens/upgrade`       |             |
| `GET`  | `/tokens/usage/history` |             |
| `GET`  | `/tokens/usage/summary` |             |

## Tools Router (`/tools`)

Defined in `apps/api/server/routers/tools.py` — 2 endpoints.

| Method | Path                        | Description |
| ------ | --------------------------- | ----------- |
| `GET`  | `/tools`                    |             |
| `POST` | `/tools/{tool_id}/generate` |             |

## Users Router (`/users`)

Defined in `apps/api/server/routers/users.py` — 8 endpoints.

| Method   | Path                 | Description |
| -------- | -------------------- | ----------- |
| `GET`    | `/users`             |             |
| `POST`   | `/users`             |             |
| `POST`   | `/users/me/password` |             |
| `GET`    | `/users/me/profile`  |             |
| `PUT`    | `/users/me/profile`  |             |
| `DELETE` | `/users/{user_id}`   |             |
| `GET`    | `/users/{user_id}`   |             |
| `PUT`    | `/users/{user_id}`   |             |

## Workspaces Router (`/workspaces`)

Defined in `apps/api/server/routers/workspaces.py` — 29 endpoints.

| Method   | Path                                           | Description                                                           |
| -------- | ---------------------------------------------- | --------------------------------------------------------------------- |
| `GET`    | `/workspaces`                                  |                                                                       |
| `POST`   | `/workspaces`                                  |                                                                       |
| `POST`   | `/workspaces/import`                           | Import workspace data from export format.                             |
| `DELETE` | `/workspaces/{workspace_id}`                   |                                                                       |
| `GET`    | `/workspaces/{workspace_id}`                   |                                                                       |
| `PUT`    | `/workspaces/{workspace_id}`                   |                                                                       |
| `GET`    | `/workspaces/{workspace_id}/activity`          | Recent activity in a workspace (training jobs, member changes, etc.). |
| `POST`   | `/workspaces/{workspace_id}/cleanup`           |                                                                       |
| `POST`   | `/workspaces/{workspace_id}/clone`             |                                                                       |
| `GET`    | `/workspaces/{workspace_id}/export`            | Export all workspace data (members, training jobs, usage).            |
| `GET`    | `/workspaces/{workspace_id}/health`            | Health check for workspace data integrity.                            |
| `POST`   | `/workspaces/{workspace_id}/invite`            |                                                                       |
| `GET`    | `/workspaces/{workspace_id}/members`           |                                                                       |
| `POST`   | `/workspaces/{workspace_id}/members`           |                                                                       |
| `POST`   | `/workspaces/{workspace_id}/members/bulk`      |                                                                       |
| `DELETE` | `/workspaces/{workspace_id}/members/{user_id}` |                                                                       |
| `GET`    | `/workspaces/{workspace_id}/notifications`     |                                                                       |
| `GET`    | `/workspaces/{workspace_id}/permissions`       |                                                                       |
| `GET`    | `/workspaces/{workspace_id}/search`            |                                                                       |
| `GET`    | `/workspaces/{workspace_id}/settings`          |                                                                       |
| `PUT`    | `/workspaces/{workspace_id}/settings`          |                                                                       |
| `POST`   | `/workspaces/{workspace_id}/share`             |                                                                       |
| `DELETE` | `/workspaces/{workspace_id}/share/{share_id}`  |                                                                       |
| `GET`    | `/workspaces/{workspace_id}/shared`            |                                                                       |
| `GET`    | `/workspaces/{workspace_id}/shared/api-keys`   |                                                                       |
| `GET`    | `/workspaces/{workspace_id}/shared/datasets`   |                                                                       |
| `GET`    | `/workspaces/{workspace_id}/shared/knowledge`  |                                                                       |
| `GET`    | `/workspaces/{workspace_id}/stats`             |                                                                       |
| `GET`    | `/workspaces/{workspace_id}/usage`             | Detailed usage metrics for a workspace.                               |

## OpenAPI Specification

| Method | Path            | Description                      |
| ------ | --------------- | -------------------------------- |
| `GET`  | `/openapi.json` | Export the OpenAPI spec as JSON. |

FastAPI automatically provides the OpenAPI spec at `/openapi.json`:

```
curl http://localhost:8000/openapi.json
```

The spec includes all routes described above, request/response schemas, and example payloads.
