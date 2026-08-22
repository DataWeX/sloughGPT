# API Routers Documentation

This document provides a brief overview of the newly added backend routers and their endpoints. All routes are served by the FastAPI application and are accessible under the base URL (e.g., `http://localhost:8000`).

## System Router (`/system`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/system/info` | Returns general system information (OS, CPU count, memory, etc.). |
| `GET` | `/system/metrics` | Returns real‑time metrics (CPU, memory, disk, GPU usage). |
| `GET` | `/system/disk` | Disk usage details per mount point. |
| `GET` | `/system/lifecycle` | Full health snapshot combining model status, inference counts, and system metrics.

## Tokenizer Router (`/tokenizer`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/tokenizer/stats` | Vocabulary size, merge count, token‑to‑id map stats. |
| `POST` | `/tokenizer/tokenize` | Accepts a string and returns an array of token IDs. |
| `POST` | `/tokenizer/detokenize` | Accepts an array of token IDs and returns the reconstructed string. |
| `GET` | `/tokenizer/vocab` | Returns the full vocabulary list. |
| `GET` | `/tokenizer/merges` | Returns BPE merge operations. |
| `GET` | `/tokenizer/samples` | Returns a handful of sample tokens/words for UI playgrounds. |
| `POST` | `/tokenizer/train` | Trains a fresh BPE tokenizer on the provided corpus. |

## Token Tree Router (`/token-tree`)

The TokenTree router exposes the tree-based BPE tokenizer (`packages/core-py/domains/training/token_tree.py`) — training, encoding, and the embedding/semantic explorer surfaces. Unless stated otherwise, request bodies are pydantic models and a missing required field returns `422`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/token-tree/stats` | Tree summary: vocab size, merge count, base tokens, embedding points/compression ratio, embed dim, trained flag. |
| `GET` | `/token-tree/vocab?limit=&offset=` | Paged vocabulary slice; each entry `{id, token, freq, is_special, is_merged}` (limit 1..500, default 50). |
| `GET` | `/token-tree/merges?top_n=&query=` | Ranked merge rules (`{rank, left, right, token, count}`); optional `query` filters by substring. |
| `GET` | `/token-tree/saved` | List saved trees (`{trees: [...]}` with name/path/vocab_size/num_merges/trained/saved_at). |
| `GET` | `/token-tree/matrix?top_k=` | Embedding-matrix overview: `{matrix, norm_min, norm_mean, norm_max, dead_tokens, live_tokens, most_energetic, least_energetic}`. `matrix` is `[rows, cols]` or `null` when embeddings are disabled. |
| `POST` | `/token-tree/save` | Save the current tree under a name. Body: `{name}`. |
| `POST` | `/token-tree/load` | Load a saved tree as current. Body: `{name}`. |
| `POST` | `/token-tree/train` | Train on `{texts?, vocab_size, embed_dim, min_frequency}`; empty `texts` uses the built-in corpus. |
| `POST` | `/token-tree/similar` | Nearest-neighbor tokens via generated embeddings. Body: `{token, top_k}`. |
| `POST` | `/token-tree/embedding` | Inspect one token's embedding: `{token, top_k}` → `{token, id, dim, norm, top, embedding_points, compression_ratio}`. |
| `POST` | `/token-tree/encode` | Encode text to token ids. Body: `{text}` → `{tokens, ids}`. |
| `POST` | `/token-tree/path` | Trace the encoder's greedy trie walk. Body: `{text}` → `{steps, ids}`. |
| `POST` | `/token-tree/decode` | Decode token ids to text. Body: `{ids}` → `{text}`. |
| `POST` | `/token-tree/lineage` | Render a token's merge lineage down to character leaves. Body: `{token}` → `{token, leaves, tree}`. |
| `DELETE` | `/token-tree/saved/{name}` | Delete a saved tree's files. |
| `POST` | `/token-tree/compare` | Diff two saved trees without changing the current one. Body: `{a, b, top_k}` → `{a, b, shared_tokens, only_a_tokens, only_b_tokens, shared_merges, only_a_merges, only_b_merges, shared_examples, only_a_examples, only_b_examples}` (each side carries `{name, stats, vocab}`; examples are `[token, freq]` pairs). |

Semantic-lookup endpoints (`similar`, `embedding`, `lineage`) return `404` for unknown tokens; `embedding` additionally returns `422` when the tree was trained without embeddings.

## Dataset Router (`/datasets`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/datasets` | List all registered datasets. |
| `POST` | `/datasets` | Create a new dataset entry. |
| `DELETE` | `/datasets/{id}` | Delete a dataset. |
| `PATCH` | `/datasets/{id}` | Update dataset metadata. |
| `POST` | `/datasets/{id}/data` | Append rows to a dataset. |
| `GET` | `/datasets/{id}/preview` | Get a preview (first N rows). |
| `POST` | `/datasets/{id}/export` | Export the dataset to a downloadable file. |

## Model Router (`/models`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/models` | List available models. |
| `POST` | `/models/load` | Load a model into memory. |
| `POST` | `/models/unload` | Unload the currently loaded model. |
| `GET` | `/models/health` | Model health/status (loaded, device, inference count). |

## Memory Router (`/memory`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/memory/stats` | Memory service stats (enabled, total facts, topic buckets, visited URLs). |
| `GET` | `/memory/list?limit=` | List stored facts (clamped 1..1000, default 100). |
| `GET` | `/memory/search?q=&limit=` | Search facts by relevance (requires `q`, limit clamped 1..100). |
| `POST` | `/memory/store` | Store a fact directly. Body: `{ "content": str, "source": str? }`. |
| `POST` | `/memory/remember` | Extract and store facts from a message. Body: `{ "message": str, "source": str? }`. |
| `POST` | `/memory/config` | Update runtime memory settings. Body: `{ "enabled": bool?, "archive_retention_days": float? }` — omitted fields are unchanged; returns the full settings snapshot. |
| `GET` | `/memory/config` | Return the current runtime memory settings snapshot (enabled, min_chars, max_facts, store_path, sync_remember, consolidation_threshold, maintenance_interval_minutes, archive_retention_days). |
| `POST` | `/memory/clear` | Clear all stored facts. |

Request bodies use pydantic models — a missing required field returns `422`. A blank/omitted `q` on `/memory/search` returns `400`. The service is fail-closed when `SLO_MEMORY_ENABLED=false` (`stats` returns `{"enabled": false}`; mutating/list endpoints return `503`).

All other existing routers (e.g., `/chat`, `/auto-train`, `/feedback`) retain their previous documentation in the autogenerated FastAPI docs (`/docs`).

## Inference Router (`/inference`, `/chat`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/inference/generate` | Generate text from a prompt. |
| `POST` | `/chat` | Chat with the model (non-streaming). |
| `POST` | `/chat/stream` | Chat with the model (SSE streaming). |
| `GET` | `/context/inspect` | Inspect the current context state. |
| `POST` | `/session/{id}/regenerate` | Regenerate the last response. |

## Souls Router (`/souls`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/souls` | List available souls. |
| `GET` | `/souls/current` | Get the currently active soul. |
| `POST` | `/souls/switch` | Switch to a different soul. |
| `GET` | `/souls/stats` | Soul system statistics. |
| `POST` | `/souls/weights` | Save trait weights. |

## Training Router (`/training`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/training/start` | Start a training job. |
| `GET` | `/training/jobs` | List all training jobs. |
| `GET` | `/training/jobs/{job_id}` | Get a specific training job. |
| `POST` | `/training/pause` | Pause the active training job. |
| `POST` | `/training/resume` | Resume the paused training job. |
| `POST` | `/training/stop` | Stop the active training job. |

## Auto-Train Router (`/auto-train`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/auto-train/start` | Start an auto-training session. |
| `GET` | `/auto-train/status` | Get auto-training status. |
| `POST` | `/auto-train/stop` | Stop the auto-training session. |
| `GET` | `/auto-train/checkpoints` | List available checkpoints. |
| `POST` | `/auto-train/checkpoints/{name}/load` | Load a checkpoint. |

## Feedback Router (`/feedback`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/feedback/record` | Record user feedback. |
| `POST` | `/feedback/workflow-record` | Record feedback for the workflow. |
| `GET` | `/feedback/history` | Get feedback history. |

## Shell Router (`/shell`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/shell/exec` | Execute a shell command. |
| `POST` | `/shell/exec/stream` | Execute a shell command with SSE streaming. |

## VM Router (`/vm`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/vm/status` | Get VM status. |
| `POST` | `/vm/start` | Start the VM. |
| `POST` | `/vm/stop` | Stop the VM. |
| `POST` | `/vm/exec` | Execute a VM instruction. |
| `GET` | `/vm/memory` | Get VM memory dump. |

## Benchmark Router (`/benchmark`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/benchmark/run` | Run a benchmark. |
| `GET` | `/benchmark/metrics` | Get model metrics. |
| `GET` | `/benchmark/{model_id}` | Get benchmark results for a model. |
| `POST` | `/benchmark/perplexity` | Calculate perplexity. |
| `GET` | `/benchmark/quality` | Get quality metrics. |
| `GET` | `/benchmark/responses` | Get logged responses. |
| `GET` | `/benchmark/stats` | Get tracker statistics. |
| `POST` | `/benchmark/clear-history` | Clear benchmark history. |

## Companion Router (`/companion`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/companion` | Get the current companion. |
| `POST` | `/companion/set-personality` | Set companion personality. |
| `POST` | `/companion/chat` | Chat with the companion. |
| `GET` | `/companion/stats` | Get companion statistics. |
| `POST` | `/companion/reset` | Reset the companion. |

## Collections Router (`/collections`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/collections` | List all collections. |
| `POST` | `/collections` | Create a collection. |
| `DELETE` | `/collections/{id}` | Delete a collection. |
| `POST` | `/collections/{id}/items` | Add an item to a collection. |
| `DELETE` | `/collections/{id}/items/{item_id}` | Remove an item from a collection. |

## World Render Router (`/world`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/world/render` | Render the world state. |
| `POST` | `/world/render/image` | Render the world as a PNG image. |
| `POST` | `/world/neural` | Process the world through the neural pipeline. |
| `POST` | `/world/tick` | Run a simulation tick. |
| `GET` | `/world/stats` | Get world rendering statistics. |

## Images Router (`/images`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/images/generate` | Generate an image from text. |
| `GET` | `/images/styles` | List available styles. |
| `POST` | `/images/variations` | Generate variations of an image. |

## Voice Router (`/voice`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/voice/tts` | Convert text to speech. |
| `GET` | `/voice/status` | Get voice synthesis status. |

## Experiments Router (`/experiments`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/experiments` | Create a new experiment. |
| `GET` | `/experiments` | List all experiments. |
| `GET` | `/experiments/{id}` | Get a specific experiment. |
| `POST` | `/experiments/{id}/log-metric` | Log a metric. |
| `POST` | `/experiments/{id}/log-param` | Log a parameter. |
| `POST` | `/experiments/{id}/complete` | Complete an experiment. |

## Learner Router (`/learner`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/learner` | Get learner state. |
| `POST` | `/learner/ingest` | Ingest training data. |
| `POST` | `/learner/train` | Train the learner. |
| `GET` | `/learner/stats` | Get learner statistics. |
| `POST` | `/learner/reset` | Reset the learner. |

## OpenAPI Specification

FastAPI automatically provides the OpenAPI spec at `/openapi.json`. You can retrieve it directly:
```
curl http://localhost:8000/openapi.json
```
The spec includes all routes described above, request/response schemas, and example payloads.
