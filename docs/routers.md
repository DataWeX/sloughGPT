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

## OpenAPI Specification

FastAPI automatically provides the OpenAPI spec at `/openapi.json`. You can retrieve it directly:
```
curl http://localhost:8000/openapi.json
```
The spec includes all routes described above, request/response schemas, and example payloads.
