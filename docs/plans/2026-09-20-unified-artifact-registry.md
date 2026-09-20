# Unified Artifact Registry — design

Date: 2026-09-20. Status: Stage 1 built (read-only index +
`GET /registry/artifacts` + `GET /registry/verify` + content fingerprints);
stages 2–3 open.

## Problem

Artifact lookup is fragmented. Each kind has its own finder with its own
filename conventions and roots: datasets (`input.txt`/`corpus.jsonl` in
`data/`, `data/datasets/`, `datasets/`, just-cache), checkpoints
(`find_checkpoint` over `CHECKPOINTS_DIR`/`TURBO_DIR`), models
(`ModelCatalog`), downloads (per-backend cache dirs + `.manifest.json`
sidecars). Gaps already caused real outages: turbo rejected valid datasets,
the catalog listed 3 of 80 entries, `REPO_ROOT` pointed outside the repo.

## What already exists (reuse, do not duplicate)

| Piece                | Location                                                                                                       | Reuse as                                                                                             |
| -------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Download manifests   | `domain/infrastructure/_internal/external_download.py` (+ `git_download.py`, `local_download.py`, `hf_hub.py`) | Per-resource `.manifest.json` (files, sizes, sha256), `is_cached`, `/`-to-`__` safe names            |
| Entry classification | `domain/training/_internal/cache_tags.py`                                                                      | `classify_kind` (dataset/adapter/system/media), `entry_tags`, `find_corpus_file`, `resolve_in_cache` |
| Model metadata store | `domain/infrastructure/_internal/model_catalog.py`                                                             | MogDB + JSON-sync persistence pattern, unique-id index                                               |
| Dataset listing      | `apps/api/server/controllers/datasets.py`                                                                      | Multi-root scan (`_entry_roots`)                                                                     |
| Corpus resolution    | `apps/api/server/training/resolution.py::resolve_legacy_corpus_path`                                           | Single shared lookup for training inputs                                                             |
| Checkpoint lookup    | `domain/training/_internal/checkpoints.py::find_checkpoint`                                                    | Checkpoint resolution                                                                                |
| Crash-safe store     | `apps/api/server/training/job_store.py` (`PersistentTrainingJobs`)                                             | Live-object + MogDB-write-through wrapper semantics                                                  |

## Proposal

New module `domain/infrastructure/_internal/artifact_registry.py`:

- **Kinds**: `model`, `dataset`, `checkpoint`, `adapter`, `weight`, `download`, `file`.
- **Record**: `artifact_id` (unique, e.g. `dataset:world-happiness`), `kind`,
  `name`, `path`, `size_bytes`, `checksum` (when known), `source`
  (`cache`|`data`|`download`|`training`), `tags`, kind-specific `meta`,
  `created_at`, `updated_at`.
- **Scan**: walk the existing roots (`models/`, `data/`, `data/datasets/`,
  just-cache `external/`, `data/user_adapters/`, checkpoint dirs) using the
  existing finders above. No new filename conventions.
- **API**: `list(kind?, tags?)`, `resolve(kind, id) -> path`, `register(record)`
  called from existing write paths (download complete, import, training
  checkpoint save). Read paths switch to `resolve()` one caller at a time.
- **Persistence**: MogDB collection + JSON sync dir, same as `ModelCatalog`.

## Staging (thin first)

1. Read-only index + `resolve()` + one `GET /registry/artifacts` endpoint.
   No behavior change for existing callers.
2. `register()` on write paths (download/import/train/checkpoint).
3. Migrate callers off bespoke finders; sunset duplicates only when the
   registry covers their tests.

## Non-goals

- No new storage layout. No new filename conventions. No training-loop
  changes. No frontend changes in stage 1.

## Open questions

- Dedupe by content hash across kinds (e.g. same `.soul` as checkpoint and
  model)? Deferred to stage 2.
- `weight` vs `model` boundary (safetensors shards vs cataloged models)?
  Weights stay loader-internal (`WeightLoaderRegistry`) until stage 2.
- Who owns `download` records after import converts them (download backend
  vs importer)? Importer registers the converted artifact, links source id.
