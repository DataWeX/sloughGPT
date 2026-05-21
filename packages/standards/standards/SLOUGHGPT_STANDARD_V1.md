# SloughGPT Standard v1

Version 1 defines **machine-readable contracts** so the same platform can run **clinical / synthetic-medical workloads** and **codebase (e.g. GitHub) training** with shared lineage, evaluation gates, and safety policies.

## Design goals

- **One inference envelope** for HTTP APIs (traceability, safety level, structured extras).
- **One dataset manifest** for every training or RAG corpus (provenance, license, PII stance).
- **One training job contract** so schedulers, CLI, and `POST /train` style jobs share fields.
- **Declarative safety** (YAML) separate from model weights so policy can change without retraining.

## Layout in this repository

| Artifact | Path |
|----------|------|
| JSON Schemas | `standards/v1/schemas/*.json` |
| Example manifests | `standards/v1/examples/` |
| This document | `standards/SLOUGHGPT_STANDARD_V1.md` |

## 1. Inference envelope (v1)

### Canonical request

Clients SHOULD send:

- **`trace_id`**: end-to-end id (generate if absent; server may echo).
- **`tenant_id`**: isolation / billing / policy namespace.
- **`model_id`**: registered artifact (e.g. registry id or file id).
- **`task_type`**: drives prompts, output schema, and safety profile (`enum` in schema).
- **`mode`**: `generate` | `chat` | `structured` (structured = JSON output validated against `output_schema_ref`).
- **`safety.level`**: `clinical_strict` | `standard` | `internal_dev` | `code_only`.
- **`input`**: mode-specific payload (`prompt` or `messages` plus optional `context` / `retrieval_query`).

This **extends** existing FastAPI bodies such as `GenerateRequest` and `ChatRequest` (`apps/api/server/main.py`): add optional top-level fields so old clients keep working.

### Canonical response

Servers SHOULD return:

- **`trace_id`**, **`model_id`**, **`model_version`** (commit, registry revision, or hash).
- **`output`**: text and/or parsed JSON (when `mode=structured`).
- **`usage`**: tokens, latency_ms.
- **`safety_flags`**: list of rule ids triggered (never empty on block/refusal).
- **`citations`**: optional list of `{source_id, snippet, span}` for RAG / medical sources.

Schema: `standards/v1/schemas/inference_request.json`, `inference_response.json`.

## 2. Dataset manifest (v1)

Every dataset version used for training or indexing MUST be describable by a **dataset manifest**:

- **`dataset_id`**, **`version`** (semver or content hash).
- **`domain`**: `medical` | `code` | `mixed` | `general`.
- **`sources`**: list of `{type, uri, commit_sha?, license_spdx?, filters?}`.
  - GitHub: `type=github`, `uri=owner/repo`, `commit_sha` mandatory for reproducibility.
- **`pii_policy`**: `none_expected` | `may_contain_phi` | `synthetic_only` | `deidentified`.
- **`splits`**: train/val/test paths or glob patterns + **`split_hash`** when pre-split.
- **`transformations`**: ordered steps (tokenize, chunk, redact, ast_chunk, etc.) with versions.

Align with existing `config/datasets.yaml` (weights, sources) and `domains/training/dataset_manager.py` (`DatasetRegistry` / `meta`): manifests are the **versioned, portable** layer; local folders stay the runtime artifact.

Schema: `standards/v1/schemas/dataset_manifest.json`.  
Example: `standards/v1/examples/dataset_manifest.github_code.example.json`.

## 3. Training job (v1)

Standard fields for orchestration (CLI, web training UI, `TrainRequest` / `TrainingRequest` in `apps/api/server/training/schemas.py`):

- **`job_id`**, **`name`**, **`status`**.
- **`dataset_ref`**: `{dataset_id, version}` pointing to a manifest.
- **`model_spec`**: base model, LoRA flag, adapter id.
- **`hyperparameters`**: epochs, batch_size, lr, max_steps, **`log_interval`**, **`eval_interval`** (tracked HTTP jobs: how often **`train_loss`** / **`eval_loss`** refresh on **`GET /training/jobs`**), …
- **`outputs`**: where checkpoints go; **`registry_target`** for promotion.
- **`gates`**: required eval suite ids and threshold handles (pass/fail before `register`).

Schema: `standards/v1/schemas/training_job.json`.

## 4. Safety policy (YAML, v1)

Policies are **named rules** applied at inference (and optionally at dataset export):

- **`policies[].id`**, **`scope`** (`input` | `output` | `both`).
- **`domains`**: which `task_type` / `safety.level` they apply to.
- **`actions`**: `block` | `redact` | `warn` | `require_disclaimer` | `require_citations`.
- **`patterns` or `classifiers`**: pluggable hooks (`phi_scanner_v1`, `code_secret_scanner_v1`).

Examples: `standards/v1/examples/safety_policy.medical.yaml`, `safety_policy.code.yaml`.

## 5. Recommended task_type registry (v1)

| `task_type` | Use | Notes |
|-------------|-----|--------|
| `clinical_assist` | Q&A on real patient-context data | `safety.level=clinical_strict`, citations strongly recommended |
| `clinical_summarization` | Summaries of notes | same; structured outputs for sections |
| `synthetic_patient_generation` | Simulated cases only | `pii_policy=synthetic_only`; block real identifiers |
| `medical_rag_qa` | Guideline-grounded answers | citations mandatory in policy |
| `code_completion` | IDE-style | `safety.level=code_only`; secret scanner on output |
| `code_rag_qa` | Repo-grounded explanation | manifest tied to commit SHA |
| `github_repo_finetune_prep` | Data ingest only | produces manifest + shards, not necessarily user-facing |

Extend as needed; unknown `task_type` MUST be rejected or defaulted with explicit `safety.level`.

## 6. Implementation checklist (repository)

1. **Inference**: `POST /v1/infer` in `apps/api/server/main.py` accepts `StandardInferenceRequest` and returns `StandardInferenceResponse` (header `X-SloughGPT-Standard: 1`). Legacy `POST /generate` is unchanged and shares `_generate_core`.
2. **Pydantic**: v1 models live in `apps/api/server/main.py` (`StandardInferenceRequest`, etc.); optional fields on `GenerateRequest` / `ChatRequest` remain future work.
3. **Middleware**: `trace_id` from body or `X-Trace-Id`; audit event `v1_infer`.
4. **Training**: `POST /train` and `POST /training/start` accept exactly one of **`dataset`**, **`manifest_uri`**, or **`dataset_ref`**. Optional **`log_interval`** / **`eval_interval`** on those JSON bodies tune live metric cadence on **`GET /training/jobs`**. Use **`POST /train/resolve`** to validate a manifest and preview `data_path` / checkpoint stem without training. Resolution: `domains/training/dataset_manifest.py` (`resolve_training_data_path`) plus `_resolve_training_inputs` in `apps/api/server/main.py`. Trainer-native **`step_*.pt`** on the API host embed **`stoi`** / **`itos`** / **`chars`** for char-LM eval parity with **`cli.py train`**; see **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*).
5. **CLI**: `python3 scripts/validate_dataset_manifest.py <path-to-manifest.json> [--resolve]` (runtime resolution). **Schema CI**: `python3 scripts/validate_standards_schemas.py` (`jsonschema` via `python3 -m pip install -e ".[dev]"` or `python3 -m pip install jsonschema`; example JSON under `standards/v1/examples/`).

## 7. Versioning

- **Standard revision**: bump `standards/SLOUGHGPT_STANDARD_V1.md` header and schemas under `standards/v1/`; keep old schemas as `standards/v0/` if needed later.
- **Breaking changes**: new major folder `standards/v2/`; servers should advertise supported revision via `GET /openapi.json` extension or `X-SloughGPT-Standard: 1`.

---

*This is a specification for the SloughGPT repo; adopt governance (legal/clinical sign-off) before production medical use.*
