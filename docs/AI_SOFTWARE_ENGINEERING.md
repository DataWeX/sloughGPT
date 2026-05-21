# AI software engineering — practices for this codebase

This document describes how SloughGPT is structured for **reliable, observable, and evolvable** AI features. It is a living guide, not a checklist to complete in one pass.

## Architecture layers

1. **HTTP boundary** — FastAPI routes validate and serialize I/O (`apps/api/server/training/router.py`, standards-based `/v1/infer`). No heavyweight model logic here beyond orchestration.
2. **Domain** — Training pipelines, manifests, evaluation (`packages/core-py/domains/training/`, `packages/core-py/domains/ml_infrastructure/`). Pure-ish logic, testable without the server.
3. **Model runtime** — Loading, generation, quantization (API globals and `packages/core-py/domains/`). Keep a single ownership path for “what model is loaded” to avoid split brain.
4. **Clients** — Web (`apps/web/lib/api.ts`), Python SDK (`packages/sdk-py/sloughgpt_sdk/`), TypeScript SDK (`packages/sdk-ts/typescript-sdk/`): mirror server field names for JSON (`snake_case` from Pydantic).

Refactor direction: **shrink `apps/api/server/main.py`** by moving more `APIRouter` modules under `apps/api/server/<domain>/` (see `apps/api/server/training/` for the pattern: `schemas.py`, `resolution.py`, `jobs.py`, `router.py`).

## Data and contracts

- **Dataset manifests** — Versioned metadata (`packages/standards/standards/v1/`); resolve to a single training file via `resolve_training_inputs`.
- **Trainer checkpoints** — Native `step_*.pt` bundles embed char vocabulary for fair `cli.py eval`; deployment exports (GGUF, `.sou`, etc.) are not drop-in for the same eval path without alignment — `docs/policies/CONTRIBUTING.md` (*Checkpoint vocabulary*).
- **Inference envelope** — `POST /v1/infer` for structured requests, tracing hooks, and future policy/retrieval fields.
- **Reject ambiguity** — Exactly one of `dataset` | `manifest_uri` | `dataset_ref` for training bodies; validate at the schema layer.

## Observability and operations

- **Metrics** — `GET /metrics/prometheus` for scrape-friendly series; HTTP middleware for request stats.
- **Logging** — Use module loggers (`logging.getLogger("sloughgpt")`); avoid `print` in server paths (background training logs exceptions).
- **Tracing** — Standard inference carries `trace_id`; propagate in new endpoints and SDK calls where useful.

## Safety and configuration

- **Secrets** — Loaded via `apps/api/server/settings.py` (`get_security_settings()`): primary API key, JWT secret, optional multi-key list (`SLOUGHGPT_*` env vars; `SLAUGHGPT_*` still honored as legacy fallbacks). Never commit real keys.
- **Auth** — Enforce consistently on mutating routes when exposing publicly.
- **Prompt/PII** — Treat prompts as sensitive data in logs and analytics; document retention if you add persistence.

## Testing

- **API** — `tests/test_server_main_api.py` for training resolve, standard infer, metrics.
- **Domain** — Prefer tests under `tests/` that call `packages/core-py/domains/` without bringing up the full app when possible.
- **Client** — SDK list/coercion helpers covered in `tests/test_sdk.py`.

## Naming and style

See `.cursor/rules/naming-conventions.mdc` for file and symbol conventions across Python and TypeScript.
