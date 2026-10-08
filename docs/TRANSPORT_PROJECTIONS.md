# Transport Projections — capability → contract → transport

> **Doctrine:** the _Endpoint & Transport Rule_ in the root
> [AGENTS.md](../AGENTS.md). **How to write a handler today:** the _Router
> Playbook_ in [PRODUCT_ENGINEERING.md](PRODUCT_ENGINEERING.md#router-playbook).
> **Endpoint inventory:** [routers.md](routers.md).
> This file is the pattern between them — what a capability _is_ before it
> becomes a route, and why "endpoint" stops being a noun you build.

## Two wrong answers, one mistake

"How does the web reach core?" has two tempting shapes. Both are conveyor-belt
thinking:

| Shape                           | What it is                          | What it costs                                                                                                     |
| ------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **N hand-laid feature routers** | one belt per feature                | the contract written N times (handler, client, docs, tests); every capability becomes an endpoint _project_       |
| **One generic tool surface**    | `GET /tools` + `POST /tools/{name}` | a god-endpoint: kills GET/POST semantics, per-route auth scopes, OpenAPI paths, cacheability, route-level metrics |

Same mistake: **transport treated as something you build.** The fix is not a
better belt — it is to stop confusing three different things that a hand-written
router fuses into one: the **contract** (what a capability means), the
**transport** (how a process boundary is crossed), and the **registration**
(how the app learns the capability exists).

## The pattern: descriptor → projection → registration

**1. Contract — declared once, per module.** A capability is a descriptor:
`{name, params (JSON Schema), result, auth_scope, idempotent, version}`. It is
the single source of truth; routes, clients, docs and tests are _derived_ from
it (the contract-first / OpenAPI / JSON-Schema school). In this repo that
descriptor is `ToolSpec` (`domain/agents/_internal/tools.py`): `parameters` is
stored, `params` derives the JSON Schema view — one declaration, two views,
never stored twice.

**2. Projections — adapters that emit the transport.** Real routes, generated
rather than hand-written: a module declares its descriptor, `create_router(spec, route, handler)`
helper emits the fragment — `GET /tools/doctor/report` for reads,
`POST /tools/doctor/check` for actions — with proper verbs, status codes,
`success_response` envelopes and `classify_and_raise` errors, all from the one
entry. FastAPI composes the fragments (`include_router` → `app.openapi()`
merges), so **the contract itself is modular**. The same descriptor projects
everywhere else:

| Projection          | Consumer                                                   | Status here                                                                          |
| ------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| HTTP route fragment | web / mobile / SDK clients                                 | ✅ `create_router(spec, route, handler)`                                             |
| SSE frame kind      | health / system / training stream (push — no request path) | ⚠️ hand-declared per stream                                                          |
| CLI subcommand      | `cli.py` shell                                             | ❌ not projected                                                                     |
| Agent tool-call     | `tools=[...]` model calls                                  | ✅ `ToolRegistry` reads `ToolSpec` directly                                          |
| TS client helper    | `http-client.ts` (the only API surface allowed)            | ✅ `gen_ts_contracts.py` → `lib/protocol/contracts.gen.ts` (rest still hand-written) |

**3. Registration — microkernel, not a switchyard.** Each module self-registers
at boot (import side-effect or a `@tool(descriptor)` decorator — the WordPress
REST / VS Code extension / POSIX loadable-module workflow). The app never
contains a central list of features to include; composition is discovered. The
workflow layer that makes it enforce itself without hierarchy: a **CI
contract-check** — every descriptor must carry JSON-Schema params, an auth
scope, and a contract test; the **OpenAPI diff is reviewed like code**
(contract testing / Pact-style consumer-driven checks).

The one-line hexagonal framing: **core = capability, transports (HTTP/SSE/CLI/
agent) = adapters, web/CLI/agents = clients of one contract.**

## Registering a capability (target workflow)

1. **Boundary test** — same process → function call; different process →
   transport crossing. A new crossing is named in a kanban card before code.
2. **Declare the descriptor** in the module — name, params, result,
   `auth_scope`, `idempotent`, `version`.
3. **Self-register at boot** — one registry entry; no edit to any central
   feature list.
4. **Emit the projections** — route fragment (via `create_router`), SSE frame,
   CLI subcommand, TS helper. Auth scope and version come from the descriptor,
   so there is _one_ auth decision and _one_ versioning story.
5. **Contract gate** — descriptor schema check + contract test + OpenAPI diff
   in CI (`scripts/check_docs_api_parity.py` is the existing parity hook to
   extend).

Registering a capability is a **registry entry, never an endpoint project**.

## Guardrails

- **No god-endpoint.** Reject both N hand-written routers and a single generic
  `POST /tools/{name}` switchyard. If single-call semantics are ever wanted,
  the honest standard is **JSON-RPC 2.0** (`/rpc`, typed versioned methods,
  introspectable — the LSP/Ethereum pattern), not an ad-hoc switchyard.
- **One envelope, one auth story, one versioning story** — those are exactly
  what a shared projection gives you; a hand-written router re-decides them
  per file (today: `schemas/common.py` imported by 58 routers).
- **Two tool tiers, one contract, never cross** — internal/model-facing
  tooling (function-calling, `domain/tools`, the inference UI's tool list) vs
  external utility tooling (Site Doctor and other app-internal helpers). Both
  are descriptors; only the first is visible to the model.
- **Grandfathered, not purged** — the 57 existing routers migrate only when
  next touched (Router Playbook rule 5).

## Where it stands

| Piece                                          | State                                                                                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Descriptor (`ToolSpec` + derived `params`)     | ✅ contract half exists                                                                                                         |
| `ToolRegistry` (agent-facing projection)       | ✅ reads the descriptor                                                                                                         |
| `create_router(spec, route, handler)` emission | ✅ `infrastructure/contract.py` — verb, 422 validation, envelope, auth scope and `x-contract` all derived from the descriptor   |
| Boot-time registration                         | ✅ generated `routers/_manifest.py` (`scripts/gen_router_manifest.py --check`) — mount order preserved, deferred imports intact |
| CI contract check (descriptor + OpenAPI diff)  | ❌ only doc/code parity exists                                                                                                  |
| TS projection of the contract                  | ❌ `http-client.ts` hand-writes every endpoint                                                                                  |

Grandfather note: `get_all_routers()`'s deferred-import list exists for a real
reason (90 s → 8 s cold start) — self-registration must preserve lazy loading,
not fight it.
