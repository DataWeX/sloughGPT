# Agents

## MANDATORY — Session Checklist
Every session MUST do these three things. No exceptions.

```
1. notes new "Session title" --tags area,subarea --status wip   ← FIRST action
2. (work happens)
3. notes edit <id> --status done --body "Summary"               ← BEFORE done
4. sync-notes-to-board                                           ← AFTER commit
```

If you skip step 1, you're doing it wrong. Go back and create the note.

## Doc-First Workflow
Before any edit, read the relevant docs for the area. Use `opencode doc-aware` to load context:
- Frontend → `docs/UI_INTEGRATION_README.md`, `docs/API.md`
- Backend → `docs/routers.md`, `docs/API.md`
- Core → `docs/DEVELOPER_GUIDE.md`, `docs/AI_SOFTWARE_ENGINEERING.md`
- SDK → `docs/API.md`
- Infra → `docs/DEPLOYMENT.md`, `docs/DEPLOYMENT_CHECKLIST.md`
- CLI → `docs/integration/CLI_README.md`
- Config → `docs/ENVIRONMENT.md`
Full map in `~/.config/opencode/agents/doc-aware-engineer.md`.

## Repo Layout & Boundaries

```
sloughGPT/
├── apps/
│   ├── api/server/            # FastAPI backend (main.py + routers/)
│   ├── web/                   # Next.js 16 frontend (app router)
│   ├── cli/                   # CLI implementation
│   └── mobile/                # React Native (no node_modules in sandbox)
└── packages/
    ├── core-py/domains/       # Python core logic (SloNet, training, inference, feedback, multimodal)
    ├── downcraft/             # Model downloader — HTTP resume, link resolution, state tracking, integrity verification
    ├── planner/               # Kanban board + notes sync (CLI + GUI)
    ├── strui/                 # @sloughgpt/strui component library
    └── mogdb/                 # Document database
```

**Core backend lives in `packages/core-py/domains/`.** Business logic goes there first; API routes in `apps/api/server/routers/` are thin adapters.

## Commands

```bash
# Start API server
make api

# Start web dev server
make web

# Start both
make stack

# Python tests (parallel)
make test-py ARGS="tests/test_file.py -x -q"

# Python syntax check
python3 -m py_compile <file>

# Frontend typecheck
cd apps/web && npm run typecheck

# Frontend tests (fast → slow)
npm run test:lib          # pure logic
npm run test:components   # UI components
npm run test:changed      # only changed files
npm run test              # full suite ~3 min

# Clear pycache (always after Python edits)
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
```

## Core Backend Conventions

### SloNet — No PyTorch for Training/Inference
- **PyTorch is NOT used** for training or inference. SloNet (`packages/core-py/domains/training/slonet.py`) is pure NumPy autograd + inference.
- `Tensor` wraps numpy arrays with autograd. Never call `torch.tensor()`.
- Generation: `generate_numpy_stream()` — token-by-token, KV cache, fused QKV, RoPE, GQA.
- Model format: `.soul` checkpoints embed soul metadata + weights. `.slnc` is mmap zero-copy loading.

### Provider Router — Image Handling
- `ProviderRouter` chains processors before the text provider.
- `VisionProcessor("multimodal")` extracts base64 images from messages, captions them via `MultimodalEngine`, injects `[Image: caption]` text.
- `KnowledgeProcessor`, `ToolUseProcessor`, `PersonalityProcessor`, `StyleProcessor` follow.
- Register via `register_provider("default", router)` in `setup_providers()`.

### SSE Envelope
All SSE endpoints emit standard envelope:
```json
{"stream":"chat","phase":"STREAMING","status":"working","data":{"token":"..."},"meta":{},"message":""}
```

### TrainingSequence Protocol
All training follows: `GENERATE_DATA → DISTILL → TRAIN → EVALUATE → DEPLOY → COMPLETE`
- Enum: `TrainingSequence` in `packages/core-py/domains/training/sequence.py`
- Router: `apps/api/server/training/router.py` (split into `execution.py`, `jobs_api.py`, `control.py`)
- Core service: `packages/core-py/domains/training/service.py` (zero HTTP deps)
- Runtime protocol: `packages/core-py/domains/training/runtime_protocol.py` (core defines interface, API layer implements)

### Critical Gotchas
- **MPS on Intel Mac**: PyTorch 2.x reports MPS available on x86_64 but crashes at runtime. `_resolve_device()` returns `"cpu"` on Intel Macs.
- **Metal accelerator overhead**: For `embed_dim ≤ 128`, Metal dispatch is slower than CPU numpy. Disable `_ACCELERATOR` during `train_step()`, `train_batch()`, `generate()`. Use `try/finally` to restore.
- **No external downloads at runtime**: SloNet trains from scratch. HF models convert to `.slnc` on first load. Never `pip install` heavy deps without asking.
- **No hardcoded paths**: Use `_DATASETS_DIR = Path(__file__).resolve().parents[4] / "datasets"` pattern.
- **ProcessGuard inference**: `ModelServer` delegates to `ProcessGuard` for subprocess isolation. Circuit breaker: 3 failures → 30s open.

## Feature Stages

All features are grouped into 4 stages. Work on **core stages first** before moving to advanced ones. Each stage must be stable before proceeding.

### Stage 1 — Core (must work)
Everything else depends on these. Break these, nothing works.

| Feature | CLI | API | Web | Core |
|---------|-----|-----|-----|------|
| System health/status | `system status/info/health/doctor` | `/system`, `/health` | `/settings` | `infrastructure/` |
| Inference | `chat`, `generate`, `serve` | `/infer`, `/inference` | `/chat`, `/infer` | `inference/` |
| Model management | `model list/status/info/download/export` | `/models` | `/models`, `/model/[id]` | `infrastructure/model_*` |
| Training | `train start/monitor/eval` | `/training/*` | `/training` | `training/slonet.py`, `training/service.py` |
| Datasets | `dataset list/stats/import/export` | `/datasets` | `/datasets` | `training/dataset.py` |
| Memory | `memory stats/search/store` | `/memory` | `/memory` | `memory/` |
| Checkpoints | `checkpoint list/load/delete` | (training sub) | (training page) | `training/checkpoint.py` |
| Logging | `logs`, `monitor` | `/dashboard` | `/monitoring` | `logging/` |
| Config | `system config` | `/config` | `/settings` | `shared/feature_flags.py` |
| Auth | `system api` | `/auth` | `/auth` | `infrastructure/auth.py` |

### Stage 2 — Intelligence (AI features)
Builds on Stage 1. These make the system smart.

| Feature | CLI | API | Web | Core |
|---------|-----|-----|-----|------|
| Knowledge base | `knowledge search/dedup/categorize/gaps` | `/knowledge` | `/knowledge`, `/kb` | `learner/`, `cognitive/rag.py` |
| Feedback loop | `feedback export/prepare` | `/feedback` | `/feedback` | `feedback/` |
| Personality | `personality list/load/create` | `/souls` | `/souls` | `ai_personality.py`, `soul/` |
| LoRA adapters | `adapter list/info/merge/delete` | `/user-adapters` | `/adapters` | `feedback/per_user_lora.py` |
| Experiments | `experiment list/create/info/metrics` | `/experiments` | `/experiments` | `benchmark/` |
| Token tree | `token-tree train/encode/decode/stats` | `/token-tree` | `/token-tree` | `training/token_tree.py` |
| Errors | `error recent/grouped/trends/clear` | `/errors` | `/errors` | (API-level) |
| Tokens billing | — | `/tokens` | `/settings` | `billing/token_service.py` |

### Stage 3 — Advanced (power features)
Requires Stage 1+2 to be stable.

| Feature | CLI | API | Web | Core |
|---------|-----|-----|-----|------|
| Agents | `agent list/create/execute/orchestrate` | `/agents` | `/agents` | `agents/` |
| Sessions | `session list/messages/search` | `/session` | `/session` | `chat/domain.py` |
| Collections | `collect file/url/rss/merge` | `/collections` | `/collections` | `collections/` |
| Companion | — | `/companion` | `/companion` | `companion.py` |
| Multimodal | — | `/multimodal` | `/multimodal` | `multimodal/` |
| Images | — | `/images` | `/images` | `multimodal/diffusion.py` |
| Voice | — | `/voice` | `/voice` | `multimodal/tts.py` |
| Vector store | — | `/vector` | `/vector` | `inference/vector_store.py` |
| Meta-weights | — | `/meta-weights` | `/meta-weights` | `feedback/meta_weights.py` |
| Learner | — | `/learn` | `/learn` | `learner/continual.py` |
| Planner | `notes`, `board` | `/api/planner/*` | `/kanban` (planner) | `packages/planner/` |

### Stage 4 — Specialized (niche/infra)
Only if specifically needed.

| Feature | CLI | API | Web | Core |
|---------|-----|-----|-----|------|
| Docker | `docker start/stop/status/build` | — | — | `Makefile` |
| World sim | `world render/tick/analyze` | `/world` | `/world` | `shell/world_*.py` |
| x86 VM | `vm run/list/debug` | `/vm` | `/vm` | `shell/vm_engine.py` |
| Shell/TUI | `shell`, `tui` | `/shell` | `/shell` | `shell/` |
| Build | `build run/init/clean` | — | — | `Makefile` |
| Security | — | `/security` | `/security` | (audit logs) |
| Docstore | — | `/docstore` | `/docstore` | `mogdb/` |
| Feeds | — | `/feeds` | — | (RSS output) |

## Downcraft — Model Downloader

`packages/downcraft/` handles downloading model files from the internet with resume support, link resolution, state persistence, and integrity verification.

### Modules

| Module | Purpose |
|--------|---------|
| `downloader.py` | HTTP downloader with byte-level resume via Range headers. Downloads to `.sgpart` temp file, atomic rename on completion. |
| `resolver.py` | Extract real download URLs from ad-heavy pages. Pure HTTP + regex + HTML parsing — no headless browser. |
| `state.py` | Persistent download state at `~/.downcraft/state.json`. Survives restarts. Tracks per-model progress, checksums, completion. |
| `verify.py` | SHA-256 file integrity verification. HuggingFace-agnostic — checks a single file against expected checksum. |

### Downcraft Rules

- **Atomic writes** — Downloads go to `.sgpart` temp files, renamed only on full completion. Never write directly to the final path.
- **Resume by default** — Always check for existing `.sgpart` and send `Range` headers. Do not re-download completed portions.
- **State flushed every chunk** — `ModelState` writes to `~/.downcraft/state.json` after every chunk. Do not buffer state in memory only.
- **No headless browser** — `resolver.py` uses regex + HTML parsing. Never add Selenium, Playwright, or Puppeteer deps.
- **HuggingFace-agnostic verification** — `verify.py` checks one file against a checksum. Model-level verification (snapshots, weight lists) lives in `domains.infrastructure.hf_hub`.
- **Checksum is SHA-256** — Always. Never MD5, CRC32, or other hashes for integrity.
- **State file is JSON** — `~/.downcraft/state.json`. Do not switch to SQLite, pickle, or other formats.
- **Thread-safe state** — `ModelState` uses a lock. Do not remove the threading primitives.
- **No network at import time** — All network calls happen inside functions, not at module level. Enables testing without connectivity.

## Frontend Conventions

- **Design system**: Noir Violet. All colors via `rgb(var(--token))`. Never hex/hsl.
- **Page wrapper**: `sl-page mx-auto max-w-4xl`. `space-y-4` between sections.
- **Typography**: `text-sm` body, `text-base font-medium` section titles, `AppRouteHeaderLead` for page titles.
- **Components**: Import from `@sloughgpt/strui`. Use `Card`, `Button`, `Dialog`, `DropdownMenu`, etc.
- **No arbitrary spacing**: Never `px-8 py-6` in page body. Use tokens: `gap-2` (8px), `p-3` (12px), `p-4` (16px).
- **States**: Every interactive element needs hover, focus-visible, disabled, loading, success.
- **Empty states**: Always explain why empty + provide next action.

## Testing Strategy

**Order matters: typecheck → targeted tests → commit.**

| Scope | Command | When |
|-------|---------|------|
| Python syntax | `python3 -m py_compile <file>` | Every Python edit |
| TS typecheck | `npm run typecheck` | Every TS edit |
| Python unit | `make test-py ARGS="tests/test_file.py -x -q"` | Single file |
| Python fast full | `make test-py-fast` | Before push |
| Frontend logic | `npm run test:lib` | lib/ changes |
| Frontend components | `npm run test:components` | UI changes |
| Changed only | `npm run test:changed` | Quick sanity |
| Full frontend | `npm run test` | Pre-push / CI only |

**pytest config**: `testpaths = packages/core-py/tests tests apps/cli/tests`, `asyncio_mode = auto`, markers: `unit`, `integration`, `e2e`, `slow`.

## Pre-commit Hooks (auto-run on `git commit`)
- `tsc --noEmit` — TypeScript type safety
- `python3 -m py_compile` — basic correctness
- `update-anchored-summary` — updates `anchored_summary.md`
- `check-todos` — no TODO/FIXME/HACK in Python
- `check-secrets` — no hardcoded secrets
- `check-feature-tags` — validates FEATURE tags

Run manually: `make precommit-run`

## Before Submitting (Mandatory)
0. **Clear pycache** — stale `.pyc` causes silent runtime crashes
1. **Syntax check** — Python: `py_compile`; TypeScript: `tsc --noEmit`
2. **Runtime test** — call the endpoint/function, verify response
3. **Log verification** — check logs for errors, not just HTTP 200
4. **Stability check** — if changing model loading/inference: `python scripts/benchmark_stability.py --runs 20`

## Engineering Rules
- **No downloads without asking** — >50MB bandwidth cost
- **No mocks** — real algorithms only, no hardcoded lookup tables or fake data
- **Infrastructure before endpoints** — core module → CLI → thin API wrapper
- **Tests alongside** — every public function gets a test; edge cases get tests
- **Document inline** — docstrings on every public function, side effects noted
- **Reversible** — can roll back if broken; deprecate before deleting
- **Formal tone** — no contractions, slang, emojis, exclamation marks
- **Concise** — no verbose summaries; state results in 1-3 bullets

## Smart Code Methodology

Frame architectural boundaries as **descriptive facts**, not prescriptive rules.

- **"Core would not know about HTTP"** — not "should not know"
- **"The service returns raw data"** — not "must not wrap responses"
- **"Dependencies flow inward"** — not "must not import outward"

When you say "should not know", you imply someone might violate it and needs enforcement. When you say "would not know", the architecture itself prevents the violation. No linters, no rules, no code reviews catching violations — the structure makes them impossible.

**Practice:** Before writing a boundary, ask: "Does this code *know* about the other layer?" If yes, the abstraction is leaky. Redraw the boundary until the dependency direction is the only one that exists.

```
# Wrong: prescriptive rule
# "Core must not import from API layer"

# Right: descriptive fact
# Core returns dicts. API wraps with success_response.
# The import direction emerges from the data flow.
```

## Multi-Agent Workflow — Git Worktrees

All frontend development runs in an isolated worktree. The main agent never pushes to it; the frontend agent never pushes to `main`.

### Layout

```
/home/mana/Documents/Default Project/sloughGPT/          ← main worktree (backend + core)
/home/mana/Documents/Default Project/sloughGPT-frontend/  ← frontend worktree (feat/frontend branch)
```

### Frontend Agent

**Workflow file**: `~/.config/opencode/agents/frontend-agent.md`

1. Frontend agent works in `../sloughGPT-frontend`
2. Commits to `feat/frontend` branch only
3. After push, notifies main agent for visual review
4. Main agent audits in browser before merge

### Main Agent — Review Protocol

When the frontend agent pushes:

```bash
cd ../sloughGPT-frontend
npm run typecheck
npm run test:changed
make web   # open in browser, screenshot pages
```

Visual audit checklist:
- [ ] No design system violations (hex colors, arbitrary spacing)
- [ ] Page layout matches Noir Violet spec
- [ ] All interactive states present (hover, focus, disabled)
- [ ] Empty states correct

### Merge

Only after review passes:

```bash
cd /home/mana/Documents/Default Project/sloughGPT
git merge feat/frontend --no-ff -m "merge: frontend from feat/frontend"
```

### Rules
- Never commit frontend changes to `main` directly
- Visual review is mandatory before merge
- Design system violations are merge blockers
- If the frontend build breaks, fix it in the frontend worktree
