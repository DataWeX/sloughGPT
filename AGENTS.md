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
Full map in `.opencode/agents/doc-aware-engineer.md`.

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
- Router: `apps/api/server/routers/auto_train.py`

### Critical Gotchas
- **MPS on Intel Mac**: PyTorch 2.x reports MPS available on x86_64 but crashes at runtime. `_resolve_device()` returns `"cpu"` on Intel Macs.
- **Metal accelerator overhead**: For `embed_dim ≤ 128`, Metal dispatch is slower than CPU numpy. Disable `_ACCELERATOR` during `train_step()`, `train_batch()`, `generate()`. Use `try/finally` to restore.
- **No external downloads at runtime**: SloNet trains from scratch. HF models convert to `.slnc` on first load. Never `pip install` heavy deps without asking.
- **No hardcoded paths**: Use `_DATASETS_DIR = Path(__file__).resolve().parents[4] / "datasets"` pattern.
- **ProcessGuard inference**: `ModelServer` delegates to `ProcessGuard` for subprocess isolation. Circuit breaker: 3 failures → 30s open.

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
