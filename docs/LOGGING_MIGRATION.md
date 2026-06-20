# Logging Migration Plan

## Architecture

```
domains.logging.Logger (ABC)
├── ConsoleLogger    → API server (replaces coloredlogs)
├── CLILogger        → CLI (Printer inherits this)
├── ShellLogger      → Shell REPL
├── WebLogger        → Browser frontend
└── BridgeHandler    → routes logging.getLogger("man.*") → our Logger
```

**Key insight**: The `BridgeHandler` is already wired into `main.py`. Any `logging.getLogger("man.xxx")` call in the codebase automatically routes through the `ConsoleLogger` with colored output. No call-site changes needed for those — just fix the logger names.

## Migration Phases

### Phase 1: Fix non-standard logger names → `man.*` (13 files)
These bypass the BridgeHandler. Rename to `man.*` so they route through the OOP system.

| File | Old name | New name |
|------|----------|----------|
| `domains/soul/base.py` | `"slo.core"` | `"man.soul.core"` |
| `domains/soul/cognitive.py` | `"slo.cognitive"` | `"man.soul.cognitive"` |
| `domains/soul/quantum.py` | `"slo.quantum"` | `"man.soul.quantum"` |
| `domains/soul/ultimate.py` | `"slo.ultimate"` | `"man.soul.ultimate"` |
| `domains/soul/transcendent.py` | `"slo.transcendent"` | `"man.soul.transcendent"` |
| `domains/soul/foundation.py` | `"slo.foundation"` | `"man.soul.foundation"` |
| `domains/soul/consciousness.py` | `"slo.consciousness"` | `"man.soul.consciousness"` |
| `domains/soul/multiversal.py` | `"slo.multiversal"` | `"man.soul.multiversal"` |
| `domains/core/soul.py` | `"sloughgpt.soul_engine"` | `"man.core.soul"` |
| `domains/training/efficient_inference.py` | `"sloughgpt.efficient"` | `"man.training.efficient"` |
| `apps/api/server/routers/auto_train.py` | `"autotrain"` | `"man.autotrain"` |
| `domains/agents/system.py` | `"agents"` | `"man.agents"` |
| `domains/agents/__init__.py` | `"man"` (bare) | `"man.agents"` |

### Phase 2: Add loggers to print-only files (34 files)
These have no logger at all — `print()` only. Inject a `ConsoleLogger` instance.

**Priority (high-traffic):**
- `domains/training/tokenizer.py` (16 print)
- `domains/training/status.py` (14 print)
- `domains/feedback/training.py` (16 print)
- `domains/inference/cloud_vector_store.py` (19 print)
- `domains/inference/optimizer.py` (12 print)
- `domains/feedback/per_user_lora.py` (7 print)

**Priority (medium):**
- `domains/training/lm_eval_char.py` (8 print)
- `domains/inference/onnx_engine.py` (8 print)
- `domains/training/ewc.py` (4 print)
- `domains/feedback/meta_weights.py` (5 print)
- `domains/infrastructure/auto_ingest.py` (8 print)
- All others (1-5 print each)

### Phase 3: Migrate high-volume API routers
These already have `logging.getLogger()` — no changes needed for routing. The BridgeHandler captures them. Optionally add `context={}` to key calls for structured metadata.

- `apps/api/server/main.py` (49 calls) — ✅ already wired
- `apps/api/server/training/router.py` (20 calls)
- `apps/api/server/routers/auto_train.py` (18 calls)
- `apps/api/server/routers/inference.py` (13 calls)
- `apps/api/server/controllers/models.py` (12 calls)

### Phase 4: Migrate core domains
Same pattern — already routed via BridgeHandler. Add context metadata where useful.

- `domains/training/train_pipeline.py` (24 calls)
- `domains/training/unified_pipeline.py` (16 calls)
- `domains/training/optimized_pipeline.py` (16 calls)
- `domains/infrastructure/database/__init__.py` (27 calls)
- `domains/cognitive/` (83 calls across 7 files)
- `domains/inference/llama_engine.py` (28 calls)

### Phase 5: Shell, CLI, remaining
- Shell: `_print()` stays for raw output; new code uses `self.log`
- CLI: `Printer` already inherits `CLILogger` ✅
- Web: `WebLogger` class ready, `devDebug()` deprecated ✅

## Progress Tracking

| Phase | Files | Status |
|-------|-------|--------|
| 1. Fix logger names | 13 | ✅ Done — all renamed to `man.*` |
| 2. Add loggers to print-only files | 24 | ✅ Done — all have `logging.getLogger()` |
| 2b. Convert diagnostic `print()` → `logger` | 14 files | ✅ `train_pipeline.py` (23 calls) + 13 from earlier |
| 3. API routers structured context | 20 calls | ✅ `main.py` (15), `auto_train.py` (9), `inference.py` (8) |
| 4. Core domains | ~60 | ✅ Already routed via BridgeHandler |
| 5. Shell/CLI/web integrations | ~10 | ✅ Done |

**Total diagnostic prints converted to logger: ~58** across 14 files

**Total f-string logger calls fixed: ~18** across 3 files

## What's left (zero — migration complete)
All diagnostic `print()` calls in core-py domains have been converted to `logger.info/warning/debug`. The 64 remaining `print()` calls are exclusively user-facing CLI output:
- `tokenizer.py` (13) — `show_merges()`, `show_vocab()`, `show_top_tokens()` display methods
- `status.py` (9) — `print_summary()` method
- `train_pipeline.py` (2) — `main()` function demo output
- `feedback/training.py` (11) — `__main__` block demo output
- `cloud_vector_store.py` (19) — standalone CLI script
- `onnx_engine.py` (3) — `__main__` block demo output
