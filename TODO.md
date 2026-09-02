# TODO: Next Stage — MogDB Migration & Data Infrastructure

## Completed ✅

### MogDB JSON Sync Engine
- [x] Created `SyncableCollection` wrapper in `packages/mogdb/src/mogdb/json_sync.py`
- [x] Added `sync_dir` parameter to `MogDB` class for global JSON sync
- [x] Atomic writes (write-to-temp + rename) for crash safety
- [x] Bootstrap: loads from JSON if collection is empty on startup
- [x] All 176 MogDB tests passing

### Router Migrations (File-based → MogDB + JSON Sync)
- [x] `apps/api/server/routers/experiments.py` → `data/experiments_mogdb/` + `data/experiments_json/`
- [x] `apps/api/server/routers/companion.py` → `data/companion_mogdb/` + `data/companion_json/`
- [x] `apps/api/server/routers/errors.py` → `data/errors_mogdb/` + `data/errors_json/`
- [x] `apps/api/server/routers/files.py` → `data/uploads_mogdb/` + `data/uploads_json/`

### Domain Module Migrations
- [x] `packages/core-py/domains/context/managers.py` (trait_weights) → `data/trait_weights_mogdb/` + `data/trait_weights_json/`
- [x] `packages/core-py/domains/feedback/model_health.py` → `data/model_health_mogdb/` + `data/model_health_json/`

### SLNC Memory-Mapped Inference (already implemented)
- [x] `domains/infrastructure/slnc/spec.py` — Binary format with tensor table, CRC32, flags
- [x] `domains/infrastructure/slnc/parser.py` — Zero-copy mmap loader, prefetch, parallel loading
- [x] `domains/infrastructure/slnc/compiler.py` — Safetensors → .slnc converter
- [x] Full integration in `slonet_provider.py` (from_slnc, lazy_from_slnc, quantization, mmap release)

### Other Features (from previous session)
- [x] Voice selection/multi-voice support (`/voice/voices`, `/voice/stt`)
- [x] Real token logprobs in SloneNet provider (forward_pass-based)
- [x] Phi-3/Llama-3 architecture detection in gguf_export
- [x] GPU matmul shader dispatch in wgpu_be.py

### Bug Fixes
- [x] Companion router `use_preset` — fixed `Field(...)` → `Query(...)` for FastAPI route registration

### Test Results
- 176 (MogDB) + 143 (slonet+kanban) + 64 (CLI) + 16 (SLNC) = **399 tests passing**

---

## Next Up 📋

### Remaining MogDB Migrations
- [ ] `data/knowledge/entries.json` → MogDB collection
- [ ] `data/knowledge/visited.json` → MogDB collection
- [ ] `data/rag_store/documents.jsonl` → MogDB collection (capped)
- [ ] `data/response_logs/*.jsonl` → MogDB collection (TTL-indexed)
- [ ] `data/model_catalog/` → MogDB collection

### Performance Optimization
- [ ] Add batch write support to `SyncableCollection` for high-throughput scenarios
- [ ] Implement lazy JSON sync (async background thread)
- [ ] Add compression for JSON sync files
- [ ] Benchmark MogDB vs file-based performance

### Data Migration Tools
- [ ] Create `scripts/migrate_legacy_json.py` to import existing JSON files into MogDB
- [ ] Add CLI command `slo db migrate` for one-time migration
- [ ] Add CLI command `slo db sync` to force JSON sync
- [ ] Add CLI command `slo db status` to show collection stats

### Documentation
- [ ] Update README.md with MogDB usage examples
- [ ] Document `SyncableCollection` API
- [ ] Add migration guide for developers

---

## Architecture Decisions

### Why MogDB as Engine (not extension files)?
- MogDB is already the primary data store for most collections
- JSON sync provides human-readable backup without separate tooling
- Single write path reduces complexity and bug surface
- Atomic writes ensure consistency between engine and sync

### Why `sync_dir` parameter?
- Opt-in per database (not all need JSON sync)
- Keeps MogDB core simple and fast
- Allows different sync strategies per database
- Backward compatible with existing MogDB usage

### JSON Sync Strategy
- **Full rewrite on every write**: Simple, safe, consistent
- **Atomic writes**: Write to temp file, then rename (no partial writes)
- **Bootstrap**: If JSON exists but collection is empty, load from JSON
- **No auto-cleanup**: JSON files persist even if collection is dropped (manual cleanup)

---

## Files Modified/Created

### New Files
- `packages/mogdb/src/mogdb/json_sync.py` — SyncableCollection wrapper
- `TODO.md` — This file

### Modified Files
- `packages/mogdb/src/mogdb/__init__.py` — Added SyncableCollection export
- `packages/mogdb/src/mogdb/database.py` — Added sync_dir parameter
- `apps/api/server/routers/experiments.py` — MogDB migration
- `apps/api/server/routers/companion.py` — MogDB migration + Query fix
- `apps/api/server/routers/errors.py` — MogDB migration
- `apps/api/server/routers/files.py` — MogDB migration
- `packages/core-py/domains/context/managers.py` — MogDB migration
- `packages/core-py/domains/feedback/model_health.py` — MogDB migration

---

*Last updated: 2026-09-02*
*Status: Ready for next tasks*
