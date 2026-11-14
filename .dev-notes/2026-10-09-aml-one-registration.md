---
title: AML — one registration binds detection AND routing (no retyped reader id)
date: 2026-10-09
status: done
branch: feat/aml-stream-loader
tags: [aml, registry, routing, api, design]
---

# Problem

`docs/FEATURES.md` promises: *"Adding a format is **one entry**, registered by
the package that owns it."* That promise is only half true.

- `register_signature(prefix, type_id, structure, mime)` → **detection**
- `register_reader(type_id, factory)` → **routing**

Two calls, and the owner must **retype the same `type_id` by hand**. Nothing
enforces they match: a typo or a drift between the two files yields
`reader_for()` → `None` → `_open()` silently falls back to a plain byte read.
No error, no log — the reader simply never fires.

Related: `_READERS` / `_STRUCTURE_READERS` ship **empty** and no production
code ever calls `register_reader`, so reader selection is currently inert in
production (only tests populate it).

# Request

> "no name for registry — the AML should be able to pass [a detection] like a
> switchyard that has the logic for maintaining routing without needing their
> id … to make it to the right station."

# Decision (conservative)

Make it **one call**: `register_signature(..., *, reader=...)`.

- The key is **derived** from the `type_id` the caller already passed — it is
  never retyped, so detection and routing cannot drift.
- **Not** a reversal of *"a registry lookup, never a switchyard"*: it is still
  a dict lookup keyed by descriptor. No `POST /tools/{name}`-style switchyard.
- **Ownership/precedence untouched**: `types.detect()` precedence and the
  `None` → plain-byte-read fallback are unchanged.
- **Both invariants preserved**:
  - idempotent — the same factory bound twice is a no-op (import-twice safe);
  - collision-detecting — a *different* factory for an already-bound
    `type_id` raises, so two packages cannot silently fight over one format.

# Circular-import note

`aml.registry` imports `aml.types` at module level, so `types` must import
`registry` **lazily inside `_bind_reader`** — the same pattern
`domain/core/_internal/stream/reader.py::_import_aml` already uses.

# Files

- `packages/aml/aml/types.py` — `Callable` import, `register_signature(..., reader=)`, `_bind_reader()`
- `packages/aml/tests/` — binding, idempotency, collision, no-signature drift
- `docs/FEATURES.md` — "one entry" bullet corrected to match reality
