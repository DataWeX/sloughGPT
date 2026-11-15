---
id: 20261104_053224_fixboot-boot-path-is-slnc-only-while-modelloader-prefers-sou
status: todo
board: 20261104_053224_fixboot-boot-path-is-slnc-only-while-modelloader-prefers-sou
title: "fix(boot): boot path is .slnc-only while ModelLoader prefers .soul"
tags: ["model", "inference", "v1", "area:models", "track:now"]
sprint: ""
gh: null
---

# fix(boot): boot path is .slnc-only while ModelLoader prefers .soul

Found while scoping card 20261101_144013. `startup._try_lazy_guard_autoload` and `_build_guard_for_model` both resolve `get_model_dir(model)/model.slnc` unconditionally, while `ModelLoader` documents `.soul`-first priority. A model installed only as a `.soul` therefore boots as 'no .slnc' (ProcessGuard skipped, no autoload) even though the loader would happily serve it.
Fix: one resolver decides artifact + format, and both paths read that decision — no second source-priority rule in startup.py. That resolver is also the `model_id` fix's home; sequence with the `_try_load_soul` card.
Gate: boot with a `.soul`-only model dir autoloads; boot with a `.slnc`-only dir still autoloads; test asserts the resolver is consulted exactly once.

---

**Prerequisite landed 2026-11-04 (`a7769616c`).** `ModelLoader._try_load_soul` now resolves a native `.soul` by header identity (exact name, `hash:<prefix>`, `dataset:<signature>`, or a family id that is only honoured when all candidates share base_model + a non-empty dataset signature) and refuses loudly otherwise — it no longer answers any request with the newest-mtime checkpoint. So flipping boot priority to `.soul` can no longer serve a TinyShakespeare checkpoint labelled Qwen.

Caveat before flipping: every current checkpoint in `models/slonet-native/` records `dataset_signature=""`, so the family aliases refuse. Boot priority flip must therefore request the checkpoint by **name** (or `SLO_NATIVE_SOUL_PATH`), not `SLO_AUTOLOAD_MODEL=sloughgpt`, until card `20261104_081626` makes training record the dataset hash.
