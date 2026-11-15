---
id: 20260924_054_migrate_final_plurals
status: done
title: Migrate final 9 plural real files (dual-tree step 2 completion)
tags: [architecture, migration, dual-tree]
---
# Dual-tree step 2 final — last 9 plural real files

## Inventory
1. domains/__init__.py (160) — BaseComponent, ComponentException, protocols, Memory, Thought, BaseDomain, DomainException + facade try/except. 18 importers `from domains import`.
2. collections/training_bridge.py (200) — twin domain/collections/_internal/training_bridge.py (minor divergence, singular canonical).
3. consciousness/__init__.py (35) — already pure re-export to domain.cognition.
4. consciousness/tests/__init__.py (0) — empty.
5. generation/__init__.py (68) — facade; singular twin is BROKEN reverse-shim (domain.generation._internal missing).
6. inference/native/__init__.py (8) — identical twin.
7. slolib/gpu/__init__.py (1240) — identical twin domain/slolib/_internal/gpu.py; startup.py prewarm string refs domains.slolib.gpu.
8. tools/tests/__init__.py (0) — empty.
9. training/huggingface/__init__.py (83) — identical twin.

## Risks
- Moving root into domain/__init__ makes every `import domain.*` execute facade imports — guarded by existing try/except (ImportError → None). Gates must verify ChatDomain/BenchmarkDomain not None where needed.
- generation singular reverse-shim broken — moving facade there FIXES it.

## Plan
branch feat/migrate-final-plurals → implement → gates → commit → merge → close.

**Completed:** merged to main `68bd7d922` (commit `bb7b0946c`); gates green; **plural reals = 0** — dual-tree step 2 COMPLETE. Fixed broken `domain/generation` reverse-shim (moved facade to singular). Retargeted startup prewarm `domains.slolib.gpu` → `domain.slolib._internal.gpu`. Note: concurrent agent created duplicate `054_migrate_trivial_plurals` (overlapping scope) — marked done/superseded.
