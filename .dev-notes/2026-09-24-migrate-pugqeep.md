---
id: 20260924_047_migrate_pugqeep_to_domain
status: done
title: Migrate pugqeep real code into domain/_internal
tags: [architecture, migration, dual-tree, pugqeep]
---
# Dual-tree step 2 — pugqeep migration

## Plan
1. Branch feat/migrate-pugqeep off main in clean worktree /home/mana/Documents/sloughGPT-pugqeep
2. Copy real packages/core-py/domains/infrastructure/pugqeep/*.py → domain/infrastructure/_internal/pugqeep/ (replace reverse shims)
3. Plural pugqeep/* → forward shims to domain.infrastructure._internal.pugqeep
4. Flip domain/infrastructure/pugqeep.py facade → from domain.infrastructure._internal.pugqeep import *
5. Retarget scripts/benchmark_block_quantization.py + tests using domains.infrastructure.pugqeep
6. Gates: ruff domain/infrastructure, pugqeep pytest suite, scatter, import probe (facade PGQ)
7. Commit, merge main, card done

## State
- done — commit 3d5dfd21a on feat/migrate-pugqeep; merged to main
- real code in domain/infrastructure/_internal/pugqeep (8501 LOC, 20 modules)
- plural tree = 3-line forward shims; facade flipped to _internal
- benchmark_block_quantization retargeted; 0 domains.infrastructure.pugqeep imports remain
- gates: ruff clean; scatter 5/5; import probe PGQ/Tree/Process OK
- pugqeep suite: 1704 passed / 21 failed — same 21 fail on parent 3c01503ac (pre-existing debt, not migration)
- follow-up candidate: fix the 21 pre-existing pugqeep failures (checkpoint decompress_tree, process timeout/stream, ProcessGroup message)
