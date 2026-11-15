---
id: 20260924_048_migrate_infra_plurals
status: done
title: Migrate slnc/quant/cache/deployment/gpu plurals into domain/_internal
tags: [architecture, migration, dual-tree]
---
# Dual-tree step 2b — remaining infrastructure plurals

## Plan
1. Branch feat/migrate-infra-plurals off main in /home/mana/Documents/sloughGPT-infra-rest
2. Copy real packages/core-py/domains/infrastructure/{slnc,quant_core,cache,deployment,gpu}/ → domain/infrastructure/_internal/ (replace reverse shims)
3. Plural → forward shims
4. Flip domain/infrastructure/quant_core facades to _internal
5. Retarget test_quantization_coverage.py domains.infrastructure.quant_core
6. Gates: ruff, targeted test suites, scatter, identity probe
7. Commit, merge main, card done

## State
- claimed; worktree created; copying modules

**Completed:** merged to main `66fa4fcf4`; ruff/scatter/identity/targeted gates green; infrastructure plural real files = 0 (30 total remain across shell/knowledge/etc).
