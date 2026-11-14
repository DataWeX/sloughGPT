---
id: 20260924_052_migrate_knowledge_plurals
title: Dual-tree step 2d: flip plural knowledge real files to forward shims
status: done
created: 2026-09-24
tags: [architecture, migration, dual-tree]
sprint: current
gh: null
body: |
  6 real files in packages/core-py/domains/knowledge/_internal/ flipped to forward shims
  targeting domain.knowledge._internal.*. 5 were byte-identical twins; knowledge.py was
  superseded real code (canonical in domain.memory.knowledge_store, re-exported by the
  domain twin). plural knowledge/__init__ was already a shim. Zero importers of domains.knowledge.
status_detail: done
---

## Result
- Branch `feat/migrate-knowledge-plurals`, worktree `/home/mana/Documents/sloughGPT-knowledge`.
- Commit `cd52f3a84` (6 files, +13/−3266); merged main (`ab01d3076` workspace nav) clean; fast-forwarded to main as `2471c4b36`.
- **Gates**: ruff clean; identity probe PASS (plural/twin resolve inside worktree, SAME objects for DataFilter/FileIndex/KnowledgeMemory/enrich_with_knowledge/extract_entities; symbol presence 12 modules); knowledge suites + consumers + contract/scatter batch **713 passed** (first run had 1 thread-timing flake `test_learner_continual::test_shutdown_stops_thread_and_saves`, green on isolation + rerun).
- Plural knowledge real files: 0 (all 7 shim).
- Note: `import *` shims expose only the twin's `__all__`/public names — knowledge twin modules mostly lack `__all__`, so star-import exposes all public names; zero importers means no blast radius anyway.
