---
id: 20260924_036_dual-domain-tree-finding
title: Structural dual package trees (domain/ vs packages/core-py/domains/) — finding + decision
status: done
sprint: current
gh: ""
body: >
  Investigate dual Python trees, import directions, shim health, broken plural-import
  tests, and .wt-merge leftovers. Deliver a finding + consolidation decision
  (singular domain/ vs dual+shims). Investigation only — no tree mutations without approval.
---

# Dual domain tree finding

## Scope
- Map both trees and who imports whom
- Shim health (Backward-compat re-exports)
- Broken test clusters (KNOWLEDGE_DIR etc.)
- Recommendation: consolidate vs dual+shims
- Explicitly NOT in scope: deleting trees, porting files, .wt-merge cleanup (propose only)

## Status
- [x] Claimed
- [ ] Import direction census
- [ ] Shim inventory
- [ ] Test-breakage map
- [ ] Decision recommendation
- [ ] Report + board done

## Released

- Claim released: user redirected this session to card 039 audit-only.

## Completion

- Finding + decision in `docs/REPO_STRUCTURE_MIGRATION.md` (consolidate onto singular `domain/`).
- Fixed 6-file cluster: learner fixture → `knowledge_store`; slonet accelerator mocks → `_internal.gpu` paths.
- Gates: 6-file cluster + scatter 5 + ruff clean; merged to main.
