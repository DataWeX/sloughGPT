---
id: 20261105_101053_featregistry-parentslineage-column-on-registry-show-each-mod
status: done
board: 20261105_101053_featregistry-parentslineage-column-on-registry-show-each-mod
title: "feat(registry): parents/lineage column on /registry — show each model's dataset family tree (compose, no new page)"
tags: []
sprint: ""
gh: null
---

# feat(registry): parents/lineage column on /registry — show each model's dataset family tree (compose, no new page)

User GO 2026-11-05 ('yes do') on the earlier proposal. Scope: extend the EXISTING /registry page with a parents/lineage view built on domain/infrastructure/_internal/artifact_registry.py provenance (parents_from_metadata, lineage, landed fddcee1a7). Surface ladder: compose onto existing journey page, no new page.tsx, no new endpoint unless boundary test says cross-process. SOP: branch feat/registry-lineage, change-scoped tests, pathspec commit, push HEAD:main.
