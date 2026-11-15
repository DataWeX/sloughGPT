---
id: 20261113_081911_port-pr-43-novel-parts-syncsqliterepository-repositoryfactor
status: done
board: 20261113_081911_port-pr-43-novel-parts-syncsqliterepository-repositoryfactor
title: "port PR 43 novel parts: SyncSQLiteRepository + RepositoryFactory + conversation/agent/trainingjob/sessioncontext repos into domain/infrastructure/_internal"
tags: []
sprint: ""
gh: "43"
---

# port PR 43 novel parts: SyncSQLiteRepository + RepositoryFactory + conversation/agent/trainingjob/sessioncontext repos into domain/infrastructure/_internal

User decision: port novel parts only. Drop the 4 classes duplicating main's entity_repositories.py (Session/Feedback/Knowledge/Dataset repos + their data models). Relocate packages/core-py/domains/infrastructure/* to domain/infrastructure/_internal/{data_models,data_repository}.py, retarget imports, rework tests/core-py/test_data_repository.py to novel classes only, facade lazy exports, run closure+ruff+contract gates+domain baseline rehash, push to feat/data-repository for fresh CI.
