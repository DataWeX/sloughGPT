# 076 — SHELL.md + DEVELOPER_GUIDE.md truth-up (seam 7)

- **Sprint**: 2026-09 (doc↔code seam series, after 071/073/074/075)
- **Status**: done
- **Outcome**: commit bf71cd6b6 on feat/routers-doc-truthup. SHELL.md
  (tree→_internal, 22→41 API methods, 148/30→18/23 counts, coverage table→
  live measure, /knowledge/list→/knowledge, {id}→{job_id}, package claim) +
  DEVELOPER_GUIDE.md (Node22/.nvmrc, DataWeX clone, make stack, flat
  tests/docs claims, real coverage config, real ServerState methods,
  orphan fragments). Guard tests/contract/test_shell_dev_docs.py (7 tests,
  mutation canary 3/3). Contract suite 42→49, ruff clean.
- **gh**: (local)
- **Branch**: feat/routers-doc-truthup (worktree sloughgpt-routers-doc)
- **Body**: Audit docs/SHELL.md (904 lines) and docs/DEVELOPER_GUIDE.md
  (633 lines) against live shell REPL commands, API routes, and paths.
  Truth-up + contract guard.
- **Blocked by**: 071/073/074/075 (same branch), 067 (merge queue)
