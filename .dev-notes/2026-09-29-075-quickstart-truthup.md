# 075 — QUICKSTART.md truth-up (seam 6)

- **Sprint**: 2026-09 (doc↔code seam series, after 071/073/074)
- **Status**: done
- **Commit**: 8e8b1e028 (QUICKSTART) + 7891218c6/172b4280a (CONTRIBUTING via
  symlink target docs/policies/CONTRIBUTING.md; root symlink restored after
  edit replaced it with a regular file)
- **Gates**: contract 42/42 (~12s), ruff clean; mutation canaries caught
  dead command, dead curl, and stale prose claim; TESTING.md count 36 -> 42
- **Findings**: QUICKSTART had ~30 stale claims (bare legacy top-level CLI
  forms,5 dead curl routes, iamtowbee URL, dead Presets/torch_runtime python,
  fabricated tree entries, make dev-stack/help); CONTRIBUTING had4 (sloughgpt
  eval x2, make help, wrong symbol/location, removed legacy server path);
  INFRASTRUCTURE.md clean (all7 tracker paths live). Guard
  tests/contract/test_quickstart_doc.py (6 tests) now covers all three docs.
- **gh**: (local)
- **Branch**: feat/routers-doc-truthup (worktree sloughgpt-routers-doc)
- **Body**: Audit QUICKSTART.md (445 lines — the 5-minute entry doc linked
  from README) plus root CONTRIBUTING.md + INFRASTRUCTURE.md against live
  commands, make/npm targets, ports, and paths. Truth-up + contract guard.
- **Blocked by**: 071/073/074 (same branch), 067 (merge queue)
