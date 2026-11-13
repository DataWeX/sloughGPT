# 074 — Repo-layout docs truth-up (README.md + STRUCTURE.md)

- **Sprint**: 2026-09 (doc↔code seam series, after 071/073)
- **Status**: done
- **gh**: (local)
- **Branch**: feat/routers-doc-truthup (worktree sloughgpt-routers-doc)
- **Commit**: 26e786a6a (gates: contract 36/36 in ~22s, ruff check+format clean;
  mutation canary caught dead path + stale count, restored green)
- **Body**: Audit README.md and docs/STRUCTURE.md against the live tree —
  path existence, counts (apps/packages/tests), command examples. Same
  pattern: truth-up + contract guard test.
  - README: `-n auto` dropped (xdist absent), consciousness 26→25, datasets/
    tree entry marked runtime; API table verified 13/13 live; CLI examples
    verified against click/argparse parsers (start/native/quick/distill/
    eval, model export -f gguf_q4_k_m in choices); port 3000 stays (next dev).
  - STRUCTURE: dead `scripts/legacy/` removed; make claims fixed to
    stack/test-repo-root/colab-smoke/colab-test (no help/dev-stack targets;
    npm dev:stack/test:repo-root verified in package.json).
  - Guard tests/contract/test_repo_layout_docs.py (7 tests): backticked
    paths, markdown links, stale strings, README API rows live (reuses
    test_routers_doc._live_routes), root/tree claims, npm+make targets,
    consciousness count == page.tsx dirs. TESTING.md contract count 29→36.
- **Blocked by**: 071/073 (same branch), 067 (merge queue)
