---
id: 20260813_005505_token-tree-compare-wiring-fix
title: Token tree compare wiring fix
status: done
tags: token-tree,cli,api
created: 2026-08-13T00:55:05.146015+00:00
---

Token tree compare wiring fix

Wired token-tree compare through all three layers and fixed a kwarg mismatch.

CLI (apps/cli/src/commands/token_tree.py): cmd_token_tree_compare prints per-side stats, vocab/merge overlap, top shared/exclusive token tables. Tests: TestCmdCompare (3) in apps/cli/tests/test_token_tree_commands.py.

API (apps/api/server/routers/token_tree.py): CompareTreesRequest {a,b,top_k}, POST /token-tree/compare -> success_response with overlap dict; 404 missing, 400 self-compare. Tests: compare mock + TestCompare (7) in tests/server/test_token_tree_router.py.

Bug fixed: manager compare(name_a, name_b, top_n=10) was called with top_k= from the API router and the CLI click wrapper — would crash in production (TypeError). Both callers now pass top_n=. CLI click wrapper at cli.py:962, router at token_tree.py compare handler. Tests set args.top_n.

Verification: 47 manager tests, 69 CLI+API tests all pass.