---
id: 20260813_042719_token-tree-cli-saved-tree-lifecycle
title: Token-tree CLI saved-tree lifecycle
status: done
tags: token-tree,cli,core
created: 2026-08-13T04:27:19.554570+00:00
---

Token-tree CLI saved-tree lifecycle

Closed the CLI gap: manager/API had saved-tree lifecycle (save/load/list_saved/delete_saved) and top_merges/search_merges but the CLI exposed neither.

Core (packages/core-py/domains/training/token_tree_manager.py): added public adopt(tree) so the CLI can hand a tree trained externally (via --tree path) to the manager before saving. Tests: TestAdopt (2).

CLI (apps/cli/src/commands/token_tree.py): added cmd_token_tree_merges (top_merges/search_merges, --top-n/--query), cmd_token_tree_saved, cmd_token_tree_save (--name/--tree), cmd_token_tree_load, cmd_token_tree_delete. All exit(2) on missing/invalid, matching existing command conventions. click wiring in apps/cli/src/cli.py. Tests: TestCmdMerges (3) + TestCmdSavedTrees (9).

Docs: docs/integration/CLI_README.md token-tree table now lists merges/compare/saved/save/load/delete.

Verification: 131 tests pass across CLI (40) + server router (42) + manager (49).