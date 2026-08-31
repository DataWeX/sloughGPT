---
id: 20260812_051051_tokentree-feature-cards-merge-rules-explorer
title: TokenTree feature cards + Merge Rules Explorer
status: done
tags: tokenizer,frontend,backend
created: 2026-08-12T05:10:51.795997+00:00
---

TokenTree feature cards + Merge Rules Explorer

Embedding-matrix overview complete end-to-end: core embedding_matrix_stats, manager matrix_summary, CLI token-tree matrix, API GET /token-tree/matrix, controller getMatrixSummary, TokenTreeMatrixCard wired into /tokenizer. Docs: /token-tree router table in docs/routers.md, token-tree CLI group in docs/integration/CLI_README.md, token-tree row in docs/API.md. Tests: core+manager+CLI+server router pytest 206 pass; frontend full suite green (matrix card 7 tests); tsc clean of new errors. Live verification: CLI smoke on token_tree_demo (1024x64, 1009 live/15 dead) and live API smoke via get_all_routers (real manager, 211x16, 85 live/126 dead, most energetic ' fox'/62/1.0).