---
id: 20260812_041246_tokentree-real-data-demo-str-input-bug-fix
title: TokenTree real-data demo + str-input bug fix
status: done
tags: training,TokenTree,pugqeep
created: 2026-08-12T04:12:46.476600+00:00
---

TokenTree real-data demo + str-input bug fix

Built scripts/token_tree_demo.py proving TokenTree on tinyshakespeare (1.1MB): 1024 tokens, 980 merges, 2.29x embedding compression via pugqeep cluster points, tree-walk encoding with subword tokens, save/load round-trip.

Fixed: train() treated a raw str as a sequence of chars (each char became its own doc -> only 38 merges). Now wraps str as one document. Cast cluster centroids to float32 (compress_cluster leaves float64 -> compression ratio went from 0.57x to 2.29x). 3 regression tests in TestStringInput; test_token_tree 35->38 passing.