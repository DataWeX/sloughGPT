---
id: 20260812_042256_embedding-learning-fix-semantic-queries-cli-tests
title: Embedding learning fix + semantic queries + CLI tests
status: done
tags: training,TokenTree,cli
created: 2026-08-12T04:22:56.944813+00:00
---

Embedding learning fix + semantic queries + CLI tests

Fixed _learn_embeddings: skipped co-occurrence signal for single-token words (jumped from leaf parent to merged token, whose embedding tensor had no gradients -> left-over raw vectors ~= random). Now jumps to each parent token, attributing all non-self rows. TokenTree.embedding() now derives from the matrix (its generation no longer needs all leaf ancestors), vocab-size-independent, no repeated L2 norm.

New query API: TokenTree.similar(token_id, top_k) returns ranked (token_id, cosine) pairs from the embedding matrix; embedding_matrix() returns (vocab_size, embed_dim) float32; embedding_points() exposes raw point cluster data for optional manual matrix builds.

Test updates: 44 token_tree tests pass (2 semantic tests hardened for cluster-reconstruction norm shrink and tiny-corpus token presence). Demo now shows nearest neighbors: 'to'->[to,out,and,in], 'the'->[the,our,of,this], 'and'->[not,but,are,go] on tinyshakespeare @ embed_dim 64.

New CLI test file apps/cli/tests/test_token_tree_commands.py (13 tests): corpus/token resolution helpers, train->load->encode/decode round-trip, stdin encode, similar, lineage, load-tree exit(2). Full CLI suite 59 passed; core-py token_tree+pugqeep all pass.