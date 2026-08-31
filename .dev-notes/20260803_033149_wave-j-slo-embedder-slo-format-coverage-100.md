---
id: 20260803_033149_wave-j-slo-embedder-slo-format-coverage-100
title: Wave J: slo_embedder + slo_format coverage 100%
status: done
tags: coverage,slo,embedder,format
created: 2026-08-03T03:31:49.948865+00:00
---

Wave J: slo_embedder + slo_format coverage 100%

Wave J complete: domains/inference/slo_embedder.py and slo_format.py both at 100% statement coverage.

## Coverage path
slo_embedder.py: 91% -> 100% (439/439 stmts); slo_format.py: 93% -> 100% (501/501 stmts).

## Bug fixes (slo_embedder.py)
1. BPETokenizer.load called on the class instead of an instance (instance method) -> always failed silently, BPE tokenizer never loaded. Fixed with instance-based bpe_tokenizer = BPETokenizer(); if bpe_tokenizer.load(bpe_path).
2. BPE sidecar load failure (corrupt json or load() returning False) left a half-initialized BPETokenizer instance as the active tokenizer because the except block did not reset bpe_tokenizer. Fixed to reset bpe_tokenizer = None on any load failure.

## New fast tests (tests/test_slo_embedder_edge.py)
- TestBpeTokenizerEdge / TestBuildVocabEdge / TestLabelByMeaningEdge (no-store + zero-vector None)
- TestTrainEmbedderWhitespaceFallback (fast whitespace tokenizer path)
- TestSaveCheckpointEdge: cleanup on params failure, bpe-save failure warns but writes file, rename+unlink both fail
- TestSloTextEmbedderLoad: invalid magic/sys-prompt -> None, no-vocab load, with-BPE load, corrupt-BPE-sidecar skip, BPE load-failure skip
- TestEmbedDimAdaptation: pad 8->32, truncate 8->4, batch

## New fast tests (test_slo_format.py additions)
- SouParser edge branches: non-numeric PARAMETER/CONTEXT/PERSONALITY kept as strings, QUANTIZATION, METADATA generic key, BEHAVIOR overflow->inf and unicode-digit->string, load/save round trip
- save_soul value-type conversion branches (tensor .data, .cpu().numpy(), .detach().cpu().numpy(), list, scalar), error-skip on unconvertible value, zero-params log, lineage-from-model when profile lineage empty, temp cleanup + unlink-error-swallowed
- load_soul v2 bad state json, generate_sample_dialogue model without generate/forward

## Regression
test_slo_embedder slow suite (29 passed, 1 skipped, 1 xpassed), test_slo_format, test_vector_store, test_slo_manager, test_embedding_service, test_embeddings, test_soul_engine, test_knowledge_memory, test_knowledge_augmenter, test_morph_tokenizer, test_bpe_tokenizer, test_tokenizer suites all green.