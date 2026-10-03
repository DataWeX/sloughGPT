"""BM25 benchmark: latency + recall of sparse retrieval vs dense.

Builds a deterministic synthetic corpus with planted relevant docs per
query, then measures:
  - index build time
  - per-query latency (p50/p99) for BM25 sparse, dense, and hybrid paths
  - recall@k and MRR@k for BM25, and MRR@k with rerank on/off
    (live config: sparse + rerank, dense off)

Usage:
    PYTHONPATH=. python scripts/benchmark_bm25.py [--docs N] [--queries N]
        [--repeats N] [--top-k K] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time

sys.path.insert(0, "packages/core-py")


def build_corpus(n_docs: int, n_queries: int, seed: int = 7):
    """Synthetic corpus: topic docs + planted keyword docs per query."""
    rng = random.Random(seed)
    topics = [
        "kernel boot loader driver firmware bios uefi grub init systemd".split(),
        "neural network gradient descent backpropagation loss optimizer adam".split(),
        "phoneme vowel consonant ipa pronunciation accent dialect prosody".split(),
        "quantization int8 int4 calibration scale zero point bits".split(),
        "retrieval rag embedding vector search index chunk rerank".split(),
        "checkpoint soul weights snapshot resume training epoch".split(),
        "tokenizer bpe merges vocab encode decode ids".split(),
        "consciousness qualia narrative belief episode reflection".split(),
        "planner kanban board card column workflow sprint".split(),
        "inference latency throughput batch gpu cpu streaming".split(),
    ]
    filler = (
        "the quick brown fox jumps over lazy dog lorem ipsum dolor sit amet "
        "consectetur adipiscing elit sed do eiusmod tempor incididunt".split()
    )
    docs: list[str] = []
    queries: list[str] = []
    relevant: dict[int, set[int]] = {}
    for qi in range(n_queries):
        topic = topics[qi % len(topics)]
        keywords = rng.sample(topic, 3)
        queries.append(" ".join(keywords))
        relevant[qi] = set()
        # Plant 3 fully-relevant docs: the query phrase kept contiguous
        # (phrase/proximity signal) embedded in shuffled topic words.
        # Distractors below share the vocabulary but never the phrase.
        for _ in range(3):
            words = rng.sample(topic, 4) + rng.sample(filler, 10)
            rng.shuffle(words)
            pos = rng.randint(0, len(words))
            words[pos:pos] = keywords
            relevant[qi].add(len(docs))
            docs.append(" ".join(words))
    # Distractors: partial overlap + pure noise.
    while len(docs) < n_docs:
        topic = rng.choice(topics)
        words = rng.sample(topic, 3) + rng.choices(filler, k=25)
        rng.shuffle(words)
        docs.append(" ".join(words))
    return docs, queries, relevant


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    k = min(len(ordered) - 1, int(len(ordered) * p / 100))
    return ordered[k]


def mrr_at_k(ranked_ids: list[int], relevant: set[int]) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="BM25 sparse vs dense retrieval benchmark")
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from domain.cognition._internal.rag import BM25Indexer, HybridRetriever, TextChunk

    docs, queries, relevant = build_corpus(args.docs, args.queries)
    chunks = [TextChunk(id=str(i), content=d, metadata={}) for i, d in enumerate(docs)]

    t0 = time.perf_counter()
    bm25 = BM25Indexer()
    bm25.index(chunks)
    index_ms = (time.perf_counter() - t0) * 1000

    retriever = HybridRetriever()
    retriever.chunks = chunks
    retriever.build_index()

    # Live config: sparse + rerank, dense off.
    live = HybridRetriever(use_dense=False)
    live.chunks = chunks
    live.build_index()

    sparse_lat: list[float] = []
    dense_lat: list[float] = []
    hybrid_lat: list[float] = []
    rerank_lat: list[float] = []
    hits = 0
    total_rel = 0
    rr_sparse = 0.0
    rr_rerank = 0.0
    for qi, q in enumerate(queries):
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            sres = bm25.score(q)[: args.top_k]
            sparse_lat.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            retriever._dense_search(q, args.top_k)
            dense_lat.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            retriever.retrieve(q, top_k=args.top_k)
            hybrid_lat.append((time.perf_counter() - t0) * 1000)

        got = {doc_id for doc_id, _ in sres}
        hits += len(got & relevant[qi])
        total_rel += len(relevant[qi])
        rr_sparse += mrr_at_k([doc_id for doc_id, _ in sres], relevant[qi])

        # Rerank on the live (sparse-only) fusion results.
        t0 = time.perf_counter()
        live.use_rerank = True
        on = [r.chunk.id for r in live.retrieve(q, top_k=args.top_k)]
        rerank_lat.append((time.perf_counter() - t0) * 1000 / max(len(on), 1))
        rr_rerank += mrr_at_k([int(i) for i in on], relevant[qi])

    nq = len(queries)
    result = {
        "docs": len(docs),
        "queries": nq,
        "repeats": args.repeats,
        "top_k": args.top_k,
        "index_ms": round(index_ms, 2),
        "sparse_p50_ms": round(statistics.median(sparse_lat), 3),
        "sparse_p99_ms": round(percentile(sparse_lat, 99), 3),
        "dense_p50_ms": round(statistics.median(dense_lat), 3),
        "dense_p99_ms": round(percentile(dense_lat, 99), 3),
        "hybrid_p50_ms": round(statistics.median(hybrid_lat), 3),
        "hybrid_p99_ms": round(percentile(hybrid_lat, 99), 3),
        "rerank_per_doc_ms": round(statistics.median(rerank_lat), 3),
        "sparse_recall_at_k": round(hits / max(total_rel, 1), 4),
        "sparse_mrr_at_k": round(rr_sparse / max(nq, 1), 4),
        "reranked_mrr_at_k": round(rr_rerank / max(nq, 1), 4),
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.json_out}")


if __name__ == "__main__":
    main()

