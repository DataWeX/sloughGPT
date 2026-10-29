#!/usr/bin/env python
"""RAG chunking benchmark: fine char windows vs legacy 512-token windows.

The retrieval injector caps the RAG block at 2000 chars split across the top
5 ranks (~400 chars each). A 512-token chunk (~3KB) means each rank shows
~13% of its chunk; the fine windows (~450 target, max 600) are sized so a
rank fits whole. This measures both chunkers on the SAME deterministic
corpus (reused from benchmark_bm25) and checks retrieval did not regress:

  - chunk size in chars: avg / median / p95 / max
  - budget fit: % of chunks <= 600 (rank hard cap) and <= 400 (rank slice)
  - retrieval: recall@5 and MRR over planted queries, fine vs legacy

Verdict (exit 1 on failure):
  * fine windows fit: every chunk <= 600 chars (atomic over-long words aside)
  * recall@5 of fine is within 2 points of legacy (no accuracy regression)

Usage:
    PYTHONPATH=. python scripts/benchmark_rag_chunking.py
        [--docs N] [--queries N] [--top-k K] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages/core-py"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_bm25 import build_corpus, percentile  # noqa: E402

from domain.cognition._internal.rag import (  # noqa: E402
    FINE_CHUNK_MAX_CHARS,
    ProductionRAG,
)


def _pad_doc(doc: str, target_chars: int = 3000) -> str:
    """Grow a corpus doc to realistic ingest size (~3KB).

    benchmark_bm25's docs are ~30 words — under BOTH chunk sizes, so every
    doc chunks to one piece and the benchmark would compare nothing. The
    padding is deterministic filler; the planted query keywords stay in the
    doc head where chunking must still land them intact.
    """
    filler = (
        " the system keeps operational notes about state transitions and"
        " scheduling behaviour under load while operators watch the dashboards"
    )
    parts = [doc]
    size = len(doc)
    i = 0
    while size < target_chars:
        sentence = f"{filler} entry {i}."
        parts.append(sentence)
        size += len(sentence) + 1
        i += 1
    return " ".join(parts)


def chunk_with(mode: str, docs: list[str]) -> tuple[ProductionRAG, list[int]]:
    """Index all docs under one chunking mode; return the RAG + chunk char lens."""
    rag = ProductionRAG()
    lens: list[int] = []
    for i, doc in enumerate(docs):
        if mode == "fine":
            ids = rag.add_document(doc, metadata={"doc_id": i}, rebuild_index=False)
        else:
            ids = rag.add_document(doc, metadata={"doc_id": i}, chunk_size=512, rebuild_index=False)
        by_id = {c.id: c for c in rag.retriever.chunks}
        lens.extend(len(by_id[cid].content) for cid in ids)
    rag.retriever.build_index()
    return rag, lens


def evaluate(rag: ProductionRAG, queries: list[str], relevant: dict, top_k: int):
    hits = 0
    mrr_sum = 0.0
    for query in queries:
        results = rag.retriever.retrieve(query, top_k=top_k)
        ranked = [r.chunk.metadata.get("doc_id") for r in results]
        first = next((i for i, d in enumerate(ranked, 1) if d in relevant[query]), None)
        if first is not None:
            hits += 1
            mrr_sum += 1.0 / first
    return hits / len(queries), mrr_sum / len(queries)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=int, default=120)
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    docs, queries, relevant_pairs = build_corpus(args.docs, args.queries)
    docs = [_pad_doc(d) for d in docs]
    # build_corpus returns relevant as {query_index: set[doc_ids]} — key by
    # query text for the evaluator.
    relevant = {queries[i]: rel for i, rel in relevant_pairs.items()}

    report: dict = {"docs": len(docs), "queries": len(queries), "top_k": args.top_k}
    verdicts: list[tuple[str, bool, str]] = []

    rags: dict[str, ProductionRAG] = {}
    for mode in ("legacy", "fine"):
        rag, lens = chunk_with(mode, docs)
        rags[mode] = rag
        stats = {
            "chunks": len(lens),
            "avg": round(statistics.mean(lens), 1),
            "median": statistics.median(lens),
            "p95": percentile(lens, 95),
            "max": max(lens),
            "fit_600_pct": round(100 * sum(1 for x in lens if x <= 600) / len(lens), 1),
            "fit_400_pct": round(100 * sum(1 for x in lens if x <= 400) / len(lens), 1),
        }
        recall, mrr = evaluate(rag, queries, relevant, args.top_k)
        stats["recall@5"] = round(recall, 3)
        stats["mrr"] = round(mrr, 3)
        report[mode] = stats

    legacy, fine = report["legacy"], report["fine"]

    print(f"{'metric':<16} {'legacy(512tok)':>14} {'fine(450chr)':>14}")
    for key in (
        "chunks",
        "avg",
        "median",
        "p95",
        "max",
        "fit_600_pct",
        "fit_400_pct",
        "recall@5",
        "mrr",
    ):
        print(f"{key:<16} {legacy[key]:>14} {fine[key]:>14}")

    overlong = fine["max"] > FINE_CHUNK_MAX_CHARS
    verdicts.append(
        (
            "fine ranks fit the 600-char cap",
            fine["fit_600_pct"] >= 99.9 and not overlong,
            f"{fine['fit_600_pct']}% fit, max={fine['max']}",
        )
    )
    verdicts.append(
        (
            "fine recall@5 within 2pts of legacy",
            fine["recall@5"] >= legacy["recall@5"] - 0.02,
            f"{fine['recall@5']} vs {legacy['recall@5']}",
        )
    )
    verdicts.append(
        (
            "fine improves whole-fit in a 400-char slice",
            fine["fit_400_pct"] >= legacy["fit_400_pct"],
            f"{fine['fit_400_pct']}% vs {legacy['fit_400_pct']}%",
        )
    )

    print("\nVerdict:")
    failed = 0
    for label, ok, detail in verdicts:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label} — {detail}")
        failed += 0 if ok else 1

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
