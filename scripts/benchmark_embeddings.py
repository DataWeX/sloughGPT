"""Embedding benchmark: throughput + cross-entry-point vector agreement.

Measures the canonical embedder (SloNet checkpoint -> word n-gram TF-IDF)
and every public entry point:

  - throughput: docs/sec + p50/p95/p99 latency per entry point
  - agreement: byte-identical vectors across entry points (the
    one-vector-space invariant — same algorithm/dimension/normalization)
  - sanity: L2 norm ~= 1 and dimension == expected for every vector

Usage:
    PYTHONPATH=. python scripts/benchmark_embeddings.py [--docs N] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np

_TOPICS = [
    "kernel boot loader driver firmware bios uefi grub init systemd".split(),
    "neural network gradient descent backpropagation loss optimizer adam".split(),
    "quantization int8 int4 calibration scale zero point bits".split(),
    "retrieval rag embedding vector search index chunk rerank".split(),
    "checkpoint soul weights snapshot resume training epoch".split(),
    "inference latency throughput batch gpu cpu streaming".split(),
]
_FILLER = (
    "the quick brown fox jumps over lazy dog lorem ipsum dolor sit amet "
    "consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore"
).split()


def build_corpus(n_docs: int, seed: int = 7) -> list[str]:
    import random

    rng = random.Random(seed)
    docs = []
    for i in range(n_docs):
        topic = _TOPICS[i % len(_TOPICS)]
        words = list(topic) + rng.choices(_FILLER, k=12)
        rng.shuffle(words)
        docs.append(f"doc {i}: " + " ".join(words))
    return docs


def _bench(fn, docs: list[str]) -> dict:
    t0 = time.perf_counter()
    for d in docs:
        fn(d)
    total = time.perf_counter() - t0

    lat_ms = []
    for d in docs[: min(200, len(docs))]:
        s = time.perf_counter()
        fn(d)
        lat_ms.append((time.perf_counter() - s) * 1000)
    lat_ms.sort()
    return {
        "docs_per_sec": round(len(docs) / total, 1),
        "p50_ms": round(statistics.median(lat_ms), 4),
        "p95_ms": round(lat_ms[int(len(lat_ms) * 0.95) - 1], 4),
        "p99_ms": round(lat_ms[int(len(lat_ms) * 0.99) - 1], 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=2000, help="corpus size")
    parser.add_argument("--json-out", type=str, default=None, help="write results JSON")
    args = parser.parse_args()

    from domain.inference._internal.embeddings import Embedder, InMemoryEmbedder
    from domain.inference._internal.text_embedder import embed_text
    from domain.inference._internal.vector_store import simple_embed

    docs = build_corpus(args.docs)

    entry_points = {
        "embed_text (canonical)": embed_text,
        "vector_store.simple_embed": simple_embed,
        "Embedder.embed_single": lambda d: Embedder(provider="n_gram").embed_single(d),
        "InMemoryEmbedder.embed": lambda d: InMemoryEmbedder().embed(d)[0],
    }

    print(f"corpus: {len(docs)} docs")
    results: dict = {"docs": len(docs), "throughput": {}, "agreement": {}, "sanity": {}}

    for name, fn in entry_points.items():
        stats = _bench(fn, docs)
        results["throughput"][name] = stats
        print(f"  {name:32s} {stats['docs_per_sec']:>10.1f} docs/s  p50={stats['p50_ms']}ms")

    # Agreement: every entry point must produce the canonical vector.
    sample = docs[:100]
    reference = [list(embed_text(d)) for d in sample]
    for name, fn in entry_points.items():
        agree = sum(1 for d, ref in zip(sample, reference, strict=True) if list(fn(d)) == ref)
        pct = 100.0 * agree / len(sample)
        results["agreement"][name] = round(pct, 1)
        print(f"  agreement {name:32s} {pct:5.1f}%")

    # Sanity: L2 norm and dimension on canonical output.
    norms = [float(np.linalg.norm(embed_text(d))) for d in sample[:20]]
    dims = {len(embed_text(d)) for d in sample[:20]}
    results["sanity"] = {
        "dim": sorted(dims),
        "norm_min": round(min(norms), 6),
        "norm_max": round(max(norms), 6),
        "norm_mean": round(statistics.mean(norms), 6),
    }
    print(f"  sanity dim={sorted(dims)} norm in [{min(norms):.4f}, {max(norms):.4f}]")

    all_agree = all(v == 100.0 for v in results["agreement"].values())
    print(f"  one-vector-space invariant: {'OK' if all_agree else 'VIOLATED'}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json_out}")

    return 0 if all_agree and dims == {384} else 1


if __name__ == "__main__":
    sys.exit(main())
