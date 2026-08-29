#!/usr/bin/env python3
"""
TokenTree proof — train the tree tokenizer on real project data and verify.

Usage:
    PYTHONPATH=packages/core-py python3 scripts/token_tree_demo.py [vocab_size] [embed_dim]

Trains a TokenTree on datasets/tinyshakespeare/input.txt (falling back to
datasets/api_conversations/input.txt), prints statistics, shows tree-walk
encodings and a merge-lineage decomposition, benchmarks parallel batch
encoding against serial, and round-trips save/load.
"""

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-py"))

from domains.training.token_tree import TokenTree


def _pick_dataset() -> Path:
    candidates = [
        REPO_ROOT / "data" / "tinyshakespeare" / "input.txt",
        REPO_ROOT / "data" / "api_conversations" / "input.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("no dataset found (expected data/*/input.txt)")


def main() -> int:
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    embed_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    dataset = _pick_dataset()
    texts = dataset.read_text(encoding="utf-8", errors="replace")
    print(f"dataset:  {dataset} ({len(texts) / 1e6:.1f} MB)")
    print(f"training: vocab_size={vocab_size} embed_dim={embed_dim} ...")

    t0 = time.perf_counter()
    tree = TokenTree().train(texts, vocab_size=vocab_size, embed_dim=embed_dim)
    t_train = time.perf_counter() - t0
    print(f"trained:  {t_train:.1f}s  ({tree.vocab_size} tokens, {len(tree.merges)} merges)")

    stats = tree.stats()
    print("\n-- stats --")
    print(f"  vocab_size                {stats['vocab_size']}")
    print(f"  num_merges                {stats['num_merges']}")
    print(f"  num_base_tokens           {stats['num_base_tokens']}")
    print(f"  embedding_points          {stats['embedding_points']}")
    print(f"  embedding_compression_ratio {stats['embedding_compression_ratio']}x")
    print(f"  library                   {stats['library']}")

    print("\n-- tree-walk encoding --")
    for phrase in [
        "to be, or not to be, that is the question",
        "the quick brown fox jumps over the lazy dog",
        "brilliant architecture with a query handler on every branch",
    ]:
        ids = tree.encode(phrase)
        tokens = [tree.itos[i].replace("</w>", "") for i in ids]
        print(f"  {phrase!r}\n    -> {tokens}  ({len(ids)} tokens, "
              f"round-trip={tree.decode(ids) == phrase.lower()})")

    common = "to" + "</w>"
    if common in tree.stoi:
        print(f"\n-- merge lineage for {common!r} --")
        print(tree.show_tree(tree.stoi[common]))

    print("\n-- embeddings generated from points --")
    emb = tree.embedding(tree.stoi.get("t", 0))
    if emb is not None:
        print(f"  token 't' embedding: shape={emb.shape} dtype={emb.dtype} "
              f"norm={float(emb @ emb) ** 0.5:.3f}")

    print("\n-- semantic query (nearest neighbors by generated embedding) --")
    if tree.embedding_points():
        for probe in ("to", "the", "and"):
            token_id = tree.stoi.get(probe + "</w>")
            if token_id is None:
                continue
            neigh = tree.similar(token_id, top_k=4)
            labels = [tree.itos[t].replace("</w>", "") for t, _ in neigh]
            scores = [f"{s:.2f}" for _, s in neigh]
            print(f"  {probe!r} -> {list(zip(labels, scores))}")

    print("\n-- parallel batch encode --")
    batch = [texts[i:i + 120] for i in range(0, min(len(texts), 120000), 120)]
    t0 = time.perf_counter()
    serial = [tree.encode(t) for t in batch]
    t_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    parallel = tree.encode_batch(batch, max_workers=8)
    t_par = time.perf_counter() - t0
    assert parallel == serial
    print(f"  {len(batch)} texts | serial {t_serial:.2f}s | "
          f"parallel {t_par:.2f}s | speedup {t_serial / max(t_par, 1e-9):.1f}x")

    print("\n-- persistence --")
    base = str(REPO_ROOT / "models" / "slonet-native" / "token_tree_demo")
    meta, points = tree.save(base)
    loaded = TokenTree.load(base)
    probe = "to be, or not to be, that is the question"
    same = loaded.decode(loaded.encode(probe)) == tree.decode(tree.encode(probe))
    print(f"  saved  {meta.name} / {points.name}")
    print(f"  loaded vocab={loaded.vocab_size} round-trip-match={same}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
