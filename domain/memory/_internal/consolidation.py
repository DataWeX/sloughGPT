"""Near-duplicate consolidation for the auto-memory layer.

Facts accumulate as the extractor re-states the same knowledge across turns
("The user prefers Zed over VS Code" vs "User prefers the editor Zed"). Each
restatement is a new fact because exact-content dedup is hash-based. This
module flags such near-duplicates using the same n-gram cosine embedding the
store uses for search, so consolidation agrees with retrieval semantics.

``plan_consolidation`` is a pure function: it reads fact dicts and returns a
removal plan. The task handler applies the plan through ``MemoryService``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from domain.memory._internal.embedding import cosine_similarity, ngram_embed

logger = logging.getLogger(__name__)


def _embed_cache(facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Precompute one n-gram embedding per fact entry id."""
    return {f["id"]: ngram_embed(f["content"]) for f in facts if f.get("id")}


def plan_consolidation(facts: List[Dict[str, Any]], threshold: float = 0.80) -> Dict[str, Any]:
    """
    Plan which facts are near-duplicates and should be removed.

    Facts are grouped by topic, then two facts are unioned when their n-gram
    cosine similarity is at or above ``threshold`` (transitively, so a chain
    of overlapping restatements collapses into one cluster). Within each
    cluster the longest fact is kept and the rest are proposed for removal.

    Args:
        facts: fact dicts from ``MemoryService.list_all`` (need ``id``,
            ``content``, ``topic``).
        threshold: min cosine similarity (0..1) to treat two facts as
            near-duplicates.

    Returns:
        dict with:
            keep_ids: entry ids to retain (original input order).
            remove_ids: entry ids to delete (original input order).
            groups: list of ``{"keep": <fact>, "duplicates": [<fact>, ...]}``
                for every cluster with at least one duplicate.
            removed_count: number of ids proposed for removal.

    Side effects:
        - none; read-only.
    """
    if not facts:
        return {"keep_ids": [], "remove_ids": [], "groups": [], "removed_count": 0}

    # Filter out facts without required fields early
    valid_facts = [f for f in facts if f.get("id") and f.get("content")]
    if not valid_facts:
        return {"keep_ids": [], "remove_ids": [], "groups": [], "removed_count": 0}

    by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for f in valid_facts:
        by_topic.setdefault(f.get("topic") or "general", []).append(f)

    cache = _embed_cache(valid_facts)
    parent = {f["id"]: f["id"] for f in valid_facts}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in by_topic.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                va, vb = cache.get(a["id"]), cache.get(b["id"])
                if va is not None and vb is not None:
                    if cosine_similarity(va, vb) >= threshold:
                        union(a["id"], b["id"])

    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for f in valid_facts:
        clusters.setdefault(find(f["id"]), []).append(f)

    input_order = [f["id"] for f in valid_facts]
    keep_set: set = set()
    remove_set: set = set()
    groups: List[Dict[str, Any]] = []
    for members in clusters.values():
        if len(members) < 2:
            keep_set.add(members[0]["id"])
            continue
        best = max(members, key=lambda m: len(m.get("content") or ""))
        keep_set.add(best["id"])
        dups = [m for m in members if m["id"] != best["id"]]
        remove_set.update(m["id"] for m in dups)
        groups.append({"keep": best, "duplicates": dups})

    keep_ids = [i for i in input_order if i in keep_set]
    remove_ids = [i for i in input_order if i in remove_set]

    return {
        "keep_ids": keep_ids,
        "remove_ids": remove_ids,
        "groups": groups,
        "removed_count": len(remove_ids),
    }
