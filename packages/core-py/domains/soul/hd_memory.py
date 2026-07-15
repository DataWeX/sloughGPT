"""
Hyperdimensional Memory Store

Integrates HD computing into SloEngine for fast semantic memory:
- O(1) similarity search via hypervector operations
- Conversation encoding as semantic vectors
- Fast retrieval without expensive embedding models
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("man.hd_memory")


@dataclass
class HDMemoryItem:
    """A memory item with hypervector representation."""

    id: str
    content: str
    hypervector: List[float]
    metadata: Dict[str, Any]
    timestamp: float
    role: str  # 'user', 'assistant', 'system'


class HDMemoryStore:
    """
    Hyperdimensional memory store for fast semantic retrieval.

    Uses holographic reduced representations (HRR) for:
    - Fast O(n) similarity search (vs O(n*d) for dense vectors)
    - No external embedding model needed
    - Robust to noise through bundle operations
    """

    def __init__(self, dim: int = 10000, max_items: int = 1000):
        self.dim = dim
        self.max_items = max_items
        self.items: List[HDMemoryItem] = []
        self.role_vectors: Dict[str, List[float]] = {}

        # Lazy import to avoid circular deps
        self._hyperdim = None
        self._initialized = False

    def _get_hyperdim(self):
        """Lazy-load hyperdimensional processor."""
        if self._hyperdim is None:
            try:
                from domains.soul.quantum import HyperdimensionalProcessor

                self._hyperdim = HyperdimensionalProcessor(dim=self.dim)
                self._initialize_role_vectors()
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize HyperdimensionalProcessor: {e}", extra={"tag": "MODEL"})
                raise
        return self._hyperdim

    def _initialize_role_vectors(self) -> None:
        """Pre-compute role-specific binding vectors."""
        hd = self._hyperdim
        self.role_vectors = {
            "user": hd.encode("ROLE_USER"),
            "assistant": hd.encode("ROLE_ASSISTANT"),
            "system": hd.encode("ROLE_SYSTEM"),
        }

    def encode_content(self, content: str) -> List[float]:
        """Encode content as hypervector."""
        hd = self._get_hyperdim()
        return hd.encode_text(content)

    def encode_with_role(self, content: str, role: str) -> List[float]:
        """Encode content with role binding for context awareness."""
        hd = self._get_hyperdim()
        content_vec = hd.encode_text(content)
        role_vec = self.role_vectors.get(role, hd.encode("ROLE_USER"))
        return hd.bind(content_vec, role_vec)

    def add(
        self, content: str, role: str = "user", metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a memory item.

        Args:
            content: Text content to encode
            role: Message role (user/assistant/system)
            metadata: Optional metadata

        Returns:
            Memory item ID
        """
        # Encode with role for context
        hypervector = self.encode_with_role(content, role)

        # Create memory item
        item_id = f"mem_{len(self.items)}_{int(time.time() * 1000)}"
        item = HDMemoryItem(
            id=item_id,
            content=content,
            hypervector=hypervector,
            metadata=metadata or {},
            timestamp=time.time(),
            role=role,
        )

        self.items.append(item)

        # Evict oldest if over capacity
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]

        return item_id

    def search(
        self, query: str, top_k: int = 5, role_filter: Optional[str] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Search memory using hypervector similarity.

        Args:
            query: Query text
            top_k: Number of results to return
            role_filter: Optional filter by role

        Returns:
            List of (id, content, similarity) tuples
        """
        if not self.items:
            return []

        hd = self._get_hyperdim()
        query_vec = hd.encode_text(query)

        # Score all items
        scored = []
        for item in self.items:
            if role_filter and item.role != role_filter:
                continue

            # Cosine similarity via HD dot product
            sim = hd.similarity(query_vec, item.hypervector)
            scored.append((item.id, item.content, sim))

        # Sort by similarity (descending)
        scored.sort(key=lambda x: x[2], reverse=True)

        return scored[:top_k]

    def get_context(self, query: str, max_chars: int = 500, include_roles: bool = True) -> str:
        """
        Get relevant context for a query.

        Args:
            query: Query text
            max_chars: Maximum characters to return
            include_roles: Include role labels in context

        Returns:
            Concatenated relevant context
        """
        results = self.search(query, top_k=10)

        if not results:
            return ""

        context_parts = []
        total_chars = 0

        for _, content, sim in results:
            if sim < 0.01:  # Skip very low-similarity items
                continue

            if total_chars + len(content) > max_chars:
                break

            if include_roles:
                # Find the item to get its role
                for item in self.items:
                    if item.content == content:
                        role_label = item.role.replace("_", " ").title()
                        context_parts.append(f"[{role_label}]: {content[:200]}")
                        break
            else:
                context_parts.append(content[:200])

            total_chars += len(content)

        return "\n".join(context_parts)

    def bundle_recent(self, n: int = 10) -> List[float]:
        """
        Bundle recent memories into single hypervector.
        Useful for representing conversation state.
        """
        hd = self._get_hyperdim()
        recent = self.items[-n:] if self.items else []
        vectors = [item.hypervector for item in recent]
        return hd.bundle(vectors) if vectors else [0] * self.dim

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_items": len(self.items),
            "max_items": self.max_items,
            "dimension": self.dim,
            "initialized": self._initialized,
            "roles": {
                role: sum(1 for item in self.items if item.role == role)
                for role in set(item.role for item in self.items)
            },
        }

    def clear(self) -> int:
        """Clear all memory items."""
        count = len(self.items)
        self.items = []
        return count

    def prune(self, similarity_threshold: float = 0.95) -> int:
        """
        Remove highly similar items (near-duplicates).

        Args:
            similarity_threshold: Items with similarity above this are pruned

        Returns:
            Number of items pruned
        """
        if len(self.items) < 2:
            return 0

        hd = self._get_hyperdim()
        to_remove = set()

        for i, item1 in enumerate(self.items):
            if item1.id in to_remove:
                continue

            for j, item2 in enumerate(self.items[i + 1 :], i + 1):
                if item2.id in to_remove:
                    continue

                sim = hd.similarity(item1.hypervector, item2.hypervector)
                if sim > similarity_threshold:
                    # Keep newer one
                    if item1.timestamp < item2.timestamp:
                        to_remove.add(item1.id)
                    else:
                        to_remove.add(item2.id)

        if to_remove:
            self.items = [item for item in self.items if item.id not in to_remove]

        return len(to_remove)


__all__ = ["HDMemoryStore", "HDMemoryItem"]
