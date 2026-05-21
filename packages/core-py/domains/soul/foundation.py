"""
Stage 1: Foundation SLO - Memory & Search

Implements:
- HaulsStore: Vector database for persistent memory
- EndicIndex: Semantic search index
- Basic storage and retrieval
"""

import hashlib
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import logging

from .base import BaseSLO, SLOConfig, Experience, Thought, EvolutionStage

logger = logging.getLogger("slo.foundation")


class HaulsStore:
    """
    Vector database for persistent memory storage.
    Simple implementation using cosine similarity.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def _simple_hash_embedding(self, text: str) -> List[float]:
        """Create simple embedding from text hash."""
        h = hashlib.sha256(text.encode()).hexdigest()
        vec = [int(h[i : i + 2], 16) / 255.0 - 0.5 for i in range(0, min(len(h), self.dim * 2), 2)]
        while len(vec) < self.dim:
            vec.append(0.0)
        return vec[: self.dim]

    def store(self, key: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Store content with embedding."""
        self.vectors[key] = self._simple_hash_embedding(content)
        self.metadata[key] = metadata or {}
        self.metadata[key]["content"] = content
        return True

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Search for similar content."""
        query_vec = self._simple_hash_embedding(query)
        results = []

        for key, vec in self.vectors.items():
            similarity = self._cosine_similarity(query_vec, vec)
            results.append((key, similarity, self.metadata.get(key, {})))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(self, key: str) -> Optional[Dict]:
        """Retrieve by key."""
        if key in self.metadata:
            return self.metadata[key]
        return None

    def delete(self, key: str) -> bool:
        """Delete by key."""
        if key in self.vectors:
            del self.vectors[key]
            del self.metadata[key]
            return True
        return False

    def count(self) -> int:
        """Count stored items."""
        return len(self.vectors)


class EndicIndex:
    """
    Semantic search index for fast retrieval.
    Uses inverted index for keyword search.
    """

    def __init__(self):
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.documents: Dict[str, str] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def add(self, doc_id: str, content: str) -> bool:
        """Add document to index."""
        self.documents[doc_id] = content
        tokens = self._tokenize(content)
        for token in set(tokens):
            if doc_id not in self.index[token]:
                self.index[token].append(doc_id)
        return True

    def search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Search documents matching query."""
        tokens = self._tokenize(query)
        doc_scores: Dict[str, int] = defaultdict(int)

        for token in tokens:
            for doc_id in self.index.get(token, []):
                doc_scores[doc_id] += 1

        results = [(doc_id, score / max(len(tokens), 1)) for doc_id, score in doc_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def get(self, doc_id: str) -> Optional[str]:
        """Get document by ID."""
        return self.documents.get(doc_id)


class FoundationSLO(BaseSLO):
    """
    Stage 1: Foundation SLO

    Capabilities:
    - Persistent memory storage (HaulsStore)
    - Semantic search (EndicIndex)
    - Basic learning and recall
    """

    def __init__(self, config: Optional[SLOConfig] = None):
        super().__init__(config)
        self.stage = EvolutionStage.FOUNDATION

        # Core systems
        self.hauls_store = HaulsStore()
        self.endic_index = EndicIndex()

        # Memory layers
        self.working_memory: List[Experience] = []
        self.long_term_memory: Dict[str, Experience] = {}

        logger.info("Foundation SLO initialized")

    def process(self, input_data: Any) -> Thought:
        """Process input through foundation systems."""
        content = str(input_data)

        # Store in working memory
        exp = Experience(
            id=self._generate_id(content),
            content=content,
            timestamp=datetime.now(),
            importance=self._calculate_importance(content),
        )
        self.experiences.append(exp)
        self.working_memory.append(exp)

        # Store in HaulsStore
        self.hauls_store.store(exp.id, content, {"timestamp": exp.timestamp.isoformat()})

        # Index for search
        self.endic_index.add(exp.id, content)

        # Search for related memories
        related = self.recall(content, limit=3)

        # Generate thought
        reasoning = [f"Processed input: {content[:100]}"]
        if related:
            reasoning.append(f"Found {len(related)} related memories")

        thought = Thought(
            content=f"Processed and stored: {content[:200]}",
            stage=self.stage,
            confidence=0.7 + len(related) * 0.05,
            reasoning=reasoning,
        )
        self.thoughts.append(thought)

        # Progress evolution
        self._evolution_progress = min(1.0, self._evolution_progress + 0.01)

        return thought

    def learn(self, experience: Experience) -> bool:
        """Learn from experience."""
        # Store in long-term memory
        self.long_term_memory[experience.id] = experience

        # Store in HaulsStore
        self.hauls_store.store(experience.id, str(experience.content), experience.to_dict())

        # Index for retrieval
        self.endic_index.add(experience.id, str(experience.content))

        # Update patterns
        content_str = str(experience.content)
        for word in content_str.split()[:10]:
            self.patterns[word.lower()] = self.patterns.get(word.lower(), 0) + 1

        return True

    def recall(self, query: str, limit: int = 10) -> List[Experience]:
        """Recall relevant experiences."""
        # Search vector store
        vector_results = self.hauls_store.search(query, top_k=limit)

        # Search keyword index
        keyword_results = self.endic_index.search(query, limit=limit)

        # Combine results
        seen_ids = set()
        results = []

        for doc_id, score, _ in vector_results:
            if doc_id in self.long_term_memory and doc_id not in seen_ids:
                results.append(self.long_term_memory[doc_id])
                seen_ids.add(doc_id)

        for doc_id, score in keyword_results:
            if doc_id in self.long_term_memory and doc_id not in seen_ids:
                results.append(self.long_term_memory[doc_id])
                seen_ids.add(doc_id)

        return results[:limit]

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content."""
        return hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12]

    def _calculate_importance(self, content: str) -> float:
        """Calculate importance score for content."""
        score = 0.5
        if len(content) > 100:
            score += 0.1
        if "?" in content:
            score += 0.1
        if any(word in content.lower() for word in ["important", "critical", "key"]):
            score += 0.2
        return min(1.0, score)

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status."""
        status = super().get_status()
        status.update(
            {
                "hauls_store_items": self.hauls_store.count(),
                "index_documents": len(self.endic_index.documents),
                "working_memory": len(self.working_memory),
                "long_term_memory": len(self.long_term_memory),
                "patterns_learned": len(self.patterns),
            }
        )
        return status
