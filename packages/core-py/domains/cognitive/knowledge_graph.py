"""
Knowledge Graph - DEPRECATED

This module is deprecated. Use domains.cognitive.knowledge_graph_v2 instead.

Canonical: KnowledgeGraphV2
"""

# Re-export v2 as the canonical implementation
from domains.cognitive.knowledge_graph_v2 import (
    KnowledgeGraph,
    RelationType,
    Entity,
    Fact,
)

import sqlite3
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


# Legacy types for backwards compatibility
class RelationType(Enum):
    IS_A = "is_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    ENABLES = "enables"
    REQUIRES = "requires"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    EXAMPLE_OF = "example_of"


class Confidence(Enum):
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


@dataclass
class KnowledgeNode:
    """Represents a concept/entity in the knowledge graph"""

    id: str
    text: str
    concept_type: str
    confidence: Confidence
    created_at: float
    last_accessed: float
    access_count: int
    metadata: Dict


@dataclass
class KnowledgeEdge:
    """Represents a relationship between nodes"""

    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float
    confidence: Confidence


class KnowledgeGraph:
    """Knowledge graph for concepts and relationships"""

    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.db_path = db_path
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                concept_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER NOT NULL,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL NOT NULL,
                confidence REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, relation_type)
            )
        """)

        self.conn.commit()

    def add_node(self, text: str, concept_type: str, metadata: Optional[Dict] = None) -> str:
        """Add a node to the graph."""
        node_id = f"node_{len(self.nodes)}_{int(time.time())}"

        node = KnowledgeNode(
            id=node_id,
            text=text,
            concept_type=concept_type,
            confidence=Confidence.MEDIUM,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0,
            metadata=metadata or {},
        )

        self.nodes[node_id] = node
        self._save_node(node)

        return node_id

    def _save_node(self, node: KnowledgeNode) -> None:
        """Save node to database."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                node.id,
                node.text,
                node.concept_type,
                node.confidence.value,
                node.created_at,
                node.last_accessed,
                node.access_count,
                json.dumps(node.metadata),
            ),
        )
        self.conn.commit()

    def add_edge(
        self, source_id: str, target_id: str, relation_type: RelationType, weight: float = 1.0
    ) -> None:
        """Add an edge between nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return

        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            confidence=Confidence.MEDIUM,
        )

        self.edges.append(edge)
        self._save_edge(edge)

    def _save_edge(self, edge: KnowledgeEdge) -> None:
        """Save edge to database."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO edges VALUES (?, ?, ?, ?, ?)
        """,
            (
                edge.source_id,
                edge.target_id,
                edge.relation_type.value,
                edge.weight,
                edge.confidence.value,
            ),
        )
        self.conn.commit()

    def get_neighbors(self, node_id: str) -> List[KnowledgeNode]:
        """Get neighboring nodes."""
        neighbor_ids = set()

        for edge in self.edges:
            if edge.source_id == node_id:
                neighbor_ids.add(edge.target_id)
            elif edge.target_id == node_id:
                neighbor_ids.add(edge.source_id)

        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

    def find_path(self, start_id: str, end_id: str) -> List[str]:
        """Find path between two nodes (simple BFS)."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return []

        visited = {start_id}
        queue = [(start_id, [start_id])]

        while queue:
            current, path = queue.pop(0)

            if current == end_id:
                return path

            for edge in self.edges:
                neighbor = None
                if edge.source_id == current and edge.target_id not in visited:
                    neighbor = edge.target_id
                elif edge.target_id == current and edge.source_id not in visited:
                    neighbor = edge.source_id

                if neighbor:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []

    def close(self) -> None:
        """Close database."""
        if self.conn:
            self.conn.close()


__all__ = ["KnowledgeGraph", "KnowledgeNode", "KnowledgeEdge", "RelationType", "Confidence"]
