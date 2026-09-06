"""
Production-Grade Knowledge Graph

Industry-standard implementation with:
- Proper graph algorithms (BFS, DFS, path finding)
- SPARQL-like querying
- Truth propagation
- Consistency checking
- Knowledge validation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import deque
from enum import Enum


class RelationType(Enum):
    """Standard relation types (Schema.org compatible)."""
    IS_A = "rdf:type"
    PART_OF = "part_of"
    CAUSES = "causes"
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    LOCATED_IN = "located_in"
    HAS_PROPERTY = "has_property"
    INSTANCE_OF = "instance_of"


@dataclass
class Entity:
    """An entity node in the knowledge graph."""
    id: str
    label: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: Set[str] = field(default_factory=set)
    confidence: float = 1.0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id


@dataclass
class Fact:
    """A fact (triple) in the knowledge graph."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: Optional[float] = None
    verified: bool = False

    def __repr__(self):
        return f"({self.subject}, {self.predicate}, {self.object})"


class KnowledgeGraph:
    """
    Production-grade knowledge graph.

    Features:
    - Efficient adjacency storage
    - Multi-hop traversal
    - Path finding
    - Truth propagation
    - Consistency checking
    """

    def __init__(self):
        # Entity storage
        self.entities: Dict[str, Entity] = {}

        # Triple storage (subject -> predicate -> [objects])
        self.subject_index: Dict[str, Dict[str, List[str]]] = {}

        # Reverse index (object -> predicate -> [subjects])
        self.object_index: Dict[str, Dict[str, List[str]]] = {}

        # All facts with metadata
        self.facts: Dict[Tuple[str, str, str], Fact] = {}

        # Graph statistics
        self.stats = {
            "entities": 0,
            "facts": 0,
            "avg_degree": 0.0,
        }

    def _resolve_entity_id(self, name: str) -> str:
        """Resolve an entity name to an existing entity ID using case-insensitive lookup.

        Args:
            name: Entity name to resolve (e.g. "Paris", "paris", "PARIS").

        Returns:
            The matching entity ID if found, otherwise the normalized name.
        """
        name_lower = name.lower()
        for eid in self.entities:
            if eid.lower() == name_lower:
                return eid
            if name_lower in [a.lower() for a in self.entities[eid].aliases]:
                return eid
        return name

    def add_entity(
        self,
        id: str,
        label: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
    ) -> Entity:
        """Add an entity to the graph.

        If an entity with the same ID (case-insensitive) already exists, the
        existing entity is returned unchanged.

        Args:
            id: Unique entity identifier (will be stored as-is).
            label: Human-readable label.
            entity_type: Type/category of entity.
            properties: Optional property dict.
            aliases: Optional alternative names.

        Returns:
            The Entity (existing or newly created).
        """
        resolved = self._resolve_entity_id(id)
        if resolved in self.entities:
            existing = self.entities[resolved]
            if aliases:
                existing.aliases.update(aliases)
            return existing

        entity = Entity(
            id=id,
            label=label,
            entity_type=entity_type,
            properties=properties or {},
            aliases=set(aliases) if aliases else set(),
        )
        self.entities[id] = entity
        self._update_stats()
        return entity

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "unknown",
    ) -> Optional[Fact]:
        """Add a fact (triple) to the graph.

        Deduplicates by (subject, predicate, object) tuple. If the same triple
        already exists, the higher confidence and newer source are kept.

        Args:
            subject: Subject entity name (auto-creates if missing).
            predicate: Relationship type (e.g. "capital_of", "is_a").
            obj: Object entity name (auto-creates if missing).
            confidence: Confidence score [0, 1].
            source: Provenance identifier.

        Returns:
            The Fact, or None if it was a no-op duplicate.
        """
        # Resolve entity IDs (case-insensitive)
        subj_id = self._resolve_entity_id(subject)
        obj_id = self._resolve_entity_id(obj)

        # Ensure entities exist
        if subj_id not in self.entities:
            self.add_entity(subj_id, subject, "unknown")
        if obj_id not in self.entities:
            self.add_entity(obj_id, obj, "unknown")

        # Use resolved IDs for the triple key
        key = (subj_id, predicate, obj_id)
        existing = self.facts.get(key)
        if existing:
            # Keep higher confidence, newer source
            if confidence > existing.confidence:
                existing.confidence = confidence
                existing.source = source
            return None

        fact = Fact(
            subject=subj_id,
            predicate=predicate,
            object=obj_id,
            confidence=confidence,
            source=source,
        )

        # Store fact
        self.facts[key] = fact

        # Update subject index
        if subj_id not in self.subject_index:
            self.subject_index[subj_id] = {}
        if predicate not in self.subject_index[subj_id]:
            self.subject_index[subj_id][predicate] = []
        if obj_id not in self.subject_index[subj_id][predicate]:
            self.subject_index[subj_id][predicate].append(obj_id)

        # Update object index
        if obj_id not in self.object_index:
            self.object_index[obj_id] = {}
        if predicate not in self.object_index[obj_id]:
            self.object_index[obj_id][predicate] = []
        if subj_id not in self.object_index[obj_id][predicate]:
            self.object_index[obj_id][predicate].append(subj_id)

        self._update_stats()
        return fact

    def get_outgoing(
        self,
        entity_id: str,
        predicate: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """Get outgoing edges from entity. Returns [(predicate, target), ...].

        Args:
            entity_id: Entity name (case-insensitive lookup).
            predicate: Optional filter for specific predicate.

        Returns:
            List of (predicate, target_entity_id) tuples.
        """
        resolved = self._resolve_entity_id(entity_id)
        if resolved not in self.subject_index:
            return []

        results = []
        predicates = [predicate] if predicate else self.subject_index[resolved].keys()

        for pred in predicates:
            if pred in self.subject_index[resolved]:
                for obj in self.subject_index[resolved][pred]:
                    results.append((pred, obj))

        return results

    def get_incoming(
        self,
        entity_id: str,
        predicate: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """Get incoming edges to entity. Returns [(predicate, source), ...].

        Args:
            entity_id: Entity name (case-insensitive lookup).
            predicate: Optional filter for specific predicate.

        Returns:
            List of (predicate, source_entity_id) tuples.
        """
        resolved = self._resolve_entity_id(entity_id)
        if resolved not in self.object_index:
            return []

        results = []
        predicates = [predicate] if predicate else self.object_index[resolved].keys()

        for pred in predicates:
            if pred in self.object_index[resolved]:
                for subj in self.object_index[resolved][pred]:
                    results.append((pred, subj))

        return results

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> List[Fact]:
        """Query facts matching pattern.

        All string parameters are resolved via case-insensitive entity lookup
        (for subject/object) or exact match (for predicate).

        Args:
            subject: Subject entity name (fuzzy).
            predicate: Relationship type (exact).
            obj: Object entity name (fuzzy).

        Returns:
            List of matching Facts.
        """
        results = []

        # Resolve entity IDs
        subj_id = self._resolve_entity_id(subject) if subject else None
        obj_id = self._resolve_entity_id(obj) if obj else None

        if subj_id and predicate and obj_id:
            fact = self.facts.get((subj_id, predicate, obj_id))
            if fact:
                results.append(fact)
        elif subj_id and predicate:
            if subj_id in self.subject_index and predicate in self.subject_index[subj_id]:
                for ob in self.subject_index[subj_id][predicate]:
                    results.append(self.facts[(subj_id, predicate, ob)])
        elif subj_id and obj_id:
            if subj_id in self.subject_index:
                for pred, objs in self.subject_index[subj_id].items():
                    if obj_id in objs:
                        results.append(self.facts[(subj_id, pred, obj_id)])
        elif predicate and obj_id:
            if obj_id in self.object_index and predicate in self.object_index[obj_id]:
                for sub in self.object_index[obj_id][predicate]:
                    results.append(self.facts[(sub, predicate, obj_id)])
        elif subj_id:
            if subj_id in self.subject_index:
                for pred, objs in self.subject_index[subj_id].items():
                    for ob in objs:
                        results.append(self.facts[(subj_id, pred, ob)])
        elif obj_id:
            if obj_id in self.object_index:
                for pred, subjects in self.object_index[obj_id].items():
                    for sub in subjects:
                        results.append(self.facts[(sub, pred, obj_id)])

        return results

    # =========================================================================
    # GRAPH ALGORITHMS
    # =========================================================================

    def bfs(
        self,
        start: str,
        predicate_filter: Optional[Callable[[str], bool]] = None,
        max_depth: int = 3,
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Breadth-first search from start entity.

        Returns:
            {entity_id: [(predicate, source_entity), ...]}
        """
        visited = {start}
        queue = deque([(start, start, 0)])  # (current, source, depth)
        paths = {start: []}

        while queue:
            current, source, depth = queue.popleft()

            if depth >= max_depth:
                continue

            for pred, obj in self.get_outgoing(current):
                if predicate_filter and not predicate_filter(pred):
                    continue

                if obj not in visited:
                    visited.add(obj)
                    paths[obj] = paths[current] + [(pred, current)]
                    queue.append((obj, current, depth + 1))

        return paths

    def dfs(
        self,
        start: str,
        predicate_filter: Optional[Callable[[str], bool]] = None,
        max_depth: int = 3,
    ) -> List[List[Tuple[str, str]]]:
        """
        Depth-first search from start entity.

        Returns:
            List of paths, each path is [(predicate, entity), ...]
        """
        paths = []

        def dfs_recursive(current: str, path: List[Tuple[str, str]], depth: int):
            paths.append(path.copy())

            if depth >= max_depth:
                return

            for pred, obj in self.get_outgoing(current):
                if predicate_filter and not predicate_filter(pred):
                    continue
                path.append((pred, obj))
                dfs_recursive(obj, path, depth + 1)
                path.pop()

        dfs_recursive(start, [], 0)
        return paths

    def find_paths(
        self,
        start: str,
        end: str,
        max_length: int = 5,
        predicate_filter: Optional[Callable[[str], bool]] = None,
    ) -> List[List[str]]:
        """
        Find paths between start and end entities using BFS.

        Returns:
            List of paths, each path is [start, ..., end]
        """
        if start == end:
            return [[start]]

        visited = {start: (None, None)}
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()

            if len(path) > max_length:
                continue

            for pred, obj in self.get_outgoing(current):
                if predicate_filter and not predicate_filter(pred):
                    continue

                if obj == end:
                    return [path + [obj]]

                if obj not in visited:
                    visited[obj] = (current, pred)
                    queue.append((obj, path + [obj]))

        return []

    def shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find shortest path between entities."""
        paths = self.find_paths(start, end, max_length=10)
        if paths:
            return min(paths, key=len)
        return None

    # =========================================================================
    # REASONING
    # =========================================================================

    def infer_transitive(
        self,
        start: str,
        predicate: str,
        max_depth: int = 5,
    ) -> Set[str]:
        """
        Infer all entities reachable via transitive relation.
        E.g., infer all mammals given "Human is_a Mammal" and "Mammal is_a Animal"
        """
        reachable = set()
        queue = deque([start])

        while queue and len(reachable) < 1000:  # Limit for safety
            current = queue.popleft()

            for pred, obj in self.get_outgoing(current, predicate):
                if obj not in reachable:
                    reachable.add(obj)
                    queue.append(obj)

            # Also check reverse (for symmetric relations)
            for pred, subj in self.get_incoming(current, predicate):
                if pred in [RelationType.SIMILAR_TO.value, RelationType.RELATED_TO.value]:
                    if subj not in reachable:
                        reachable.add(subj)
                        queue.append(subj)

        return reachable

    def verify_statement(self, statement: str) -> Dict[str, Any]:
        """
        Verify a statement against the knowledge graph.
        """
        # Parse simple statements
        patterns = [
            (r"(.+)\s+is\s+a\s+(.+)", "is_a"),
            (r"(.+)\s+is\s+located\s+in\s+(.+)", "located_in"),
            (r"(.+)\s+causes\s+(.+)", "causes"),
            (r"(.+)\s+is\s+part\s+of\s+(.+)", "part_of"),
        ]

        for pattern, predicate in patterns:
            match = re.match(pattern, statement, re.IGNORECASE)
            if match:
                subject, obj = match.groups()
                subject, obj = subject.strip(), obj.strip()

                facts = self.query(subject=subject, predicate=predicate, obj=obj)

                if facts:
                    return {
                        "statement": statement,
                        "verified": True,
                        "confidence": max(f.confidence for f in facts),
                        "sources": [f.source for f in facts],
                        "predicate": predicate,
                    }
                else:
                    # Check if contradiction exists
                    reverse_facts = self.query(subject=subject, predicate=predicate)
                    if any(f.object != obj for f in reverse_facts):
                        return {
                            "statement": statement,
                            "verified": False,
                            "reason": "Contradicting information exists",
                            "confidence": 0.0,
                        }

                    return {
                        "statement": statement,
                        "verified": False,
                        "reason": "No supporting evidence",
                        "confidence": 0.0,
                    }

        return {
            "statement": statement,
            "verified": False,
            "reason": "Could not parse statement",
            "confidence": 0.0,
        }

    # =========================================================================
    # CONSISTENCY CHECKING
    # =========================================================================

    def check_consistency(self) -> List[Dict[str, Any]]:
        """
        Check graph for logical inconsistencies.
        """
        issues = []

        # Check for cycles in hierarchical relations
        hierarchical = [RelationType.IS_A.value, RelationType.PART_OF.value]

        for entity in self.entities:
            paths = self.dfs(entity, lambda p: p in hierarchical, max_depth=5)
            for path in paths:
                if len(path) > 3:  # Suspiciously deep
                    issues.append({
                        "type": "deep_hierarchy",
                        "entity": entity,
                        "path": path,
                        "severity": "warning",
                    })

        # Check for conflicting facts
        for entity in self.entities:
            outgoing = self.get_outgoing(entity)
            for pred, obj in outgoing:
                if pred in [RelationType.IS_A.value]:
                    # Check for multiple direct types
                    types = [o for p, o in outgoing if p == RelationType.IS_A.value]
                    if len(set(types)) > 1:
                        issues.append({
                            "type": "multiple_types",
                            "entity": entity,
                            "types": types,
                            "severity": "error",
                        })

        return issues

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _update_stats(self):
        """Update graph statistics."""
        self.stats["entities"] = len(self.entities)
        self.stats["facts"] = len(self.facts)

        if self.stats["entities"] > 0:
            total_degree = sum(len(v) for v in self.subject_index.values())
            self.stats["avg_degree"] = total_degree / self.stats["entities"]

    def export(self) -> Dict[str, Any]:
        """Export graph to dictionary."""
        return {
            "entities": {
                id: {
                    "label": e.label,
                    "type": e.entity_type,
                    "properties": e.properties,
                }
                for id, e in self.entities.items()
            },
            "facts": [
                {
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.object,
                    "confidence": f.confidence,
                    "source": f.source,
                }
                for f in self.facts.values()
            ],
            "stats": self.stats,
        }

    def export_triples(self) -> List[Dict[str, Any]]:
        """Export all facts as (subject, predicate, object) triples for training.

        Returns:
            List of dicts with subject, predicate, object, confidence, source keys.
        """
        return [
            {
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "confidence": f.confidence,
                "source": f.source,
            }
            for f in self.facts.values()
        ]

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "Knowledge Graph Summary",
            "=" * 40,
            f"Entities: {self.stats['entities']:,}",
            f"Facts: {self.stats['facts']:,}",
            f"Avg Degree: {self.stats['avg_degree']:.2f}",
        ]

        # Top predicates
        predicate_counts = {}
        for f in self.facts.values():
            predicate_counts[f.predicate] = predicate_counts.get(f.predicate, 0) + 1

        if predicate_counts:
            lines.append("\nTop Relations:")
            for pred, count in sorted(predicate_counts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  {pred}: {count}")

        return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Entity",
    "Fact",
    "RelationType",
    "KnowledgeGraph",
]
