"""Entity and fact extraction from conversations.

Byte-identical with the canonical implementation; re-exported from
:mod:`domain.knowledge._internal.entity_extractor` to keep a single source
of truth.
"""

from domain.knowledge._internal.entity_extractor import (
    _is_valid_entity,
    extract_and_store,
    extract_entities,
    extract_facts_from_conversation,
    extract_facts_neural,
    extract_relationships,
)

__all__ = [
    "_is_valid_entity",
    "extract_entities",
    "extract_facts_from_conversation",
    "extract_facts_neural",
    "extract_relationships",
    "extract_and_store",
]
