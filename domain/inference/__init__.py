"""inference — Model inference, vector store, soul format.

Public API:
    VectorStoreType, VectorEntry, QueryResult, VectorStore
    InMemoryVectorStore, MogDBVectorStore, simple_embed
    SloProfile, PersonalityCore, BehavioralTraits, CognitiveSignature, EmotionalRange
    GenerationParams, ContextParams, SouParser, create_soul_profile
    save_soul, load_soul, write_v3_sou, generate_sample_dialogue
"""

from domain.inference._internal.vector_store import (
    VectorStoreType,
    VectorEntry,
    QueryResult,
    VectorStore,
    InMemoryVectorStore,
    MogDBVectorStore,
    simple_embed,
)
from domain.inference._internal.slo_format import (
    SloProfile,
    PersonalityCore,
    BehavioralTraits,
    CognitiveSignature,
    EmotionalRange,
    GenerationParams,
    ContextParams,
    SouParser,
    create_soul_profile,
    save_soul,
    load_soul,
    write_v3_sou,
    generate_sample_dialogue,
)

__all__ = [
    "VectorStoreType",
    "VectorEntry",
    "QueryResult",
    "VectorStore",
    "InMemoryVectorStore",
    "MogDBVectorStore",
    "simple_embed",
    "SloProfile",
    "PersonalityCore",
    "BehavioralTraits",
    "CognitiveSignature",
    "EmotionalRange",
    "GenerationParams",
    "ContextParams",
    "SouParser",
    "create_soul_profile",
    "save_soul",
    "load_soul",
    "write_v3_sou",
    "generate_sample_dialogue",
]
