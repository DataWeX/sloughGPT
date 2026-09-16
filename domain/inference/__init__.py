"""inference — Model inference, vector store, soul format.

Public API:
    VectorStoreType, VectorEntry, QueryResult, VectorStore
    InMemoryVectorStore, MogDBVectorStore, simple_embed
    SloProfile, PersonalityCore, BehavioralTraits, CognitiveSignature, EmotionalRange
    GenerationParams, ContextParams, SouParser, create_soul_profile
    save_soul, load_soul, write_v3_sou, generate_sample_dialogue
"""

from domain.inference._internal.slo_format import (
    BehavioralTraits,
    CognitiveSignature,
    ContextParams,
    EmotionalRange,
    GenerationParams,
    PersonalityCore,
    SloProfile,
    SouParser,
    create_soul_profile,
    generate_sample_dialogue,
    load_soul,
    save_soul,
    write_v3_sou,
)
from domain.inference._internal.vector_store import (
    InMemoryVectorStore,
    MogDBVectorStore,
    QueryResult,
    VectorEntry,
    VectorStore,
    VectorStoreType,
    simple_embed,
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
