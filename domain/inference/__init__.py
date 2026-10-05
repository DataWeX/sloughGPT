"""inference — Model inference, vector store, soul format.

Public API:
    VectorStoreType, VectorEntry, QueryResult, VectorStore
    InMemoryVectorStore, MogDBVectorStore, simple_embed
    SloProfile, PersonalityCore, BehavioralTraits, CognitiveSignature, EmotionalRange
    GenerationParams, ContextParams, SouParser, create_soul_profile
    save_soul, load_soul, write_v3_sou, generate_sample_dialogue
    soul_path, soul_meta_path, soul_read_candidates, is_soul_file
    classify_soul, SoulIdentity, SoulVariant
    SOUL_PROVENANCE, SOUL_PROVENANCE_{TRAINING,EXPORT,DISTILLATION}
    train_embedder, SloTextEmbedder
    create_vector_store, get_slo_manager, PDFVLMProcessor, _ngram_embed
"""

from domain.inference._internal.slo_format import (
    SOUL_PROVENANCE,
    SOUL_PROVENANCE_DISTILLATION,
    SOUL_PROVENANCE_EXPORT,
    SOUL_PROVENANCE_TRAINING,
    BehavioralTraits,
    CognitiveSignature,
    ContextParams,
    EmotionalRange,
    GenerationParams,
    PersonalityCore,
    SloProfile,
    SoulIdentity,
    SoulVariant,
    SouParser,
    classify_soul,
    create_soul_profile,
    generate_sample_dialogue,
    is_soul_file,
    load_soul,
    save_soul,
    soul_meta_path,
    soul_path,
    soul_read_candidates,
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
    "soul_path",
    "soul_meta_path",
    "soul_read_candidates",
    "is_soul_file",
    "classify_soul",
    "SoulIdentity",
    "SoulVariant",
    "SOUL_PROVENANCE",
    "SOUL_PROVENANCE_TRAINING",
    "SOUL_PROVENANCE_EXPORT",
    "SOUL_PROVENANCE_DISTILLATION",
    "write_v3_sou",
    "generate_sample_dialogue",
    "train_embedder",
    "SloTextEmbedder",
    "create_vector_store",
    "get_slo_manager",
    "PDFVLMProcessor",
    "_ngram_embed",
    "SloNetChatProvider",
    "VectorEntry",
]


def __getattr__(name):
    _lazy = {
        "train_embedder": "domain.inference._internal.slo_embedder",
        "SloTextEmbedder": "domain.inference._internal.slo_embedder",
        "_EMBEDDER_PATH": "domain.inference._internal.slo_embedder",
        "create_vector_store": "domain.inference._internal.vector_store",
        "_ngram_embed": "domain.inference._internal.vector_store",
        "get_slo_manager": "domain.inference._internal.slo_manager",
        "PDFVLMProcessor": "domain.inference._internal.pdf_vlm",
        "SloNetChatProvider": "domain.inference._internal.slonet_provider",
        "VectorEntry": "domain.inference._internal.vector_store",
        "PineconeVectorStore": "domain.inference._internal.vector_stores.pinecone_store",
        "configure_api_provider": "domain.inference._internal.api_provider",
        "get_engine": "domain.inference._internal.native.engine",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
