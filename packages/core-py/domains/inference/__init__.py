"""Backward-compatibility shim — imports from the new ``domain.inference`` package."""

from domain.inference import (
    VectorStoreType,
    VectorEntry,
    QueryResult,
    VectorStore,
    InMemoryVectorStore,
    MogDBVectorStore,
    simple_embed,
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

from domain.inference._internal import slo_format, slo_manager, slonet_provider  # noqa: F401

# Allow submodule access (domains.X.Y) for test mocking
import importlib as _importlib
def __getattr__(name):
    try:
        return _importlib.import_module(f"domain.{name}")
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        return _importlib.import_module(f"domain.inference._internal.{name}")
    except (ImportError, ModuleNotFoundError):
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
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
