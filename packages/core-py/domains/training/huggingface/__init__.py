# HuggingFace Integration Module - LOCAL MODELS ONLY

"""
Load and serve models from HuggingFace locally.

Quick Start:
    from domains.training.huggingface import HFClient

    # Load model locally (PREFERRED)
    client = HFClient("mistralai/Mistral-7B-Instruct-v0.2", mode="local")
    print(client("Tell me a story"))

    # API mode is FALLBACK only (requires HF_API_KEY)
    client = HFClient("meta-llama/Llama-2-7b-chat-hf", mode="api")
"""

from .local_loader import (
    HFLocalConfig,
    HuggingFaceLocalLoader,
    HuggingFaceLocalClient,
    download_model,
    load_model,
    generate_local,
)

# Back-compat name used in older docs / archives
LocalModelLoader = HuggingFaceLocalLoader

from .model_map import (
    ModelSize,
    HFModelInfo as ModelInfo,
    HF_MODELS,
    get_model_info,
    search_models,
    get_recommended_quantization,
    get_model_requirements,
    map_to_sloughgpt_config,
)

MODEL_REGISTRY = HF_MODELS  # legacy alias (not every call site needs the training registry)

from .client import (
    HFClient,
    get_model_memory,
    list_models,
    generate,
    chat,
)


__all__ = [
    # Primary - Local Loading
    "LocalModelLoader",
    "HFClient",
    "get_model_memory",
    "list_models",
    "generate",
    "chat",
    # Local Loader
    "HFLocalConfig",
    "HuggingFaceLocalLoader",
    "HuggingFaceLocalClient",
    "download_model",
    "load_model",
    "generate_local",
    # Model Map
    "ModelSize",
    "ModelInfo",
    "HF_MODELS",
    "MODEL_REGISTRY",
    "get_model_info",
    "search_models",
    "get_recommended_quantization",
    "get_model_requirements",
    "map_to_sloughgpt_config",
]
