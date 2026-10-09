# HuggingFace Integration Module - LOCAL MODELS ONLY

"""
Load and serve models from HuggingFace locally.

Quick Start:
    from domain.training._internal.huggingface import HFClient

    # Load model locally (PREFERRED)
    client = HFClient("mistralai/Mistral-7B-Instruct-v0.2", mode="local")
    print(client("Tell me a story"))

    # API mode is FALLBACK only (requires HF_API_KEY)
    client = HFClient("meta-llama/Llama-2-7b-chat-hf", mode="api")
"""

from __future__ import annotations

from .local_loader import (
    HFLocalConfig,
    HuggingFaceLocalClient,
    HuggingFaceLocalLoader,
    download_model,
    generate_local,
    load_model,
)

# Back-compat name used in older docs / archives
LocalModelLoader = HuggingFaceLocalLoader

import logging

from .model_map import (
    HF_MODELS,
    ModelSize,
    get_model_info,
    get_model_requirements,
    get_recommended_quantization,
    map_to_sloughgpt_config,
    search_models,
)
from .model_map import (
    HFModelInfo as ModelInfo,
)

logger = logging.getLogger("slo.huggingface")

MODEL_REGISTRY = HF_MODELS  # legacy alias (not every call site needs the training registry)

from .client import (
    HFClient,
    chat,
    generate,
    get_model_memory,
    list_models,
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
