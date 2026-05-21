# SloughGPT HuggingFace Integration
# Easy-to-use wrapper for HF model loading

"""
Quick Start:
    from sloughgpt import HFClient

    # Use HF Inference API (no download)
    client = HFClient("meta-llama/Llama-2-7b-chat-hf", mode="api")
    print(client("Hello!"))

    # Or load locally
    client = HFClient("mistralai/Mistral-7B-Instruct-v0.2", mode="local")
    print(client("Tell me a story"))

Environment Variables:
    HF_API_KEY     - HuggingFace API token for inference API
    HF_TOKEN      - Alternative to HF_API_KEY
    HF_CACHE_DIR  - Cache directory for models (default: ~/.cache/huggingface)
"""

from .api_loader import HuggingFaceAPILoader as _APILoader
from .local_loader import HuggingFaceLocalLoader as _LocalLoader
from .model_map import get_model_requirements, HF_MODELS


class HFClient:
    """
    Unified client for HuggingFace models.

    Supports two modes:
    - "api": Use HF Inference API (no download, pay-per-use)
    - "local": Download and run locally

    Args:
        model: Model name (e.g., "meta-llama/Llama-2-7b-chat-hf")
        mode: "api" or "local"
        **kwargs: Additional config options
    """

    def __init__(self, model: str, mode: str = "api", **kwargs):
        self.model = model
        self.mode = mode

        if mode == "api":
            from .api_loader import HFAPIConfig

            config = HFAPIConfig(model=model, **kwargs)
            self._client = _APILoader(config)
        elif mode == "local":
            from .local_loader import HFLocalConfig

            config = HFLocalConfig(model=model, **kwargs)
            self._client = _LocalLoader(config)
            self._client.load()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'api' or 'local'")

    def __call__(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        return self._client.generate(prompt, **kwargs)

    def chat(self, messages, **kwargs) -> str:
        """Chat with the model."""
        return self._client.chat(messages, **kwargs)

    def __repr__(self):
        return f"HFClient(model='{self.model}', mode='{self.mode}')"


def get_model_memory(model: str, precision: str = "bf16") -> dict:
    """
    Get memory requirements for a model.

    Args:
        model: Model name
        precision: Precision (fp32, bf16, q4_k_m, etc.)

    Returns:
        Dictionary with memory estimates
    """
    return get_model_requirements(model)


def list_models(organization: str = None, size: str = None) -> list:
    """
    List available models.

    Args:
        organization: Filter by organization (e.g., "meta", "mistralai")
        size: Filter by size (small, medium, large, xlarge)

    Returns:
        List of model names
    """
    from .model_map import search_models, ModelSize

    size_enum = None
    if size:
        size_enum = ModelSize[size.upper()]

    results = search_models(organization=organization, size=size_enum)
    return [r.model_id for r in results]


# Convenience functions


def generate(prompt: str, model: str = "meta-llama/Llama-2-7b-chat-hf", **kwargs) -> str:
    """Quick generate via API."""
    client = HFClient(model, mode="api")
    return client(prompt, **kwargs)


def chat(messages: list, model: str = "meta-llama/Llama-2-7b-chat-hf", **kwargs) -> str:
    """Quick chat via API."""
    client = HFClient(model, mode="api")
    return client.chat(messages, **kwargs)


# Export everything needed
__all__ = [
    "HFClient",
    "get_model_memory",
    "list_models",
    "generate",
    "chat",
    "HF_MODELS",
]
