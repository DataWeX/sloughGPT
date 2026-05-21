"""
Model Registry - Dynamically loadable models for SloughGPT

Models are auto-discovered from the models directory and HuggingFace Hub.
"""

from typing import Dict, Any, Optional, Type, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import importlib.util
import logging

logger = logging.getLogger("sloughgpt.models")


def _get_nn():
    """Lazily get torch.nn module."""
    import torch

    return torch.nn


class ModelSource(Enum):
    """Source of the model."""

    LOCAL = "local"
    HUGGINGFACE = "huggingface"


@dataclass
class ModelInfo:
    """Information about a model."""

    id: str
    name: str
    description: str
    model_class: Optional[Type] = None
    default_config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    source: ModelSource = ModelSource.LOCAL
    hf_model_id: Optional[str] = None


def BaseModel(config: Dict[str, Any]):
    """Factory function to create base model class (lazy torch import)."""
    nn = _get_nn()

    class _BaseModel(nn.Module):
        def __init__(self, config: Dict[str, Any]):
            super().__init__()
            self.config = config

        def forward(self, x, targets=None):
            raise NotImplementedError

        def get_num_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

    return _BaseModel(config)


class ModelRegistry:
    """Registry for dynamically loadable models."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models: Dict[str, ModelInfo] = {}
            cls._instance._discovered = False
        return cls._instance

    def _ensure_discovered(self):
        """Lazily discover models when needed."""
        if not self._discovered:
            self._discover_models()
            self._discover_hf_models()
            self._discovered = True

    def _discover_models(self):
        """Discover models from domains/training/models/ only."""
        models_dir = Path("domains/training/models")

        if not models_dir.exists():
            return

        for model_file in models_dir.glob("*.py"):
            if model_file.name.startswith("_"):
                continue

            self._load_from_file(model_file)

    def _discover_hf_models(self):
        """Discover and register HuggingFace models."""
        try:
            from .huggingface.model_map import HF_MODELS, get_model_requirements

            for model_id, hf_info in HF_MODELS.items():
                self._models[f"hf/{model_id}"] = ModelInfo(
                    id=f"hf/{model_id}",
                    name=hf_info.name,
                    description=hf_info.description,
                    model_class=None,
                    default_config={
                        "model": hf_info.model_id,
                        "source": "huggingface",
                    },
                    tags=["huggingface", "hf"] + hf_info.tags,
                    source=ModelSource.HUGGINGFACE,
                    hf_model_id=hf_info.model_id,
                )
        except ImportError:
            logger.debug("HuggingFace integration not available")

    def _load_from_file(self, model_file: Path):
        """Load model classes from a file."""
        nn = _get_nn()

        try:
            spec = importlib.util.spec_from_file_location(
                f"models.{model_file.stem}", model_file
            )

            if not spec or not spec.loader:
                return

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue

                attr = getattr(module, attr_name)

                if not isinstance(attr, type):
                    continue

                if not issubclass(attr, nn.Module):
                    continue

                if attr is nn.Module or attr is BaseModel:
                    continue

                if not hasattr(attr, "forward"):
                    continue

                model_id = f"{model_file.stem}-{attr_name.lower()}"

                if model_id in self._models:
                    continue

                config = {}
                if hasattr(module, "DEFAULT_CONFIG"):
                    config = getattr(module, "DEFAULT_CONFIG")

                desc = attr.__doc__ or f"Model from {model_file.name}"
                desc = desc.strip().split("\n")[0][:80]

                self._models[model_id] = ModelInfo(
                    id=model_id,
                    name=attr_name,
                    description=desc,
                    model_class=attr,
                    default_config=config,
                    tags=[model_file.stem],
                    source=ModelSource.LOCAL,
                )

        except Exception as e:
            logger.debug(f"Error loading {model_file}: {e}")

    def register_model(self, model_info: ModelInfo):
        """Register a custom model."""
        self._models[model_info.id] = model_info

    def get(self, model_id: str) -> Optional[ModelInfo]:
        self._ensure_discovered()
        return self._models.get(model_id)

    def get_hf_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a HuggingFace model by ID (with or without hf/ prefix)."""
        self._ensure_discovered()
        if model_id.startswith("hf/"):
            return self._models.get(model_id)
        return self._models.get(f"hf/{model_id}")

    def list_models(self, source: ModelSource = None) -> List[ModelInfo]:
        """List all models, optionally filtered by source."""
        self._ensure_discovered()
        if source:
            return [m for m in self._models.values() if m.source == source]
        return list(self._models.values())

    def list_hf_models(self) -> List[ModelInfo]:
        """List all HuggingFace models."""
        return self.list_models(ModelSource.HUGGINGFACE)

    def list_local_models(self) -> List[ModelInfo]:
        """List all local models."""
        return self.list_models(ModelSource.LOCAL)

    def create_model(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        """Create a model instance."""
        self._ensure_discovered()
        model_info = self.get(model_id)

        if not model_info:
            available = ", ".join(self._models.keys())
            raise ValueError(f"Model '{model_id}' not found. Available: {available}")

        if model_info.source == ModelSource.HUGGINGFACE:
            raise ValueError(
                f"Model '{model_id}' is a HuggingFace model. "
                "Use load_hf_model() instead."
            )

        final_config = {**model_info.default_config, **(config or {})}
        return model_info.model_class(**final_config)

    def load_hf_model(
        self,
        model_id: str,
        mode: str = "local",
        **kwargs,
    ):
        """Load a HuggingFace model (api or local mode)."""
        model_info = self.get_hf_model(model_id)

        if not model_info:
            raise ValueError(
                f"HuggingFace model '{model_id}' not found. "
                f"Available: {', '.join(m.id for m in self.list_hf_models())}"
            )

        from .huggingface import HFClient

        return HFClient(model_info.hf_model_id, mode=mode, **kwargs)


registry = ModelRegistry()


def get_available_models(source: ModelSource = None) -> List[ModelInfo]:
    """Get list of available models."""
    return registry.list_models(source)


def get_available_hf_models() -> List[ModelInfo]:
    """Get list of available HuggingFace models."""
    return registry.list_hf_models()


def create_model(model_id: str, config: Optional[Dict[str, Any]] = None):
    """Create a local model instance."""
    return registry.create_model(model_id, config)


def load_hf_model(
    model_id: str,
    mode: str = "local",
    **kwargs,
):
    """Load a HuggingFace model."""
    return registry.load_hf_model(model_id, mode, **kwargs)


__all__ = [
    "ModelRegistry",
    "ModelInfo",
    "ModelSource",
    "BaseModel",
    "registry",
    "get_available_models",
    "get_available_hf_models",
    "create_model",
    "load_hf_model",
]
