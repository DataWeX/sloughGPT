"""
domains/models/ - Pluggable model backends (native SloNet)

**ModelInterface** is the generic contract (forward, generate, load, ...) for any
neural backend in this repo.

**SloughGPTModel** is the default first-party **implementation** — a native
SloNet Transformer (RoPE, RMSNorm, SwiGLU, KV-cache).

External architectures plug in via **ModelLoader.register** or loaders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Callable
import logging
import numpy as np

from domains.training.slonet import (
    SloTransformer, SloNet, Tensor,
    SloRMSNorm, SloMultiHeadAttention, SloFeedForward,
    SloTransformerBlock, SloDropout,
    export_to_sou, import_from_sou,
    cross_entropy, softmax,
    zeros, ones, randn, tensor,
    no_grad, topk, multinomial,
)

logger = logging.getLogger("sloughgpt.models")


class ModelInterface(ABC):
    """Generic interface every pluggable model backend must implement."""

    @abstractmethod
    def load(self, path: str, device: str = "cpu", **kwargs) -> "ModelInterface":
        pass

    @abstractmethod
    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def forward(
        self, input_ids: np.ndarray, targets: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, np.ndarray], **kwargs) -> None:
        pass

    @abstractmethod
    def num_parameters(self) -> int:
        pass

    @abstractmethod
    def config(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def to(self, device: str) -> "ModelInterface":
        pass

    @abstractmethod
    def eval(self) -> "ModelInterface":
        pass

    @abstractmethod
    def train_mode(self) -> "ModelInterface":
        pass


class ModelLoader:
    """Pluggable model loader - dispatches to the right backend.

    Supports:
    - .sou/.soul/.slo files (Slo Unit format, native, pure NumPy)
    - .gguf llama.cpp format
    - External model types registered via ModelLoader.register()
    """

    _registry: Dict[str, type] = {}
    _loader_funcs: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, model_class: type):
        cls._registry[name] = model_class
        logger.info("Registered model backend: %s", name, extra={"tag": "MODEL"})

    @classmethod
    def register_loader(cls, suffix: str, loader_func: Callable):
        cls._loader_funcs[suffix] = loader_func
        logger.info("Registered loader for: %s", suffix, extra={"tag": "MODEL"})

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> ModelInterface:
        from pathlib import Path
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in cls._loader_funcs:
            return cls._loader_funcs[suffix](path, device, **kwargs)
        if suffix in (".sou", ".soul", ".slo"):
            return cls._load_sou(path, device, **kwargs)
        elif suffix == ".gguf":
            return cls._load_gguf(path, device, **kwargs)
        else:
            return cls._load_sou(path, device, **kwargs)

    @classmethod
    def _load_sou(cls, path: str, device: str, **kwargs) -> ModelInterface:
        from domains.infrastructure.weight_loader import SoulWeightLoader

        loader = SoulWeightLoader(path)
        meta = loader.load_metadata()
        soul = meta.pop("soul")
        cfg = soul.metadata.get("config", {})
        model_type = cfg.get("model_type", "sloughgpt")

        if model_type == "sloughgpt":
            model = SloughGPTModel(
                vocab_size=meta["vocab_size"], n_embed=meta["n_embed"],
                n_layer=meta["n_layer"], n_head=meta["n_head"],
                n_kv_head=cfg.get("n_kv_head"),
                block_size=meta["block_size"],
                max_seq_len=cfg.get("max_seq_len", 2048),
            )
        else:
            model = cls._load_external_model(model_type, cfg)

        loader.load(model)
        model._soul = soul
        return model

    @classmethod
    def _load_gguf(cls, path: str, device: str, **kwargs):
        try:
            from llama_cpp import Llama
            return Llama(model_path=path, n_ctx=kwargs.get("n_ctx", 2048))
        except ImportError:
            raise NotImplementedError(
                "GGUF loading requires llama-cpp-python. Install: pip install llama-cpp-python"
            )

    @classmethod
    def _load_external_model(cls, model_type: str, config: Dict[str, Any]) -> ModelInterface:
        if model_type in cls._registry:
            model_class = cls._registry[model_type]
            return model_class(**config)
        supported = list(cls._registry.keys())
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported external models: {supported}. "
            f"Register with ModelLoader.register('{model_type}', YourModelClass)"
        )


# =============================================================================
# SloughGPTModel — native SloNet Transformer
# Architecture: RoPE, RMSNorm, SwiGLU, KV-cache, GQA
# =============================================================================

class SloughGPTModel(SloTransformer, ModelInterface):
    """First-party SloNet Transformer implementing ModelInterface.

    Native SloNet decoder-only causal LM with RoPE, RMSNorm, SwiGLU, KV-cache.
    Drop-in for the original torch-based SloughGPTModel.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        n_embed: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        n_kv_head: Optional[int] = None,
        dropout: float = 0.1,
        block_size: int = 128,
        max_seq_len: int = 2048,
        use_sdpa: bool = True,
        use_flash: bool = False,
        tie_weights: bool = True,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__(
            vocab_size=vocab_size, n_embed=n_embed, n_layer=n_layer,
            n_head=n_head, n_kv_head=n_kv_head, block_size=block_size,
            max_seq_len=max_seq_len, dropout=dropout, use_rope=True,
            tie_weights=tie_weights, intermediate_size=intermediate_size,
            soul_name="SloughGPT",
        )
        self._device = "cpu"
        self._soul = None
        self._config = {
            "vocab_size": vocab_size, "n_embed": n_embed, "n_layer": n_layer,
            "n_head": n_head, "n_kv_head": n_kv_head or n_head,
            "dropout": dropout, "block_size": block_size,
            "max_seq_len": max_seq_len, "use_sdpa": use_sdpa,
            "use_flash": use_flash, "model_type": "sloughgpt",
        }

    def load(self, path: str, device: str = "cpu", **kwargs) -> "SloughGPTModel":
        loaded = ModelLoader.load(path, device=device, config=self._config, **kwargs)
        self.load_state_dict(loaded.state_dict())
        self._soul = getattr(loaded, "_soul", None)
        return self

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        result = super().generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            eos_token=0,
        )
        return result.data

    def forward(
        self, input_ids, targets=None, **kwargs
    ):
        from domains.training.slonet import Tensor as SloTensor
        logits_t, loss_t = super().forward(input_ids, targets)
        if loss_t is not None:
            if isinstance(loss_t, (np.ndarray, float, np.floating)):
                loss_t = SloTensor(np.array(loss_t, dtype=np.float32))
            if isinstance(logits_t, (np.ndarray, float, np.floating)):
                logits_t = SloTensor(np.array(logits_t, dtype=np.float32))
            return logits_t, loss_t
        return logits_t, None

    def state_dict(self) -> Dict[str, np.ndarray]:
        sd = super().state_dict()
        sd["config"] = self._config
        return sd

    def load_state_dict(self, state_dict: Dict[str, np.ndarray], strict: bool = True, **kwargs) -> None:
        filtered = {k: v for k, v in state_dict.items() if k != "config"}
        super().load_state_dict(filtered, strict=strict)

    def num_parameters(self) -> int:
        return sum(int(np.prod(p.data.shape)) for p in self.parameters())

    def config(self) -> Dict[str, Any]:
        return self._config.copy()

    def to(self, device: str) -> "SloughGPTModel":
        self._device = device
        return self

    def eval(self) -> "SloughGPTModel":
        super().eval()
        return self

    def train_mode(self) -> "SloughGPTModel":
        super().train(True)
        return self

    def clear_kv_cache(self):
        super().clear_kv_cache()

    def get_model_size_mb(self) -> float:
        total = sum(p.data.nbytes for p in self.parameters())
        return total / (1024 * 1024)

    def freeze_embeddings(self) -> "SloughGPTModel":
        self.layers[0].weight.requires_grad = False
        return self


# Re-export SloNet components as aliases for backward compatibility
RMSNorm = SloRMSNorm
SloughGPTAttention = SloMultiHeadAttention
SloughGPTBlock = SloTransformerBlock
SwiGLU = SloFeedForward

def rotate_half(x):
    from domains.training.slonet import _rotate_half
    return _rotate_half(x)

def apply_rotary_pos_emb(q, k, cos, sin):
    from domains.training.slonet import _apply_rope
    return _apply_rope(q, k, cos, sin)


__all__ = [
    "ModelInterface", "ModelLoader",
    "SloughGPTModel", "RMSNorm",
    "SloughGPTAttention", "SloughGPTBlock", "SwiGLU",
    "rotate_half", "apply_rotary_pos_emb",
]
