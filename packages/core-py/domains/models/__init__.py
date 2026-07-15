"""
domains/models/ - Pluggable model backends (native SloNet, no PyTorch)

**ModelInterface** is the generic contract (forward, generate, load, ...) for any
neural backend in this repo.

**SloughGPTModel** is the default first-party **implementation** — a native
SloNet Transformer (RoPE, RMSNorm, SwiGLU, KV-cache). No PyTorch dependency.

External architectures plug in via **ModelLoader.register** or loaders.
"""

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
    - .slo files (Slo Unit format, native)
    - .pt/.pth PyTorch checkpoints (converted to numpy on load)
    - .gguf llama.cpp format
    - HuggingFace model IDs (optional, requires transformers)
    - External model types registered via ModelLoader.register()
    """

    _registry: Dict[str, type] = {}
    _loader_funcs: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, model_class: type):
        cls._registry[name] = model_class
        logger.info(f"Registered model backend: {name}", extra={"tag": "MODEL"})

    @classmethod
    def register_loader(cls, suffix: str, loader_func: Callable):
        cls._loader_funcs[suffix] = loader_func
        logger.info(f"Registered loader for: {suffix}", extra={"tag": "MODEL"})

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> ModelInterface:
        from pathlib import Path
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in cls._loader_funcs:
            return cls._loader_funcs[suffix](path, device, **kwargs)
        if suffix == ".slo":
            return cls._load_sou(path, device, **kwargs)
        elif suffix == ".gguf":
            return cls._load_gguf(path, device, **kwargs)
        elif suffix in (".pt", ".pth"):
            return cls._load_torch(path, device, **kwargs)
        elif "/" in path or path.startswith("hf://"):
            return cls._load_huggingface(path, device, **kwargs)
        else:
            return cls._load_torch(path, device, **kwargs)

    @classmethod
    def _load_sou(cls, path: str, device: str, **kwargs) -> ModelInterface:
        from domains.inference import load_soul
        soul, state_dict = load_soul(path)
        cfg = state_dict.get("config", {}) if isinstance(state_dict, dict) else {}
        model_type = cfg.get("model_type", "sloughgpt")
        vocab_size = cfg.get("vocab_size", 256)
        n_embed = cfg.get("n_embed", 256)
        n_layer = cfg.get("n_layer", 6)
        n_head = cfg.get("n_head", 8)
        block_size = cfg.get("block_size", 128)
        if model_type == "sloughgpt":
            model = SloughGPTModel(
                vocab_size=vocab_size, n_embed=n_embed, n_layer=n_layer,
                n_head=n_head, n_kv_head=cfg.get("n_kv_head"),
                block_size=block_size, max_seq_len=cfg.get("max_seq_len", 2048),
            )
        else:
            model = cls._load_external_model(model_type, cfg)
        if isinstance(state_dict, dict):
            filtered = {k: v for k, v in state_dict.items() if k not in ("config", "metadata")}
            model.load_state_dict(filtered, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        model._soul = soul
        return model

    @classmethod
    def _load_torch(cls, path: str, device: str, **kwargs) -> ModelInterface:
        """Load a PyTorch .pt file, converting weights to numpy arrays."""
        try:
            from domains.training.slonet_compat import torch
        except ImportError:
            raise ImportError(
                "PyTorch required to load .pt files. Install: pip install torch"
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model", checkpoint)
            config = checkpoint.get("config", {})
        else:
            state_dict = checkpoint
            config = {}
        model_type = config.get("model_type", "sloughgpt")
        state_dict_np = {}
        for k, v in state_dict.items():
            if isinstance(v, np.ndarray):
                state_dict_np[k] = v.astype(np.float32)
            elif hasattr(v, 'numpy'):
                state_dict_np[k] = v.cpu().numpy().astype(np.float32)
            else:
                state_dict_np[k] = np.array(v, dtype=np.float32)
        if model_type == "sloughgpt":
            model = SloughGPTModel(
                vocab_size=config.get("vocab_size", 256),
                n_embed=config.get("n_embed", 256),
                n_layer=config.get("n_layer", 6),
                n_head=config.get("n_head", 8),
                n_kv_head=config.get("n_kv_head"),
                block_size=config.get("block_size", 128),
                max_seq_len=config.get("max_seq_len", 2048),
            )
        else:
            model = cls._load_external_model(model_type, config)
        model.load_state_dict(state_dict_np, strict=False)
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
    def _load_huggingface(cls, model_id: str, device: str, **kwargs) -> ModelInterface:
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
            logger.info(f"Loading from HuggingFace: {model_id}", extra={"tag": "MODEL"})
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, config=config, trust_remote_code=True,
                device_map=device if device != "cpu" else None, **kwargs
            )
            return HuggingFaceWrapper(model)
        except ImportError:
            raise ImportError(
                "HuggingFace loading requires transformers. Install: pip install transformers"
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


class HuggingFaceWrapper(ModelInterface):
    """Wrapper for HuggingFace transformers models (optional torch dependency)."""

    def __init__(self, model):
        self._model = model
        self._config = getattr(model, "config", {})

    def load(self, path: str, device: str = "cpu", **kwargs) -> "HuggingFaceWrapper":
        from transformers import AutoModelForCausalLM
        self._model = AutoModelForCausalLM.from_pretrained(path)
        from domains.training.slonet_compat import torch
        self._model = self._model.to(torch.device(device))
        return self

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        from domains.training.slonet_compat import torch
        inp = torch.from_numpy(input_ids.astype(np.int64))
        with torch.no_grad():
            output = self._model.generate(
                inp, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=True, **kwargs
            )
        return output.cpu().numpy()

    def forward(
        self, input_ids: np.ndarray, targets: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        from domains.training.slonet_compat import torch
        inp = torch.from_numpy(input_ids.astype(np.int64))
        tgt = torch.from_numpy(targets.astype(np.int64)) if targets is not None else None
        outputs = self._model(inp, labels=tgt)
        logits = outputs.logits.cpu().numpy()
        loss = getattr(outputs, "loss", None)
        return logits, loss.cpu().numpy() if loss is not None else None

    def state_dict(self) -> Dict[str, np.ndarray]:
        sd = self._model.state_dict()
        return {k: v.cpu().numpy() for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, np.ndarray], **kwargs) -> None:
        from domains.training.slonet_compat import torch
        torch_sd = {k: torch.from_numpy(v) for k, v in state_dict.items()}
        self._model.load_state_dict(torch_sd, **kwargs)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    def config(self) -> Dict[str, Any]:
        return self._config

    def to(self, device: str) -> "HuggingFaceWrapper":
        from domains.training.slonet_compat import torch
        self._model = self._model.to(torch.device(device))
        return self

    def eval(self) -> "HuggingFaceWrapper":
        self._model.eval()
        return self

    def train_mode(self) -> "HuggingFaceWrapper":
        self._model.train()
        return self


# =============================================================================
# SloughGPTModel — native SloNet Transformer (no PyTorch)
# Architecture: RoPE, RMSNorm, SwiGLU, KV-cache, GQA
# =============================================================================

class SloughGPTModel(SloTransformer, ModelInterface):
    """First-party SloNet Transformer implementing ModelInterface.

    Native SloNet decoder-only causal LM with RoPE, RMSNorm, SwiGLU, KV-cache.
    No PyTorch dependency. Drop-in for the old torch-based SloughGPTModel.
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
        return self

    def train_mode(self) -> "SloughGPTModel":
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
    "ModelInterface", "ModelLoader", "HuggingFaceWrapper",
    "SloughGPTModel", "RMSNorm",
    "SloughGPTAttention", "SloughGPTBlock", "SwiGLU",
    "rotate_half", "apply_rotary_pos_emb",
]
