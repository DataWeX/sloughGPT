"""
CTransformProvider — thin wrapper making NativeEngine look like SlonetChatProvider.

The NPU calls provider.generate(), provider.tokenize(), provider._model.forward_pass().
Both SlonetChatProvider (numpy) and CTransformProvider (C/NativeEngine) satisfy this interface.
"""

import logging
import numpy as np
from typing import Optional, Dict, List

logger = logging.getLogger("slo.inference.ct_transform")


class CTransformProvider:
    """Wraps NativeEngine behind the SlonetChatProvider interface.

    The NPU doesn't know or care which backend is running. It calls:
      provider.generate(prompt, ...)
      provider.tokenize(text)
      provider._model.forward_pass(input_ids)

    This wrapper delegates all of those to the NativeEngine (C/Accelerate).
    """

    def __init__(self, engine, model_id: str = "c-model"):
        from .native.engine import NativeEngine
        self._engine: NativeEngine = engine
        self._model_id = model_id
        self._model = engine  # NPU accesses provider._model.forward_pass()
        self._model_lock = None
        self._tokenizer = None
        self._device = "cpu"
        self._quant_engine = None

        # Try to load a tokenizer for encode/decode
        try:
            from ..tokenizer import get_tokenizer
            self._tokenizer = get_tokenizer()
        except Exception:
            self._tokenizer = None

    @classmethod
    def from_slnc(cls, slnc_path: str, model_id: str = "c-model",
                  seq_capacity: int = 2048) -> "CTransformProvider":
        """Load from .slnc file via the NativeEngine."""
        from .native.engine import NativeEngine
        from ..infrastructure.slnc.parser import SLNCParser

        parser = SLNCParser(slnc_path)
        config = parser.config
        tensors = parser.get_weights_dict()

        engine = NativeEngine()
        info = engine.load_from_slnc(tensors, config, seq_capacity=seq_capacity)

        logger.info("CTransformProvider loaded %s via NativeEngine: %s", model_id, info)
        return cls(engine, model_id=model_id)

    @classmethod
    def from_slo(cls, slo_model, model_id: str = "slo-model",
                 seq_capacity: int = 2048) -> "CTransformProvider":
        """Bridge: load SloTransformer into NativeEngine for C acceleration."""
        from .native.engine import NativeEngine

        engine = NativeEngine()
        info = engine.load_from_slo(slo_model, seq_capacity=seq_capacity)

        logger.info("CTransformProvider bridged %s via NativeEngine: %s", model_id, info)
        return cls(engine, model_id=model_id)

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 1.0,
                 top_k: int = None, top_p: float = None, repetition_penalty: float = 1.0,
                 **kwargs) -> str:
        """Generate text via NativeEngine."""
        messages = [{"role": "user", "content": prompt}]
        return self._engine.generate(
            messages, max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p if top_p is not None else 0.9,
            top_k=top_k if top_k is not None else 50,
        )

    def tokenize(self, text: str) -> List[int]:
        if self._tokenizer is not None:
            return self._tokenizer.encode(text)
        return self._engine._tokenize_simple(text)

    def detokenize(self, token_ids: List[int]) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids)
        return self._engine._detokenize_simple(token_ids)

    def embed(self, text: str, layer: int = -1) -> np.ndarray:
        """Embed via forward pass — returns last-position logits as a rough embedding."""
        tokens = self.tokenize(text)
        input_ids = np.array([tokens], dtype=np.int64)
        result = self._engine.forward_pass(input_ids)
        return result.logits[0, -1, :]

    def metadata(self) -> Dict:
        config = self._engine._config or {}
        return {
            "model_id": self._model_id,
            "architecture": "NativeEngine",
            "total_params": 0,
            "n_layer": config.get("num_hidden_layers", 0),
            "n_embed": config.get("hidden_size", 0),
            "n_head": config.get("num_attention_heads", 0),
            "vocab_size": config.get("vocab_size", 0),
            "max_seq_len": config.get("max_position_embeddings", 2048),
            "device": self._device,
            "quantized": False,
            "has_tokenizer": self._tokenizer is not None,
            "engine": "c",
        }
