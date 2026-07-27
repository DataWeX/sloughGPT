"""
engine.py - High-level transformer engine with chat template and sampling.

Wraps the C forward pass with:
  - SLNC weight loading via weight_mapper
  - Chat template formatting (Qwen, LLaMA, GPT-2)
  - Top-p / top-k / temperature sampling
  - Token-by-token generation with KV cache
"""

import ctypes
import logging
import time
from typing import Optional, List, Dict, Tuple, Generator

import numpy as np

from . import bindings as B
from .weight_mapper import map_slnc_to_native

logger = logging.getLogger("slo.inference.native.engine")

IMS = "<|im_start|>"
IME = "<|im_end|>"


def _detect_model_type(config: dict) -> str:
    arch = config.get("architectures", [""])[0].lower()
    if "qwen" in arch:
        return "qwen2"
    if "llama" in arch or "mistral" in arch or "phi" in arch:
        return "llama"
    if "gpt2" in arch or "openai" in arch:
        return "gpt2"
    return "qwen2"


def _format_chat_qwen(messages: List[Dict[str, str]], system: str = "") -> str:
    parts = []
    if system:
        parts.append(f"{IMS}system\n{system}{IME}")
    for msg in messages:
        role = msg.get("role", "user")
        content_m = msg.get("content", "")
        parts.append(f"{IMS}{role}\n{content_m}{IME}")
    parts.append(IMS + "assistant\n")
    return "".join(parts)


def _format_chat_llama(messages: List[Dict[str, str]], system: str = "") -> str:
    parts = []
    if system:
        parts.append(f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n")
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content_m = msg.get("content", "")
        if role == "user":
            if i == 0 and system:
                parts.append(f"{content_m} [/INST] ")
            else:
                parts.append(f"[INST] {content_m} [/INST] ")
        elif role == "assistant":
            parts.append(f"{content_m} </s>")
    return "".join(parts)


def _format_chat_gpt2(messages: List[Dict[str, str]], system: str = "") -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content_m = msg.get("content", "")
        parts.append(f"{role}: {content_m}\n")
    parts.append("Assistant:")
    return "".join(parts)


def format_chat(messages: List[Dict[str, str]], model_type: str, system: str = "") -> str:
    if model_type == "qwen2":
        return _format_chat_qwen(messages, system)
    elif model_type == "llama":
        return _format_chat_llama(messages, system)
    else:
        return _format_chat_gpt2(messages, system)


def sample_token(logits: np.ndarray, temperature: float = 1.0,
                 top_p: float = 0.9, top_k: int = 50,
                 rng: Optional[np.random.Generator] = None) -> int:
    if temperature <= 0.01:
        return int(np.argmax(logits))
    if rng is None:
        rng = np.random.default_rng()

    logits = logits / temperature

    if top_k > 0 and top_k < len(logits):
        idx = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -1e9)
        mask[idx] = logits[idx]
        logits = mask

    if top_p < 1.0:
        sorted_idx = np.argsort(-logits)
        sorted_logits = logits[sorted_idx]
        cumsum = np.cumsum(np.exp(sorted_logits))
        cutoff = cumsum[-1] * top_p
        for i in range(len(sorted_logits)):
            if cumsum[i] >= cutoff:
                logits[sorted_idx[i+1:]] = -1e9
                break

    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


class TransformerEngine:
    def __init__(self):
        self._lib = B.load_lib()
        self._weights = None
        self._cache = None
        self._model_type = "qwen2"
        self._config = {}
        self._loaded = False

    def load_from_slnc(self, slnc_tensors: dict, slnc_config: dict,
                       seq_capacity: int = 2048) -> dict:
        self._config = slnc_config
        self._model_type = _detect_model_type(slnc_config)
        n_layers = slnc_config.get("num_hidden_layers", 12)
        hidden = slnc_config.get("hidden_size", 768)
        n_heads = slnc_config.get("num_attention_heads", 12)
        n_kv_heads = slnc_config.get("num_key_value_heads", n_heads)
        head_dim = hidden // n_heads
        ff_dim = slnc_config.get("intermediate_size", hidden * 4)
        vocab = slnc_config.get("vocab_size", 50257)

        flat, info = map_slnc_to_native(
            slnc_tensors, n_layers, hidden, n_heads, n_kv_heads, head_dim, ff_dim, vocab
        )

        cfg = B.load_lib()._Config()
        cfg.n_layers = n_layers
        cfg.hidden_dim = hidden
        cfg.n_heads = n_heads
        cfg.n_kv_heads = n_kv_heads
        cfg.head_dim = head_dim
        cfg.ff_dim = ff_dim
        cfg.vocab_size = vocab
        cfg.block_size = slnc_config.get("max_position_embeddings", seq_capacity)
        cfg.rope_base = slnc_config.get("rope_theta", 10000.0)
        cfg.rope_theta = slnc_config.get("rope_theta", 10000.0)

        self._weights = B.load_lib()._Weights()
        flat_arr = np.ascontiguousarray(flat, dtype=np.float32)
        flat_ct = flat_arr.ctypes.data_as(B.load_lib()._Config._fields_[0][1])
        rc = self._lib.transformer_load_weights(
            self._weights, flat_ct, len(flat), cfg
        )
        if rc != 0:
            raise RuntimeError(f"transformer_load_weights failed: {rc}")

        self._cache = B.load_lib()._KVCache()
        self._lib.transformer_kv_cache_init(self._cache, cfg, seq_capacity)
        self._loaded = True

        return {"model_type": self._model_type, "layers": n_layers, "hidden": hidden}

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 128,
                 temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50,
                 system: str = "") -> str:
        if not self._loaded:
            raise RuntimeError("No model loaded")

        prompt = format_chat(messages, self._model_type, system)
        logger.info("prompt: %r", prompt[:200])

        tokens = self._tokenize_simple(prompt)
        logger.info("tokenized: %d tokens", len(tokens))

        self._lib.transformer_kv_cache_reset(self._cache)

        logits_buf = np.zeros(self._config.get("vocab_size", 50257), dtype=np.float32)
        rng = np.random.default_rng()

        t0 = time.perf_counter()
        for i, tok in enumerate(tokens):
            self._lib.transformer_forward_step(
                self._weights, self._cache, tok, i,
                logits_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
        t_prompt = time.perf_counter() - t0
        logger.info("prompt eval: %.3fs (%d tokens)", t_prompt, len(tokens))

        generated = []
        t0 = time.perf_counter()
        for step in range(max_tokens):
            tok = sample_token(logits_buf, temperature, top_p, top_k, rng)
            if tok == self._config.get("eos_token_id", 2):
                break
            generated.append(tok)
            self._lib.transformer_forward_step(
                self._weights, self._cache, tok, len(tokens) + step,
                logits_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
        t_gen = time.perf_counter() - t0

        n_gen = len(generated)
        result = self._detokenize_simple(generated)
        logger.info("generated %d tokens in %.3fs (%.1f tok/s)", n_gen, t_gen,
                     n_gen / t_gen if t_gen > 0 else 0)
        return result

    def generate_stream(self, messages: List[Dict[str, str]], max_tokens: int = 128,
                        temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50,
                        system: str = "") -> Generator[str, None, None]:
        if not self._loaded:
            raise RuntimeError("No model loaded")

        prompt = format_chat(messages, self._model_type, system)
        tokens = self._tokenize_simple(prompt)
        self._lib.transformer_kv_cache_reset(self._cache)

        logits_buf = np.zeros(self._config.get("vocab_size", 50257), dtype=np.float32)
        rng = np.random.default_rng()

        for i, tok in enumerate(tokens):
            self._lib.transformer_forward_step(
                self._weights, self._cache, tok, i,
                logits_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )

        for step in range(max_tokens):
            tok = sample_token(logits_buf, temperature, top_p, top_k, rng)
            if tok == self._config.get("eos_token_id", 2):
                break
            piece = self._detokenize_simple([tok])
            yield piece
            self._lib.transformer_forward_step(
                self._weights, self._cache, tok, len(tokens) + step,
                logits_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )

    def _tokenize_simple(self, text: str) -> list:
        try:
            from ..tokenizer import get_tokenizer
            tok = get_tokenizer()
            ids = tok.encode(text)
            return ids if isinstance(ids, list) else list(ids)
        except Exception:
            return list(text.encode("utf-8", errors="replace"))

    def _detokenize_simple(self, tokens: list) -> str:
        try:
            from ..tokenizer import get_tokenizer
            tok = get_tokenizer()
            return tok.decode(tokens)
        except Exception:
            return "".join(chr(t) if 32 <= t < 127 else "?" for t in tokens)

    def reset_cache(self):
        if self._cache:
            self._lib.transformer_kv_cache_reset(self._cache)

    @property
    def loaded(self) -> bool:
        return self._loaded


_engine = None

def get_engine() -> TransformerEngine:
    global _engine
    if _engine is None:
        _engine = TransformerEngine()
    return _engine


class NativeTransformerProvider:
    """Async provider wrapping the C-accelerated TransformerEngine.

    Implements the ModelProvider protocol for integration into the provider chain.
    Uses Apple Accelerate BLAS via the native C forward pass.
    """

    def __init__(self, engine: TransformerEngine, model_id: str = "native-c"):
        self._engine = engine
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self):
        from domains.models.provider import ModelCapabilities
        return ModelCapabilities(chat=True, streaming=True, embedding=False, vision=False)

    async def chat_stream(self, messages, max_tokens=512, temperature=0.8,
                          top_p=0.9, top_k=50, cancel_event=None,
                          session_id=None, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()

        def _gen():
            return self._engine.generate_stream(
                messages, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
            )

        try:
            gen = await loop.run_in_executor(None, _gen)
            for piece in gen:
                if cancel_event is not None and cancel_event.is_set():
                    break
                yield piece
        except Exception as e:
            logger.warning("Native C generation error: %s", e, extra={"tag": "MODEL"})

    async def chat(self, messages, max_tokens=512, temperature=0.8, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        def _gen():
            return self._engine.generate(messages, max_tokens=max_tokens,
                                         temperature=temperature)
        return await loop.run_in_executor(None, _gen)

    def embed(self, text: str) -> list:
        return []

    @property
    def metadata(self) -> dict:
        return {
            "model_id": self._model_id,
            "type": "native-c",
            "model_type": self._engine._model_type,
            "loaded": self._engine.loaded,
            "vocab_size": self._engine._config.get("vocab_size", 0),
            "layers": self._engine._config.get("num_hidden_layers", 0),
        }
