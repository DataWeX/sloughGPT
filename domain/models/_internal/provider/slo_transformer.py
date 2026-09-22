"""SloTransformerProvider — __slots__, module-level."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from .protocols import ModelCapabilities

logger = logging.getLogger("slo.models.provider")


class SloTransformerProvider:
    __slots__ = ("_model", "_stoi", "_itos", "_model_id_str", "_bos", "_eos")

    def __init__(self, model, stoi: dict, itos: dict, model_id_str: str = "soultransformer"):
        self._model = model
        self._stoi = stoi
        self._itos = {int(k): v for k, v in itos.items()}
        self._model_id_str = model_id_str
        self._bos = stoi.get(" ", 0)
        self._eos = stoi.get("<PAD>", 0)

    @property
    def model_id(self) -> str:
        return self._model_id_str

    @property
    def capabilities(self):
        return ModelCapabilities(chat=True, streaming=True, embedding=False, vision=False)

    def _encode(self, text: str) -> list:
        return [self._stoi.get(c, self._bos) for c in text.lower()]

    def _decode(self, ids: list) -> str:
        return "".join(self._itos.get(i, "?") for i in ids)

    def _messages_to_prompt(self, messages: list) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    async def chat_stream(self, messages: list, max_tokens: int = 512, temperature: float = 0.7, cancel_event=None, session_id: str | None = None, **kwargs) -> AsyncIterator[str]:
        prompt = self._messages_to_prompt(messages)
        input_ids = self._encode(prompt)
        if not input_ids:
            input_ids = [self._bos]
        import numpy as np

        inp = np.array([input_ids], dtype=np.int64)
        loop = asyncio.get_event_loop()

        def _gen():
            return self._model.generate(inp, max_new_tokens=max_tokens, temperature=temperature, top_k=kwargs.get("top_k", 40), top_p=kwargs.get("top_p", 0.95), repetition_penalty=kwargs.get("repetition_penalty", 1.1), eos_token=self._eos)

        try:
            out = await loop.run_in_executor(None, _gen)
        except Exception as e:
            logger.warning("SloTransformer generation error: %s", e, extra={"tag": "MODEL"})
            return
        if out is None:
            return
        out_ids = out.data.flatten().tolist()
        if self._eos in out_ids:
            out_ids = out_ids[: out_ids.index(self._eos)]
        text = self._decode(out_ids)
        if text:
            yield text

    async def chat(self, messages, max_tokens=512, temperature=0.8, **kwargs):
        chunks = []
        async for chunk in self.chat_stream(messages, max_tokens, temperature, **kwargs):
            chunks.append(chunk)
        return "".join(chunks)

    def embed(self, text: str) -> list:
        return []

    @property
    def metadata(self) -> dict:
        return {"model_id": self._model_id_str, "vocab_size": len(self._stoi), "type": "soultransformer", "n_layer": self._model.n_layer, "n_embed": self._model.n_embed, "n_head": self._model.n_head}

    @classmethod
    def load_from_sou(cls, path: str, model_id_str: str = "") -> "SloTransformerProvider":
        from domain.inference import load_soul
        from domain.infrastructure._internal.weight_loader import infer_arch_from_state_dict

        soul, sd = load_soul(path)
        if isinstance(sd, dict) and "tok_emb.weight" not in sd:
            sd = sd.get("weights", sd)
            if not isinstance(sd, dict):
                sd = sd.state_dict() if hasattr(sd, "state_dict") else {}
        chars = ["<PAD>", "<UNK>"] + list(" abcdefghijklmnopqrstuvwxyz0123456789.,!?-'")
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = dict(enumerate(chars))
        meta_vocab = getattr(soul, "vocab_size", None) or (soul.metadata or {}).get("vocab_size")
        if meta_vocab and meta_vocab > len(chars):
            extra = ["_"] * (meta_vocab - len(chars))
            chars = chars + extra
            stoi = {ch: i for i, ch in enumerate(chars)}
            itos = dict(enumerate(chars))
        arch = infer_arch_from_state_dict(sd)
        from domain.training._internal.slonet import SloTransformer

        model = SloTransformer(vocab_size=arch["vocab_size"], n_embed=arch["n_embed"], n_layer=arch["n_layer"], n_head=arch["n_head"], dropout=0.0, tie_weights=False)
        model.load_state_dict(sd, strict=False)
        logger.info("Loaded SloTransformer from %s (vocab=%d, n_embed=%d, n_layer=%d, n_head=%d)", path, arch["vocab_size"], arch["n_embed"], arch["n_layer"], arch["n_head"], extra={"tag": "MODEL"})
        return cls(model, stoi, itos, model_id_str=model_id_str)
