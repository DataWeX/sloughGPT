"""HuggingFace Local Loader - Download and run models locally.

Device auto-detection and dtype resolution do not require PyTorch.
The underlying ``transformers`` model still needs torch at runtime,
but this module itself imports neither torch nor slonet_compat.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path

import logging

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore

logger = logging.getLogger("slo.hf_loader")


@dataclass
class HFLocalConfig:
    """Configuration for local HF model loading."""

    model: str
    device: str = "auto"
    dtype: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    cache_dir: Optional[str] = None
    local_files_only: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0


class HuggingFaceLocalLoader:
    """Load and run HuggingFace models locally."""

    def __init__(self, config: HFLocalConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._determine_device()

    def _determine_device(self):
        """Auto-detect the best device.

        PyTorch is not required for detection: without torch installed the
        only usable device is CPU. Explicit ``config.device`` values are
        preserved.
        """
        if self.config.device == "auto":
            self.config.device = "cpu"

    def _get_dtype(self):
        """Get dtype from config string (string form for transformers)."""
        dtype_map = {
            "float32": "float32",
            "float16": "float16",
            "half": "float16",
            "bfloat16": "bfloat16",
            "auto": "auto",
        }
        return dtype_map.get(self.config.dtype, "float32")

    def load(self) -> None:
        """Download and load the model."""
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError(
                "transformers is not installed. Install it with: pip install transformers"
            )
        cache_dir = self.config.cache_dir or os.getenv(
            "HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface")
        )

        logger.info("Loading model: %s", self.config.model,
            extra={"tag": "TRAIN"},)
        logger.info("Device: %s", self.config.device,
            extra={"tag": "TRAIN"},)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            cache_dir=cache_dir,
            local_files_only=self.config.local_files_only,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "pretrained_model_name_or_path": self.config.model,
            "cache_dir": cache_dir,
            "local_files_only": self.config.local_files_only,
        }

        if self.config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        elif self.config.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            dtype = self._get_dtype()
            if dtype != "auto":
                load_kwargs["dtype"] = dtype
            load_kwargs["device_map"] = None

        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        if not self.config.load_in_8bit and not self.config.load_in_4bit:
            if self.config.device != "cpu":
                self.model = self.model.to(self.config.device)

        self.model.eval()
        logger.info("Model loaded successfully!",
            extra={"tag": "TRAIN"},)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text from prompt."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.config.device != "cpu":
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": top_p or self.config.top_p,
            "repetition_penalty": repetition_penalty or self.config.repetition_penalty,
            "do_sample": True,
        }
        gen_kwargs.update(kwargs)

        outputs = self.model.generate(**inputs, **gen_kwargs)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Chat with the model using messages format."""
        prompt = self._format_chat_prompt(messages)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"
            elif role == "system":
                formatted += f"System: {content}\n"
        formatted += "Assistant:"
        return formatted

    def unload(self):
        """Unload the model to free memory."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None


class HuggingFaceLocalClient(HuggingFaceLocalLoader):
    """Alias for HuggingFaceLocalLoader for compatibility."""

    pass


def download_model(model: str, cache_dir: Optional[str] = None) -> str:
    """Download a model without loading it."""
    cache_dir = cache_dir or os.getenv("HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface"))
    logger.info("Downloading %s...", model,
        extra={"tag": "TRAIN"},)
    AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    AutoModelForCausalLM.from_pretrained(model, cache_dir=cache_dir)
    logger.info("Downloaded to %s", cache_dir,
        extra={"tag": "TRAIN"},)
    return cache_dir


def load_model(config: HFLocalConfig) -> HuggingFaceLocalLoader:
    """Load a model with the given config."""
    loader = HuggingFaceLocalLoader(config)
    loader.load()
    return loader


def generate_local(
    prompt: str,
    model: str = "gpt2",
    device: str = "auto",
    **kwargs,
) -> str:
    """Quick generate with local model."""
    config = HFLocalConfig(model=model, device=device, **kwargs)
    loader = HuggingFaceLocalLoader(config)
    loader.load()
    return loader.generate(prompt)


__all__ = [
    "HFLocalConfig",
    "HuggingFaceLocalLoader",
    "HuggingFaceLocalClient",
    "download_model",
    "load_model",
    "generate_local",
]
