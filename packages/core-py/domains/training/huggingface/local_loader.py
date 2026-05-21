"""HuggingFace Local Loader - Download and run models locally."""

import os
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path

from domains.training.slonet_compat import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore


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
        """Auto-detect the best device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.config.device = "cuda"
            elif torch.backends.mps.is_available():
                self.config.device = "mps"
            else:
                self.config.device = "cpu"

    def _get_dtype(self):
        """Get torch dtype from config string."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "half": torch.half,
            "bfloat16": torch.bfloat16,
            "auto": "auto",
        }
        return dtype_map.get(self.config.dtype, torch.float16)

    def load(self) -> None:
        """Download and load the model."""
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError(
                "transformers is not installed. Install it with: pip install transformers"
            )
        cache_dir = self.config.cache_dir or os.getenv(
            "HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface")
        )

        print(f"Loading model: {self.config.model}")
        print(f"Device: {self.config.device}")

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
        print("Model loaded successfully!")

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

        with torch.no_grad():
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HuggingFaceLocalClient(HuggingFaceLocalLoader):
    """Alias for HuggingFaceLocalLoader for compatibility."""

    pass


def download_model(model: str, cache_dir: Optional[str] = None) -> str:
    """Download a model without loading it."""
    cache_dir = cache_dir or os.getenv("HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface"))
    print(f"Downloading {model}...")
    AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    AutoModelForCausalLM.from_pretrained(model, cache_dir=cache_dir)
    print(f"Downloaded to {cache_dir}")
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
