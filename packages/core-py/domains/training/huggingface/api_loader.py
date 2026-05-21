"""HuggingFace Inference API Loader - OPTIONAL FALLBACK ONLY.

This is for testing/development when local models aren't available.
For production, use HuggingFaceLocalLoader from local_loader.py.

WARNING: This calls external HuggingFace API endpoints.
Set HF_API_KEY or HF_TOKEN environment variable to use.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import requests


HF_API_BASE = "https://api-inference.huggingface.co/models"


@dataclass
class HFAPIConfig:
    """Configuration for HF Inference API."""

    model: str
    api_key: Optional[str] = None
    timeout: int = 60
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0


class HuggingFaceAPILoader:
    """Load models via HuggingFace Inference API."""

    def __init__(self, config: HFAPIConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to HF Inference API."""
        url = f"{HF_API_BASE}/{self.config.model}"
        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=self.config.timeout,
        )

        if response.status_code == 503:
            raise RuntimeError(f"Model {self.config.model} is loading. Please wait and try again.")

        if response.status_code == 401:
            raise RuntimeError("Invalid or missing HuggingFace API token.")

        response.raise_for_status()
        return response.json()

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
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
                "temperature": temperature or self.config.temperature,
                "top_p": top_p or self.config.top_p,
                "repetition_penalty": repetition_penalty or self.config.repetition_penalty,
                "return_full_text": False,
            },
            "options": {"use_cache": True},
        }
        payload["parameters"].update(kwargs)

        result = self._make_request(payload)

        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        elif isinstance(result, dict):
            return result.get("generated_text", "")
        return str(result)

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


class HFInferenceClient(HuggingFaceAPILoader):
    """Alias for HuggingFaceAPILoader for compatibility."""

    pass


def create_api_client(model: str, **kwargs) -> HuggingFaceAPILoader:
    """Create an HF API client."""
    config = HFAPIConfig(model=model, **kwargs)
    return HuggingFaceAPILoader(config)


def generate_via_api(
    prompt: str,
    model: str = "gpt2",
    api_key: Optional[str] = None,
    **kwargs,
) -> str:
    """Quick generate via HF API."""
    client = create_api_client(model, api_key=api_key)
    return client.generate(prompt, **kwargs)


def chat_via_api(
    messages: List[Dict[str, str]],
    model: str = "gpt2",
    api_key: Optional[str] = None,
    **kwargs,
) -> str:
    """Quick chat via HF API."""
    client = create_api_client(model, api_key=api_key)
    return client.chat(messages, **kwargs)


__all__ = [
    "HFAPIConfig",
    "HuggingFaceAPILoader",
    "HFInferenceClient",
    "create_api_client",
    "generate_via_api",
    "chat_via_api",
]
