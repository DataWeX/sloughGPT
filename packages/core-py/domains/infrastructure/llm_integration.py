"""
LLM Integration - ARCHIVED (2026-03-18)

This code has been moved to archive/third_party_apis/ for reference.
SloughGPT now focuses on LOCAL MODEL INFRASTRUCTURE only.

To restore:
    mv archive/third_party_apis/llm_integration.py domains/infrastructure/
    mv archive/third_party_apis/api_loader.py domains/training/huggingface/
"""

# RESTORED FROM ARCHIVE - Original third-party integration

import os
from typing import Dict, Optional
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        if not self.is_available():
            return "OpenAI API key not available"

        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            return f"Error: {str(e)}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        if not self.is_available():
            return "Anthropic API key not available"

        try:
            import requests

            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            data = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            result = response.json()
            return result.get("content", [{}])[0].get("text", "")
        except Exception as e:
            return f"Error: {str(e)}"


class LocalProvider(LLMProvider):
    """Local model provider (Ollama, LM Studio, etc.)"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def is_available(self) -> bool:
        try:
            import requests

            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            import requests

            data = {"model": "llama2", "prompt": prompt, "stream": False}
            response = requests.post(f"{self.base_url}/api/generate", json=data, timeout=60)
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            return f"Error: {str(e)}"


class LLMIntegration:
    """Unified LLM integration"""

    def __init__(self, provider: str = "openai"):
        self.providers: Dict[str, LLMProvider] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "local": LocalProvider(),
        }
        self.current_provider = self.providers.get(provider, self.providers["openai"])

    def set_provider(self, provider: str) -> None:
        if provider in self.providers:
            self.current_provider = self.providers[provider]

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        return self.current_provider.generate_response(prompt, max_tokens)

    def is_available(self) -> bool:
        return self.current_provider.is_available()


__all__ = ["LLMProvider", "OpenAIProvider", "AnthropicProvider", "LocalProvider", "LLMIntegration"]
