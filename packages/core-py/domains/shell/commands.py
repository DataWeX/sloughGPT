"""
ShellCommands — delegates to existing domains for all operations.

Each command maps to real backend endpoints or domain functions.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("man.shell.commands")

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Default API base — overridden via ShellCommands(api_base=...) or SLGPT_API env var
_DEFAULT_API_BASE = "http://localhost:8000"


def _resolve_api_base() -> str:
    """Resolve API base from env var or default."""
    return os.environ.get("SLGPT_API", _DEFAULT_API_BASE)


def _api_get(path: str, api_base: str | None = None) -> dict[str, Any] | list:
    import requests
    base = api_base or _resolve_api_base()
    try:
        r = requests.get(f"{base}{path}", timeout=10)
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}", "detail": r.text[:200]}
    except Exception as e:
        return {"error": str(e)}


def _api_post(path: str, data: dict | None = None, api_base: str | None = None, timeout: int = 120) -> dict[str, Any] | list:
    import requests
    base = api_base or _resolve_api_base()
    try:
        r = requests.post(f"{base}{path}", json=data or {}, timeout=timeout)
        if r.status_code in (200, 201):
            return r.json()
        return {"error": f"HTTP {r.status_code}", "detail": r.text[:200]}
    except Exception as e:
        return {"error": str(e)}


def _api_delete(path: str, api_base: str | None = None) -> dict[str, Any]:
    import requests
    base = api_base or _resolve_api_base()
    try:
        r = requests.delete(f"{base}{path}", timeout=10)
        return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


class ShellCommands:
    """All commands are thin wrappers delegating to real domains."""

    def __init__(self, api_base: str | None = None):
        self.api_base = api_base or _resolve_api_base()

    def ps(self) -> list[dict[str, Any]]:
        """List running training jobs (processes)."""
        result = _api_get("/training/jobs", self.api_base)
        if isinstance(result, list):
            return result
        return []

    def kill(self, job_id: str) -> dict[str, Any]:
        """Stop a training job."""
        return _api_post(f"/training/jobs/{job_id}/stop", api_base=self.api_base)

    def models(self) -> list[dict[str, Any]]:
        """List available models."""
        result = _api_get("/models", self.api_base)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("models", [])
        return []

    def load_model(self, model_name: str) -> dict[str, Any]:
        """Load a model by name."""
        return _api_post("/models/load", {"model_id": model_name, "device": "cpu"}, api_base=self.api_base, timeout=600)

    def unload_model(self) -> dict[str, Any]:
        """Unload the current model."""
        return _api_post("/models/unload", api_base=self.api_base)

    def souls(self) -> list[dict[str, Any]]:
        """List available souls."""
        result = _api_get("/souls", self.api_base)
        if isinstance(result, dict):
            return result.get("souls", [])
        return []

    def switch_soul(self, name: str) -> dict[str, Any]:
        """Switch to a soul."""
        return _api_post("/souls/switch", {"name": name}, api_base=self.api_base)

    def current_soul(self) -> dict[str, Any]:
        """Get the current soul."""
        result = _api_get("/souls/current", self.api_base)
        if isinstance(result, dict):
            return result
        return {"name": "unknown"}

    def health(self) -> dict[str, Any]:
        """System health check."""
        result = _api_get("/health", self.api_base)
        if isinstance(result, dict):
            return result
        return {"status": "unknown"}

    def health_detailed(self) -> dict[str, Any]:
        """Detailed system health."""
        result = _api_get("/health/detailed", self.api_base)
        if isinstance(result, dict):
            return result
        return {"status": "unknown"}

    def datasets(self) -> list[dict[str, Any]]:
        """List available datasets."""
        result = _api_get("/datasets", self.api_base)
        if isinstance(result, dict):
            return result.get("datasets", [])
        return []

    def list_knowledge(self, query: str = "") -> list[dict[str, Any]]:
        """List knowledge base entries."""
        if query:
            result = _api_get(f"/knowledge/search?query={query}", self.api_base)
            if isinstance(result, dict):
                return result.get("results", [])
        else:
            result = _api_get("/knowledge", self.api_base)
            if isinstance(result, list):
                return result
        return []

    def add_knowledge(self, content: str, topic: str = "shell") -> dict[str, Any]:
        """Add a fact to the knowledge base."""
        return _api_post("/knowledge", {"content": content, "topic": topic, "source": "shell"}, api_base=self.api_base)

    def knowledge_stats(self) -> dict[str, Any]:
        """Knowledge base statistics."""
        result = _api_get("/knowledge/stats", self.api_base)
        if isinstance(result, dict):
            return result
        return {"total_items": 0}

    def checkpoints(self) -> list[dict[str, Any]]:
        """List saved checkpoints."""
        result = _api_get("/auto-train/checkpoints", self.api_base)
        if isinstance(result, dict):
            return result.get("checkpoints", [])
        return []

    def load_checkpoint(self, name: str) -> dict[str, Any]:
        """Load a checkpoint."""
        return _api_post(f"/auto-train/checkpoints/{name}/load", api_base=self.api_base)

    def delete_checkpoint(self, name: str) -> dict[str, Any]:
        """Delete a checkpoint."""
        return _api_delete(f"/auto-train/checkpoints/{name}", self.api_base)

    def finetuned_models(self) -> list[dict[str, Any]]:
        """List fine-tuned models."""
        result = _api_get("/training/finetuned-models", self.api_base)
        if isinstance(result, dict):
            return result.get("models", [])
        return []

    def load_finetuned(self, name: str) -> dict[str, Any]:
        """Load a fine-tuned model."""
        return _api_post(f"/training/finetuned-models/{name}/load", api_base=self.api_base)

    def delete_finetuned(self, name: str) -> dict[str, Any]:
        """Delete a fine-tuned model."""
        return _api_delete(f"/training/finetuned-models/{name}", self.api_base)

    def generate(self, prompt: str, max_tokens: int = 100) -> dict[str, Any]:
        """Generate text via the inference endpoint."""
        return _api_post("/inference/generate", {
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
        }, api_base=self.api_base)

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Send a chat message."""
        return _api_post("/chat", {"messages": messages}, api_base=self.api_base)

    def system_metrics(self) -> dict[str, Any]:
        """System metrics (CPU, memory, disk)."""
        result = _api_get("/system/metrics", self.api_base)
        if isinstance(result, dict):
            return result
        return {}

    def tokenizer_stats(self) -> dict[str, Any]:
        """Tokenizer statistics."""
        result = _api_get("/tokenizer/stats", self.api_base)
        if isinstance(result, dict):
            return result
        return {}

    def feedback_stats(self) -> dict[str, Any]:
        """Feedback pipeline statistics."""
        result = _api_get("/workflow/status", self.api_base)
        if isinstance(result, dict):
            return result
        return {}

    def feedback_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent feedback records."""
        result = _api_get(f"/feedback?limit={limit}", self.api_base)
        if isinstance(result, list):
            return result
        return []

    def conversations(self, limit: int = 20) -> list[dict[str, Any]]:
        """List chat sessions."""
        result = _api_get("/chat/sessions", self.api_base)
        if isinstance(result, list):
            return result[:limit]
        if isinstance(result, dict):
            return result.get("sessions", [])[:limit]
        return []

    def benchmark_history(self) -> list[dict[str, Any]]:
        """Benchmark quality metrics history."""
        result = _api_get("/benchmark/metrics", self.api_base)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("metrics", [])
        return []

    def training_status(self) -> dict[str, Any]:
        """Current training status."""
        result = _api_get("/training/status", self.api_base)
        if isinstance(result, dict):
            return result
        return {}

    def auto_train_status(self) -> dict[str, Any]:
        """Auto-train pipeline status."""
        result = _api_get("/auto-train/status", self.api_base)
        if isinstance(result, dict):
            return result
        return {}
