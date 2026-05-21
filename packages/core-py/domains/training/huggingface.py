"""
HuggingFace Integration - Ported from recovered huggingface_integration.py
"""

from pathlib import Path
from typing import Dict, List, Optional


try:
    from huggingface_hub import HfApi, Repository

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HfApi = None
    Repository = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForCausalLM = None


class HuggingFaceManager:
    """Manages Hugging Face integration for datasets and models."""

    def __init__(self, models_dir: str = "hf_models"):
        self.api = HfApi() if HF_AVAILABLE else None
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def search_models(
        self, query: str, limit: int = 10, task: str = "text-generation"
    ) -> List[Dict]:
        """Search for models on Hugging Face."""
        if not HF_AVAILABLE or self.api is None:
            return []

        try:
            models = self.api.list_models(search=query, limit=limit, task=task)

            return [
                {
                    "id": model.id,
                    "author": getattr(model, "author", "Unknown"),
                    "downloads": getattr(model, "downloads", 0) or 0,
                    "likes": getattr(model, "likes", 0) or 0,
                    "library_name": getattr(model, "library_name", "Unknown"),
                }
                for model in models
            ]
        except Exception:
            return []

    def download_model(self, model_id: str, destination: Optional[str] = None) -> Optional[Path]:
        """Download a model from Hugging Face."""
        if not TRANSFORMERS_AVAILABLE:
            return None

        dest = Path(destination) if destination else self.models_dir / model_id.replace("/", "_")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_id)

            tokenizer.save_pretrained(str(dest))
            model.save_pretrained(str(dest))

            return dest
        except Exception:
            return None

    def upload_model(
        self, model_path: str, repo_id: str, commit_message: str = "Upload model"
    ) -> bool:
        """Upload a model to Hugging Face Hub."""
        if not HF_AVAILABLE:
            return False

        try:
            api = HfApi()
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
            return True
        except Exception:
            return False

    def list_downloaded_models(self) -> List[str]:
        """List downloaded models."""
        if not self.models_dir.exists():
            return []

        return [d.name for d in self.models_dir.iterdir() if d.is_dir()]


class HuggingFaceDatasetManager:
    """Manages Hugging Face datasets."""

    def __init__(self):
        self.api = HfApi() if HF_AVAILABLE else None

    def search_datasets(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for datasets on Hugging Face."""
        if not HF_AVAILABLE or self.api is None:
            return []

        try:
            datasets = self.api.list_datasets(search=query, limit=limit)

            return [
                {
                    "id": ds.id,
                    "downloads": getattr(ds, "downloads", 0) or 0,
                    "likes": getattr(ds, "likes", 0) or 0,
                }
                for ds in datasets
            ]
        except Exception:
            return []


__all__ = ["HuggingFaceManager", "HuggingFaceDatasetManager"]
