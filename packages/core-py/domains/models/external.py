"""
External Model Import Interface for SloughGPT.

Provides a standardized way to import any model from various sources:
- HuggingFace Hub (models, safetensors)
- Local files (.slo, .pt, .safetensors, .gguf)
- Online URLs (direct download)
- Ollama (local LLM server)
- llama.cpp (native binaries)

Usage:
    from domains.models.external import ModelImporter

    # From HuggingFace
    model = ModelImporter.from_huggingface("gpt2")
    model = ModelImporter.from_huggingface("meta-llama/Llama-2-7b")

    # From local file
    model = ModelImporter.from_local("model.slo")
    model = ModelImporter.from_local("model.pt")

    # From Ollama
    model = ModelImporter.from_ollama("llama2")

    # Auto-detect source
    model = ModelImporter.auto_load("gpt2")  # HuggingFace
    model = ModelImporter.auto_load("model.slo")  # Local
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from domains.training.slonet_compat import torch

logger = logging.getLogger("sloughgpt.external")

if TYPE_CHECKING:
    from domains.models import ModelInterface

SOURCE_HUGGINGFACE = "huggingface"
SOURCE_LOCAL = "local"
SOURCE_OLLAMA = "ollama"
SOURCE_LLAMACPP = "llamacpp"
SOURCE_ONLINE = "online"


class ModelImporter:
    """
    Unified interface for importing models from various sources.

    Supports:
    - HuggingFace Hub (models, safetensors)
    - Local files (.slo, .pt, .safetensors, .gguf)
    - Ollama (local LLM server)
    - llama.cpp binaries
    """

    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, loader_func: Any) -> None:
        """Register a custom model loader."""
        cls._registry[name] = loader_func
        logger.info(f"Registered model loader: {name}")

    @classmethod
    def from_huggingface(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs
    ) -> "ModelInterface":
        """
        Load a model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID (e.g., "gpt2", "meta-llama/Llama-2-7b")
            device: Device to load model on ("cpu", "cuda")
            **kwargs: Additional arguments for AutoModelForCausalLM.from_pretrained

        Returns:
            Model wrapped in ModelInterface
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
            from domains.models import HuggingFaceWrapper, SloughGPTModel

            logger.info(f"Loading from HuggingFace: {model_id}")

            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model_type = getattr(config, "model_type", "").lower()

            if model_type in ("llama", "mistral", "gpt2", "opt", "bloom", "falcon"):
                logger.info(f"Using HuggingFace wrapper for: {model_type}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    trust_remote_code=True,
                    device_map=device if device != "cpu" else None,
                    **kwargs
                )
                return HuggingFaceWrapper(model)
            else:
                logger.info(f"Auto-detected architecture: {model_type}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    trust_remote_code=True,
                    device_map=device if device != "cpu" else None,
                    **kwargs
                )
                return HuggingFaceWrapper(model)

        except ImportError:
            raise ImportError(
                "HuggingFace loading requires transformers. "
                "Install: pip install transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load from HuggingFace: {e}")
            raise

    @classmethod
    def from_local(
        cls,
        path: str,
        device: str = "cpu",
        **kwargs
    ) -> "ModelInterface":
        """
        Load a model from a local file.

        Supports:
        - .slo files (SloughGPT Slo Unit format)
        - .pt/.pth files (PyTorch checkpoints)
        - .safetensors files
        - .gguf files (via llama-cpp-python)

        Args:
            path: Path to model file
            device: Device to load model on
            **kwargs: Additional arguments

        Returns:
            Model wrapped in ModelInterface
        """
        from domains.models import ModelLoader, SloughGPTModel

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        suffix = p.suffix.lower()
        logger.info(f"Loading local model: {path}")

        if suffix == ".gguf":
            try:
                from llama_cpp import Llama
                return Llama(model_path=str(p), n_ctx=kwargs.get("n_ctx", 2048))
            except ImportError:
                raise ImportError(
                    "GGUF loading requires llama-cpp-python. "
                    "Install: pip install llama-cpp-python"
                )

        return ModelLoader.load(path, device=device, **kwargs)

    @classmethod
    def from_ollama(
        cls,
        model_name: str,
        base_url: str = "http://localhost:11434",
        **kwargs
    ) -> "OllamaWrapper":
        """
        Load a model via Ollama API.

        Args:
            model_name: Ollama model name (e.g., "llama2", "mistral")
            base_url: Ollama server URL
            **kwargs: Additional arguments

        Returns:
            OllamaWrapper instance
        """
        return OllamaWrapper(model_name, base_url, **kwargs)

    @classmethod
    def from_llamacpp(
        cls,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        **kwargs
    ) -> Any:
        """
        Load a model via llama.cpp bindings.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context size
            n_gpu_layers: Number of layers to offload to GPU
            **kwargs: Additional llama.cpp arguments

        Returns:
            Llama instance
        """
        try:
            from llama_cpp import Llama
            return Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "llama.cpp loading requires llama-cpp-python. "
                "Install: pip install llama-cpp-python"
            )

    @classmethod
    def from_url(
        cls,
        url: str,
        output_path: Optional[str] = None,
        device: str = "cpu",
        **kwargs
    ) -> "ModelInterface":
        """
        Download and load a model from a URL.

        Args:
            url: Direct URL to model file
            output_path: Local path to save downloaded file
            device: Device to load model on
            **kwargs: Additional arguments

        Returns:
            Model wrapped in ModelInterface
        """
        import tempfile
        import urllib.request

        if output_path is None:
            output_path = Path(tempfile.gettempdir()) / Path(url).name

        logger.info(f"Downloading model from: {url}")

        try:
            urllib.request.urlretrieve(url, output_path)
            logger.info(f"Downloaded to: {output_path}")
            return cls.from_local(output_path, device=device, **kwargs)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    @classmethod
    def auto_load(
        cls,
        source: str,
        device: str = "cpu",
        **kwargs
    ) -> "ModelInterface":
        """
        Auto-detect source and load model.

        Detects:
        - HuggingFace model IDs (e.g., "gpt2", "meta-llama/...")
        - Local file paths
        - URLs (http://, https://)
        - Ollama models (ollama://...)

        Args:
            source: Model source (ID, path, or URL)
            device: Device to load model on
            **kwargs: Additional arguments

        Returns:
            Model wrapped in ModelInterface
        """
        source = str(source).strip()

        if source.startswith(("http://", "https://")):
            logger.info(f"Detected URL source: {source}")
            return cls.from_url(source, device=device, **kwargs)

        if source.startswith("ollama://"):
            model_name = source.replace("ollama://", "")
            logger.info(f"Detected Ollama source: {model_name}")
            return cls.from_ollama(model_name, **kwargs)

        if "/" in source or source.replace("-", "").replace("_", "").isalnum():
            if "/" in source or len(source) > 3:
                logger.info(f"Detected HuggingFace source: {source}")
                return cls.from_huggingface(source, device=device, **kwargs)

        p = Path(source)
        if p.exists():
            logger.info(f"Detected local file: {source}")
            return cls.from_local(source, device=device, **kwargs)

        if p.suffix in (".slo", ".pt", ".pth", ".safetensors", ".gguf"):
            logger.info(f"Detected local file by extension: {source}")
            return cls.from_local(source, device=device, **kwargs)

        logger.info(f"Assuming HuggingFace source: {source}")
        return cls.from_huggingface(source, device=device, **kwargs)

    @classmethod
    def list_supported_local_formats(cls) -> List[str]:
        """List supported local file formats."""
        return [".slo", ".pt", ".pth", ".safetensors", ".gguf"]

    @classmethod
    def list_supported_online_sources(cls) -> List[str]:
        """List supported online sources."""
        return ["HuggingFace Hub", "Direct URL"]


class OllamaWrapper:
    """
    Wrapper for Ollama API.

    Provides a ModelInterface-compatible wrapper for Ollama models.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._config = {
            "model": model_name,
            "base_url": base_url,
            **kwargs
        }
        self._device = "cpu"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text using Ollama API."""
        import json
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "")
        except urllib.error.URLError as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(
                f"Ollama server not available at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )

    def __call__(self, prompt: str, **kwargs) -> str:
        """Shorthand for generate."""
        return self.generate(prompt, **kwargs)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Ollama models don't have a state dict."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], **kwargs) -> None:
        """No-op for Ollama models."""
        pass

    def num_parameters(self) -> int:
        """Unknown for Ollama models."""
        return 0

    def config(self) -> Dict[str, Any]:
        """Return configuration."""
        return self._config.copy()

    def to(self, device: str) -> "OllamaWrapper":
        """No-op for Ollama models."""
        self._device = device
        return self

    def eval(self) -> "OllamaWrapper":
        """No-op for Ollama models."""
        return self

    def train_mode(self) -> "OllamaWrapper":
        """No-op for Ollama models."""
        return self


class LLamaCppWrapper:
    """
    Wrapper for llama.cpp (llama-cpp-python) models.

    Provides a ModelInterface-compatible wrapper for GGUF models.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        **kwargs
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama.cpp loading requires llama-cpp-python. "
                "Install: pip install llama-cpp-python"
            )

        self._llama = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            **kwargs
        )
        self._config = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
        }
        self._device = "cpu"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        **kwargs
    ) -> str:
        """Generate text using llama.cpp."""
        output = self._llama(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            **kwargs
        )
        return output["choices"][0]["text"]

    def __call__(self, prompt: str, **kwargs) -> str:
        """Shorthand for generate."""
        return self.generate(prompt, **kwargs)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """llama.cpp models don't expose state dict."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], **kwargs) -> None:
        """No-op for llama.cpp models."""
        pass

    def num_parameters(self) -> int:
        """Unknown for llama.cpp models."""
        return 0

    def config(self) -> Dict[str, Any]:
        """Return configuration."""
        return self._config.copy()

    def to(self, device: str) -> "LLamaCppWrapper":
        """No-op for llama.cpp models."""
        self._device = device
        return self

    def eval(self) -> "LLamaCppWrapper":
        """No-op for llama.cpp models."""
        return self

    def train_mode(self) -> "LLamaCppWrapper":
        """No-op for llama.cpp models."""
        return self


__all__ = [
    "ModelImporter",
    "OllamaWrapper",
    "LLamaCppWrapper",
    "SOURCE_HUGGINGFACE",
    "SOURCE_LOCAL",
    "SOURCE_OLLAMA",
    "SOURCE_LLAMACPP",
    "SOURCE_ONLINE",
]
