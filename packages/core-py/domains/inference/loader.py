"""
Model Loader Module
Provides model loading and inference functionality.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import logging

from domains.training.slonet_compat import torch
nn = torch.nn

from .slo_format import SloProfile, SouParser, save_soul
from .quantization import SouModelQuantizer, QuantizationType


logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Inference configuration."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "fp32"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.0
    num_beams: int = 1


class SouModelLoader:
    """Loader for .slo Slo Unit files."""

    def __init__(self, sou_file: Union[str, SloProfile]):
        if isinstance(sou_file, str):
            self.slo = SouParser.load(sou_file)
        else:
            self.slo = sou_file

        gen = getattr(self.slo, 'generation', None) or {}
        ctx = getattr(self.slo, 'context', None) or {}

        if hasattr(gen, 'temperature'):
            temp = gen.temperature
            top_p = gen.top_p
            max_len = gen.max_tokens
        else:
            temp = gen.get('temperature', 0.7) if isinstance(gen, dict) else 0.7
            top_p = gen.get('top_p', 0.9) if isinstance(gen, dict) else 0.9
            max_len = gen.get('max_tokens', 2048) if isinstance(gen, dict) else 2048

        self.config = InferenceConfig(
            temperature=temp,
            top_p=top_p,
            max_length=max_len,
        )

        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None

    def load_model(self):
        """Load the model."""
        base = getattr(self.slo, 'base_model', 'nanogpt')
        logger.info(f"Loading soul: {getattr(self.slo, 'name', 'unknown')} ({base})")

        # Try to load from training models
        try:
            from ..training.models.nanogpt import NanoGPT

            # Create placeholder model
            self.model = NanoGPT(vocab_size=5000, n_embed=256, n_layer=6, n_head=8)
        except Exception as e:
            logger.warning(f"Could not load NanoGPT: {e}")

        return self.model

    def load_tokenizer(self):
        """Load the tokenizer."""
        # Placeholder
        return None

    def quantize(self, quantization_type: QuantizationType = QuantizationType.Q4_K):
        """Quantize the model."""
        if self.model is None:
            raise RuntimeError("Load model before quantizing")

        quantizer = SouModelQuantizer(quantization_type)
        self.model = quantizer.quantize_model(self.model)

        return self.model


class SouInferenceEngine:
    """Inference engine for .slo models."""

    def __init__(self, loader: SouModelLoader):
        self.loader = loader
        self.model = loader.model
        self.config = loader.config

        if self.model is None:
            loader.load_model()
            self.model = loader.model

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate text from prompt."""
        if self.model is None:
            raise RuntimeError("No model loaded")

        max_length = max_length or self.config.max_length
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p

        # Simple character-level generation
        # Real implementation would use proper tokenization
        device = (
            next(self.model.parameters()).device
            if hasattr(self.model, "parameters")
            else torch.device("cpu")
        )

        chars = sorted(set(prompt))
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}

        idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long).to(device)

        self.model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                idx_cond = idx[:, -128:]
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)

        result = "".join([itos.get(i, "") for i in idx[0].tolist()])
        return result[len(prompt) :]

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat completion."""
        # Convert messages to prompt
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return self.generate(prompt, **kwargs)


def load_model(sou_file: str, quantize: Optional[str] = None):
    """Load model from .slo file."""
    loader = SouModelLoader(sou_file)
    loader.load_model()

    if quantize:
        qtype = QuantizationType(quantize)
        loader.quantize(qtype)

    return SouInferenceEngine(loader)


def generate(prompt: str, sou_file: Optional[str] = None, **kwargs):
    """Generate text from prompt."""
    if sou_file:
        engine = load_model(sou_file)
        return engine.generate(prompt, **kwargs)
    else:
        raise ValueError("sou_file required")


def chat(messages: List[Dict[str, str]], sou_file: Optional[str] = None, **kwargs):
    """Chat completion."""
    if sou_file:
        engine = load_model(sou_file)
        return engine.chat(messages, **kwargs)
    else:
        raise ValueError("sou_file required")
