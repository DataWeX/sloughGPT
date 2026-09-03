"""HuggingFace model map - Registry of available HF models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any


class ModelSize(Enum):
    """Model size categories."""

    SMALL = "small"  # < 1B params
    MEDIUM = "medium"  # 1-7B params
    LARGE = "large"  # 7-13B params
    XLARGE = "xlarge"  # > 13B params


@dataclass
class HFModelInfo:
    """Information about a HuggingFace model."""

    model_id: str
    name: str
    description: str
    size: ModelSize
    params: int
    context_length: int
    recommended_quantization: str
    memory_fp16_gb: float
    memory_int8_gb: float
    memory_q4_gb: float
    organization: str
    tags: List[str]


HF_MODELS = {
    "gpt2": HFModelInfo(
        model_id="gpt2",
        name="GPT-2",
        description="OpenAI GPT-2 (124M params)",
        size=ModelSize.SMALL,
        params=124_000_000,
        context_length=1024,
        recommended_quantization="fp16",
        memory_fp16_gb=0.5,
        memory_int8_gb=0.3,
        memory_q4_gb=0.2,
        organization="openai",
        tags=["gpt", "causal-lm"],
    ),
    "gpt2-medium": HFModelInfo(
        model_id="gpt2-medium",
        name="GPT-2 Medium",
        description="OpenAI GPT-2 Medium (355M params)",
        size=ModelSize.SMALL,
        params=355_000_000,
        context_length=1024,
        recommended_quantization="fp16",
        memory_fp16_gb=1.0,
        memory_int8_gb=0.6,
        memory_q4_gb=0.4,
        organization="openai",
        tags=["gpt", "causal-lm"],
    ),
    "gpt2-large": HFModelInfo(
        model_id="gpt2-large",
        name="GPT-2 Large",
        description="OpenAI GPT-2 Large (774M params)",
        size=ModelSize.SMALL,
        params=774_000_000,
        context_length=1024,
        recommended_quantization="fp16",
        memory_fp16_gb=2.0,
        memory_int8_gb=1.0,
        memory_q4_gb=0.6,
        organization="openai",
        tags=["gpt", "causal-lm"],
    ),
    "microsoft/phi-2": HFModelInfo(
        model_id="microsoft/phi-2",
        name="Phi-2",
        description="Microsoft Phi-2 (2.7B params)",
        size=ModelSize.MEDIUM,
        params=2_700_000_000,
        context_length=2048,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=6.0,
        memory_int8_gb=3.5,
        memory_q4_gb=2.0,
        organization="microsoft",
        tags=["phi", "code", "reasoning"],
    ),
    "mistralai/Mistral-7B-Instruct-v0.2": HFModelInfo(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        name="Mistral 7B Instruct v0.2",
        description="Mistral 7B Instruct v0.2 (7.3B params)",
        size=ModelSize.MEDIUM,
        params=7_300_000_000,
        context_length=32768,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=14.6,
        memory_int8_gb=7.5,
        memory_q4_gb=4.0,
        organization="mistralai",
        tags=["mistral", "chat", "instruction"],
    ),
    "meta-llama/Llama-2-7b-chat-hf": HFModelInfo(
        model_id="meta-llama/Llama-2-7b-chat-hf",
        name="LLaMA-2 7B Chat",
        description="Meta LLaMA-2 7B Chat (7B params)",
        size=ModelSize.MEDIUM,
        params=7_000_000_000,
        context_length=4096,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=14.0,
        memory_int8_gb=7.0,
        memory_q4_gb=4.0,
        organization="meta",
        tags=["llama", "chat", "instruction"],
    ),
    "meta-llama/Llama-2-13b-chat-hf": HFModelInfo(
        model_id="meta-llama/Llama-2-13b-chat-hf",
        name="LLaMA-2 13B Chat",
        description="Meta LLaMA-2 13B Chat (13B params)",
        size=ModelSize.LARGE,
        params=13_000_000_000,
        context_length=4096,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=26.0,
        memory_int8_gb=13.0,
        memory_q4_gb=7.0,
        organization="meta",
        tags=["llama", "chat", "instruction"],
    ),
    "codellama/CodeLlama-7b-Instruct-hf": HFModelInfo(
        model_id="codellama/CodeLlama-7b-Instruct-hf",
        name="Code LLaMA 7B Instruct",
        description="Code LLaMA 7B Instruct (7B params)",
        size=ModelSize.MEDIUM,
        params=7_000_000_000,
        context_length=16384,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=14.0,
        memory_int8_gb=7.0,
        memory_q4_gb=4.0,
        organization="codellama",
        tags=["codellama", "code", "instruction"],
    ),
    "Qwen/Qwen2-0.5B-Instruct": HFModelInfo(
        model_id="Qwen/Qwen2-0.5B-Instruct",
        name="Qwen2 0.5B Instruct",
        description="Qwen2 0.5B Instruct (0.5B params)",
        size=ModelSize.SMALL,
        params=500_000_000,
        context_length=32768,
        recommended_quantization="fp16",
        memory_fp16_gb=1.2,
        memory_int8_gb=0.7,
        memory_q4_gb=0.4,
        organization="qwen",
        tags=["qwen", "chat", "multilingual"],
    ),
    "Qwen/Qwen2-1.5B-Instruct": HFModelInfo(
        model_id="Qwen/Qwen2-1.5B-Instruct",
        name="Qwen2 1.5B Instruct",
        description="Qwen2 1.5B Instruct (1.5B params)",
        size=ModelSize.SMALL,
        params=1_500_000_000,
        context_length=32768,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=3.2,
        memory_int8_gb=1.8,
        memory_q4_gb=1.0,
        organization="qwen",
        tags=["qwen", "chat", "multilingual"],
    ),
    "Qwen/Qwen2-7B-Instruct": HFModelInfo(
        model_id="Qwen/Qwen2-7B-Instruct",
        name="Qwen2 7B Instruct",
        description="Qwen2 7B Instruct (7B params)",
        size=ModelSize.MEDIUM,
        params=7_000_000_000,
        context_length=32768,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=14.0,
        memory_int8_gb=7.0,
        memory_q4_gb=4.0,
        organization="qwen",
        tags=["qwen", "chat", "multilingual"],
    ),
    "google/gemma-2b-it": HFModelInfo(
        model_id="google/gemma-2b-it",
        name="Gemma 2B Instruct",
        description="Google Gemma 2B Instruct (2B params)",
        size=ModelSize.SMALL,
        params=2_000_000_000,
        context_length=8192,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=4.2,
        memory_int8_gb=2.5,
        memory_q4_gb=1.5,
        organization="google",
        tags=["gemma", "chat", "instruction"],
    ),
    "google/gemma-7b-it": HFModelInfo(
        model_id="google/gemma-7b-it",
        name="Gemma 7B Instruct",
        description="Google Gemma 7B Instruct (7B params)",
        size=ModelSize.MEDIUM,
        params=7_000_000_000,
        context_length=8192,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=14.0,
        memory_int8_gb=7.0,
        memory_q4_gb=4.0,
        organization="google",
        tags=["gemma", "chat", "instruction"],
    ),
    "deepseek-ai/DeepSeek-Coder-1.3B-instruct": HFModelInfo(
        model_id="deepseek-ai/DeepSeek-Coder-1.3B-instruct",
        name="DeepSeek Coder 1.3B",
        description="DeepSeek Coder 1.3B Instruct (1.3B params)",
        size=ModelSize.SMALL,
        params=1_300_000_000,
        context_length=16384,
        recommended_quantization="fp16",
        memory_fp16_gb=2.8,
        memory_int8_gb=1.5,
        memory_q4_gb=0.9,
        organization="deepseek",
        tags=["coder", "code", "instruction"],
    ),
    "deepseek-ai/DeepSeek-Coder-7B-instruct-v1.5": HFModelInfo(
        model_id="deepseek-ai/DeepSeek-Coder-7B-instruct-v1.5",
        name="DeepSeek Coder 7B",
        description="DeepSeek Coder 7B Instruct v1.5 (7B params)",
        size=ModelSize.MEDIUM,
        params=7_000_000_000,
        context_length=16384,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=14.0,
        memory_int8_gb=7.0,
        memory_q4_gb=4.0,
        organization="deepseek",
        tags=["coder", "code", "instruction"],
    ),
    "tiiuae/falcon-7b-instruct": HFModelInfo(
        model_id="tiiuae/falcon-7b-instruct",
        name="Falcon 7B Instruct",
        description="Falcon 7B Instruct (7B params)",
        size=ModelSize.MEDIUM,
        params=7_000_000_000,
        context_length=2048,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=14.0,
        memory_int8_gb=7.0,
        memory_q4_gb=4.0,
        organization="tiiuae",
        tags=["falcon", "chat", "instruction"],
    ),
    "mistralai/Mistral-7B-v0.1": HFModelInfo(
        model_id="mistralai/Mistral-7B-v0.1",
        name="Mistral 7B",
        description="Mistral 7B Base (7.3B params)",
        size=ModelSize.MEDIUM,
        params=7_300_000_000,
        context_length=32768,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=14.6,
        memory_int8_gb=7.5,
        memory_q4_gb=4.0,
        organization="mistralai",
        tags=["mistral", "base", "causal-lm"],
    ),
    "microsoft/Phi-3-mini-128k-instruct": HFModelInfo(
        model_id="microsoft/Phi-3-mini-128k-instruct",
        name="Phi-3 Mini 128K",
        description="Microsoft Phi-3 Mini 128K Instruct (3.8B params)",
        size=ModelSize.SMALL,
        params=3_800_000_000,
        context_length=131072,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=8.0,
        memory_int8_gb=4.5,
        memory_q4_gb=2.5,
        organization="microsoft",
        tags=["phi", "chat", "instruction", "long-context"],
    ),
    "stabilityai/stablelm-2-1_6b-chat": HFModelInfo(
        model_id="stabilityai/stablelm-2-1_6b-chat",
        name="StableLM-2 1.6B",
        description="StableLM-2 1.6B Chat (1.6B params)",
        size=ModelSize.SMALL,
        params=1_600_000_000,
        context_length=4096,
        recommended_quantization="fp16",
        memory_fp16_gb=3.4,
        memory_int8_gb=2.0,
        memory_q4_gb=1.2,
        organization="stabilityai",
        tags=["stablelm", "chat", "instruction"],
    ),
    "allenai/OLMo-1.7-7b": HFModelInfo(
        model_id="allenai/OLMo-1.7-7b",
        name="OLMo 1.7 7B",
        description="AllenAI OLMo 1.7 7B (7B params)",
        size=ModelSize.MEDIUM,
        params=7_000_000_000,
        context_length=4096,
        recommended_quantization="q4_k_m",
        memory_fp16_gb=14.0,
        memory_int8_gb=7.0,
        memory_q4_gb=4.0,
        organization="allenai",
        tags=["olmo", "open", "base"],
    ),
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": HFModelInfo(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        name="TinyLlama 1.1B",
        description="TinyLlama 1.1B Chat (1.1B params) - fast chat model",
        size=ModelSize.SMALL,
        params=1_100_000_000,
        context_length=2048,
        recommended_quantization="fp16",
        memory_fp16_gb=2.5,
        memory_int8_gb=1.4,
        memory_q4_gb=0.8,
        organization="TinyLlama",
        tags=["tinyllama", "chat", "instruction"],
    ),
}


def get_model_info(model_id: str) -> Optional[HFModelInfo]:
    """Get information about a model."""
    return HF_MODELS.get(model_id)


def search_models(
    organization: Optional[str] = None,
    size: Optional[ModelSize] = None,
    tags: Optional[List[str]] = None,
) -> List[HFModelInfo]:
    """Search for models by organization, size, or tags."""
    results = list(HF_MODELS.values())

    if organization:
        results = [m for m in results if m.organization == organization]

    if size:
        results = [m for m in results if m.size == size]

    if tags:
        results = [m for m in results if any(t in m.tags for t in tags)]

    return results


def get_recommended_quantization(model_id: str) -> str:
    """Get recommended quantization for a model."""
    model = get_model_info(model_id)
    if model:
        return model.recommended_quantization
    return "q4_k_m"


def get_model_requirements(model_id: str, precision: str = "bf16") -> Dict[str, Any]:
    """Get memory requirements for a model."""
    model = get_model_info(model_id)

    if not model:
        return {
            "model_id": model_id,
            "precision": precision,
            "memory_gb": "unknown",
            "params": "unknown",
        }

    precision_map = {
        "fp32": model.memory_fp16_gb * 2,
        "fp16": model.memory_fp16_gb,
        "bf16": model.memory_fp16_gb,
        "int8": model.memory_int8_gb,
        "q4_k_m": model.memory_q4_gb,
        "q4": model.memory_q4_gb,
        "q8": model.memory_int8_gb,
    }

    memory = precision_map.get(precision, model.memory_fp16_gb)

    return {
        "model_id": model_id,
        "name": model.name,
        "params": model.params,
        "precision": precision,
        "memory_gb": memory,
        "context_length": model.context_length,
        "size": model.size.value if model.size else "unknown",
    }


def map_to_sloughgpt_config(model_id: str) -> Dict[str, Any]:
    """Map HuggingFace model to SloughGPT config."""
    model = get_model_info(model_id)

    if not model:
        return {}

    if model.size == ModelSize.SMALL:
        return {
            "n_embed": 256,
            "n_layer": 6,
            "n_head": 8,
            "block_size": 512,
        }
    elif model.size == ModelSize.MEDIUM:
        return {
            "n_embed": 512,
            "n_layer": 12,
            "n_head": 16,
            "block_size": 1024,
        }
    elif model.size == ModelSize.LARGE:
        return {
            "n_embed": 768,
            "n_layer": 20,
            "n_head": 24,
            "block_size": 1024,
        }
    else:
        return {
            "n_embed": 1024,
            "n_layer": 24,
            "n_head": 32,
            "block_size": 2048,
        }


__all__ = [
    "ModelSize",
    "HFModelInfo",
    "HF_MODELS",
    "get_model_info",
    "search_models",
    "get_recommended_quantization",
    "get_model_requirements",
    "map_to_sloughgpt_config",
]
