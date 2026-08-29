"""
GGUF Export Module for SloughGPT.

Handles GGUF export for multiple model architectures, optimized for llama.rn.
Auto-detects model architecture from tensor names.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("slo.gguf_export")

# Model-like objects expose ``state_dict()`` returning a dict of arrays.
# Arrays may be numpy ndarrays or SloNet/Torch tensors; conversion is
# handled at export time via ``_as_float16``.
ModelLike = Any
StateDict = Dict[str, Any]

QUANTIZATION_TYPES = {
    "F32": "32-bit float (full precision)",
    "F16": "16-bit float",
    "Q8_0": "8-bit integer (high quality)",
    "Q5_1": "5-bit integer (balanced)",
    "Q5_0": "5-bit integer",
    "Q4_1": "4-bit integer with improved quality",
    "Q4_0": "4-bit integer (smallest, fastest)",
    "Q4_K_M": "4-bit medium (RECOMMENDED for mobile)",
    "Q4_K_S": "4-bit small",
    "Q3_K_M": "3-bit medium",
    "Q3_K_S": "3-bit small",
    "Q2_K": "2-bit medium",
}

MOBILE_RECOMMENDED = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0"]


class GGUFExportConfig:
    """Configuration for GGUF export."""

    def __init__(
        self,
        model_name: str = "sloughgpt",
        model_version: str = "1.0",
        quantization: str = "Q4_K_M",
        use_gpu: bool = False,
        n_ctx: int = 2048,
        rope_freq_base: float = 10000.0,
        rope_freq_scale: float = 1.0,
        architecture: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.quantization = quantization
        self.use_gpu = use_gpu
        self.n_ctx = n_ctx
        self.rope_freq_base = rope_freq_base
        self.rope_freq_scale = rope_freq_scale
        self.architecture = architecture


class TensorMapping:
    """Base class for tensor name mappings."""

    def __init__(self, name: str, gguf_type: str = "llama"):
        self.name = name
        self.gguf_type = gguf_type

    def get_tensor_map(self) -> Dict[str, str]:
        """Return dict mapping model tensor names to GGUF tensor names."""
        raise NotImplementedError

    def get_block_prefix(self) -> str:
        """Return the prefix for transformer block tensors."""
        raise NotImplementedError

    def has_rope(self) -> bool:
        """Return True if this architecture uses RoPE."""
        raise NotImplementedError

    def has_position_embeddings(self) -> bool:
        """Return True if this architecture uses position embeddings."""
        raise NotImplementedError

    def get_special_tensors(self) -> Dict[str, str]:
        """Return any special tensor mappings."""
        return {}

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        """Return mapping for transformer block tensors."""
        return {}

    def get_fused_qkv_keys(self) -> List[Tuple[str, str]]:
        """Return list of (source_suffix, arch_name) for fused QKV tensors.

        Architectures like Falcon/GPT-NeoX/Bloom use a single fused
        ``query_key_value.weight`` that must be split into Q/K/V parts.
        Default: no fused QKV.
        """
        return []


class SloughGPTMapping(TensorMapping):
    """Tensor mapping for SloughGPTModel architecture."""

    def __init__(self):
        super().__init__("sloughgpt", "llama")

    def get_tensor_map(self) -> Dict[str, str]:
        return {
            "tok_emb.weight": "token_embd.weight",
            "lm_head.weight": "output.weight",
            "norm.weight": "output_norm.weight",
        }

    def get_block_prefix(self) -> str:
        return "blocks."

    def has_rope(self) -> bool:
        return True

    def has_position_embeddings(self) -> bool:
        return False

    def get_special_tensors(self) -> Dict[str, str]:
        return {
            "rope.cos": "rope.cos",
            "rope.sin": "rope.sin",
        }

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        mapping = {}
        for i in range(n_layers):
            prefix = f"blocks.{i}."

            mapping[f"{prefix}norm1.weight"] = f"blk.{i}.attn_norm.weight"
            mapping[f"{prefix}norm2.weight"] = f"blk.{i}.ffn_norm.weight"

            mapping[f"{prefix}attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
            mapping[f"{prefix}attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
            mapping[f"{prefix}attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
            mapping[f"{prefix}attn.o_proj.weight"] = f"blk.{i}.attn_output.weight"

            mapping[f"{prefix}mlp.w1.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"{prefix}mlp.w2.weight"] = f"blk.{i}.ffn_down.weight"
            mapping[f"{prefix}mlp.w3.weight"] = f"blk.{i}.ffn_up.weight"

        return mapping


class LLaMAMapping(TensorMapping):
    """Tensor mapping for LLaMA architecture."""

    def __init__(self):
        super().__init__("llama", "llama")

    def get_tensor_map(self) -> Dict[str, str]:
        return {
            "model.embed_tokens.weight": "token_embd.weight",
            "lm_head.weight": "output.weight",
            "model.norm.weight": "output_norm.weight",
        }

    def get_block_prefix(self) -> str:
        return "model.layers."

    def has_rope(self) -> bool:
        return True

    def has_position_embeddings(self) -> bool:
        return False

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        mapping = {}
        for i in range(n_layers):
            prefix = f"model.layers.{i}."

            mapping[f"{prefix}input_layernorm.weight"] = f"blk.{i}.attn_norm.weight"
            mapping[f"{prefix}post_attention_layernorm.weight"] = f"blk.{i}.ffn_norm.weight"

            mapping[f"{prefix}self_attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
            mapping[f"{prefix}self_attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
            mapping[f"{prefix}self_attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
            mapping[f"{prefix}self_attn.o_proj.weight"] = f"blk.{i}.attn_output.weight"

            mapping[f"{prefix}mlp.gate_proj.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"{prefix}mlp.up_proj.weight"] = f"blk.{i}.ffn_up.weight"
            mapping[f"{prefix}mlp.down_proj.weight"] = f"blk.{i}.ffn_down.weight"

        return mapping


class MistralMapping(TensorMapping):
    """Tensor mapping for Mistral architecture."""

    def __init__(self):
        super().__init__("mistral", "mistral")

    def get_tensor_map(self) -> Dict[str, str]:
        return {
            "model.embed_tokens.weight": "token_embd.weight",
            "lm_head.weight": "output.weight",
            "model.norm.weight": "output_norm.weight",
        }

    def get_block_prefix(self) -> str:
        return "model.layers."

    def has_rope(self) -> bool:
        return True

    def has_position_embeddings(self) -> bool:
        return False

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        mapping = {}
        for i in range(n_layers):
            prefix = f"model.layers.{i}."

            mapping[f"{prefix}input_layernorm.weight"] = f"blk.{i}.attn_norm.weight"
            mapping[f"{prefix}post_attention_layernorm.weight"] = f"blk.{i}.ffn_norm.weight"

            mapping[f"{prefix}self_attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
            mapping[f"{prefix}self_attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
            mapping[f"{prefix}self_attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
            mapping[f"{prefix}self_attn.o_proj.weight"] = f"blk.{i}.attn_output.weight"

            mapping[f"{prefix}mlp.gate_proj.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"{prefix}mlp.up_proj.weight"] = f"blk.{i}.ffn_up.weight"
            mapping[f"{prefix}mlp.down_proj.weight"] = f"blk.{i}.ffn_down.weight"

        return mapping


class GPT2Mapping(TensorMapping):
    """Tensor mapping for GPT-2 architecture."""

    def __init__(self):
        super().__init__("gpt2", "gpt2")

    def get_tensor_map(self) -> Dict[str, str]:
        return {
            "wte.weight": "token_embd.weight",
            "lm_head.weight": "output.weight",
            "ln_f.weight": "output_norm.weight",
        }

    def get_block_prefix(self) -> str:
        return "h."

    def has_rope(self) -> bool:
        return False

    def has_position_embeddings(self) -> bool:
        return True

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        mapping = {}
        for i in range(n_layers):
            prefix = f"h.{i}."

            mapping[f"{prefix}ln_1.weight"] = f"blk.{i}.attn_norm.weight"
            mapping[f"{prefix}ln_2.weight"] = f"blk.{i}.ffn_norm.weight"

            mapping[f"{prefix}attn.c_attn.weight"] = f"blk.{i}.attn_q.weight"
            mapping[f"{prefix}attn.c_proj.weight"] = f"blk.{i}.attn_output.weight"

            mapping[f"{prefix}mlp.c_fc.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"{prefix}mlp.c_proj.weight"] = f"blk.{i}.ffn_down.weight"

        return mapping


class OPTMapping(TensorMapping):
    """Tensor mapping for OPT (Meta) architecture."""

    def __init__(self):
        super().__init__("opt", "llama")

    def get_tensor_map(self) -> Dict[str, str]:
        return {
            "model.embed_tokens.weight": "token_embd.weight",
            "lm_head.weight": "output.weight",
            "model.decoder.final_layer_norm.weight": "output_norm.weight",
        }

    def get_block_prefix(self) -> str:
        return "model.decoder.layers."

    def has_rope(self) -> bool:
        return False

    def has_position_embeddings(self) -> bool:
        return True

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        mapping = {}
        for i in range(n_layers):
            prefix = f"model.decoder.layers.{i}."

            mapping[f"{prefix}self_attn.layer_norm.weight"] = f"blk.{i}.attn_norm.weight"
            mapping[f"{prefix}final_layer_norm.weight"] = f"blk.{i}.ffn_norm.weight"

            mapping[f"{prefix}self_attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
            mapping[f"{prefix}self_attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
            mapping[f"{prefix}self_attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
            mapping[f"{prefix}self_attn.out_proj.weight"] = f"blk.{i}.attn_output.weight"

            mapping[f"{prefix}fc1.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"{prefix}fc2.weight"] = f"blk.{i}.ffn_down.weight"

        return mapping


class FalconMapping(TensorMapping):
    """Tensor mapping for Falcon architecture."""

    def __init__(self):
        super().__init__("falcon", "llama")

    def get_tensor_map(self) -> Dict[str, str]:
        return {
            "transformer.word_embeddings.weight": "token_embd.weight",
            "lm_head.weight": "output.weight",
            "transformer.ln_f.weight": "output_norm.weight",
        }

    def get_block_prefix(self) -> str:
        return "transformer.h."

    def has_rope(self) -> bool:
        return False

    def has_position_embeddings(self) -> bool:
        return True

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        mapping = {}
        for i in range(n_layers):
            prefix = f"transformer.h.{i}."

            mapping[f"{prefix}ln_attn.weight"] = f"blk.{i}.attn_norm.weight"
            mapping[f"{prefix}ln_mlp.weight"] = f"blk.{i}.ffn_norm.weight"

            mapping[f"{prefix}self_attention.query_key_value.weight"] = f"blk.{i}.attn_q.weight"
            mapping[f"{prefix}self_attention.dense.weight"] = f"blk.{i}.attn_output.weight"

            mapping[f"{prefix}mlp.dense_h_to_4h.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"{prefix}mlp.dense_4h_to_h.weight"] = f"blk.{i}.ffn_down.weight"

        return mapping

    def get_fused_qkv_keys(self) -> List[Tuple[str, str]]:
        """Return (source_key_pattern, architecture_name) for fused QKV splits.

        Falcon uses fused query_key_value.weight that must be split into Q/K/V.
        """
        return [("query_key_value.weight", "falcon")]


class GPTNeoXMapping(TensorMapping):
    """Tensor mapping for GPT-NeoX architecture."""

    def __init__(self):
        super().__init__("gpt_neox", "llama")

    def get_tensor_map(self) -> Dict[str, str]:
        return {
            "embed_in.weight": "token_embd.weight",
            "embed_out.weight": "output.weight",
            "final_layer_norm.weight": "output_norm.weight",
        }

    def get_block_prefix(self) -> str:
        return "layers."

    def has_rope(self) -> bool:
        return False

    def has_position_embeddings(self) -> bool:
        return True

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        mapping = {}
        for i in range(n_layers):
            prefix = f"layers.{i}."

            mapping[f"{prefix}input_layer_norm.weight"] = f"blk.{i}.attn_norm.weight"
            mapping[f"{prefix}post_attention_layer_norm.weight"] = f"blk.{i}.ffn_norm.weight"

            mapping[f"{prefix}attention.query_key_value.weight"] = f"blk.{i}.attn_q.weight"
            mapping[f"{prefix}attention.dense.weight"] = f"blk.{i}.attn_output.weight"

            mapping[f"{prefix}mlp.dense_h_to_4h.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"{prefix}mlp.dense_4h_to_h.weight"] = f"blk.{i}.ffn_down.weight"

        return mapping

    def get_fused_qkv_keys(self) -> List[Tuple[str, str]]:
        """GPT-NeoX uses fused query_key_value that must be split into Q/K/V."""
        return [("query_key_value.weight", "gpt_neox")]


class BloomMapping(TensorMapping):
    """Tensor mapping for Bloom architecture (BigScience)."""

    def __init__(self):
        super().__init__("bloom", "llama")

    def get_tensor_map(self) -> Dict[str, str]:
        return {
            "word_embeddings.weight": "token_embd.weight",
            "lm_head.weight": "output.weight",
            "ln_f.weight": "output_norm.weight",
        }

    def get_block_prefix(self) -> str:
        return "h."

    def has_rope(self) -> bool:
        return False

    def has_position_embeddings(self) -> bool:
        return True

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        mapping = {}
        for i in range(n_layers):
            prefix = f"h.{i}."

            mapping[f"{prefix}ln_1.weight"] = f"blk.{i}.attn_norm.weight"
            mapping[f"{prefix}ln_2.weight"] = f"blk.{i}.ffn_norm.weight"

            mapping[f"{prefix}self_attention.query_key_value.weight"] = f"blk.{i}.attn_q.weight"
            mapping[f"{prefix}self_attention.dense.weight"] = f"blk.{i}.attn_output.weight"

            mapping[f"{prefix}mlp.dense_h_to_4h.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"{prefix}mlp.dense_4h_to_h.weight"] = f"blk.{i}.ffn_down.weight"

        return mapping

    def get_fused_qkv_keys(self) -> List[Tuple[str, str]]:
        """Bloom uses fused query_key_value that must be split into Q/K/V."""
        return [("query_key_value.weight", "bloom")]


ARCHITECTURE_MAPPINGS: Dict[str, TensorMapping] = {
    "sloughgpt": SloughGPTMapping(),
    "llama": LLaMAMapping(),
    "mistral": MistralMapping(),
    "gpt2": GPT2Mapping(),
    "opt": OPTMapping(),
    "falcon": FalconMapping(),
    "gpt_neox": GPTNeoXMapping(),
    "bloom": BloomMapping(),
}


class PhiMapping(TensorMapping):
    """Tensor mapping for Phi series (Microsoft)."""

    def __init__(self):
        super().__init__("phi", "llama")

    def get_tensor_map(self) -> Dict[str, str]:
        return {
            "model.embed_tokens.weight": "token_embd.weight",
            "lm_head.weight": "output.weight",
            "model.final_layernorm.weight": "output_norm.weight",
        }

    def get_block_prefix(self) -> str:
        return "model.h."

    def has_rope(self) -> bool:
        return True

    def has_position_embeddings(self) -> bool:
        return False

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        mapping = {}
        for i in range(n_layers):
            prefix = f"model.h.{i}."

            mapping[f"{prefix}ln_1.weight"] = f"blk.{i}.attn_norm.weight"
            mapping[f"{prefix}ln_2.weight"] = f"blk.{i}.ffn_norm.weight"

            mapping[f"{prefix}attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
            mapping[f"{prefix}attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
            mapping[f"{prefix}attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
            mapping[f"{prefix}attn.dense.weight"] = f"blk.{i}.attn_output.weight"

            mapping[f"{prefix}mlp.gate_proj.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"{prefix}mlp.up_proj.weight"] = f"blk.{i}.ffn_up.weight"
            mapping[f"{prefix}mlp.down_proj.weight"] = f"blk.{i}.ffn_down.weight"

        return mapping


class GemmaMapping(TensorMapping):
    """Tensor mapping for Gemma (Google)."""

    def __init__(self):
        super().__init__("gemma", "gemma")

    def get_tensor_map(self) -> Dict[str, str]:
        return {
            "model.embed_tokens.weight": "token_embd.weight",
            "lm_head.weight": "output.weight",
            "model.norm.weight": "output_norm.weight",
        }

    def get_block_prefix(self) -> str:
        return "model.layers."

    def has_rope(self) -> bool:
        return True

    def has_position_embeddings(self) -> bool:
        return False

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        mapping = {}
        for i in range(n_layers):
            prefix = f"model.layers.{i}."

            mapping[f"{prefix}input_layernorm.weight"] = f"blk.{i}.attn_norm.weight"
            mapping[f"{prefix}post_attention_layernorm.weight"] = f"blk.{i}.ffn_norm.weight"

            mapping[f"{prefix}self_attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
            mapping[f"{prefix}self_attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
            mapping[f"{prefix}self_attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
            mapping[f"{prefix}self_attn.o_proj.weight"] = f"blk.{i}.attn_output.weight"

            mapping[f"{prefix}mlp.gate_proj.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"{prefix}mlp.up_proj.weight"] = f"blk.{i}.ffn_up.weight"
            mapping[f"{prefix}mlp.down_proj.weight"] = f"blk.{i}.ffn_down.weight"

        return mapping


class QwenMapping(TensorMapping):
    """Tensor mapping for Qwen (Alibaba)."""

    def __init__(self):
        super().__init__("qwen", "qwen")

    def get_tensor_map(self) -> Dict[str, str]:
        return {
            "transformer.wte.weight": "token_embd.weight",
            "lm_head.weight": "output.weight",
            "transformer.ln_f.weight": "output_norm.weight",
        }

    def get_block_prefix(self) -> str:
        return "transformer.h."

    def has_rope(self) -> bool:
        return True

    def has_position_embeddings(self) -> bool:
        return False

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        mapping = {}
        for i in range(n_layers):
            prefix = f"transformer.h.{i}."

            mapping[f"{prefix}ln_1.weight"] = f"blk.{i}.attn_norm.weight"
            mapping[f"{prefix}ln_2.weight"] = f"blk.{i}.ffn_norm.weight"

            mapping[f"{prefix}attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
            mapping[f"{prefix}attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
            mapping[f"{prefix}attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
            mapping[f"{prefix}attn.c_proj.weight"] = f"blk.{i}.attn_output.weight"

            mapping[f"{prefix}mlp.w1.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"{prefix}mlp.w2.weight"] = f"blk.{i}.ffn_down.weight"

        return mapping


class DeepseekMapping(TensorMapping):
    """Tensor mapping for Deepseek (Deepseek AI)."""

    def __init__(self):
        super().__init__("deepseek", "llama")

    def get_tensor_map(self) -> Dict[str, str]:
        return {
            "model.embed_tokens.weight": "token_embd.weight",
            "lm_head.weight": "output.weight",
            "model.norm.weight": "output_norm.weight",
        }

    def get_block_prefix(self) -> str:
        return "model.layers."

    def has_rope(self) -> bool:
        return True

    def has_position_embeddings(self) -> bool:
        return False

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        mapping = {}
        for i in range(n_layers):
            prefix = f"model.layers.{i}."

            mapping[f"{prefix}input_layernorm.weight"] = f"blk.{i}.attn_norm.weight"
            mapping[f"{prefix}post_attention_layernorm.weight"] = f"blk.{i}.ffn_norm.weight"

            mapping[f"{prefix}self_attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
            mapping[f"{prefix}self_attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
            mapping[f"{prefix}self_attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
            mapping[f"{prefix}self_attn.o_proj.weight"] = f"blk.{i}.attn_output.weight"

            mapping[f"{prefix}mlp.gate_proj.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"{prefix}mlp.up_proj.weight"] = f"blk.{i}.ffn_up.weight"
            mapping[f"{prefix}mlp.down_proj.weight"] = f"blk.{i}.ffn_down.weight"

        return mapping


class YiMapping(TensorMapping):
    """Tensor mapping for Yi (01.AI)."""

    def __init__(self):
        super().__init__("yi", "llama")

    def get_tensor_map(self) -> Dict[str, str]:
        return {
            "model.embed_tokens.weight": "token_embd.weight",
            "lm_head.weight": "output.weight",
            "model.norm.weight": "output_norm.weight",
        }

    def get_block_prefix(self) -> str:
        return "model.layers."

    def has_rope(self) -> bool:
        return True

    def has_position_embeddings(self) -> bool:
        return False

    def get_block_mapping(self, n_layers: int = 100) -> Dict[str, str]:
        mapping = {}
        for i in range(n_layers):
            prefix = f"model.layers.{i}."

            mapping[f"{prefix}input_layernorm.weight"] = f"blk.{i}.attn_norm.weight"
            mapping[f"{prefix}post_attention_layernorm.weight"] = f"blk.{i}.ffn_norm.weight"

            mapping[f"{prefix}self_attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
            mapping[f"{prefix}self_attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
            mapping[f"{prefix}self_attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
            mapping[f"{prefix}self_attn.o_proj.weight"] = f"blk.{i}.attn_output.weight"

            mapping[f"{prefix}mlp.gate_proj.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"{prefix}mlp.up_proj.weight"] = f"blk.{i}.ffn_up.weight"
            mapping[f"{prefix}mlp.down_proj.weight"] = f"blk.{i}.ffn_down.weight"

        return mapping


ARCHITECTURE_MAPPINGS.update({
    "phi": PhiMapping(),
    "gemma": GemmaMapping(),
    "qwen": QwenMapping(),
    "starcoder": DeepseekMapping(),  # Starcoder uses same structure
    "deepseek": DeepseekMapping(),
    "yi": YiMapping(),
})


def _as_float16(tensor) -> Optional[np.ndarray]:
    """Convert a tensor (numpy / SloNet / torch) to a float16 numpy array.

    Returns None for values that cannot be converted.
    """
    if isinstance(tensor, np.ndarray):
        if tensor.dtype == object:
            return None
        return tensor.astype(np.float16)
    data = getattr(tensor, "data", None)  # SloNet Tensor
    if isinstance(data, np.ndarray):
        return data.astype(np.float16)
    if hasattr(tensor, "detach") and hasattr(tensor, "numpy"):  # torch Tensor
        return tensor.detach().cpu().numpy().astype(np.float16)
    try:
        arr = np.asarray(tensor)
    except Exception:
        return None
    if arr.dtype == object:
        return None
    return arr.astype(np.float16)


def detect_architecture(state_dict: StateDict) -> Optional[TensorMapping]:
    """Auto-detect model architecture from tensor names."""
    keys = list(state_dict.keys())

    patterns = {
        "sloughgpt": ["tok_emb.weight", "norm.weight", "mlp.w1.weight"],
        "llama": ["model.embed_tokens.weight", "model.norm.weight", "model.layers."],
        "mistral": ["model.embed_tokens.weight", "input_layernorm.weight"],
        "gpt2": ["wte.weight", "ln_f.weight", "h."],
        "opt": ["model.embed_tokens.weight", "model.decoder."],
        "falcon": ["transformer.word_embeddings.weight", "transformer.h."],
        "gpt_neox": ["embed_in.weight", "final_layer_norm.weight", "layers."],
        "bloom": ["word_embeddings.weight", "ln_f.weight", "h."],
    }

    scores = {}
    for arch, arch_patterns in patterns.items():
        scores[arch] = sum(1 for p in arch_patterns if any(p in k for k in keys))

    max_score = max(scores.values())
    if max_score == 0:
        logger.warning("Could not detect architecture, defaulting to SloughGPT",
            extra={"tag": "TRAIN"},)
        return SloughGPTMapping()

    detected = max(scores, key=scores.get)
    logger.info("Detected architecture: %s (score: %s)", detected, max_score,
        extra={"tag": "TRAIN"},)
    return ARCHITECTURE_MAPPINGS.get(detected, SloughGPTMapping())


def register_architecture(name: str, mapping: TensorMapping) -> None:
    """Register a custom architecture mapping."""
    ARCHITECTURE_MAPPINGS[name] = mapping
    logger.info("Registered custom architecture: %s", name,
        extra={"tag": "TRAIN"},)


def get_tensor_mapping(model: ModelLike) -> Dict[str, str]:
    """Get GGUF tensor name mapping for the model architecture."""
    state_dict = model.state_dict()
    mapping = detect_architecture(state_dict)

    if mapping is None:  # pragma: no cover (detect_architecture never returns None)
        mapping = SloughGPTMapping()

    tensor_map = mapping.get_tensor_map()
    n_layers = count_layers(state_dict, mapping.get_block_prefix())
    tensor_map.update(mapping.get_block_mapping(n_layers))

    return tensor_map


def count_layers(state_dict: StateDict, block_prefix: str) -> int:
    """Count the number of transformer blocks."""
    n_layer = 0
    for key in state_dict.keys():
        if block_prefix in key:
            if "layers" in block_prefix:
                parts = key.split(block_prefix)
                if len(parts) > 1:
                    layer_num = int(parts[1].split(".")[0])
                    n_layer = max(n_layer, layer_num + 1)
            else:
                parts = key.split(block_prefix)
                if len(parts) > 1:
                    layer_num = int(parts[1].split(".")[0])
                    n_layer = max(n_layer, layer_num + 1)
    return n_layer


def get_block_mapping(model: ModelLike = None, n_layers: int = 100) -> Dict[str, str]:
    """Get GGUF tensor name mapping for transformer blocks."""
    if model is not None:
        state_dict = model.state_dict()
        mapping = detect_architecture(state_dict)
        if mapping:
            n_layers = count_layers(state_dict, mapping.get_block_prefix())
            return mapping.get_block_mapping(n_layers)

    mapping = SloughGPTMapping()
    return mapping.get_block_mapping(n_layers)


def export_to_gguf(
    model: ModelLike,
    output_path: str,
    tokenizer: Optional[Any] = None,
    config: Optional[GGUFExportConfig] = None,
) -> str:
    """Export model to GGUF format for llama.rn.

    Supports multiple architectures:
    - SloughGPTModel (default)
    - LLaMA
    - Mistral
    - Custom architectures via register_architecture()

    Args:
        model: The model to export
        output_path: Output GGUF file path
        tokenizer: Optional tokenizer for vocab
        config: Export configuration

    Returns:
        Path to exported GGUF file
    """
    if config is None:
        config = GGUFExportConfig()

    try:
        from gguf import GGUFWriter
    except ImportError:
        raise ImportError("gguf not installed. Run: pip install gguf")

    model.eval()
    state_dict = model.state_dict()

    if config.architecture:
        if config.architecture not in ARCHITECTURE_MAPPINGS:
            raise ValueError(f"Unknown architecture: {config.architecture}. Available: {list(ARCHITECTURE_MAPPINGS.keys())}")
        mapping = ARCHITECTURE_MAPPINGS[config.architecture]
    else:
        mapping = detect_architecture(state_dict) or SloughGPTMapping()

    n_layer = count_layers(state_dict, mapping.get_block_prefix())

    config_dict = model._config if getattr(model, "_config", None) else {}

    vocab_size = config_dict.get("vocab_size", getattr(model, "vocab_size", 256))
    n_embed = config_dict.get("n_embed", getattr(model, "n_embed", 256))
    n_head = config_dict.get("n_head", getattr(model, "n_head", 8))
    n_kv_head = config_dict.get("n_kv_head", getattr(model, "n_kv_head", n_head))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = GGUFWriter(output_path, mapping.gguf_type)

    writer.add_name(config.model_name)
    writer.add_description(f"{mapping.name} exported with {config.quantization}")

    writer.add_vocab_size(vocab_size)
    writer.add_context_length(config.n_ctx)
    writer.add_embedding_length(n_embed)
    writer.add_block_count(n_layer)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_kv_head)

    hidden_dim = int(n_embed * 8 // 3)
    hidden_dim = ((hidden_dim + 63) // 64) * 64
    writer.add_feed_forward_length(hidden_dim)
    writer.add_rope_freq_base(config.rope_freq_base)

    writer.add_tokenizer_model(mapping.gguf_type)

    if tokenizer is not None:
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
            token_list = [tokenizer.decode([i]) if hasattr(tokenizer, 'decode') else chr(i) for i in range(len(vocab))]
        else:
            token_list = [chr(i) if i < 256 else f"<0x{i:02X}>" for i in range(vocab_size)]
    else:
        token_list = [chr(i) if i < 256 else f"<0x{i:02X}>" for i in range(vocab_size)]

    writer.add_token_list(token_list)

    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    writer.add_add_space_prefix(False)

    tensor_map = mapping.get_tensor_map()
    block_map = mapping.get_block_mapping(n_layer)
    tensor_map.update(block_map)

    fused_suffixes = mapping.get_fused_qkv_keys()
    fused_set = set()
    for suffix, _arch in fused_suffixes:
        for key in list(tensor_map.keys()):
            if key.endswith(suffix):
                fused_set.add(key)

    for key, tensor in state_dict.items():
        tensor_np = _as_float16(tensor)
        if tensor_np is None:
            continue

        if key in fused_set:
            qkv = tensor_np
            dim0 = qkv.shape[0]
            third = dim0 // 3
            q_part = qkv[:third]
            k_part = qkv[third:2 * third]
            v_part = qkv[2 * third:]

            q_target = tensor_map[key]
            k_target = q_target.replace(".attn_q.", ".attn_k.")
            v_target = q_target.replace(".attn_q.", ".attn_v.")

            writer.add_tensor(q_target, q_part)
            writer.add_tensor(k_target, k_part)
            writer.add_tensor(v_target, v_part)
        else:
            mapped_key = tensor_map.get(key, key)
            writer.add_tensor(mapped_key, tensor_np)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.flush()

    logger.info("Exported GGUF (%s): %s", mapping.name, output_path,
        extra={"tag": "TRAIN"},)
    return output_path


def export_to_gguf_fp16(model, output_path, tokenizer=None, architecture=None):
    """Export to GGUF FP16 (no quantization)."""
    return export_to_gguf(
        model, output_path, tokenizer,
        GGUFExportConfig(quantization="F16", architecture=architecture)
    )


def export_to_gguf_q4_k_m(model, output_path, tokenizer=None, architecture=None):
    """Export to GGUF Q4_K_M (recommended for mobile)."""
    return export_to_gguf(
        model, output_path, tokenizer,
        GGUFExportConfig(quantization="Q4_K_M", architecture=architecture)
    )


def quantize_gguf(input_path: str, output_path: str, quantization: str = "Q4_K_M") -> str:
    """Quantize GGUF file using llama.cpp."""
    try:
        import subprocess
        import shutil

        llama_quantize = shutil.which("llama-quantize")
        if llama_quantize is None:
            logger.warning("llama-quantize not found. Install llama.cpp",
                extra={"tag": "TRAIN"},)
            return input_path

        result = subprocess.run(
            [llama_quantize, input_path, output_path, quantization],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("Quantized: %s -> %s", input_path, output_path,
                extra={"tag": "TRAIN"},)
            return output_path
        else:
            logger.error("Quantization failed: %s", result.stderr,
                extra={"tag": "TRAIN"},)
            return input_path
    except Exception as e:
        logger.error("Quantization error: %s", e,
            extra={"tag": "TRAIN"},)
        return input_path


def get_model_info_gguf(gguf_path: str) -> Dict[str, Any]:
    """Get metadata from GGUF file."""
    try:
        from gguf import GGUFReader
        reader = GGUFReader(gguf_path)
        return {
            "path": gguf_path,
            "file_size_mb": round(Path(gguf_path).stat().st_size / (1024 * 1024), 2),
        }
    except Exception:
        return {}


def estimate_memory_requirements(
    vocab_size: int,
    n_layer: int,
    n_embed: int,
    n_ctx: int,
    quantization: str = "Q4_K_M"
) -> Dict[str, float]:
    """Estimate memory requirements for GGUF model."""
    bytes_per_param = {
        "F32": 4.0, "F16": 2.0, "Q8_0": 1.0, "Q5_K_M": 0.7, "Q5_1": 0.6,
        "Q5_0": 0.5, "Q4_1": 0.5, "Q4_0": 0.4, "Q4_K_M": 0.45, "Q4_K_S": 0.4,
        "Q3_K_M": 0.35, "Q3_K_S": 0.3, "Q2_K": 0.25,
    }
    bpp = bytes_per_param.get(quantization, 0.45)
    params = vocab_size * n_embed + n_layer * (4 * n_embed * n_embed) + n_embed * vocab_size
    model_mem = params * bpp / (1024 * 1024)
    kv_mem = 2 * n_layer * n_embed * 2 * n_ctx / (1024 * 1024)
    return {
        "model_mb": round(model_mem, 2),
        "kv_cache_mb": round(kv_mem, 2),
        "total_mb": round(model_mem + kv_mem, 2)
    }


def list_available_quantizations() -> List[Tuple[str, str, bool]]:
    """List available GGUF quantization types."""
    return [(n, d, n in MOBILE_RECOMMENDED) for n, d in QUANTIZATION_TYPES.items()]


def list_supported_architectures() -> List[str]:
    """List supported model architectures."""
    return list(ARCHITECTURE_MAPPINGS.keys())


__all__ = [
    "GGUFExportConfig",
    "TensorMapping",
    "SloughGPTMapping",
    "LLaMAMapping",
    "MistralMapping",
    "QUANTIZATION_TYPES",
    "MOBILE_RECOMMENDED",
    "ARCHITECTURE_MAPPINGS",
    "detect_architecture",
    "register_architecture",
    "export_to_gguf",
    "export_to_gguf_fp16",
    "export_to_gguf_q4_k_m",
    "quantize_gguf",
    "get_model_info_gguf",
    "estimate_memory_requirements",
    "list_available_quantizations",
    "list_supported_architectures",
]
