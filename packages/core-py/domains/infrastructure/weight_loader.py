"""
Unified weight loading infrastructure.

Data flow:
    file → parse() → state_dict + config
                    ↓
           build_load_plan() → LoadPlan (arch detect, O(1) mapping)
                    ↓
           load_into_model() → mmap → transpose → parameter buffer

Format dispatch:
    WeightLoaderRegistry自动检测文件格式并路由到正确的加载器。
    添加新格式只需 register_loader(suffix, loader_class)。

Architecture:
    LoadPlan / TensorMapping — pre-computed data structures
    build_load_plan()         — shared plan builder (used by all formats)
    load_into_model()         — generic loader (reads from dict, writes to params)
    DirectWeightLoader        — SLNC-specific: mmap→param single-pass
    WeightLoaderRegistry      — format auto-detection + dispatch
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import numpy as np

logger = logging.getLogger("slo.infrastructure.weight_loader")


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TensorMapping:
    """Pre-computed mapping for a single file tensor → model parameter."""

    param_name: str
    needs_transpose: bool
    canonical: str


@dataclass
class LoadPlan:
    """Pre-computed loading plan. Built once, used for O(1) per-tensor lookups."""

    tensor_map: Dict[str, TensorMapping]
    tied_weights: List[Tuple[str, str]]
    synthesized_params: List[Tuple[str, str, str]]
    fused_qkv: Dict[str, List[str]]
    n_layer: int
    n_embed: int
    arch_name: str


@dataclass
class WeightLoadResult:
    """Standardized result from weight loading."""

    success: bool
    n_written: int = 0
    n_fused: int = 0
    timing: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


# ── Plan Builder (shared by all formats) ────────────────────────────────────

# Canonical → SloTransformer mapping (single source of truth)
_ARCH_TO_SLONET: Dict[str, Optional[str]] = {
    "embed.token": "tok_emb.weight",
    "embed.pos": "pos_emb.weight",
    "layers.{i}.attn_norm.weight": "blocks.{i}.attn_norm.weight",
    "layers.{i}.attn_norm.bias": None,
    "layers.{i}.q.weight": "blocks.{i}.attn.q_proj.weight",
    "layers.{i}.q.bias": "blocks.{i}.attn.q_proj.bias",
    "layers.{i}.k.weight": "blocks.{i}.attn.k_proj.weight",
    "layers.{i}.k.bias": "blocks.{i}.attn.k_proj.bias",
    "layers.{i}.v.weight": "blocks.{i}.attn.v_proj.weight",
    "layers.{i}.v.bias": "blocks.{i}.attn.v_proj.bias",
    "layers.{i}.qkv.weight": None,
    "layers.{i}.qkv.bias": None,
    "layers.{i}.o_proj.weight": "blocks.{i}.attn.o_proj.weight",
    "layers.{i}.o_proj.bias": "blocks.{i}.attn.o_proj.bias",
    "layers.{i}.ff_norm.weight": "blocks.{i}.ff_norm.weight",
    "layers.{i}.ff_norm.bias": None,
    "layers.{i}.ffn.down.weight": "blocks.{i}.ff.w2.weight",
    "layers.{i}.ffn.down.bias": "blocks.{i}.ff.w2.bias",
    "final_norm.weight": "norm.weight",
    "final_norm.bias": None,
}

_SWIGLU_MAP: Dict[str, Optional[str]] = {
    "layers.{i}.ffn.gate.weight": "blocks.{i}.ff.w1.weight",
    "layers.{i}.ffn.gate.bias": "blocks.{i}.ff.w1.bias",
    "layers.{i}.ffn.up.weight": "blocks.{i}.ff.w3.weight",
    "layers.{i}.ffn.up.bias": "blocks.{i}.ff.w3.bias",
}

_GELU_MAP: Dict[str, Optional[str]] = {
    "layers.{i}.ffn.up.weight": "blocks.{i}.ff.w1.weight",
    "layers.{i}.ffn.up.bias": "blocks.{i}.ff.w1.bias",
}

_NORM_BIAS_MAP: Dict[str, Optional[str]] = {
    "layers.{i}.attn_norm.bias": "blocks.{i}.attn_norm.bias",
    "layers.{i}.ff_norm.bias": "blocks.{i}.ff_norm.bias",
    "final_norm.bias": "norm.bias",
}

_NO_TRANSPOSE: Set[str] = {"embed.token", "embed.pos", "lm_head"}


def build_load_plan(
    state_dict: Dict[str, np.ndarray],
    n_layer: int,
    config: dict,
) -> LoadPlan:
    """Pre-compute loading plan from state dict + config.

    Detects architecture, maps file tensors → model parameters, identifies
    tied/synthesized/fused-QKV params. Single source of truth for all formats.
    """
    _t0 = time.monotonic()

    from domains.infrastructure.arch_config import build_arch
    arch = build_arch(
        name=config.get("architectures", ["unknown"])[0],
        config=config,
        weight_keys=set(state_dict.keys()),
    )

    W = arch.weight_map

    # Build canonical → slo_target mapping
    slo_map = dict(_ARCH_TO_SLONET)
    slo_map.update(_SWIGLU_MAP if arch.activation == "swiglu" else _GELU_MAP)
    if arch.norm == "layer_norm":
        slo_map.update(_NORM_BIAS_MAP)

    # Build file_tensor → TensorMapping
    tensor_map: Dict[str, TensorMapping] = {}
    for canonical, slo_target in slo_map.items():
        if slo_target is None:
            continue
        mapped_hf_key = W.get(canonical)
        if mapped_hf_key is None:
            continue
        do_transpose = arch.transpose_weights and canonical not in _NO_TRANSPOSE
        if "{i}" in mapped_hf_key:
            for i in range(n_layer):
                concrete = mapped_hf_key.replace("{i}", str(i))
                slo_key = slo_target.replace("{i}", str(i))
                if concrete in state_dict:
                    arr = state_dict[concrete]
                    tensor_map[concrete] = TensorMapping(
                        param_name=slo_key,
                        needs_transpose=do_transpose and arr.ndim == 2,
                        canonical=canonical,
                    )
        else:
            if mapped_hf_key in state_dict:
                arr = state_dict[mapped_hf_key]
                tensor_map[mapped_hf_key] = TensorMapping(
                    param_name=slo_target,
                    needs_transpose=do_transpose and arr.ndim == 2,
                    canonical=canonical,
                )

    # Fused QKV
    fused_qkv: Dict[str, List[str]] = {}
    for canonical in ["layers.{i}.qkv.weight", "layers.{i}.qkv.bias"]:
        mapped = W.get(canonical, "")
        if not mapped:
            continue
        suffix = "bias" if "bias" in canonical else "weight"
        global_key = mapped.replace("{i}", "")
        if global_key in state_dict:
            fused_qkv[global_key] = [
                f"blocks.{i}.attn.{p}.{suffix}"
                for i in range(n_layer)
                for p in ["q_proj", "k_proj", "v_proj"]
            ]
        for i in range(n_layer):
            concrete = mapped.replace("{i}", str(i))
            if concrete in state_dict:
                fused_qkv[concrete] = [
                    f"blocks.{i}.attn.q_proj.{suffix}",
                    f"blocks.{i}.attn.k_proj.{suffix}",
                    f"blocks.{i}.attn.v_proj.{suffix}",
                ]

    # Tied weights
    loaded_params = {tm.param_name for tm in tensor_map.values()}
    tied: List[Tuple[str, str]] = []
    if "lm_head.weight" not in loaded_params and "tok_emb.weight" in loaded_params:
        tied.append(("lm_head.weight", "tok_emb.weight"))

    # Synthesized params (GELU w3)
    synthesized: List[Tuple[str, str, str]] = []
    if arch.activation != "swiglu":
        for i in range(n_layer):
            w1_key = f"blocks.{i}.ff.w1.weight"
            w3_key = f"blocks.{i}.ff.w3.weight"
            w3_bias_key = f"blocks.{i}.ff.w3.bias"
            if w1_key in loaded_params:
                synthesized.append((w3_key, "0", w1_key))
                synthesized.append((w3_bias_key, "1", w1_key))

    plan = LoadPlan(
        tensor_map=tensor_map,
        tied_weights=tied,
        synthesized_params=synthesized,
        fused_qkv=fused_qkv,
        n_layer=n_layer,
        n_embed=arch.n_embed,
        arch_name=arch.name,
    )

    logger.info(
        "build_load_plan: arch=%s mapped=%d fused_qkv=%d tied=%d synth=%d (%.3fs)",
        arch.name, len(tensor_map), len(fused_qkv), len(tied),
        len(synthesized), time.monotonic() - _t0,
        extra={"tag": "INFRA"},
    )
    return plan


# ── Architecture Inference ───────────────────────────────────────────────────

def infer_arch_from_state_dict(state_dict: Dict[str, np.ndarray]) -> dict:
    """Infer model architecture from a state dict's tensor shapes.

    Centralizes the duplicated arch detection logic found in
    routers/souls.py, models/provider.py, and controllers/models.py.

    Args:
        state_dict: Dict mapping param names → numpy arrays

    Returns:
        dict with keys: vocab_size, n_embed, n_layer, n_head, intermediate_size
    """
    result = {
        "vocab_size": 256,
        "n_embed": 128,
        "n_layer": 1,
        "n_head": 8,
        "intermediate_size": 512,
    }

    # vocab_size and n_embed from tok_emb
    tok_emb = state_dict.get("tok_emb.weight")
    if tok_emb is not None and tok_emb.ndim == 2:
        result["vocab_size"] = tok_emb.shape[0]
        result["n_embed"] = tok_emb.shape[1]

    # n_layer from max block index
    n_layer = 1
    for key in state_dict:
        if key.startswith("blocks.") and ".attn_norm.weight" in key:
            try:
                idx = int(key.split(".")[1])
                n_layer = max(n_layer, idx + 1)
            except (ValueError, IndexError):
                pass
    result["n_layer"] = n_layer

    # n_head from q_proj shape
    n_embed = result["n_embed"]
    q_w = state_dict.get("blocks.0.attn.q_proj.weight")
    if q_w is None:
        q_w = state_dict.get("blocks.0.q_proj.weight")
    if q_w is not None and q_w.ndim == 2:
        head_dim = n_embed // 8
        if head_dim > 0:
            detected = q_w.shape[0] // head_dim
            if detected >= 1:
                result["n_head"] = detected

    # intermediate_size from w1/gate_proj shape
    for key in state_dict:
        if "mlp.w1.weight" in key or "mlp.gate_proj.weight" in key:
            shape = state_dict[key].shape
            if len(shape) >= 2:
                result["intermediate_size"] = shape[0]
            break

    return result


def build_model_from_config(config: dict, _lazy: bool = True):
    """Construct a SloTransformer from a config dict (SLNC or .soul metadata).

    Auto-detects: RoPE vs absolute pos, RMSNorm vs LayerNorm, SwiGLU vs GELU,
    GQA (n_kv_head < n_head), eps, rope_base.

    Args:
        config: Dict with keys like hidden_size, num_hidden_layers, etc.
        _lazy: If True, skip weight initialization (for loading pre-trained weights)

    Returns:
        SloTransformer instance (uninitialized if _lazy=True)
    """
    from domains.training.slonet import SloTransformer

    n_embed = config.get("n_embd", config.get("hidden_size", 768))
    n_head = config.get("n_head", config.get("num_attention_heads", 12))
    n_layer = config.get("n_layer", config.get("num_hidden_layers", 12))
    vocab_size = config.get("vocab_size", 50257)
    intermediate_size = config.get("n_inner") or config.get("intermediate_size", n_embed * 4)
    max_pos = config.get("n_positions", config.get("max_position_embeddings", 1024))

    # Auto-detect positional encoding
    has_rope = config.get("rope_theta") is not None or config.get("position_embedding_type") == "rope"
    use_abs_pos = not has_rope

    # Auto-detect norm type
    has_rms = config.get("rms_norm_eps") is not None
    norm_type = "rms_norm" if has_rms else "layer_norm"
    if config.get("layer_norm_type"):
        norm_type = config["layer_norm_type"]

    # Auto-detect GQA
    n_kv_head = config.get("num_key_value_heads", n_head)

    # Auto-detect activation
    hidden_act = config.get("hidden_act", "gelu")
    activation = "silu" if hidden_act == "silu" else "gelu"

    hf_eps = config.get("rms_norm_eps", 1e-5)

    return SloTransformer(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        intermediate_size=intermediate_size,
        block_size=max_pos,
        max_seq_len=max_pos,
        use_rope=not use_abs_pos,
        rope_base=config.get("rope_theta", 10000.0),
        dropout=0.0,
        eps=hf_eps,
        tie_weights=True,
        use_abs_pos_emb=use_abs_pos,
        norm_type=norm_type,
        activation=activation,
        _lazy=_lazy,
    )


# ── Generic Loader ───────────────────────────────────────────────────────────

def load_into_model(
    model,
    plan: LoadPlan,
    tensor_data: Dict[str, np.ndarray],
) -> WeightLoadResult:
    """Load pre-read tensor data into model parameters using a LoadPlan.

    This is format-agnostic: any format that provides a state_dict can use this.

    Args:
        model: SloTransformer (constructed, _lazy=True)
        plan: Pre-computed load plan
        tensor_data: Dict mapping file tensor names → numpy arrays

    Returns:
        WeightLoadResult with timing and counts
    """
    _t0 = time.monotonic()
    param_map = dict(model._named_parameters())

    # Direct writes
    n_written = 0
    for file_name, mapping in plan.tensor_map.items():
        param = param_map.get(mapping.param_name)
        if param is None:
            continue
        arr = tensor_data.get(file_name)
        if arr is None:
            continue
        if mapping.needs_transpose and arr.ndim == 2:
            param.data[:] = arr.T
        else:
            param.data[:] = arr
        n_written += 1

    _t_direct = time.monotonic()

    # Fused QKV
    n_fused = 0
    for hf_key, param_names in plan.fused_qkv.items():
        arr = tensor_data.get(hf_key)
        if arr is None:
            continue
        is_bias = arr.ndim == 1
        if is_bias:
            q, k, v = np.split(arr, 3, axis=0)
        else:
            q, k, v = np.split(arr.T, 3, axis=0)
        for pname, chunk in zip(param_names, [q, k, v]):
            p = param_map.get(pname)
            if p is not None:
                p.data[:] = chunk
                n_fused += 1

    _t_fused = time.monotonic()

    # Tied weights
    for dest, src in plan.tied_weights:
        if dest in param_map and src in param_map:
            param_map[dest].data[:] = param_map[src].data

    # Synthesized params
    for param_name, fill_val, _shape_ref in plan.synthesized_params:
        p = param_map.get(param_name)
        if p is not None:
            p.data[:] = 0 if fill_val == "0" else 1

    _t_end = time.monotonic()

    return WeightLoadResult(
        success=True,
        n_written=n_written,
        n_fused=n_fused,
        timing={
            "direct": _t_direct - _t0,
            "fused_qkv": _t_fused - _t_direct,
            "tied_synth": _t_end - _t_fused,
            "total": _t_end - _t0,
        },
    )


# ── SLNC Direct Loader ──────────────────────────────────────────────────────

class DirectWeightLoader:
    """SLNC-specific loader: mmap → parameter single-pass.

    Avoids the intermediate state_dict copy by reading from mmap
    directly into parameter buffers during the load phase.
    """

    def __init__(self, parser, state_dict: Dict[str, np.ndarray], config: dict):
        self._parser = parser
        self._state_dict = state_dict
        self._plan = build_load_plan(
            state_dict,
            config.get("n_layer", config.get("num_hidden_layers", 12)),
            config,
        )

    @classmethod
    def _from_plan(cls, parser, plan: LoadPlan, state_dict: Dict[str, np.ndarray]) -> "DirectWeightLoader":
        """Construct from a pre-built plan (avoids rebuilding it)."""
        loader = cls.__new__(cls)
        loader._parser = parser
        loader._state_dict = state_dict
        loader._plan = plan
        return loader

    @property
    def plan(self) -> LoadPlan:
        return self._plan

    def load(self, model, max_workers: Optional[int] = None) -> WeightLoadResult:
        """Load weights directly from mmap into model parameters."""
        _t0 = time.monotonic()
        plan = self._plan
        param_map = dict(model._named_parameters())
        parser = self._parser

        n_written = 0
        for file_name, mapping in plan.tensor_map.items():
            if file_name not in parser.tensor_names:
                continue
            arr = parser.read_tensor_region(file_name)
            param = param_map.get(mapping.param_name)
            if param is None:
                continue

            if mapping.needs_transpose and arr.ndim == 2:
                param.data[:] = arr.T
            else:
                param.data[:] = arr
            n_written += 1

        _t_direct = time.monotonic()

        n_fused = 0
        for hf_key, param_names in plan.fused_qkv.items():
            if hf_key not in parser.tensor_names:
                continue
            arr = parser.read_tensor_region(hf_key)

            if arr.ndim == 1:
                q, k, v = np.split(arr, 3, axis=0)
            else:
                q, k, v = np.split(arr.T, 3, axis=0)

            for pname, chunk in zip(param_names, [q, k, v]):
                p = param_map.get(pname)
                if p is not None:
                    p.data[:] = chunk
                    n_fused += 1

        _t_fused = time.monotonic()

        for dest, src in plan.tied_weights:
            if dest in param_map and src in param_map:
                param_map[dest].data[:] = param_map[src].data

        for param_name, fill_val, _ in plan.synthesized_params:
            p = param_map.get(param_name)
            if p is not None:
                p.data[:] = 0 if fill_val == "0" else 1

        _t_end = time.monotonic()

        return WeightLoadResult(
            success=True,
            n_written=n_written,
            n_fused=n_fused,
            timing={
                "direct": _t_direct - _t0,
                "fused_qkv": _t_fused - _t_direct,
                "tied_synth": _t_end - _t_fused,
                "total": _t_end - _t0,
            },
        )


# ── Soul Loader ──────────────────────────────────────────────────────────────

class SoulWeightLoader:
    """Load .soul checkpoints (native SloNet training output).

    Weights are already in SloNet format — no conversion needed.
    Just load state_dict directly into model parameters.
    """

    def __init__(self, soul_path: str, **kwargs):
        self._soul_path = soul_path

    def load_metadata(self) -> dict:
        """Load just the metadata (no weights) for model construction."""
        from domains.inference.slo_format import load_soul
        soul, _ = load_soul(self._soul_path)
        cfg = soul.metadata.get("config", {})
        return {
            "vocab_size": soul.metadata.get("vocab_size", cfg.get("vocab_size", 256)),
            "n_embed": cfg.get("n_embed", 128),
            "n_layer": cfg.get("n_layer", 4),
            "n_head": cfg.get("n_head", 4),
            "block_size": cfg.get("block_size", 128),
            "soul": soul,
        }

    def load(self, model) -> WeightLoadResult:
        from domains.inference.slo_format import load_soul

        _t0 = time.monotonic()
        soul, state_dict = load_soul(self._soul_path)
        _t_load = time.monotonic()

        model.load_state_dict(state_dict)
        _t_apply = time.monotonic()

        return WeightLoadResult(
            success=True,
            n_written=len(state_dict),
            timing={
                "load_soul": _t_load - _t0,
                "apply": _t_apply - _t_load,
                "total": _t_apply - _t0,
            },
        )


# ── Format Registry ──────────────────────────────────────────────────────────

class WeightLoaderRegistry:
    """Auto-detect file format and route to the correct loader.

    Usage:
        registry = WeightLoaderRegistry()
        registry.register_loader(".slnc", SLNCLoader)
        result = registry.load_file("model.slnc", model)
    """

    def __init__(self):
        self._loaders: Dict[str, Type] = {}
        self._default: Optional[Type] = None

    def register_loader(self, suffix: str, loader_class: Type):
        """Register a loader class for a file suffix."""
        self._loaders[suffix] = loader_class
        logger.info("Registered weight loader: %s → %s", suffix, loader_class.__name__,
                     extra={"tag": "INFRA"})

    def set_default(self, loader_class: Type):
        """Set fallback loader for unregistered suffixes."""
        self._default = loader_class

    def get_loader(self, file_path: str) -> Optional[Type]:
        """Get loader class for a file path by suffix."""
        from pathlib import Path
        suffix = Path(file_path).suffix.lower()
        return self._loaders.get(suffix) or self._default

    def load_file(self, file_path: str, model, **kwargs) -> WeightLoadResult:
        """Auto-detect format and load weights into model.

        Args:
            file_path: Path to model file
            model: SloTransformer (constructed, _lazy=True)
            **kwargs: Passed to the loader

        Returns:
            WeightLoadResult
        """
        loader_class = self.get_loader(file_path)
        if loader_class is None:
            return WeightLoadResult(
                success=False,
                error=f"No loader registered for {file_path}",
            )

        _t0 = time.monotonic()
        try:
            loader = loader_class(file_path, **kwargs)
            result = loader.load(model)
            result.timing["total"] = time.monotonic() - _t0
            return result
        except Exception as e:
            return WeightLoadResult(
                success=False,
                error=str(e),
                timing={"total": time.monotonic() - _t0},
            )


# Global registry singleton
_registry: Optional[WeightLoaderRegistry] = None


def get_weight_loader_registry() -> WeightLoaderRegistry:
    """Get or create the global weight loader registry.

    On first call, registers built-in formats:
    - .slnc → DirectWeightLoader (mmap→param single-pass)
    - .soul → SoulWeightLoader (native training checkpoints)
    """
    global _registry
    if _registry is None:
        _registry = WeightLoaderRegistry()
        _registry.register_loader(".slnc", DirectWeightLoader)
        _registry.register_loader(".soul", SoulWeightLoader)
    return _registry
