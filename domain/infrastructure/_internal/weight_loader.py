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

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

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

    tensor_map: dict[str, TensorMapping]
    tied_weights: list[tuple[str, str]]
    synthesized_params: list[tuple[str, str, str]]
    fused_qkv: dict[str, list[str]]
    n_layer: int
    n_embed: int
    arch_name: str


@dataclass
class WeightLoadResult:
    """Standardized result from weight loading."""

    success: bool
    n_written: int = 0
    n_fused: int = 0
    timing: dict[str, float] = field(default_factory=dict)
    error: str | None = None


# ── Plan Builder (shared by all formats) ────────────────────────────────────

# Canonical → SloTransformer mapping (single source of truth)
_ARCH_TO_SLONET: dict[str, str | None] = {
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

_SWIGLU_MAP: dict[str, str | None] = {
    "layers.{i}.ffn.gate.weight": "blocks.{i}.ff.w1.weight",
    "layers.{i}.ffn.gate.bias": "blocks.{i}.ff.w1.bias",
    "layers.{i}.ffn.up.weight": "blocks.{i}.ff.w3.weight",
    "layers.{i}.ffn.up.bias": "blocks.{i}.ff.w3.bias",
}

_GELU_MAP: dict[str, str | None] = {
    "layers.{i}.ffn.up.weight": "blocks.{i}.ff.w1.weight",
    "layers.{i}.ffn.up.bias": "blocks.{i}.ff.w1.bias",
}

_NORM_BIAS_MAP: dict[str, str | None] = {
    "layers.{i}.attn_norm.bias": "blocks.{i}.attn_norm.bias",
    "layers.{i}.ff_norm.bias": "blocks.{i}.ff_norm.bias",
    "final_norm.bias": "norm.bias",
}

_NO_TRANSPOSE: set[str] = {"embed.token", "embed.pos", "lm_head"}


# ── Shape contracts (portable loader) ───────────────────────────────────────


class ShapeMismatchError(ValueError):
    """A file tensor cannot be assigned to its model parameter.

    Raised BEFORE any parameter is written (plan-time validation), or at
    write time if a shape slips through. Carries the full issue list so one
    exception explains every mismatch, not just the first.
    """

    def __init__(self, issues: list[tuple[str, str, tuple, tuple]]):
        self.issues = issues
        lines = [f"shape mismatch loading {len(issues)} tensor(s):"]
        for param_name, canonical, have, want in issues:
            lines.append(
                f"  {param_name} (canonical={canonical}): "
                f"file {have} vs model {want}"
            )
        lines.append(
            "FFN dims may pad to the next 64-multiple (align_dim_ff); "
            "everything else must match exactly"
        )
        super().__init__("\n".join(lines))

    @classmethod
    def single(cls, param_name: str, canonical: str, have: tuple, want: tuple) -> ShapeMismatchError:
        return cls([(param_name, canonical, have, want)])


# FFN canonicals whose buffers are SIMD-aligned (may legitimately differ by
# the align_dim_ff rounding — zero-padded at assign time).
_FFN_SWIGLU_CANONICALS: frozenset[str] = frozenset(
    {
        "layers.{i}.ffn.gate.weight",
        "layers.{i}.ffn.gate.bias",
        "layers.{i}.ffn.up.weight",
        "layers.{i}.ffn.up.bias",
        "layers.{i}.ffn.down.weight",
        "layers.{i}.ffn.down.bias",
    }
)
_FFN_GELU_CANONICALS: frozenset[str] = frozenset(
    {
        "layers.{i}.ffn.up.weight",
        "layers.{i}.ffn.up.bias",
        "layers.{i}.ffn.down.weight",
        "layers.{i}.ffn.down.bias",
    }
)


@dataclass(frozen=True)
class ArchShapePolicy:
    """Per-architecture shape rules for the loader.

    The ONE place per architecture where load-time quirks live. Today that
    is only "which FFN canonicals may take alignment padding"; fused-QKV
    layouts or quant packing get entries here later — never in the assign
    loops.
    """

    name: str
    ffn_pad_canonicals: frozenset[str]


SHAPE_POLICIES: dict[str, ArchShapePolicy] = {
    "Qwen2ForCausalLM": ArchShapePolicy("Qwen2ForCausalLM", _FFN_SWIGLU_CANONICALS),
    "Qwen3ForCausalLM": ArchShapePolicy("Qwen3ForCausalLM", _FFN_SWIGLU_CANONICALS),
    "GPT2LMHeadModel": ArchShapePolicy("GPT2LMHeadModel", _FFN_GELU_CANONICALS),
}
# Unknown arch: permit the union — alignment rules are structural, and
# fail-loud still happens for any non-alignable difference.
_DEFAULT_SHAPE_POLICY = ArchShapePolicy("default", _FFN_SWIGLU_CANONICALS | _FFN_GELU_CANONICALS)


def get_shape_policy(arch_name: str) -> ArchShapePolicy:
    """Resolve the shape policy for an architecture name (exact key first)."""
    return SHAPE_POLICIES.get(arch_name, _DEFAULT_SHAPE_POLICY)


def _is_align_pad(have: tuple, want: tuple) -> bool:
    """True when ``want`` is ``have`` with exactly one axis rounded up by
    the FFN SIMD alignment (``align_dim_ff``)."""
    from domain.shared import align_dim_ff

    if len(have) != len(want):
        return False
    diffs = [i for i in range(len(have)) if have[i] != want[i]]
    if len(diffs) != 1:
        return False
    i = diffs[0]
    return want[i] > have[i] and want[i] == align_dim_ff(have[i])


def _assign_checked(param, arr: np.ndarray, param_name: str, canonical: str, policy) -> bool:
    """Write ``arr`` into ``param.data`` under the shape contract.

    Returns True when zero-padding for FFN alignment was applied. Raises
    ShapeMismatchError for anything not covered by the policy.
    """
    dst = param.data
    have, want = arr.shape, dst.shape
    if have == want:
        dst[...] = arr
        return False
    if canonical in policy.ffn_pad_canonicals and _is_align_pad(have, want):
        dst[...] = 0
        dst[tuple(slice(0, s) for s in have)] = arr
        logger.debug(
            "FFN alignment pad: %s %s -> %s (zero-filled)", param_name, have, want
        )
        return True
    raise ShapeMismatchError.single(param_name, canonical, have, want)


def _validate_plan_shapes(plan: LoadPlan, param_map: dict, shapes, policy) -> None:
    """Plan-time validation: every mapped tensor checked against the model
    BEFORE any parameter is written. ``shapes(file_name)`` returns the file
    tensor's shape or None when absent."""
    issues: list[tuple[str, str, tuple, tuple]] = []
    for file_name, mapping in plan.tensor_map.items():
        param = param_map.get(mapping.param_name)
        if param is None:
            continue
        have = shapes(file_name)
        if have is None:
            continue
        if mapping.needs_transpose and len(have) == 2:
            have = (have[1], have[0])
        want = param.data.shape
        if have == want:
            continue
        if mapping.canonical in policy.ffn_pad_canonicals and _is_align_pad(have, want):
            continue
        issues.append((mapping.param_name, mapping.canonical, have, want))
    if issues:
        raise ShapeMismatchError(issues)


def build_load_plan(
    state_dict: dict[str, np.ndarray],
    n_layer: int,
    config: dict,
) -> LoadPlan:
    """Pre-compute loading plan from state dict + config.

    Detects architecture, maps file tensors → model parameters, identifies
    tied/synthesized/fused-QKV params. Single source of truth for all formats.
    """
    _t0 = time.monotonic()

    from .arch_config import build_arch

    # ── Native identity path (.soul / native .slnc keys) ──────────────────
    # File tensor names already match SloTransformer params — no remapping,
    # no transpose. This is the owned-architecture load plan (goal 12).
    if "tok_emb.weight" in state_dict:
        tensor_map: dict[str, TensorMapping] = {
            key: TensorMapping(param_name=key, needs_transpose=False, canonical=key)
            for key in state_dict
        }
        loaded_params = set(tensor_map)
        tied: list[tuple[str, str]] = []
        if "lm_head.weight" not in loaded_params and "tok_emb.weight" in loaded_params:
            tied.append(("lm_head.weight", "tok_emb.weight"))

        synthesized: list[tuple[str, str, str]] = []
        has_w3 = any(k.endswith(".ff.w3.weight") for k in loaded_params)
        if not has_w3:
            for i in range(n_layer):
                w1_key = f"blocks.{i}.ff.w1.weight"
                if w1_key in loaded_params:
                    synthesized.append((f"blocks.{i}.ff.w3.weight", "0", w1_key))
                    synthesized.append((f"blocks.{i}.ff.w3.bias", "1", w1_key))

        n_embed_arr = state_dict.get("tok_emb.weight")
        n_embed = int(n_embed_arr.shape[1]) if n_embed_arr is not None and n_embed_arr.ndim == 2 else 0

        plan = LoadPlan(
            tensor_map=tensor_map,
            tied_weights=tied,
            synthesized_params=synthesized,
            fused_qkv={},
            n_layer=n_layer,
            n_embed=n_embed,
            arch_name="native",
        )
        logger.info(
            "build_load_plan: arch=native mapped=%d tied=%d synth=%d (%.3fs)",
            len(tensor_map),
            len(tied),
            len(synthesized),
            time.monotonic() - _t0,
            extra={"tag": "INFRA"},
        )
        return plan

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
    tensor_map: dict[str, TensorMapping] = {}
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
    fused_qkv: dict[str, list[str]] = {}
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
    tied: list[tuple[str, str]] = []
    if "lm_head.weight" not in loaded_params and "tok_emb.weight" in loaded_params:
        tied.append(("lm_head.weight", "tok_emb.weight"))

    # Synthesized params (GELU w3)
    synthesized: list[tuple[str, str, str]] = []
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
        arch.name,
        len(tensor_map),
        len(fused_qkv),
        len(tied),
        len(synthesized),
        time.monotonic() - _t0,
        extra={"tag": "INFRA"},
    )
    return plan


# ── Architecture Inference ───────────────────────────────────────────────────


def _split_fused_qkv_weight(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a fused QKV weight into q, k, v chunks in (out, in) layout.

    Handles both storage layouts:
    - HF/GPT-2 ``(n, 3n)``: transpose to ``(3n, n)``, then split rows.
    - Stacked ``(3n, n)``: split rows, then transpose each chunk.
    """
    rows, cols = arr.shape[:2]
    if rows and rows == 3 * cols:
        return tuple(c.T for c in np.split(arr, 3, axis=0))  # type: ignore[return-value]
    return np.split(arr.T, 3, axis=0)  # type: ignore[return-value]


def infer_arch_from_state_dict(
    state_dict: dict[str, np.ndarray], config: dict | None = None
) -> dict:
    """Infer model architecture dims from config (preferred) + tensor shapes.

    Centralizes the duplicated arch detection logic found in
    routers/souls.py, models/provider.py, and controllers/models.py.

    Config-first: when ``config`` (config.json) is provided its values win;
    tensor shapes fill gaps. Structural defaults (n_embed=128 etc.) only
    apply when neither source has the value — and log a warning, since they
    are last-ditch placeholders, not real architecture.

    Args:
        state_dict: Dict mapping param names → numpy arrays
        config: Optional model config dict (same keys as dims_from_config).

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

    # Config first — authoritative when present.
    if config:
        n_embed_cfg = config.get("n_embd", config.get("hidden_size"))
        if n_embed_cfg is not None:
            result["n_embed"] = n_embed_cfg
        if config.get("vocab_size") is not None:
            result["vocab_size"] = config["vocab_size"]
        n_layer_cfg = config.get("n_layer", config.get("num_hidden_layers"))
        if n_layer_cfg is not None:
            result["n_layer"] = n_layer_cfg
        n_head_cfg = config.get("n_head", config.get("num_attention_heads"))
        if n_head_cfg is not None:
            result["n_head"] = n_head_cfg
        inter_cfg = config.get("n_inner") or config.get("intermediate_size")
        if inter_cfg is not None:
            result["intermediate_size"] = inter_cfg

    # Shapes fill gaps (and are the sole source when no config is given).
    tok_emb = state_dict.get("tok_emb.weight")
    if tok_emb is not None and tok_emb.ndim == 2:
        if config is None or config.get("vocab_size") is None:
            result["vocab_size"] = int(tok_emb.shape[0])
        if (
            config is None
            or (config.get("n_embd") is None and config.get("hidden_size") is None)
        ):
            result["n_embed"] = int(tok_emb.shape[1])

    # n_layer from max block index
    if config is None or (
        config.get("n_layer") is None and config.get("num_hidden_layers") is None
    ):
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
    if q_w is not None and q_w.ndim == 2 and (
        config is None
        or (config.get("n_head") is None and config.get("num_attention_heads") is None)
    ):
        head_dim = n_embed // 8
        if head_dim > 0:
            detected = q_w.shape[0] // head_dim
            if detected >= 1:
                result["n_head"] = detected

    # intermediate_size from w1/gate_proj shape
    if config is None or (not config.get("n_inner") and not config.get("intermediate_size")):
        for key in state_dict:
            if (
                "mlp.w1.weight" in key
                or "mlp.gate_proj.weight" in key
                or "ff.w1.weight" in key
                or "ff.gate_proj.weight" in key
            ):
                shape = state_dict[key].shape
                if len(shape) >= 2:
                    result["intermediate_size"] = shape[0]
                break

    if config is None and tok_emb is None:
        logger.warning(
            "infer_arch_from_state_dict: no config and no tok_emb — "
            "returning structural defaults (n_embed=128, intermediate=512); "
            "pass config.json for real dims"
        )

    return result


def dims_from_config(config: dict) -> dict:
    """Resolve every construction dim/flag from a config dict — config-first.

    Single extraction point shared by model construction, plan validation,
    and callers that previously re-derived dims (or silently fell back to
    hardcoded defaults). Unknown keys keep the historical defaults.

    Args:
        config: HF or native model config (config.json / .soul metadata).

    Returns:
        Dict with keys: n_embed, n_head, n_layer, vocab_size,
        intermediate_size, max_pos, max_seq_len, use_abs_pos, norm_type,
        n_kv_head, activation, eps, rope_base.
    """
    n_embed = config.get("n_embd", config.get("hidden_size", 768))
    n_head = config.get("n_head", config.get("num_attention_heads", 12))
    n_layer = config.get("n_layer", config.get("num_hidden_layers", 12))
    vocab_size = config.get("vocab_size", 50257)
    intermediate_size = config.get("n_inner") or config.get("intermediate_size", n_embed * 4)
    max_pos = (
        config.get("n_positions")
        or config.get("max_position_embeddings")
        or config.get("block_size")
        or 1024
    )
    max_seq_len = config.get("max_seq_len") or max_pos

    # Auto-detect positional encoding (native soul sets use_rope explicitly)
    if config.get("use_rope") is not None:
        use_abs_pos = not bool(config["use_rope"])
    else:
        has_rope = (
            config.get("rope_theta") is not None
            or config.get("position_embedding_type") == "rope"
        )
        use_abs_pos = not has_rope

    # Auto-detect norm type (native soul sets norm_type explicitly)
    if config.get("norm_type"):
        norm_type = config["norm_type"]
    elif config.get("layer_norm_type"):
        norm_type = config["layer_norm_type"]
    elif config.get("rms_norm_eps") is not None:
        norm_type = "rms_norm"
    else:
        norm_type = "layer_norm"

    # Auto-detect GQA
    n_kv_head = config.get("num_key_value_heads", n_head)

    # Auto-detect activation
    hidden_act = config.get("hidden_act", "gelu")
    activation = "silu" if hidden_act == "silu" else "gelu"

    eps = config.get("rms_norm_eps", 1e-5)

    return {
        "n_embed": n_embed,
        "n_head": n_head,
        "n_layer": n_layer,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "max_pos": max_pos,
        "max_seq_len": max_seq_len,
        "use_abs_pos": use_abs_pos,
        "norm_type": norm_type,
        "n_kv_head": n_kv_head,
        "activation": activation,
        "eps": eps,
        "rope_base": config.get("rope_theta", 10000.0),
    }


def build_model_from_config(config: dict, _lazy: bool = True):
    """Construct a SloTransformer from a config dict (SLNC or .soul metadata).

    Auto-detects: RoPE vs absolute pos, RMSNorm vs LayerNorm, SwiGLU vs GELU,
    GQA (n_kv_head < n_head), eps, rope_base. All dim extraction lives in
    ``dims_from_config`` (config-first, one source of truth).

    Args:
        config: Dict with keys like hidden_size, num_hidden_layers, etc.
        _lazy: If True, skip weight initialization (for loading pre-trained weights)

    Returns:
        SloTransformer instance (uninitialized if _lazy=True)
    """
    from domain.training._internal.slonet import SloTransformer

    d = dims_from_config(config)
    use_abs_pos = d["use_abs_pos"]

    return SloTransformer(
        vocab_size=d["vocab_size"],
        n_embed=d["n_embed"],
        n_layer=d["n_layer"],
        n_head=d["n_head"],
        n_kv_head=d["n_kv_head"],
        intermediate_size=d["intermediate_size"],
        block_size=d["max_pos"],
        max_seq_len=d["max_seq_len"],
        use_rope=not use_abs_pos,
        rope_base=d["rope_base"],
        dropout=0.0,
        eps=d["eps"],
        tie_weights=True,
        use_abs_pos_emb=use_abs_pos,
        norm_type=d["norm_type"],
        activation=d["activation"],
        _lazy=_lazy,
    )


# ── Generic Loader ───────────────────────────────────────────────────────────


def load_into_model(
    model,
    plan: LoadPlan,
    tensor_data: dict[str, np.ndarray],
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
    policy = get_shape_policy(plan.arch_name)

    def _shape_of(file_name: str) -> tuple | None:
        arr = tensor_data.get(file_name)
        return None if arr is None else arr.shape

    _validate_plan_shapes(plan, param_map, _shape_of, policy)

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
            _assign_checked(param, arr.T, mapping.param_name, mapping.canonical, policy)
        else:
            _assign_checked(param, arr, mapping.param_name, mapping.canonical, policy)
        n_written += 1

    _t_direct = time.monotonic()

    # Fused QKV
    n_fused = 0
    for hf_key, param_names in plan.fused_qkv.items():
        arr = tensor_data.get(hf_key)
        if arr is None:
            continue
        is_bias = arr.ndim == 1
        canonical = "layers.{i}.qkv.bias" if is_bias else "layers.{i}.qkv.weight"
        if is_bias:
            q, k, v = np.split(arr, 3, axis=-1)
        else:
            q, k, v = _split_fused_qkv_weight(arr)
        for pname, chunk in zip(param_names, [q, k, v], strict=False):
            p = param_map.get(pname)
            if p is not None:
                _assign_checked(p, chunk, pname, canonical, policy)
                n_fused += 1

    _t_fused = time.monotonic()

    # Tied weights
    for dest, src in plan.tied_weights:
        if dest in param_map and src in param_map:
            _assign_checked(
                param_map[dest], param_map[src].data, dest, "lm_head.tie", policy
            )

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

    def __init__(self, parser, state_dict: dict[str, np.ndarray], config: dict):
        self._parser = parser
        self._state_dict = state_dict
        self._plan = build_load_plan(
            state_dict,
            config.get("n_layer", config.get("num_hidden_layers", 12)),
            config,
        )

    @classmethod
    def _from_plan(
        cls, parser, plan: LoadPlan, state_dict: dict[str, np.ndarray]
    ) -> DirectWeightLoader:
        """Construct from a pre-built plan (avoids rebuilding it)."""
        loader = cls.__new__(cls)
        loader._parser = parser
        loader._state_dict = state_dict
        loader._plan = plan
        return loader

    @property
    def plan(self) -> LoadPlan:
        return self._plan

    def load(self, model, max_workers: int | None = None) -> WeightLoadResult:
        """Load weights directly from mmap into model parameters."""
        _t0 = time.monotonic()
        plan = self._plan
        param_map = dict(model._named_parameters())
        parser = self._parser
        policy = get_shape_policy(plan.arch_name)

        def _shape_of(file_name: str) -> tuple | None:
            if file_name not in parser.tensor_names:
                return None
            info = parser.get_tensor_info(file_name)
            return tuple(info[1])

        _validate_plan_shapes(plan, param_map, _shape_of, policy)

        n_written = 0
        for file_name, mapping in plan.tensor_map.items():
            if file_name not in parser.tensor_names:
                continue
            arr = parser.read_tensor_region(file_name)
            param = param_map.get(mapping.param_name)
            if param is None:
                continue

            if mapping.needs_transpose and arr.ndim == 2:
                _assign_checked(param, arr.T, mapping.param_name, mapping.canonical, policy)
            else:
                _assign_checked(param, arr, mapping.param_name, mapping.canonical, policy)
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
                q, k, v = _split_fused_qkv_weight(arr)

            for pname, chunk in zip(param_names, [q, k, v], strict=False):
                p = param_map.get(pname)
                if p is not None:
                    _assign_checked(p, chunk, pname, "layers.{i}.qkv", policy)
                    n_fused += 1

        _t_fused = time.monotonic()

        for dest, src in plan.tied_weights:
            if dest in param_map and src in param_map:
                _assign_checked(
                    param_map[dest], param_map[src].data, dest, "lm_head.tie", policy
                )

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
        from domain.inference._internal.slo_format import load_soul

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
        from domain.inference._internal.slo_format import load_soul

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
        self._loaders: dict[str, type] = {}
        self._default: type | None = None

    def register_loader(self, suffix: str, loader_class: type):
        """Register a loader class for a file suffix (case-insensitive)."""
        self._loaders[suffix.lower()] = loader_class
        logger.info(
            "Registered weight loader: %s → %s",
            suffix,
            getattr(loader_class, "__name__", type(loader_class).__name__),
            extra={"tag": "INFRA"},
        )

    def set_default(self, loader_class: type):
        """Set fallback loader for unregistered suffixes."""
        self._default = loader_class

    def get_loader(self, file_path: str) -> type | None:
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
_registry: WeightLoaderRegistry | None = None


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
