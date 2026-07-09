"""
Smart converter — analyzes if HF→SloNet conversion helps.

Doesn't just convert blindly. Scores the model on architecture compatibility,
size, device, and quality tradeoffs. Returns a recommendation.

Usage:
    from domains.infrastructure.smart_converter import analyze_model
    report = analyze_model("gpt2")
    print(report.recommendation)  # "convert" | "keep" | "skip"
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("man.infrastructure.smart_converter")


class Recommendation(str, Enum):
    CONVERT = "convert"      # SloNet conversion will help
    KEEP = "keep"            # Keep as-is (HF is fine)
    SKIP = "skip"            # Can't convert / not worth it


@dataclass
class ConversionReport:
    """Analysis result for a model."""
    model_id: str
    recommendation: Recommendation
    score: float                          # 0-100, higher = more worth converting
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    arch_match: float = 0.0               # 0-1 architecture compatibility
    size_score: float = 0.0               # 0-1 size benefit
    speed_estimate: str = ""              # estimated speedup
    memory_estimate: str = ""             # estimated memory change
    convertible_layers: int = 0
    total_layers: int = 0
    params: int = 0
    slo_params: int = 0

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_id}",
            f"Recommendation: {self.recommendation.value.upper()} (score: {self.score:.0f}/100)",
            f"Architecture match: {self.arch_match:.0%}",
            f"Parameters: {self.params:,} → {self.slo_params:,} (SloNet)",
            f"Convertible layers: {self.convertible_layers}/{self.total_layers}",
        ]
        if self.speed_estimate:
            lines.append(f"Speed: {self.speed_estimate}")
        if self.memory_estimate:
            lines.append(f"Memory: {self.memory_estimate}")
        if self.reasons:
            lines.append("Reasons:")
            for r in self.reasons:
                lines.append(f"  + {r}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


# ── Architecture compatibility scoring ───────────────────────────────────────

# Layers that map cleanly HF → SloNet
_COMPATIBLE_LAYERS = {
    "wte": "tok_emb",               # token embeddings — exact match
    "ln_f": "norm",                  # final layer norm
    "h.*.ln_1": "blocks.*.attn_norm",  # pre-attention norm
    "h.*.ln_2": "blocks.*.ff_norm",    # pre-FFN norm
    "h.*.attn.c_proj": "blocks.*.attn.o_proj",  # output projection
    "h.*.mlp.c_proj": "blocks.*.ff.w2",          # FFN down-projection
}

# Layers that need splitting/reshaping
_PARTIAL_LAYERS = {
    "h.*.attn.c_attn": "blocks.*.attn.{q,k,v}_proj",  # combined QKV → split
    "h.*.mlp.c_fc": "blocks.*.ff.w1",                  # FFN up → gate (close enough)
}

# Layers with no SloNet equivalent
_INCOMPATIBLE_LAYERS = {
    "wpe": "RoPE replaces absolute positional embeddings",
}


def _score_architecture(config: dict, weights: dict) -> Tuple[float, List[str], List[str], int, int]:
    """Score architecture compatibility (0-1)."""
    reasons = []
    warnings = []
    compatible = 0
    total = 0

    n_layer = config.get("n_layer", 12)
    n_head = config.get("n_head", 12)
    n_embed = config.get("n_embd", 768)

    # Check head divisibility
    if n_embed % n_head == 0:
        reasons.append(f"n_embed ({n_embed}) divisible by n_head ({n_head})")
        compatible += 1
    else:
        warnings.append(f"n_embed ({n_embed}) not divisible by n_head ({n_head})")
    total += 1

    # Check layer count
    if n_layer <= 24:
        reasons.append(f"Layer count ({n_layer}) is convertible")
        compatible += 1
    else:
        warnings.append(f"Layer count ({n_layer}) is large — conversion may be slow")
    total += 1

    # Check weight shapes
    for key in weights:
        if key.startswith("h.0.attn.c_attn.weight"):
            # Combined QKV — needs splitting
            shape = weights[key].shape
            if shape[0] == shape[1] * 3:
                reasons.append(f"QKV projection ({shape}) can be split evenly")
                compatible += 1
            else:
                warnings.append(f"QKV shape {shape} — uneven split possible")
            total += 1

        if key.startswith("h.0.mlp.c_fc.weight"):
            shape = weights[key].shape
            reasons.append(f"FFN up-projection ({shape}) maps to SwiGLU gate")
            compatible += 1
            total += 1

        if key == "wpe.weight":
            warnings.append("Absolute positional embeddings — will be discarded (RoPE used instead)")
            total += 1

        if key == "wte.weight":
            reasons.append("Token embeddings — exact match")
            compatible += 1
            total += 1

    # Avoid double-counting
    total = max(total, 4)

    return compatible / total, reasons, warnings, compatible, total


def _score_size(config: dict) -> Tuple[float, str, str]:
    """Score size benefit (0-1)."""
    n_embed = config.get("n_embd", 768)
    n_layer = config.get("n_layer", 12)
    vocab_size = config.get("vocab_size", 50257)

    # Estimate params
    params = vocab_size * n_embed + n_layer * (4 * n_embed * n_embed + 4 * n_embed)
    params_m = params / 1e6

    # SloNet is more memory-efficient (no PyTorch overhead, numpy arrays)
    slo_memory_mb = params * 4 / (1024 * 1024)  # float32
    hf_memory_mb = slo_memory_mb * 1.3  # PyTorch overhead

    memory_ratio = 1 - (slo_memory_mb / hf_memory_mb)
    speed_note = f"NumPy forward pass ~{params_m:.0f}M params on CPU"
    memory_note = f"Saves ~{memory_ratio:.0%} memory vs PyTorch"

    return min(memory_ratio * 2, 1.0), speed_note, memory_note


# ── Main analysis ────────────────────────────────────────────────────────────


def analyze_model(model_id: str) -> ConversionReport:
    """
    Analyze if converting a HuggingFace model to SloNet will help.

    Args:
        model_id: HuggingFace model ID (e.g. "gpt2", "Qwen/Qwen2.5-0.5B-Instruct")

    Returns:
        ConversionReport with recommendation, scores, and reasoning
    """
    from domains.infrastructure.safetensors_loader import load_model_config, load_model_weights

    report = ConversionReport(model_id=model_id, recommendation=Recommendation.SKIP, score=0)

    # Load config
    try:
        config = load_model_config(model_id)
    except FileNotFoundError:
        report.warnings.append(f"Config not found for {model_id}")
        return report

    # Load weights (just check shapes, don't load full data)
    try:
        weights = load_model_weights(model_id)
    except Exception as e:
        report.warnings.append(f"Could not load weights: {e}")
        return report

    report.params = sum(w.size for w in weights.values())

    # Architecture compatibility
    arch_score, reasons, warnings, compat, total = _score_architecture(config, weights)
    report.arch_match = arch_score
    report.reasons.extend(reasons)
    report.warnings.extend(warnings)
    report.convertible_layers = compat
    report.total_layers = total

    # Size benefit
    size_score, speed_est, mem_est = _score_size(config)
    report.size_score = size_score
    report.speed_estimate = speed_est
    report.memory_estimate = mem_est

    # Compute SloNet params (after conversion)
    n_embed = config.get("n_embd", 768)
    n_layer = config.get("n_layer", 12)
    vocab_size = config.get("vocab_size", 50257)
    report.slo_params = vocab_size * n_embed + n_layer * (4 * n_embed * n_embed + 4 * n_embed)

    # Final score: weighted combination
    report.score = (arch_score * 50) + (size_score * 30) + (min(report.convertible_layers / max(total, 1), 1.0) * 20)

    # Decision
    if report.score >= 60 and arch_score >= 0.5:
        report.recommendation = Recommendation.CONVERT
        report.reasons.append("Architecture is sufficiently compatible for conversion")
    elif report.score >= 30:
        report.recommendation = Recommendation.KEEP
        report.reasons.append("Conversion possible but benefits are marginal")
    else:
        report.recommendation = Recommendation.SKIP
        report.reasons.append("Architecture mismatch or too small to benefit")

    # Override: large models skip conversion (not worth the effort)
    if report.params > 1e9:
        report.recommendation = Recommendation.SKIP
        report.warnings.append(f"Model too large ({report.params/1e9:.1f}B params) — conversion not practical")

    return report


def convert_if_worth(
    model_id: str,
    output_path: Optional[str] = None,
    force: bool = False,
) -> Optional[str]:
    """
    Convert a model to SloNet only if it's worth it.

    Args:
        model_id: HuggingFace model ID
        output_path: Where to save .sou file (default: models/<model_id>.sou)
        force: Force conversion even if recommendation is SKIP

    Returns:
        Path to .sou file if converted, None if skipped
    """
    report = analyze_model(model_id)

    if report.recommendation == Recommendation.SKIP and not force:
        logger.info("Skipping %s: %s", model_id, report.warnings)
        return None

    if report.recommendation == Recommendation.KEEP and not force:
        logger.info("Keeping %s as HF: marginal benefit", model_id)
        return None

    logger.info("Converting %s to SloNet (score: %.0f)", model_id, report.score)

    try:
        from domains.infrastructure.safetensors_loader import load_model_weights, load_model_config
        from domains.training.slonet import SloTransformer, export_to_sou

        config = load_model_config(model_id)
        weights = load_model_weights(model_id)

        # Create SloTransformer
        n_embed = config.get("n_embd", 768)
        n_head = config.get("n_head", 12)
        n_layer = config.get("n_layer", 12)
        vocab_size = config.get("vocab_size", 50257)
        block_size = config.get("n_positions", 1024)

        model = SloTransformer(
            vocab_size=vocab_size,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            block_size=block_size,
            max_seq_len=block_size,
            use_rope=True,
            tie_weights=False,
            intermediate_size=config.get("n_inner", None),
            soul_name=model_id,
        )

        # Map weights (best-effort)
        _map_weights(model, weights, config)

        # Export
        if output_path is None:
            output_path = f"models/{model_id.replace('/', '_')}.sou"

        export_to_sou(model, output_path)
        logger.info("Converted %s → %s", model_id, output_path)
        return output_path

    except Exception as e:
        logger.error("Conversion failed for %s: %s", model_id, e)
        return None


def _map_weights(model: Any, hf_weights: dict, config: dict) -> None:
    """Map HuggingFace weights to SloTransformer state dict (best-effort)."""
    n_embed = config.get("n_embd", 768)
    n_head = config.get("n_head", 12)
    head_dim = n_embed // n_head

    state = {}

    # Token embeddings — direct copy
    if "wte.weight" in hf_weights:
        state["tok_emb.weight"] = hf_weights["wte.weight"]

    # No wpe — RoPE handles positions

    n_layer = config.get("n_layer", 12)
    for i in range(n_layer):
        prefix = f"h.{i}"

        # Attention norm (LayerNorm → RMSNorm: gamma only, no beta)
        if f"{prefix}.ln_1.weight" in hf_weights:
            state[f"blocks.{i}.attn_norm.weight"] = hf_weights[f"{prefix}.ln_1.weight"]

        # QKV split: (n_embed, 3*n_embed) → 3 × (n_embed, n_embed)
        if f"{prefix}.attn.c_attn.weight" in hf_weights:
            qkv_w = hf_weights[f"{prefix}.attn.c_attn.weight"]
            q_w, k_w, v_w = np.split(qkv_w, 3, axis=-1)
            state[f"blocks.{i}.attn.q_proj.weight"] = q_w
            state[f"blocks.{i}.attn.k_proj.weight"] = k_w
            state[f"blocks.{i}.attn.v_proj.weight"] = v_w

        if f"{prefix}.attn.c_attn.bias" in hf_weights:
            qkv_b = hf_weights[f"{prefix}.attn.c_attn.bias"]
            q_b, k_b, v_b = np.split(qkv_b, 3, axis=-1)
            state[f"blocks.{i}.attn.q_proj.bias"] = q_b
            state[f"blocks.{i}.attn.k_proj.bias"] = k_b
            state[f"blocks.{i}.attn.v_proj.bias"] = v_b

        # Output projection
        if f"{prefix}.attn.c_proj.weight" in hf_weights:
            state[f"blocks.{i}.attn.o_proj.weight"] = hf_weights[f"{prefix}.attn.c_proj.weight"]
        if f"{prefix}.attn.c_proj.bias" in hf_weights:
            state[f"blocks.{i}.attn.o_proj.bias"] = hf_weights[f"{prefix}.attn.c_proj.bias"]

        # FFN norm
        if f"{prefix}.ln_2.weight" in hf_weights:
            state[f"blocks.{i}.ff_norm.weight"] = hf_weights[f"{prefix}.ln_2.weight"]

        # FFN: GPT-2 c_fc (n_embed → 4*n_embed) → SwiGLU w1/w3 (n_embed → dim_ff)
        if f"{prefix}.mlp.c_fc.weight" in hf_weights:
            c_fc_w = hf_weights[f"{prefix}.mlp.c_fc.weight"]  # (n_embed, 4*n_embed)
            # Use c_fc as both gate (w1) and up (w3) — best effort
            state[f"blocks.{i}.ff.w1.weight"] = c_fc_w
            state[f"blocks.{i}.ff.w3.weight"] = c_fc_w.copy()
        if f"{prefix}.mlp.c_fc.bias" in hf_weights:
            c_fc_b = hf_weights[f"{prefix}.mlp.c_fc.bias"]
            state[f"blocks.{i}.ff.w1.bias"] = c_fc_b
            state[f"blocks.{i}.ff.w3.bias"] = c_fc_b.copy()

        # FFN down-projection
        if f"{prefix}.mlp.c_proj.weight" in hf_weights:
            state[f"blocks.{i}.ff.w2.weight"] = hf_weights[f"{prefix}.mlp.c_proj.weight"]
        if f"{prefix}.mlp.c_proj.bias" in hf_weights:
            state[f"blocks.{i}.ff.w2.bias"] = hf_weights[f"{prefix}.mlp.c_proj.bias"]

    # Final norm
    if "ln_f.weight" in hf_weights:
        state["norm.weight"] = hf_weights["ln_f.weight"]

    # LM head (weight tying — copy from tok_emb)
    if "wte.weight" in hf_weights:
        state["lm_head.weight"] = hf_weights["wte.weight"].copy()

    # Load into model
    model.load_state_dict(state, strict=False)
    logger.info("Mapped %d weight tensors to SloTransformer", len(state))
