#!/usr/bin/env python3
"""
Benchmark quantized vs float32 inference on SloTransformer.

Measures:
  - Memory savings (fp32 bytes vs quantized bytes)
  - Per-token latency (fp32 vs int8 vs int4)
  - Output quality (cosine similarity to fp32 baseline)
  - Model load time (with and without pre-quantized weights)

Usage:
  python scripts/benchmark_quantized_inference.py [--steps 20]
"""

import argparse
import time
import sys
import numpy as np

sys.path.insert(0, "packages/core-py")

from domains.training.slonet import SloTransformer
from domains.infrastructure.quantization import Quantine, walk_slo_linears


def create_model(vocab=32000, embed=256, layers=4, heads=8, seq_len=128):
    """Create a SloTransformer model for benchmarking."""
    return SloTransformer(
        vocab_size=vocab,
        n_embed=embed,
        n_layer=layers,
        n_head=heads,
        intermediate_size=embed * 4,
        block_size=seq_len,
        max_seq_len=seq_len,
        use_rope=True,
        dropout=0.0,
        tie_weights=False,
        use_abs_pos_emb=False,
        norm_type="rms_norm",
    )


def quantize_model(model, bits, mode="symmetric"):
    """Quantize all SloLinear layers in-place. Returns (engine, count)."""
    engine = Quantine(bits=bits, mode=mode)
    layers = walk_slo_linears(model)
    count = 0
    for name, module in layers.items():
        info = engine.quantize(f"{name}.weight", module.weight.data.copy())
        if info.is_quantized:
            module.set_quantized_weight(info)
            count += 1
    return engine, count


def count_linear_bytes(model):
    """Count total float32 bytes for all SloLinear weights."""
    total = 0
    for _, module in walk_slo_linears(model).items():
        total += module.weight.data.nbytes
        if hasattr(module, "bias") and module.use_bias:
            total += module.bias.data.nbytes
    return total


def count_quantized_bytes(model):
    """Count total bytes used by quantized weights."""
    total = 0
    for _, module in walk_slo_linears(model).items():
        if module._quant_info is not None:
            total += module._quant_info.array.nbytes
            if hasattr(module, "bias") and module.use_bias:
                total += module.bias.data.nbytes
    return total


def benchmark_generate(model, input_ids, num_steps=10, label=""):
    """Measure per-token generation latency."""
    latencies = []
    for _ in range(3):
        model.generate_numpy(input_ids, max_new_tokens=2)

    for _ in range(num_steps):
        t0 = time.perf_counter()
        _ = model.generate_numpy(input_ids, max_new_tokens=10)
        latencies.append(time.perf_counter() - t0)

    latencies = np.array(latencies) * 1000
    print(f"  {label}: {latencies.mean():.1f}ms avg, {latencies.std():.1f}ms std, "
          f"min={latencies.min():.1f}ms, max={latencies.max():.1f}ms")
    return latencies


def _run(model, x):
    """Run model forward, extract logits from tuple return."""
    out = model(x)
    if isinstance(out, tuple):
        return out[0].data
    return out.data if hasattr(out, 'data') else out


def benchmark_forward(model, input_ids, num_steps=30, label=""):
    """Measure single forward pass latency."""
    for _ in range(3):
        _run(model, input_ids)
    latencies = []
    for _ in range(num_steps):
        t0 = time.perf_counter()
        _ = _run(model, input_ids)
        latencies.append(time.perf_counter() - t0)
    latencies = np.array(latencies) * 1000
    print(f"  {label}: {latencies.mean():.1f}ms avg, {latencies.std():.1f}ms std, "
          f"min={latencies.min():.1f}ms, max={latencies.max():.1f}ms")
    return latencies


def main():
    parser = argparse.ArgumentParser(description="Benchmark quantized inference")
    parser.add_argument("--steps", type=int, default=20, help="Number of timing steps")
    args = parser.parse_args()

    print("=" * 60)
    print("Quantized Inference Benchmark")
    print("=" * 60)

    # ── Model sizes ──────────────────────────────────────────────────
    for label, embed, layers, heads in [
        ("Tiny (d=64, L=2)", 64, 2, 4),
        ("Small (d=128, L=4)", 128, 4, 4),
    ]:
        print(f"\n--- {label} ---")
        model = create_model(embed=embed, layers=layers, heads=heads)
        inp = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

        fp32_bytes = count_linear_bytes(model)
        print(f"  FP32 weight memory: {fp32_bytes / 1024:.1f} KB")

        # int8
        qmodel = create_model(embed=embed, layers=layers, heads=heads)
        qmodel.load_state_dict(model.state_dict(), strict=False)
        _, cnt8 = quantize_model(qmodel, bits=8)
        q8_bytes = count_quantized_bytes(qmodel)
        ratio8 = fp32_bytes / max(q8_bytes, 1)
        print(f"  INT8 weight memory:  {q8_bytes / 1024:.1f} KB ({ratio8:.1f}x compression, {cnt8} layers)")

        # int4
        qmodel4 = create_model(embed=embed, layers=layers, heads=heads)
        qmodel4.load_state_dict(model.state_dict(), strict=False)
        _, cnt4 = quantize_model(qmodel4, bits=4)
        q4_bytes = count_quantized_bytes(qmodel4)
        ratio4 = fp32_bytes / max(q4_bytes, 1)
        print(f"  INT4 weight memory:  {q4_bytes / 1024:.1f} KB ({ratio4:.1f}x compression, {cnt4} layers)")

        # Quality
        logits_fp32 = _run(model, inp)
        logits_i8 = _run(qmodel, inp)
        logits_i4 = _run(qmodel4, inp)

        def cosine(a, b):
            return float(np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b)))

        print(f"  Cosine sim vs FP32: INT8={cosine(logits_fp32, logits_i8):.5f}, INT4={cosine(logits_fp32, logits_i4):.5f}")

    # ── Latency ──────────────────────────────────────────────────────
    print(f"\n--- Latency (Small model, {args.steps} steps) ---")
    model = create_model(embed=128, layers=4, heads=4)
    inp = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

    model(inp)
    l_fp32 = benchmark_forward(model, inp, args.steps, "FP32 forward")

    qmodel = create_model(embed=128, layers=4, heads=4)
    qmodel.load_state_dict(model.state_dict(), strict=False)
    quantize_model(qmodel, bits=8)
    l_i8 = benchmark_forward(qmodel, inp, args.steps, "INT8 forward")

    qmodel4 = create_model(embed=128, layers=4, heads=4)
    qmodel4.load_state_dict(model.state_dict(), strict=False)
    quantize_model(qmodel4, bits=4)
    l_i4 = benchmark_forward(qmodel4, inp, args.steps, "INT4 forward")

    # Speed ratio vs fp32
    print(f"\n  Speed ratio (lower=faster):")
    print(f"    INT8/FP32: {l_i8.mean() / l_fp32.mean():.2f}x")
    print(f"    INT4/FP32: {l_i4.mean() / l_fp32.mean():.2f}x")

    # ── Pre-quantized load time ──────────────────────────────────────
    print(f"\n--- Pre-quantized weight load time ---")
    import tempfile, os
    from pathlib import Path

    model = create_model(embed=128, layers=4, heads=4)
    engine = Quantine(bits=8, mode="symmetric")
    layers = walk_slo_linears(model)
    tensor_infos = {}
    for name, module in layers.items():
        info = engine.quantize(f"{name}.weight", module.weight.data.copy())
        if info.is_quantized:
            tensor_infos[name] = info

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        npz_path = f.name
    try:
        engine.save_weights(npz_path, tensor_infos)
        file_size = os.path.getsize(npz_path)
        print(f"  Quantized weights file size: {file_size / 1024:.1f} KB")

        t0 = time.perf_counter()
        for _ in range(10):
            _ = engine.load_weights(npz_path)
        load_ms = (time.perf_counter() - t0) * 100
        print(f"  Load time: {load_ms:.1f}ms avg (10 runs)")
    finally:
        os.unlink(npz_path)

    # ── AVX2 check ──────────────────────────────────────────────────
    print(f"\n--- Hardware acceleration ---")
    try:
        from domains.infrastructure.quant_core.wrapper import HAS_AVX2
        if HAS_AVX2:
            print("  AVX2 kernels: AVAILABLE (int8/int4 GEMM accelerated)")
        else:
            print("  AVX2 kernels: NOT AVAILABLE (using numpy fallback)")
    except Exception:
        print("  AVX2 kernels: NOT AVAILABLE (import failed)")

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
