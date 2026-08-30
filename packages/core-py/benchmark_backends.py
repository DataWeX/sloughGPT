"""Benchmark: VectorBE vs NumpyBE on Qwen model."""

import os
import sys
import time
import numpy as np

SLNC = "/home/mana/Documents/Default Project/sloughGPT/models/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/model.slnc"


def load_model():
    from domains.infrastructure.slnc.parser import SLNCParser
    from domains.infrastructure.arch_config import build_arch
    parser = SLNCParser(SLNC)
    config = parser.config
    keys = set(parser._tensor_map.keys())
    arch = build_arch(config.get("_name_or_path", "model"), config, keys)
    weights = parser.get_weights_dict_parallel()
    return weights, arch


def bench_forward(name, backend, token_ids, warmup=2, runs=5):
    # Warmup
    for _ in range(warmup):
        backend.forward(token_ids)

    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        backend.forward(token_ids)
        times.append(time.perf_counter() - t0)

    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    print(f"  {name:20s}  {avg:8.1f}ms ± {std:5.1f}ms")
    return avg


def bench_generate(name, backend, token_ids, max_tokens=20):
    t0 = time.perf_counter()
    result, metrics = backend.generate(token_ids, max_new_tokens=max_tokens, temperature=0.0)
    elapsed = (time.perf_counter() - t0) * 1000
    tps = metrics["n_tokens"] / (elapsed / 1000)
    print(f"  {name:20s}  {elapsed:8.0f}ms  {tps:5.1f} tok/s  ({metrics['n_tokens']} tokens)")
    return tps


def main():
    print("Loading model...")
    weights, arch = load_model()
    print(f"  {arch.n_layers} layers, {arch.hidden_size} hidden, {arch.n_heads} heads\n")

    # Import and register backends
    from domains.infrastructure.numpy_backend import NumpyBE
    from domains.infrastructure.vector_backend import VectorBE

    np_be = NumpyBE.from_weights(weights, arch)
    vec_be = VectorBE.from_weights(weights, arch)

    print(f"NumpyBE: {np_be.backend_name()}")
    print(f"VectorBE: {vec_be.backend_name()}\n")

    # Test 1: Forward pass (seq=1)
    print("=== Forward Pass (seq=1) ===")
    ids_1 = np.array([[1]], dtype=np.int64)
    t_np = bench_forward("NumpyBE", np_be, ids_1)
    t_vec = bench_forward("VectorBE", vec_be, ids_1)
    print(f"  Speedup: {t_np/t_vec:.2f}x\n")

    # Test 2: Forward pass (seq=128)
    print("=== Forward Pass (seq=128) ===")
    ids_128 = np.arange(128, dtype=np.int64).reshape(1, -1)
    t_np = bench_forward("NumpyBE", np_be, ids_128)
    t_vec = bench_forward("VectorBE", vec_be, ids_128)
    print(f"  Speedup: {t_np/t_vec:.2f}x\n")

    # Test 3: Generation
    print("=== Generation (20 tokens) ===")
    ids_gen = np.array([[1, 2, 3]], dtype=np.int64)
    tps_np = bench_generate("NumpyBE", np_be, ids_gen, 20)
    tps_vec = bench_generate("VectorBE", vec_be, ids_gen, 20)
    print(f"  Speedup: {tps_vec/tps_np:.2f}x")


if __name__ == "__main__":
    main()
