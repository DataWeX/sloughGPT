# Porting to ASM & WASM (brief plan)

## Goal
Create a minimal inference engine that can run in the browser (WASM) and later
in native environments with optional ASM-level kernels.

## Phase 1 — WASM inference prototype
1. Export a tiny model (small config, fixed vocab) to a portable format.
2. Implement a minimal tokenizer (char-level first).
3. Build a WASM runtime that supports:
   - Embedding lookup
   - LayerNorm/RMSNorm
   - MLP (GELU/SwiGLU)
   - Self-attention (single-head first)
4. Validate with golden outputs from Python.

## Phase 2 — Optimize
1. Add KV-cache for fast autoregressive decoding.
2. Fuse ops (matmul + bias + activation) where possible.
3. Move hot paths to SIMD (WASM SIMD).

## Phase 3 — ASM-native kernels
1. Identify bottlenecks via profiling.
2. Implement fused kernels (matmul, softmax) in assembly or intrinsics.
3. Integrate optional native backend with runtime dispatch.

## Constraints
- Start with **small models** and **short context**.
- Keep deterministic tests for numerical correctness.
- Prioritize inference first; training later.
