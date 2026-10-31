---
id: 20261013_cpu_kernel_fusion_opencl
title: "CPU kernels: fuse numpy to one pass first, then revisit OpenCL"
status: todo
date: 2026-10-13
sprint: current
gh: ""
tags: [performance, gpu, numpy, quant_core, opencl, accelerator, kernels, benchmark]
---

# CPU kernels: fuse first, then revisit OpenCL

**Not started.** Findings from the session investigating *"can SlonNet replace
torch-class libraries / can CUDA run on CPU"*. Emerged out of the
`fix/test-drift-shell-repl` work; **no code written for this yet.**

---

## The answer to the question

**CUDA emulation on CPU is always worse than writing good CPU kernels.** The
CPU-vs-GPU gap is **kernel fusion and threading**, not the instruction set.
PTX/SASS has no CPU target to compile to, and even if it did, emulation could
not supply the memory bandwidth that is the GPU's actual advantage.

What CUDA really provides is *native code compiled for the target*. That is
obtainable on a CPU without emulation — and it is the piece we are missing.

---

## Corrected measurement: int8 decode GEMM

> **Correction owned:** an earlier message reported *"11,348 GB/s"* — that was
> **MB/s mislabelled as GB/s**. The real figure is **~11.4 GB/s**.

int8 decode GEMM, `K=4096`, weight **16.8 MB**, i5-9300H (4C/8T, AVX2, no AVX-512):

| path | time | bandwidth | % of CPU DRAM ceiling* |
|---|---|---|---|
| **AVX2 C kernel** (`quant_core`) | **1.48 ms** | **11.4 GB/s** | **27 %** |
| numpy int32 fallback | 22.3 ms | 0.75 GB/s | **1.8 %** |
| — | | | |
| GTX 1050 peak (GDDR5) | — | 112 GB/s | — (different memory) |

\* DDR4-2666 dual-channel ≈ **41.9 GB/s** theoretical.

**Derived:**
- **C kernel is 15.1× numpy.**
- numpy's int8 path runs at **1.8 % of CPU DRAM ceiling** — 56× off. It is
  catastrophically memory-pass-bound (int8→int32 promotion, multiple passes,
  no vectorization).
- The C kernel at **27 %** still has **3.7× headroom** to the CPU DRAM ceiling.
- GPU 112 GB/s = **2.67× the entire CPU DRAM ceiling**, and **9.8× the C
  kernel**. This is why the GPU wins at memory-bound work — it is a bandwidth
  machine, not an instruction-set machine.

**Read alongside:** fp32 dense GEMM measures **254.5 GFLOP/s (2048³)** and
**218.8 GFLOP/s (4096³)** ≈ **97 % of the i5-9300H's ~262 GFLOP/s FP32 peak**.
Same CPU, same library — one path saturated, one path at 1.8 %.

---

## Why we chose numpy (and where that choice stops being right)

**The case for numpy is sound — five real reasons:**

1. **Project rule: build from scratch, import last.** torch was purged
   (`81616a6f7`; the `make_torch_forward` orphan). numpy is the floor that
   needs no framework and no vendor runtime.
2. **The `_Accelerator` contract is numpy-shaped.** Every op takes and returns
   `np.ndarray`, so numpy is the *reference implementation* the contract was
   written against — any backend must express identical semantics, and numpy
   gives correctness by construction.
3. **numpy already delegates dense GEMM to threaded BLAS** — measured at
   **97 % of CPU peak**. For compute-bound dense GEMM numpy is *already
   optimal*; hand-written C would gain essentially nothing.
4. **Portability, no hardware gate.** numpy runs on any box; CUDA does not.
   The backend ladder (CUDA 200 → Metal 100 → OpenCL 50 → CPU 10/20) degrades
   gracefully *because* numpy is the universal floor.
5. **Correctness oracle.** Every other kernel (AVX2 C, CuPy, OpenCL) is diffed
   against numpy. You need it for testing regardless.

**But the numbers bound exactly where numpy stops being the right answer:**

| work type | numpy result | verdict |
|---|---|---|
| fp32 dense GEMM — compute-bound, BLAS takes it | 254 GFLOP/s ≈ **97 % peak** | **numpy wins — keep it** |
| int8 decode GEMM — memory-bound, no int8 BLAS path | 0.75 GB/s = **1.8 % DRAM** | **numpy loses 15× → C kernel** |
| fused norm+activation — elementwise, never touches BLAS | 4455 µs / 524 KB | **numpy loses ~40× → C kernel** |

### The rule that falls out

> **numpy is right when the work is compute-bound and BLAS takes it.
> numpy is wrong when the work is memory-pass-bound**, because every chained
> numpy op is a *separate pass over memory* with a malloc/free each.

That is precisely the matmul → bias → activation gap: **three memory
round-trips instead of one.** It is also why *"install OpenBLAS"* was a
no-op — BLAS was never in the loop for elementwise work.

**So the answer is a boundary, not a verdict:**
**numpy** for BLAS-backed GEMM + as the correctness floor,
**`quant_core` C** for memory-bound and fused ops.

---

## Finding: the `fused_*` ops are not fused (≈40× gap)

`_CPUBackend` (`domain/slolib/_internal/gpu.py:723`) overrides **only**
detection (`has_openblas`/`has_simd`/`openblas_threads`), metadata
(`vram_gb`, `compute_tier`, `memory_hint`), and `softmax`. Everything else —
`matmul`, `gelu`, `layer_norm`, `rms_norm`, and **every `fused_*` op** —
inherits base-class defaults.

Those defaults are multi-statement numpy allocating ~10 temporaries, under
docstrings claiming *"one pass instead of two"*:

```python
def fused_layernorm_gelu(self, x, weight, bias, eps=1e-5):
    """Fused layer_norm + gelu — one pass over x instead of two."""
    mean   = x.mean(...)                       # pass 1
    var    = x.var(...)                        # pass 2
    normed = (x - mean) / np.sqrt(var + eps)   # pass 3 + alloc
    scaled = normed * weight + bias            # pass 4 + alloc
    return 0.5*scaled*(1 + np.tanh(...))       # passes 5-7 + allocs
```

Measured:

```
(64, 2048)  tensor = 524 KB   fused_layernorm_gelu = 4455 us   ~824 MB/s eff.
(1, 2048)   tensor =   8 KB   fused_layernorm_gelu =  115 us
            fused_add_mul    =   52 us / 3.6 us   (base default: a + b*c)
```

524 KB in **4.5 ms** is nowhere near DRAM limits ⇒ **not bandwidth-bound —
it is malloc/free and Python-call overhead** from chained temporaries. A real
single pass needs ~100 µs. **≈40× off**, on the **decode hot path**, per token,
per layer.

The false docstrings are the same class of fiction as `make_torch_forward` —
and they are exactly what hid this gap.

**Prior art proving the approach works:** note
`2026-09-29-card093-fix-decode-gemv.md` — a sweep of **48 fusions took
33.1 → 0.142 ms/call**. Fusion sweeps are a proven, huge lever in this tree.

---

## Finding: OpenCL-on-CPU and WebGPU-on-CPU — phase 2

- **`_OpenCLBackend` (`gpu.py:1054`)** is the one principled *"same source, two
  targets"* route already in the tree: **pocl** or Intel's OpenCL CPU runtime
  would make `.is_available()` true on a GPU-less machine and run the
  *identical* kernels on CPU.
  - **UNVERIFIED:** I have **not** checked its op coverage. If it inherits the
    unfused base defaults, installing pocl buys little. **Gate for phase 2.**
- **`wgpu_be.py` degrades to plain numpy, not to WebGPU-on-CPU.** It drives its
  own C `gpu_engine.py` via ctypes; when `_has_gpu` is False or shader warmup
  fails, `backend_name()` returns `gpu(fallback-numpy)`.
- **The box already has the software-Vulkan path:** `libvulkan.so.1.3.275` and
  **`lvp_icd.json` (Mesa lavapipe, Vulkan-on-CPU)** are installed. Python
  `wgpu`/`wgpu_native` are **not**. WebGPU-on-CPU is reachable via lavapipe
  if we ever choose that route.

---

## THE PLAN (the ordering is the point)

> **Phase 1 — make numpy do one pass instead of three. Nothing else first.**
> **Phase 2 — then revisit OpenCL (pocl) as an idea.**

### Phase 1 — fused CPU kernels (the 40× lever)
1. **Extend `domain/infrastructure/_internal/quant_core/`** with a fused
   norm+activation kernel, reusing the proven pattern already there:
   ```
   matmul_int8.c / matmul_int4.c    ← C source
   *.so + *.so.buildflags           ← prebuilt, flags recorded
   wrapper.py                       ← ctypes + numpy fallback + HAS_AVX2/HAS_AVX512
   ```
   Zero new dependencies, matches *"build from scratch, import last"*.
   Alternative considered and rejected for now: **numba** — `@njit(parallel)`
   → LLVM → AVX2+threads and the same source runs on NVIDIA via
   `@numba.cuda.jit` (genuinely one source, two native targets), but ~100 MB
   incl. llvmpipe, not installed, plus first-call JIT latency.
2. Wire it into `_CPUBackend` — it currently overrides none of these ops.
   **Prefer landing in the base class so `_OpenCLBackend` inherits it too.**
3. **Make the docstrings true** (or reword to "chain of numpy passes") — same
   fiction class as `make_torch_forward`.
4. **Repair `has_openblas()`** to see numpy's bundled BLAS — fixes the
   `_detect_best_backend()` priority score (10 → 20+threads).
5. Wire `HAS_AVX2`/`HAS_AVX512` (already exported by `quant_core.wrapper`)
   into `_CPUBackend.has_simd()` — its docstring claims AVX2 detection its
   code does not do (currently just `arch in ("x86_64","AMD64") → True`).
6. Raise the C decode GEMM's **27 % → closer to the 41.9 GB/s ceiling**
   (3.7× headroom). Benchmark before/after:
   `scripts/benchmark_slonet.py`, `scripts/benchmark_latency.py`, recorded via
   `scripts/benchmark_results.py`.

### Phase 2 — revisit OpenCL (only after Phase 1)
- First **verify `_OpenCLBackend`'s op coverage** — if it inherits the unfused
  defaults, Phase 1's base-class fix already covers it.
- Then evaluate `pocl` as the CPU OpenCL runtime.

---

## Recurring pitfalls to avoid on revisit

- **Unit labels.** `11,348` was **MB/s written as GB/s**. Check magnitude
  against the memory ceiling before quoting — 112 GB/s (GPU GDDR5) and
  41.9 GB/s (CPU DDR4) are the two rulers.
- **`find_library(...) → None` does not mean "not installed."** Bundled,
  hash-renamed, RPATH-loaded libs are invisible to it — this one line produced
  two false inferences. Verify with `ldd` / `np.show_config()`.
- **A wrong detector is not a performance problem.** The priority score being
  `10` vs `20` costs nothing at runtime; do not add a system dependency to
  paper over it.
- **Do not verify by counting installs** — verify by measuring throughput. The
  254 GFLOP/s number is what settled OpenBLAS.

## Machine facts (re-measure if hardware changes)
- i5-9300H, 4C/8T, AVX2, **no AVX-512** → `HAS_AVX2=True`, `HAS_AVX512=False`
- DDR4-2666 dual-channel ≈ **41.9 GB/s** CPU DRAM ceiling
- NVIDIA GTX 1050 **3GB** (Pascal, 640 cores, **112 GB/s GDDR5**, no tensor cores)
- CuPy installed (`_CUDABackend` is CuPy, **never torch**) · torch **not**
  installed, not a dependency
- OpenBLAS present only as numpy's bundled `libscipy_openblas64_-f48b354e.so`
- Vulkan loader + **lavapipe** present; Python `wgpu` absent
