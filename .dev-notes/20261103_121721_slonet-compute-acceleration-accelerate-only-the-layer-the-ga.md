---
id: 20261103_121721_slonet-compute-acceleration-accelerate-only-the-layer-the-ga
status: done
board: 20261103_121721_slonet-compute-acceleration-accelerate-only-the-layer-the-ga
title: "SloNet compute acceleration — accelerate ONLY the layer the gap benchmark proves binding (blocked on gap benchmark)"
tags: ["sloNet", "benchmark", "quantization"]
sprint: ""
gh: null
---

# SloNet compute acceleration — accelerate ONLY the layer the gap benchmark proves binding (blocked on gap benchmark)

Self-sufficiency thread (user Q 2026-11-03): own compute implementation. Do NOT pre-build: Card A (20261103_121700) names the binding layer — kernel work (slonet_kernels.py) vs quantization (gguf_export.py, benchmark_quantization.py) vs other. Every change benchmark-gated per SOP (benchmark_slonet.py / benchmark_slonet_training.py — regression => revert). BLOCKED on 20261103_121700. Out of scope: Qwen autoload path untouched; the served-model swap is a separate later ruling behind the existing model-load state machine.
