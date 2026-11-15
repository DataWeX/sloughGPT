---
id: 20261114_091409_quantine-canonical-scalar-quantization-owner-converge-the-sc
status: done
board: 20261114_091409_quantine-canonical-scalar-quantization-owner-converge-the-sc
title: "Quantine = canonical scalar-quantization owner: converge the scattered implementations (Quantine / block_q8 / Quantize8Bit)"
tags: ["area:compression"]
sprint: ""
gh: null
---

# Quantine = canonical scalar-quantization owner: converge the scattered implementations (Quantine / block_q8 / Quantize8Bit)

Decision (user, 2026-11-14): Quantine should own everything quantization-related; no scattered implementations of the same math.

Today: three independent scalar quantizers — (1) Quantine per-tensor symmetric/asymmetric/clip + calibration + fused hot-path kernel (domain/infrastructure/_internal/quantization.py), (2) pugqeep compress_block_q8/decompress_block_q8 per-block min/max affine (pugqeep/compressor.py + codecs.py BQ8 framing), (3) Quantize8Bit example strategy (pugqeep/generic.py). Zero imports between them. VQ cluster/Huffman/DEFLATE are NOT scalar quantization — they stay in pugqeep.

Plan: extract a stateless numpy-only numeric core (scale_and_codes / dequantize, parameterized by granularity per-tensor|per-block and symmetry) into quant_core/; re-point all three call sites; engine API and BQ4/BQ8 wire framing byte-stable.

Gates: benchmark_block_quantization fidelity within tolerance (Q8 cosine ~0.99999), test_quantization_coverage (35) + pugqeep round-trip suites green, import direction pugqeep -> quant_core only (never the serving engine), domain-baseline rehash if internals move.

Branch: feat/quantine-quant-core (audit first — expected thin overlap; honest 'not worth extracting' verdict allowed with evidence).
