# Quantization Benchmark Report

- **Model**: tiny
- **Quick**: True
- **Precisions**: int8, int4
- **Python**: 3.12.3
- **CPU**: x86_64
- **Cores**: 8

## int8 Results (7/7 passed)

### throughput_vs_length [PASS]

```json
{
  "10": {
    "non_quantized_tps": 252.0,
    "quantized_tps": 334.0,
    "speedup": 1.33
  },
  "20": {
    "non_quantized_tps": 246.9,
    "quantized_tps": 306.8,
    "speedup": 1.24
  },
  "50": {
    "non_quantized_tps": 341.2,
    "quantized_tps": 504.9,
    "speedup": 1.48
  }
}
```

### throughput_vs_prompt [PASS]

```json
{
  "10": {
    "prompt_tokens": 10,
    "non_quantized_tps": 727.4,
    "quantized_tps": 485.7,
    "speedup": 0.67,
    "non_quantized_total_s": 0.069,
    "quantized_total_s": 0.103
  },
  "50": {
    "prompt_tokens": 50,
    "non_quantized_tps": 208.2,
    "quantized_tps": 438.2,
    "speedup": 2.1,
    "non_quantized_total_s": 0.24,
    "quantized_total_s": 0.114
  },
  "100": {
    "prompt_tokens": 100,
    "non_quantized_tps": 163.1,
    "quantized_tps": 372.5,
    "speedup": 2.28,
    "non_quantized_total_s": 0.307,
    "quantized_total_s": 0.134
  }
}
```

### temperature_impact [PASS]

```json
{
  "temp=0.0": {
    "non_quantized_tps": 214.9,
    "quantized_tps": 493.0,
    "speedup": 2.29
  },
  "temp=0.5": {
    "non_quantized_tps": 318.5,
    "quantized_tps": 447.6,
    "speedup": 1.41
  },
  "temp=1.0": {
    "non_quantized_tps": 540.7,
    "quantized_tps": 495.5,
    "speedup": 0.92
  }
}
```

### regression [PASS]

```json
{
  "non_quantized_tps": 353.5,
  "quantized_tps": 569.1,
  "speedup": 1.61
}
```

### memory_usage [PASS]

```json
{
  "non_quantized_rss_mb": 65,
  "quantized_rss_mb": 65,
  "rss_delta_mb": 0,
  "non_quantized_weight_mb": 4.1,
  "quantized_weight_mb": 1.0,
  "weight_compression": 4.0,
  "quantized_layers": 29
}
```

### quality_degradation [PASS]

```json
{
  "per_prompt": {
    "0": {
      "prompt": "The capital of France is",
      "logit_cosine": 0.9997,
      "token_agreement": 0.02,
      "nq_perplexity": 377.48,
      "q_perplexity": 377.59,
      "nq_tokens": 50,
      "q_tokens": 50
    },
    "1": {
      "prompt": "def fibonacci(n):",
      "logit_cosine": 0.9993,
      "token_agreement": 0.0,
      "nq_perplexity": 319.67,
      "q_perplexity": 320.57,
      "nq_tokens": 50,
      "q_tokens": 50
    },
    "2": {
      "prompt": "Once upon a time",
      "logit_cosine": 0.9996,
      "token_agreement": 0.02,
      "nq_perplexity": 607.76,
      "q_perplexity": 610.66,
      "nq_tokens": 50,
      "q_tokens": 50
    },
    "3": {
      "prompt": "The quick brown fox",
      "logit_cosine": 0.9996,
      "token_agreement": 0.0,
      "nq_perplexity": 306.32,
      "q_perplexity": 306.5,
      "nq_tokens": 50,
      "q_tokens": 50
    }
  },
  "avg_logit_cosine": 0.9996,
  "avg_token_agreement": 0.01,
  "nq_perplexity": 370.47,
  "q_perplexity": 372.03,
  "perplexity_ratio": 1.004,
  "ppl_passage_tokens": 534,
  "short_prompt_nq_perplexity": 402.81,
  "short_prompt_q_perplexity": 403.83
}
```

### cold_vs_warm [PASS]

```json
{
  "n_runs": 5,
  "non_quantized": {
    "cold_s": 0.121,
    "warm_median_s": 0.087,
    "cold_warmup_ratio": 1.39,
    "variance_s": 0.024
  },
  "quantized": {
    "cold_s": 0.117,
    "warm_median_s": 0.097,
    "cold_warmup_ratio": 1.21,
    "variance_s": 0.01
  }
}
```

## int4 Results (7/7 passed)

### throughput_vs_length [PASS]

```json
{
  "10": {
    "non_quantized_tps": 218.2,
    "quantized_tps": 275.8,
    "speedup": 1.26
  },
  "20": {
    "non_quantized_tps": 439.3,
    "quantized_tps": 292.9,
    "speedup": 0.67
  },
  "50": {
    "non_quantized_tps": 765.2,
    "quantized_tps": 417.5,
    "speedup": 0.55
  }
}
```

### throughput_vs_prompt [PASS]

```json
{
  "10": {
    "prompt_tokens": 10,
    "non_quantized_tps": 1314.0,
    "quantized_tps": 487.0,
    "speedup": 0.37,
    "non_quantized_total_s": 0.038,
    "quantized_total_s": 0.103
  },
  "50": {
    "prompt_tokens": 50,
    "non_quantized_tps": 324.8,
    "quantized_tps": 393.7,
    "speedup": 1.21,
    "non_quantized_total_s": 0.154,
    "quantized_total_s": 0.127
  },
  "100": {
    "prompt_tokens": 100,
    "non_quantized_tps": 210.7,
    "quantized_tps": 274.0,
    "speedup": 1.3,
    "non_quantized_total_s": 0.237,
    "quantized_total_s": 0.182
  }
}
```

### temperature_impact [PASS]

```json
{
  "temp=0.0": {
    "non_quantized_tps": 469.0,
    "quantized_tps": 398.8,
    "speedup": 0.85
  },
  "temp=0.5": {
    "non_quantized_tps": 293.9,
    "quantized_tps": 398.2,
    "speedup": 1.35
  },
  "temp=1.0": {
    "non_quantized_tps": 492.9,
    "quantized_tps": 392.1,
    "speedup": 0.8
  }
}
```

### regression [PASS]

```json
{
  "non_quantized_tps": 547.4,
  "quantized_tps": 393.5,
  "speedup": 0.72
}
```

### memory_usage [PASS]

```json
{
  "non_quantized_rss_mb": 256,
  "quantized_rss_mb": 256,
  "rss_delta_mb": 0,
  "non_quantized_weight_mb": 4.1,
  "quantized_weight_mb": 0.5,
  "weight_compression": 8.0,
  "quantized_layers": 29
}
```

### quality_degradation [PASS]

```json
{
  "per_prompt": {
    "0": {
      "prompt": "The capital of France is",
      "logit_cosine": 0.9262,
      "token_agreement": 0.0,
      "nq_perplexity": 377.48,
      "q_perplexity": 401.34,
      "nq_tokens": 50,
      "q_tokens": 50
    },
    "1": {
      "prompt": "def fibonacci(n):",
      "logit_cosine": 0.895,
      "token_agreement": 0.0,
      "nq_perplexity": 319.67,
      "q_perplexity": 349.84,
      "nq_tokens": 50,
      "q_tokens": 50
    },
    "2": {
      "prompt": "Once upon a time",
      "logit_cosine": 0.9564,
      "token_agreement": 0.0,
      "nq_perplexity": 607.76,
      "q_perplexity": 729.46,
      "nq_tokens": 50,
      "q_tokens": 50
    },
    "3": {
      "prompt": "The quick brown fox",
      "logit_cosine": 0.9057,
      "token_agreement": 0.0,
      "nq_perplexity": 306.32,
      "q_perplexity": 316.63,
      "nq_tokens": 50,
      "q_tokens": 50
    }
  },
  "avg_logit_cosine": 0.9208,
  "avg_token_agreement": 0.0,
  "nq_perplexity": 370.47,
  "q_perplexity": 371.64,
  "perplexity_ratio": 1.003,
  "ppl_passage_tokens": 534,
  "short_prompt_nq_perplexity": 402.81,
  "short_prompt_q_perplexity": 449.32
}
```

### cold_vs_warm [PASS]

```json
{
  "n_runs": 5,
  "non_quantized": {
    "cold_s": 0.079,
    "warm_median_s": 0.082,
    "cold_warmup_ratio": 0.96,
    "variance_s": 0.029
  },
  "quantized": {
    "cold_s": 0.121,
    "warm_median_s": 0.12,
    "cold_warmup_ratio": 1.01,
    "variance_s": 0.003
  }
}
```

## Precision Comparison

| Metric | int8 | int4 |
|--------|------|------ |
| Gen geomean | 1.35x | 0.77x |
| Prompt geomean | 1.47x | 0.83x |
| Temp geomean | 1.44x | 0.97x |
| Weight compression | 4.0x | 8.0x |
| Logit cosine | 0.9996 | 0.9208 |
| Token agreement | 0.01 | 0.0 |
| PPL ratio (Q/NQ) | 1.00 | 1.00 |
| Cold start (s) | 0.121s | 0.079s |
| Warm median (s) | 0.087s | 0.082s |
| Tests passed | 7/7 | 7/7 |

## Recommendations

Best precision per model (score = 0.4*quality + 0.4*compression + 0.2*speed, metrics normalized to the best candidate).

| Model | Precision | Score | Logit cosine | Compression | Gen geomean |
|-------|-----------|-------|--------------|-------------|-------------|
| tiny | int4 | 0.883 | 0.9208 | 8.0x | 0.7743x |

---

## Real-Model Results — Qwen/Qwen2.5-0.5B-Instruct (2026-08-05)

Cached 500M model (28 layers / 169 weight tensors via `model.slnc`), same 7-test
benchmark with `--model Qwen/Qwen2.5-0.5B-Instruct --bits 8,4 --quick`. The
quantized forward path uses the AVX2 C kernels with per-channel scales for both
int8 and int4. The headline perplexity ratio is scored over a fixed 94-token
passage (single teacher-forced forward per model); short-prompt perplexity is
reported per prompt.

| Metric | int8 | int4 |
|--------|------|------|
| Gen geomean | 3.15x | 3.72x |
| Prompt geomean | 3.01x | 4.48x |
| Temp geomean | 4.74x | 4.72x |
| Regression speedup | 10.76x | 4.52x |
| Weight compression | 4.0x (1884→471 MB) | 8.0x (1884→236 MB) |
| Logit cosine | 0.9868 (floor 0.95) | 0.8366 (floor 0.85) |
| Token agreement | 7% | 2.5% |
| **PPL ratio (Q/NQ, 94-token passage)** | **1.02** | **2.58** |
| Cold start (generation, s) | NQ 18.0 / Q 3.0 | NQ 16.6 / Q 3.3 |
| Tests passed | **7/7** | 6/7 |
| **Recommended** | **int8** | — |

### Quality detail (per prompt)

| Precision | Prompt | Logit cosine | NQ PPL | Q PPL |
|-----------|--------|--------------|--------|-------|
| int8 | "The capital of France is" | 0.9941 | 43.5 | 87.7 |
| int8 | "def fibonacci(n):" | 0.9777 | 16.4 | 33.4 |
| int8 | "Once upon a time" | 0.9893 | 10.8 | 118.4 |
| int8 | "The quick brown fox" | 0.9860 | 8.2 | 51.5 |
| int4 | "The capital of France is" | 0.9280 | 43.5 | 104.3 |
| int4 | "def fibonacci(n):" | 0.7270 | 16.4 | 557.3 |
| int4 | "Once upon a time" | 0.8635 | 10.8 | 480.9 |
| int4 | "The quick brown fox" | 0.8277 | 8.2 | 2072.0 |

### Verdict

- **int8 is the correct precision for Qwen2.5-0.5B** — 7/7 PASS, cosine 0.987,
  ppl ratio 1.02; matches the plan guidance (under 1B params → int8).
- **int4 fails quality on this model** — average cosine 0.8366 is below the 0.85
  floor and the ppl ratio 2.58 exceeds 1.5. Round-to-nearest per-channel int4 is
  too lossy at 500M scale; a calibrated scheme (GPTQ/AWQ-style) would be needed.
- Per-token ppl on the 4-6 token prompts is sampling-noise dominated (int4 Q ppl
  ranges 104-2072 across prompts), which is why the headline ratio is scored on
  the fixed longer passage.

---

## Legacy GPT-2 Run (2026-07-13, macOS, superseded)

Prior real-model run preserved for reference:

| Metric | int4 | int8 |
|--------|------|------|
| Speedup | 1.2-1.6x | 1.7-2.65x |
| Weight Compression | 8.0x (579→72 MB) | 4.0x (579→145 MB) |
| Quality (Token Agreement) | 0% | 41% |
| Tests Passed | 6/7 (int4 quality failed) | 6/7 |
| Recommended | No (GPT-2) | Yes |

---

## Key Insights

1. **Int8 preserves the logit distribution on both fixtures** — tiny cosine 0.9996 /
   ppl ratio 1.00; Qwen 0.9868 / 1.02. Int8 is the safe default for <1B models.
2. **Int4 clears the quality floor only on the tiny fixture** — cosine 0.9208
   (floor 0.85) with per-channel scaling; on Qwen-0.5B it fails (0.8366, ppl 2.58).
3. **Quantized inference is faster than the float32 path on the real model** —
   the AVX2 C kernels deliver 3.15x (int8) / 3.72x (int4) generation speedup on
   Qwen-0.5B; int4 is the fastest at the cost of quality.
4. **Weight compression is the headline win** — 4.0x int8 / 8.0x int4 packed
   weights on both tiny (4.1 MB) and Qwen (1884 MB) scales.
5. **Perplexity must be scored over a longer passage** — the 4-6 token prompts
   give sampling-noise-dominated ratios (int4 Q ppl 104-2072 across prompts); the
   fixed 94-token passage yields stable int8 1.02 / int4 2.58. Fixed in the
   benchmark's headline metric (2026-08-05).
6. **Log-softmax math bug fixed** — the step-14 perplexity metric previously used
   an unstable formula that could produce ppl below 1.0; corrected to
   `m - logsumexp(m)` with `m = logits - max` (2026-08-05).
7. **Int4 quantization was per-tensor until 2026-08-05** — outlier rows dominated
   the scale at 4 bits (tiny cosine 0.8524, Qwen logits broken at -0.40). Per-row
   scaling raised tiny cosine to 0.9208 and Qwen to 0.928 on a short prompt.
8. **Guidance holds**: <1B → int8; ≥1B → evaluate int4 with a calibrated scheme
   (round-to-nearest per-channel is too lossy at 500M).

---

## Usage

```bash
# Default: int8 tiny-fixture benchmark
python scripts/benchmark_quantization.py --quick

# Multi-precision comparison + recommendation
python scripts/benchmark_quantization.py --quick --bits 8,4

# Full benchmark
python scripts/benchmark_quantization.py --bits 8

# Real cached model (e.g. Qwen2.5-0.5B-Instruct)
python scripts/benchmark_quantization.py --model Qwen/Qwen2.5-0.5B-Instruct --bits 8,4

# CI gate: memory + quality only, exit 0/1
python scripts/benchmark_quantization.py --validate --bits 8,4
python scripts/benchmark_quantization.py --validate --model Qwen/Qwen2.5-0.5B-Instruct --bits 8

# Baseline regression check (exits 1 on regression)
python scripts/benchmark_quantization.py --bits 8,4 --baseline quantization_baseline.json

# CI gate + baseline in one command (validate subset never false-fails 'passed')
python scripts/benchmark_quantization.py --validate --bits 8,4 --baseline quantization_baseline.json

# Enforced automatically on every push by the reusable-ci-core.yml
# 'quantization-gate' job (numpy-only, deterministic tiny fixture, no downloads).

# JSON / markdown report / CSV export / per-layer stats
python scripts/benchmark_quantization.py --quick --json
python scripts/benchmark_quantization.py --quick --report
python scripts/benchmark_quantization.py --quick --csv
python scripts/benchmark_quantization.py --quick --per-layer
```

---

## Configuration

```bash
# Server config (apps/api/server/config.py)
quantize_slonet: bool = True   # int8 enabled by default
quant_bits: int = 8
quant_mode: str = "symmetric"

# Environment variables
SLO_QUANTIZE=1          # Enable quantization
SLO_QUANT_BITS=8        # 8 or 4
SLO_QUANT_MODE=symmetric
```

---

## Files

- `scripts/benchmark_quantization.py` — Benchmark script (7 tests; `--bits 8|4|8,4`, `--models`, `--quick`, `--json`, `--report`, `--validate`, `--per-layer`, `--csv`, `--baseline`)
- `quantization_baseline.json` — Committed regression baseline (tiny int8/int4 + Qwen int8 headline metrics; created by `--baseline`, checked by `--validate --baseline`)
- `.github/workflows/reusable-ci-core.yml` — `quantization-gate` job runs `--validate --bits 8,4 --baseline` on every push
- `packages/core-py/tests/test_quantization_benchmark.py` — Synthetic-weight unit tests
- `packages/core-py/tests/test_quantization_benchmark_e2e.py` — End-to-end suite (74 e2e tests + 5 baseline-comparison unit tests)
- `docs/features/QUANTIZATION_BENCHMARK_PLAN.md` — Test plan (steps 1-14, complete)
- `docs/features/QUANTIZATION_BENCHMARK_REPORT.md` — This report

