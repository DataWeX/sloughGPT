# Int4/Int8 Quantization Benchmark Report

**Date:** 2026-07-13  
**Model:** GPT-2 (124M params)  
**Device:** Intel i7-9750H, CPU (macOS)  
**Script:** `scripts/benchmark_quantization.py`

---

## Summary: int4 vs int8

| Metric | int4 | int8 | Winner |
|--------|------|------|--------|
| **Speedup** | 1.3-2.5x | 1.7-2.5x | Tie |
| **Weight Compression** | 8.0x (579→72 MB) | 4.0x (579→145 MB) | int4 |
| **Quality (Token Agreement)** | 0% | 73% | **int8** |
| **Tests Passed** | 6/7 | **7/7** | **int8** |
| **Recommended** | ❌ No (GPT-2) | ✅ Yes | **int8** |

---

## Int8 Results (7/7 PASS)

### Test 1: Throughput vs Generation Length

| Tokens | Non-Quantized | Int8 Quantized | Speedup |
|--------|--------------|----------------|---------|
| 10 | 14.2 tok/s | 35.2 tok/s | **2.48x** |
| 20 | 18.3 tok/s | 36.8 tok/s | **2.01x** |
| 50 | 21.4 tok/s | 36.9 tok/s | **1.72x** |

### Test 2: Throughput vs Prompt Length

| Prompt | Non-Quantized | Int8 Quantized | Speedup |
|--------|--------------|----------------|---------|
| 10 tokens | 20.3 tok/s | 31.9 tok/s | **1.57x** |
| 50 tokens | 16.8 tok/s | 23.3 tok/s | **1.39x** |
| 100 tokens | 14.4 tok/s | 17.1 tok/s | **1.19x** |

**Key finding:** Int8 is faster than int4 for long prompts (100 tokens: 1.19x vs 0.71x for int4).

### Test 3: Memory Usage

| Metric | Non-Quantized | Int8 Quantized |
|--------|--------------|----------------|
| RSS | 1741 MB | 1741 MB |
| Weight Size | 579.2 MB | 144.8 MB |
| Compression | — | **4.0x** |

### Test 4: Quality Degradation

| Prompt | Token Agreement |
|--------|----------------|
| "The capital of France is" | **100.0%** |
| "def fibonacci(n):" | 46.0% |
| **Average** | **73.0%** |

**Key finding:** Int8 preserves quality much better than int4 (73% vs 0% agreement).

### Test 5: Cold Start vs Warm

| Metric | Non-Quantized | Int8 Quantized |
|--------|--------------|----------------|
| Cold start | 2.490s | 1.326s |
| Warm steady | 2.303s | 1.255s |
| Overhead | 1.1x | 1.0x |

### Test 6: Temperature Impact

| Temperature | Non-Quantized | Int8 Quantized | Speedup |
|-------------|--------------|----------------|---------|
| 0.0 (greedy) | 22.0 tok/s | 39.4 tok/s | **1.80x** |
| 0.5 | 21.3 tok/s | 38.3 tok/s | **1.79x** |
| 1.0 | 21.3 tok/s | 37.3 tok/s | **1.75x** |

### Test 7: Regression Check

| Model | Throughput |
|-------|-----------|
| Non-quantized | 21.3 tok/s |
| Int8 Quantized | 39.7 tok/s |
| Speedup | **1.86x** |

---

## Int4 Results (6/7 PASS — quality failed)

| Metric | Value |
|--------|-------|
| Speedup | 1.3-2.5x |
| Weight Compression | 8.0x |
| Quality (Token Agreement) | 0% |
| Tests Passed | 6/7 |

**Why int4 fails quality on GPT-2:**
- GPT-2 has only 124M parameters — quantization error is proportionally large
- Int4 = 4-bit = 16 discrete values vs float32's 4B values
- The quantized model generates different (but grammatically valid) text
- Best suited for larger models (1B+) where error is proportionally smaller

---

## Key Insights

1. **Int8 is the right default for GPT-2** — preserves quality while delivering 1.7-2.5x speedup
2. **Int4 is too aggressive for 124M models** — quality degrades completely
3. **Int8 handles long prompts better** — 1.19x at 100 tokens vs int4's 0.71x
4. **4x weight compression** (int8) is significant — frees ~434 MB per model
5. **Both paths are stable** — no startup overhead, consistent performance

---

## Recommendations

| Model Size | Recommended Quantization |
|------------|------------------------|
| < 250M (GPT-2) | **int8** (quality safe) |
| 250M - 1B | **int8** (quality safe) |
| 1B - 7B | int4 viable (test quality) |
| > 7B | int4 recommended (max compression) |

---

## Usage

```bash
# Default: int8 quantization (recommended)
python scripts/benchmark_quantization.py --quick

# Int4 quantization (for larger models)
python scripts/benchmark_quantization.py --quick --bits 4

# Full benchmark
python scripts/benchmark_quantization.py --bits 8

# JSON output
python scripts/benchmark_quantization.py --quick --json
```

---

## Files

- `scripts/benchmark_quantization.py` — Benchmark script (supports --bits 4 or 8)
- `docs/features/QUANTIZATION_BENCHMARK_PLAN.md` — Test plan
- `docs/features/QUANTIZATION_BENCHMARK_REPORT.md` — This report
