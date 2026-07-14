# Int4 Quantization Benchmark Plan

**Status:** Ready to implement  
**Created:** 2026-07-13  
**Purpose:** Thoroughly benchmark the int4 quantized `generate_numpy` path vs non-quantized baseline

---

## Context

We wired int4 quantization into `generate_numpy()` in `slonet.py`. Initial benchmarks showed:
- Non-quantized: 23.5 tok/s (50 tokens)
- Int4 quantized: 35.9 tok/s (50 tokens)
- Speedup: 1.53x

These were quick 3-run tests. We need thorough benchmarking before declaring this production-ready.

---

## What to Benchmark

### 1. Throughput vs Generation Length

**Why:** Quantization overhead is fixed per-step. Longer generations amortize the overhead better.

**Test:** Generate 10, 20, 50, 100, 200 tokens. Measure tok/s for each.

**Expected:** Quantized advantage grows with length (fixed overhead amortized).

### 2. Throughput vs Prompt Length

**Why:** First step processes the full prompt. Quantized QKV may be slower for long prompts.

**Test:** Prompt lengths 10, 50, 100, 200, 500 tokens. Generate 50 tokens. Measure total time and per-step time.

**Expected:** Quantized slower for prompt processing, faster for generation.

### 3. Memory Usage

**Why:** Int4 should use ~8x less weight memory. But does it actually reduce RSS?

**Test:** Measure RSS before/after loading quantized model. Compare peak memory during generation.

**Method:** `psutil.Process().memory_info().rss` at key points.

### 4. Quality Degradation

**Why:** Int4 quantization introduces error. How much quality do we lose?

**Test:** Generate 200 tokens from 10 prompts. Compare:
- Token-level agreement (% identical tokens)
- Cosine similarity of logit vectors
- Perplexity on held-out text
- Human-readable quality (side-by-side output)

### 5. Steady-State vs Cold Start

**Why:** First call may include JIT compilation, cache warming, etc.

**Test:** 20 runs. Discard first 3 as warmup. Report:
- Cold start time (first run)
- Warm steady-state (median of remaining)
- Variance (std dev)

### 6. Temperature Impact

**Why:** Greedy path (temp=0) has argmax fast path. Sampling path (temp>0) has different overhead.

**Test:** Temperature 0, 0.5, 1.0. Measure tok/s for each.

**Expected:** Greedy slightly faster due to argmax fast path.

### 7. Regression Check

**Why:** Ensure quantized path doesn't slow down non-quantized models.

**Test:** Run generate_numpy on a non-quantized model. Verify same performance as before.

### 8. Multi-Model Comparison

**Why:** Different model sizes may show different quantization benefits.

**Test:** GPT-2 (124M), and optionally a smaller custom model if available.

---

## Implementation Plan

### New File: `scripts/benchmark_quantization.py`

```python
"""
Int4 Quantization Benchmark — Comprehensive comparison of quantized vs non-quantized generate_numpy.

Tests:
  1. Throughput vs generation length (10-200 tokens)
  2. Throughput vs prompt length (10-500 tokens)
  3. Memory usage (RSS delta)
  4. Quality degradation (token agreement, logit cosine similarity)
  5. Cold start vs steady state (20 runs, warmup analysis)
  6. Temperature impact (greedy vs sampling)
  7. Regression check (non-quantized model unaffected)

Usage:
    python scripts/benchmark_quantization.py                    # All tests, GPT-2
    python scripts/benchmark_quantization.py --model gpt2       # Specific model
    python scripts/benchmark_quantization.py --quick            # Reduced runs
    python scripts/benchmark_quantization.py --json             # Machine-readable output
"""
```

**Structure:**
```python
@dataclass
class BenchmarkResult:
    test_name: str
    metrics: Dict[str, Any]
    passed: bool
    details: str

class QuantizationBenchmark:
    def __init__(self, model_name: str = "gpt2", quick: bool = False):
        self.model_name = model_name
        self.quick = quick
        self.results: List[BenchmarkResult] = []
        
    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmark tests."""
        self.results.append(self.test_throughput_vs_length())
        self.results.append(self.test_throughput_vs_prompt())
        self.results.append(self.test_memory_usage())
        self.results.append(self.test_quality_degradation())
        self.results.append(self.test_cold_vs_warm())
        self.results.append(self.test_temperature_impact())
        self.results.append(self.test_regression())
        return self.results
        
    def test_throughput_vs_length(self) -> BenchmarkResult:
        """Test 1: Generate different lengths, measure tok/s."""
        
    def test_throughput_vs_prompt(self) -> BenchmarkResult:
        """Test 2: Different prompt lengths, measure generation time."""
        
    def test_memory_usage(self) -> BenchmarkResult:
        """Test 3: Measure RSS delta for quantized vs non-quantized."""
        
    def test_quality_degradation(self) -> BenchmarkResult:
        """Test 4: Compare outputs, logits, perplexity."""
        
    def test_cold_vs_warm(self) -> BenchmarkResult:
        """Test 5: 20 runs, analyze cold start vs steady state."""
        
    def test_temperature_impact(self) -> BenchmarkResult:
        """Test 6: Greedy vs sampling throughput."""
        
    def test_regression(self) -> BenchmarkResult:
        """Test 7: Non-quantized model performance unchanged."""
        
    def print_report(self):
        """Print human-readable report."""
        
    def to_json(self) -> str:
        """Machine-readable output."""
```

**Key implementation details:**

1. **Model loading:** Load GPT-2 via `SloNetChatProvider('gpt2')`, get `provider._model`
2. **Quantization:** Use `quantize_state_dict()` + manual wiring (same as our test)
3. **Timing:** `time.perf_counter()` with 3 warmup runs, 5 measured runs, report median
4. **Memory:** `psutil.Process().memory_info().rss` at start, after load, during generation
5. **Quality:** Compare `out1` vs `out2` token-by-token, compute logit cosine similarity
6. **Regression:** Run non-quantized model, verify same tok/s as baseline

---

## Expected Results Table

| Test | Non-Quantized | Quantized | Target |
|------|--------------|-----------|--------|
| Throughput (50 tok) | 23.5 tok/s | 35.9 tok/s | >1.3x speedup |
| Throughput (200 tok) | ~20 tok/s | ~38 tok/s | >1.5x speedup |
| Memory (RSS delta) | ~548 MB | ~120 MB | >4x reduction |
| Quality (token agreement) | 100% | >70% | Acceptable |
| Cold start | ~15s | ~8s | <2x improvement |
| Steady state | 23.5 tok/s | 35.9 tok/s | >1.3x |
| Greedy vs sampling | ~22 tok/s | ~34 tok/s | <10% difference |

---

## Output Format

### Human-readable (default)
```
=== Int4 Quantization Benchmark ===
Model: gpt2 (124M params)
Device: CPU (Intel i7-9750H)

Test 1: Throughput vs Generation Length
  Length    Non-Quantized    Quantized    Speedup
  10 tok    28.3 tok/s       42.1 tok/s   1.49x
  20 tok    25.1 tok/s       38.7 tok/s   1.54x
  50 tok    23.5 tok/s       35.9 tok/s   1.53x
  100 tok   21.2 tok/s       34.1 tok/s   1.61x
  200 tok   19.8 tok/s       33.2 tok/s   1.68x

Test 2: Throughput vs Prompt Length
  Prompt    Process Time    Gen Time (50 tok)
  10 tok    0.12s           2.15s
  50 tok    0.45s           2.18s
  100 tok   0.89s           2.21s
  200 tok   1.78s           2.25s

Test 3: Memory Usage
  Non-Quantized RSS: 548 MB
  Quantized RSS:     120 MB
  Savings:           428 MB (78%)

Test 4: Quality Degradation
  Token agreement:  78.3%
  Logit cosine:     0.942
  Perplexity delta: +12.3%

Test 5: Cold vs Warm
  Cold start:   8.2s (first run)
  Warm steady:  2.8s (median of runs 4-20)
  Variance:     0.3s (std dev)

Test 6: Temperature Impact
  Greedy (temp=0):     35.9 tok/s
  Sampling (temp=0.5): 34.2 tok/s
  Sampling (temp=1.0): 33.8 tok/s

Test 7: Regression Check
  Non-quantized tok/s: 23.4 (baseline: 23.5) — PASS

Overall: 7/7 tests passed
```

### Machine-readable (--json)
```json
{
  "model": "gpt2",
  "device": "cpu",
  "timestamp": "2026-07-13T12:00:00Z",
  "results": [
    {
      "test": "throughput_vs_length",
      "passed": true,
      "metrics": {
        "10_tok": {"non_quantized": 28.3, "quantized": 42.1, "speedup": 1.49},
        "50_tok": {"non_quantized": 23.5, "quantized": 35.9, "speedup": 1.53},
        "200_tok": {"non_quantized": 19.8, "quantized": 33.2, "speedup": 1.68}
      }
    }
  ]
}
```

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/benchmark_quantization.py` | CREATE | Main benchmark script |
| `scripts/benchmark_quantization_report.md` | CREATE (auto) | Generated report |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| All 7 tests pass | 7/7 |
| Speedup at 50 tok | >1.3x |
| Memory reduction | >4x |
| Quality (token agreement) | >70% |
| Regression check | Within 5% of baseline |
| Report generation | Human-readable + JSON |

---

## Execution Order

1. Create `scripts/benchmark_quantization.py` with all 7 tests
2. Run with `--quick` first to verify it works
3. Run full benchmark
4. Review results
5. If any test fails, investigate and fix
6. Generate final report
