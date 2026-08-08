"""
End-to-end tests for scripts/benchmark_quantization.py.

Runs the real tiny-model benchmark in-process (no downloads, deterministic)
and asserts the deterministic gates:
  - all 7 tests pass, in the stable run order
  - weight compression (int8 ~4x, int4 ~8x) on the real SloLinear layers
  - logit-cosine quality floors (int8 >= 0.95, int4 >= 0.85)
  - seeded tiny-model determinism (identical weights and logits)
  - JSON report shape (model/quick/bits/tiny/results/passed/total)
  - tiny-scale speed gate is informational within a sanity band
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import benchmark_quantization as bq  # noqa: E402

from domains.infrastructure.quantization import walk_slo_linears  # noqa: E402

EXPECTED_TEST_NAMES = [
    "throughput_vs_length",
    "throughput_vs_prompt",
    "temperature_impact",
    "regression",
    "memory_usage",
    "quality_degradation",
    "cold_vs_warm",
]


def _quiet(fn):
    """Run callable with stdout suppressed (benchmark prints its report)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def _quantized_weight_bytes(model):
    """Sum of packed quantized weight bytes for the quantized linear layers."""
    total = 0
    for m in walk_slo_linears(model).values():
        info = getattr(m, "_quant_info", None)
        if info is not None and info.is_quantized:
            total += info.array.nbytes
        else:
            total += m.weight.data.nbytes
    return total


@pytest.fixture(scope="module")
def int8_bench():
    """One real quick int8 run shared by all int8 assertions."""
    bench = bq.QuantizationBenchmark(tiny=True, quick=True, bits=8)
    _quiet(bench.run_all)
    return bench


def _result(bench, name):
    return next(r for r in bench.results if r.name == name)


class TestEndToEndInt8:
    """Full in-process quick int8 benchmark run."""

    def test_all_tests_pass(self, int8_bench):
        assert [r.name for r in int8_bench.results] == EXPECTED_TEST_NAMES
        assert all(r.passed for r in int8_bench.results)
        assert len(int8_bench.results) == 7

    def test_int8_weight_compression(self, int8_bench):
        metrics = _result(int8_bench, "memory_usage").metrics
        compression = metrics["weight_compression"]
        assert 3.5 <= compression <= 5.5, f"int8 compression {compression}x != ~4x"
        assert metrics["quantized_layers"] > 0

    def test_int8_quality_floor(self, int8_bench):
        metrics = _result(int8_bench, "quality_degradation").metrics
        assert metrics["avg_logit_cosine"] >= 0.95
        assert metrics["avg_token_agreement"] >= 0.0

    def test_throughput_results_carry_metrics(self, int8_bench):
        metrics = _result(int8_bench, "throughput_vs_length").metrics
        assert len(metrics) == 3  # quick mode lengths
        for length, point in metrics.items():
            assert point["speedup"] > 0
            assert point["non_quantized_tps"] > 0
            assert point["quantized_tps"] > 0

    def test_json_report_shape(self, int8_bench):
        data = json.loads(int8_bench.to_json())
        assert data["tiny"] is True
        assert data["quick"] is True
        assert data["bits"] == 8
        assert data["passed"] == data["total"] == 7
        names = [r["name"] for r in data["results"]]
        assert names == EXPECTED_TEST_NAMES
        assert all(r["passed"] for r in data["results"])
        assert all("metrics" in r and "details" in r for r in data["results"])

    def test_json_header_contract(self, int8_bench):
        """JSON run includes device and timestamp fields per the output contract."""
        data = json.loads(int8_bench.to_json())
        assert isinstance(data["device"], str) and data["device"]
        assert isinstance(data["timestamp"], str) and data["timestamp"]
        assert data["timestamp"].endswith("Z")

    def test_markdown_report(self, int8_bench, tmp_path):
        path = int8_bench.write_report(tmp_path / "report.md")
        assert path.exists()
        text = path.read_text()
        assert text.startswith("# Int8 Quantization Benchmark Report")
        assert "- **Model**: tiny (in-process)" in text
        assert "- **Bits**: 8" in text
        assert "- **Result**: 7/7 tests passed" in text
        for name in EXPECTED_TEST_NAMES:
            assert f"## {name}" in text
        compression = _result(int8_bench, "memory_usage").metrics["weight_compression"]
        assert f"Weight compression: {compression}x" in text
        assert "Avg logit cosine" in text
        assert "## Notes" in text


class TestEndToEndInt4:
    """int4 path: quantize the real tiny model and assert compression/quality."""

    def test_int4_compression_and_quality(self):
        bench4 = bq.QuantizationBenchmark(tiny=True, quick=True, bits=4)
        bench4.model, _ = _quiet(bench4._load_model)
        model = bench4.model
        bench4.quant_model = _quiet(lambda: bench4._quantize_model(model))

        nq_bytes = sum(m.weight.data.nbytes for m in walk_slo_linears(model).values())
        q_bytes = _quantized_weight_bytes(bench4.quant_model)
        compression = nq_bytes / max(q_bytes, 1)
        assert 6.0 <= compression <= 10.0, f"int4 compression {compression:.1f}x != ~8x"

        prompt = bench4._encode("The capital of France is")
        cosine = bench4._logit_cosine(model, bench4.quant_model, prompt)
        assert cosine >= 0.85, f"int4 logit cosine {cosine:.4f} below 0.85 floor"


class TestMultiPrecision:
    """Test --bits 8,4 multi-precision comparison mode."""

    def test_comparison_table_structure(self):
        """_comparison_table produces valid markdown with both precision rows."""
        run8 = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=8).to_json())
        run4 = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=4).to_json())
        table = bq._comparison_table([run8, run4])
        assert "## Precision Comparison" in table
        assert "int8" in table
        assert "int4" in table
        assert "Gen geomean" in table
        assert "Weight compression" in table
        assert "Logit cosine" in table
        assert "Token agreement" in table
        assert "Cold start (s)" in table
        assert "Warm median (s)" in table
        assert "Tests passed" in table

    def test_comparison_json_structure(self):
        """_comparison_json produces a dict keyed by precision."""
        run8 = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=8).to_json())
        run4 = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=4).to_json())
        comp = bq._comparison_json([run8, run4])
        assert "int8" in comp
        assert "int4" in comp
        assert "bits" in comp["int8"]
        assert "passed" in comp["int8"]
        assert "gen_geomean" in comp["int8"]
        assert "weight_compression" in comp["int8"]
        assert "logit_cosine" in comp["int8"]
        assert "token_agreement" in comp["int8"]
        assert "cold_start_s" in comp["int8"]
        assert "warm_median_s" in comp["int8"]

    def test_parse_bits_single(self):
        """_parse_bits handles a single value."""
        assert bq._parse_bits("8") == [8]
        assert bq._parse_bits("4") == [4]

    def test_parse_bits_multi(self):
        """_parse_bits handles comma-separated values."""
        assert bq._parse_bits("8,4") == [8, 4]
        assert bq._parse_bits("4,8") == [4, 8]
        assert bq._parse_bits(" 8 , 4 ") == [8, 4]

    def test_parse_bits_invalid(self):
        """_parse_bits rejects invalid input."""
        with pytest.raises(Exception):
            bq._parse_bits("abc")

    def test_parse_models_single(self):
        """_parse_models handles a single model id."""
        assert bq._parse_models("tiny") == ["tiny"]

    def test_parse_models_multi(self):
        """_parse_models handles comma-separated ids and strips whitespace."""
        assert bq._parse_models("tiny,Qwen/Qwen2.5") == ["tiny", "Qwen/Qwen2.5"]
        assert bq._parse_models("  tiny , Qwen  ") == ["tiny", "Qwen"]

    def test_parse_models_empty_rejected(self):
        """_parse_models rejects empty input."""
        with pytest.raises(ValueError):
            bq._parse_models("")
        with pytest.raises(ValueError):
            bq._parse_models("  ,  ")

    def test_model_comparison_table_structure(self):
        """_model_comparison_table produces a markdown table per model."""
        run_a = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=8).to_json())
        run_a["model"] = "tiny"
        run_b = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=8).to_json())
        run_b["model"] = "other"
        table = bq._model_comparison_table([run_a, run_b])
        assert "## Model Comparison" in table
        assert "Model" in table
        assert "tiny" in table
        assert "other" in table
        assert "/ cos" in table
        assert "/ gen" in table

    def test_model_comparison_table_skips_foreign_precision(self):
        """Cells for a model without a given precision render as em-dash."""
        run_a8 = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=8).to_json())
        run_a8["model"] = "tiny"
        run_b4 = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=4).to_json())
        run_b4["model"] = "other"
        table = bq._model_comparison_table([run_a8, run_b4])
        rows = [ln for ln in table.splitlines() if ln.startswith("| ")]
        assert any("—" in ln for ln in rows), "missing cell should render as em-dash"

    def test_model_comparison_table_dedupes_precision_columns(self):
        """Shared precisions across models produce one column each, not duplicates."""
        run_a8 = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=8).to_json())
        run_a8["model"] = "tiny"
        run_a4 = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=4).to_json())
        run_a4["model"] = "tiny"
        run_b8 = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=8).to_json())
        run_b8["model"] = "other"
        run_b4 = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=4).to_json())
        run_b4["model"] = "other"
        table = bq._model_comparison_table([run_a8, run_a4, run_b8, run_b4])
        header = table.splitlines()[2]
        assert header.count("int8") == 1, header
        assert header.count("int4") == 1, header

    def test_model_comparison_json_structure(self):
        """_model_comparison keys by model then precision."""
        run_a = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=8).to_json())
        run_a["model"] = "tiny"
        run_b = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=4).to_json())
        run_b["model"] = "other"
        comp = bq._model_comparison([run_a, run_b])
        assert "tiny" in comp
        assert "other" in comp
        assert "int8" in comp["tiny"]
        assert "int4" in comp["other"]
        assert "weight_compression" in comp["tiny"]["int8"]
        assert "avg_logit_cosine" in comp["tiny"]["int8"]
        assert "avg_token_agreement" in comp["tiny"]["int8"]
        assert "cold_start_s" in comp["tiny"]["int8"]
        assert "warm_median_s" in comp["tiny"]["int8"]

    def test_int8_beats_int4_compression(self):
        """int8 should have ~4x compression, int4 ~8x."""
        bench8 = bq.QuantizationBenchmark(tiny=True, quick=True, bits=8)
        bench8.model, _ = _quiet(bench8._load_model)
        bench8.quant_model = _quiet(lambda: bench8._quantize_model(bench8.model))
        nq8 = sum(m.weight.data.nbytes for m in walk_slo_linears(bench8.model).values())
        q8 = _quantized_weight_bytes(bench8.quant_model)

        bench4 = bq.QuantizationBenchmark(tiny=True, quick=True, bits=4)
        bench4.model = bench8.model
        bench4.quant_model = _quiet(lambda: bench4._quantize_model(bench4.model))
        q4 = _quantized_weight_bytes(bench4.quant_model)

        assert q4 < q8, f"int4 packed ({q4}) should be smaller than int8 ({q8})"
        ratio = q8 / max(q4, 1)
        assert ratio >= 1.5, f"int4 should be ~2x smaller than int8, got {ratio:.2f}x"


class TestValidateMode:
    """Test --validate CI mode."""

    def test_validate_exits_clean(self):
        """Validate mode runs quality + compression only, all pass."""
        bench = bq.QuantizationBenchmark(tiny=True, quick=True, bits=8)
        with contextlib.redirect_stdout(io.StringIO()):
            bench.model, _ = bench._load_model()
            bench.quant_model = bench._quantize_model(bench.model)
            bench.results.append(bench.test_memory_usage())
            bench.results.append(bench.test_quality_degradation())
        assert all(r.passed for r in bench.results)
        assert len(bench.results) == 2

    def test_validate_int4_quality_above_floor(self):
        """Validate mode quality check passes for int4 (cosine >= 0.85)."""
        bench = bq.QuantizationBenchmark(tiny=True, quick=True, bits=4)
        with contextlib.redirect_stdout(io.StringIO()):
            bench.model, _ = bench._load_model()
            bench.quant_model = bench._quantize_model(bench.model)
            bench.results.append(bench.test_quality_degradation())
        assert bench.results[0].passed
        assert bench.results[0].metrics["avg_logit_cosine"] >= 0.85

    def test_validate_json_output(self):
        """Validate + --json produces valid JSON with 2 results."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/benchmark_quantization.py",
             "--validate", "--bits", "8", "--json"],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "runs" in data
        assert len(data["runs"]) == 1
        run = data["runs"][0]
        assert run["passed"] == 2
        assert run["total"] == 2
        names = [r["name"] for r in run["results"]]
        assert "memory_usage" in names
        assert "quality_degradation" in names

    def test_validate_exits_1_on_failure(self):
        """Validate mode exits 1 when quality check fails."""
        import subprocess, sys
        # Force failure by monkeypatching the quality floor to impossibly high
        bench = bq.QuantizationBenchmark(tiny=True, quick=True, bits=4)
        with contextlib.redirect_stdout(io.StringIO()):
            bench.model, _ = bench._load_model()
            bench.quant_model = bench._quantize_model(bench.model)
        # Patch the quality test to use a floor of 1.0 (impossible)
        original_test = bench.test_quality_degradation
        def forced_fail():
            r = original_test()
            r.passed = False
            r.metrics["avg_logit_cosine"] = 0.85
            return r
        bench.test_quality_degradation = forced_fail
        with contextlib.redirect_stdout(io.StringIO()):
            bench.results.append(bench.test_memory_usage())
            bench.results.append(bench.test_quality_degradation())
        n_pass = sum(1 for r in bench.results if r.passed)
        assert n_pass < len(bench.results), "Expected at least one failure"


class TestBaselineCompare:
    """Direct tests for _compare_baselines regression gating."""

    def _make_run(self, passed=7, total=7, tiny=False, cosine=0.95,
                  ppl_ratio=1.0, compression=4.0, tag="tiny:int8"):
        """Build a headline-metrics dict for one model:int key."""
        return {tag: {
            "tiny": tiny, "passed": passed, "total": total,
            "gen_geomean": 1.5, "prompt_geomean": 1.5, "temp_geomean": 1.5,
            "weight_compression": compression,
            "avg_logit_cosine": cosine, "avg_token_agreement": 0.5,
            "perplexity_ratio": ppl_ratio,
            "cold_start_s": 1.0, "warm_median_s": 0.5,
        }}

    def test_validate_subset_does_not_regress_passed(self):
        """A 2-test validate subset must not flag 'passed' vs a 7-test baseline."""
        cur = self._make_run(passed=2, total=2)
        base = self._make_run(passed=7, total=7)
        result = bq._compare_baselines(cur, base)
        assert result["regressions"] == []
        assert result["deltas"]["tiny:int8"]["passed"]["current"] == 2

    def test_full_run_passed_regression_detected(self):
        """Same test set with a dropped pass count must regress."""
        cur = self._make_run(passed=6, total=7)
        base = self._make_run(passed=7, total=7)
        result = bq._compare_baselines(cur, base)
        assert any(r["metric"] == "passed" for r in result["regressions"])

    def test_cosine_regression_detected(self):
        """Cosine below baseline minus tolerance must regress."""
        cur = self._make_run(cosine=0.88)
        base = self._make_run(cosine=0.98)
        result = bq._compare_baselines(cur, base)
        assert any(r["metric"] == "avg_logit_cosine" for r in result["regressions"])

    def test_perplexity_ratio_regression_detected(self):
        """Perplexity ratio (lower is better) beyond +tolerance must regress."""
        cur = self._make_run(ppl_ratio=2.0)
        base = self._make_run(ppl_ratio=1.0)
        result = bq._compare_baselines(cur, base)
        assert any(r["metric"] == "perplexity_ratio" for r in result["regressions"])

    def test_improved_metrics_no_regression(self):
        """Better-than-baseline metrics never regress."""
        cur = self._make_run(passed=7, cosine=0.99, ppl_ratio=0.9, compression=4.2)
        base = self._make_run(passed=7, cosine=0.95, ppl_ratio=1.0, compression=4.0)
        result = bq._compare_baselines(cur, base)
        assert result["regressions"] == []


class TestModelsCli:
    """CLI-level --models multi-model comparison."""

    def _run(self, *args):
        import subprocess, sys
        return subprocess.run(
            [sys.executable, "scripts/benchmark_quantization.py", *args],
            capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parents[3]),
        )

    def test_models_json_has_model_comparison(self):
        """--models --json emits a model_comparison block keyed by model."""
        result = self._run("--models", "tiny,tiny", "--bits", "8,4",
                           "--json", "--quick")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert len(data["runs"]) == 4
        assert set(r["model"] for r in data["runs"]) == {"tiny"}
        assert list(data["model_comparison"].keys()) == ["tiny"]
        assert "int8" in data["model_comparison"]["tiny"]
        assert "int4" in data["model_comparison"]["tiny"]
        assert data["comparison"] is None

    def test_models_validate_json_valid(self):
        """--models --validate --json stays valid JSON."""
        result = self._run("--models", "tiny,tiny", "--bits", "8",
                           "--validate", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert all(r["passed"] == r["total"] for r in data["runs"])

    def test_models_report_contains_comparison(self, tmp_path):
        """--models --report writes a model comparison section."""
        report = tmp_path / "mc.md"
        result = self._run("--models", "tiny,tiny", "--bits", "8,4",
                           "--quick", "--report", str(report))
        assert result.returncode == 0, result.stderr
        text = report.read_text()
        assert "## Model Comparison" in text
        assert "- **Model**: tiny, tiny" in text

    def test_models_mutually_exclusive_with_model(self):
        """--model and --models together are rejected."""
        result = self._run("--model", "tiny", "--models", "tiny", "--quick")
        assert result.returncode != 0
        assert "mutually exclusive" in result.stderr


class TestCsvOutput:
    """--csv export: one row per run."""

    def test_csv_rows_for_each_run(self):
        """Multi-precision run yields one CSV row per run with header."""
        bench8 = bq.QuantizationBenchmark(tiny=True, quick=True, bits=8)
        bench4 = bq.QuantizationBenchmark(tiny=True, quick=True, bits=4)
        _quiet(bench8.run_all)
        _quiet(bench4.run_all)
        text = bq._csv_output([
            json.loads(bench8.to_json()),
            json.loads(bench4.to_json()),
        ])
        import csv
        lines = text.splitlines()
        assert lines[0].startswith("model,bits,quick,tiny,passed,total")
        rows = list(csv.DictReader(lines))
        assert len(rows) == 2
        assert [r["bits"] for r in rows] == ["8", "4"]
        assert all(r["passed"] == "7" and r["total"] == "7" for r in rows)
        assert rows[0]["weight_compression"] == "4.0"
        assert rows[1]["weight_compression"] == "8.0"
        assert all(r["gen_geomean"] for r in rows)
        assert all(r["warm_median_s"] for r in rows)
        assert all(r["avg_token_agreement"] for r in rows)

    def test_csv_contains_model_and_tiny_flags(self):
        """CSV carries model id and tiny flag per row."""
        run = json.loads(bq.QuantizationBenchmark(tiny=True, quick=True, bits=8).to_json())
        run["model"] = "Qwen/Qwen2.5"
        text = bq._csv_output([run])
        import csv
        row = next(csv.DictReader(text.splitlines()))
        assert row["model"] == "Qwen/Qwen2.5"
        assert row["tiny"] == "True"

    def test_csv_cli_writes_file(self, tmp_path):
        """--csv PATH writes a parseable CSV file."""
        import subprocess
        out = tmp_path / "bench.csv"
        result = subprocess.run(
            [sys.executable, "scripts/benchmark_quantization.py",
             "--models", "tiny", "--bits", "8,4", "--quick",
             "--csv", str(out)],
            capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        assert result.returncode == 0, result.stderr
        import csv
        rows = list(csv.DictReader(out.read_text().splitlines()))
        assert len(rows) == 2
        assert {r["bits"] for r in rows} == {"8", "4"}


class TestTinyModelInvariants:
    """Determinism and gate behavior of the benchmark fixtures."""

    def test_seeded_tiny_models_are_identical(self):
        a = bq._create_tiny_model()
        b = bq._create_tiny_model()
        wa = {n: m.weight.data for n, m in walk_slo_linears(a).items()}
        wb = {n: m.weight.data for n, m in walk_slo_linears(b).items()}
        assert set(wa) == set(wb)
        for name in wa:
            np.testing.assert_array_equal(wa[name], wb[name])

        ids = np.array([ord(c) % 256 for c in "hello world"], dtype=np.int64)
        bench = bq.QuantizationBenchmark(tiny=True, quick=True, bits=8)
        la = bench._logits(a, ids)
        lb = bench._logits(b, ids)
        np.testing.assert_array_equal(la, lb)

    def test_tiny_speed_gate_is_informational(self):
        bench = bq.QuantizationBenchmark(tiny=True, quick=True, bits=8)
        ok_high, note = bench._speed_gate(0.5)
        assert ok_high is True
        assert "informational" in note
        ok_low, _ = bench._speed_gate(0.05)
        assert ok_low is False

    def test_quantized_linear_uses_quantized_weights(self, int8_bench):
        quantized = sum(
            1
            for m in walk_slo_linears(int8_bench.quant_model).values()
            if getattr(m, "_quant_info", None) is not None and m._quant_info.is_quantized
        )
        assert quantized > 0


class TestCachedModelDiscovery:
    """Test cached-model listing and missing-model error handling."""

    def test_list_cached_models_returns_list(self):
        """_list_cached_models returns a list of model ids (possibly empty)."""
        cached = bq._list_cached_models()
        assert isinstance(cached, list)
        assert all(isinstance(m, str) and "/" in m for m in cached)

    def test_missing_model_raises_with_cached_hint(self):
        """Loading an uncached model raises a helpful FileNotFoundError."""
        bench = bq.QuantizationBenchmark(
            model_name="no/such-model-12345", tiny=False, bits=8, quick=True
        )
        with pytest.raises(FileNotFoundError) as exc:
            bench._load_model()
        msg = str(exc.value)
        assert "no cached model.slnc" in msg
        assert "Use --model" in msg

    def test_list_cached_models_includes_local_cache(self):
        """The project-local models/hf-cache/hub dir is scanned."""
        cached = bq._list_cached_models()
        if cached:
            # Every returned id must resolve to an existing model.slnc
            from domains.infrastructure.safetensors_loader import _get_model_dir
            for mid in cached:
                assert (_get_model_dir(mid) / "model.slnc").exists(), f"{mid} missing"


class TestWeightCosines:
    """Test per-layer weight-fidelity cosine."""

    def test_int8_weight_cosine_near_one(self, int8_bench):
        """int8 per-layer weight cosine should be >= 0.99."""
        nq_layers = walk_slo_linears(int8_bench.model)
        q_layers = walk_slo_linears(int8_bench.quant_model)
        for name in nq_layers:
            cos = int8_bench._weight_cosine(nq_layers[name], q_layers[name])
            assert cos >= 0.99, f"{name} cosine {cos:.4f}"

    def test_int4_weight_cosine_above_floor(self):
        """int4 per-layer weight cosine should be >= 0.95."""
        bench = bq.QuantizationBenchmark(tiny=True, quick=True, bits=4)
        with contextlib.redirect_stdout(io.StringIO()):
            bench.model, _ = bench._load_model()
            bench.quant_model = bench._quantize_model(bench.model)
        nq_layers = walk_slo_linears(bench.model)
        q_layers = walk_slo_linears(bench.quant_model)
        for name in nq_layers:
            cos = bench._weight_cosine(nq_layers[name], q_layers[name])
            assert cos >= 0.95, f"{name} cosine {cos:.4f}"

    def test_identical_layers_cosine_is_one(self, int8_bench):
        """Comparing a layer with itself yields cosine 1.0."""
        layer_name = next(iter(walk_slo_linears(int8_bench.model)))
        m = walk_slo_linears(int8_bench.model)[layer_name]
        assert int8_bench._weight_cosine(m, m) == 1.0

    def test_quantized_weights_dequantize_to_finite(self, int8_bench):
        """Dequantized weights are finite and shape-match the originals."""
        nq_layers = walk_slo_linears(int8_bench.model)
        q_layers = walk_slo_linears(int8_bench.quant_model)
        for name in nq_layers:
            w_nq = np.asarray(nq_layers[name].weight.data)
            q_info = getattr(q_layers[name], "_quant_info", None)
            if q_info is not None and q_info.is_quantized:
                w_q = q_info.as_float()
                assert w_q.shape == w_nq.shape, f"{name} shape mismatch"
                assert np.all(np.isfinite(w_q)), f"{name} has non-finite values"

    def test_json_per_layer_structure(self):
        """--per-layer --json emits a per_layer dict in the JSON output."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/benchmark_quantization.py",
             "--per-layer", "--bits", "8", "--json", "--quick"],
            capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "per_layer" in data
        pl = data["per_layer"]["int8"]
        assert "layers" in pl
        assert len(pl["layers"]) > 0
        first = pl["layers"][0]
        assert set(first) == {"layer", "fp32_kb", "quant_kb", "ratio", "weight_cosine"}
        assert pl["total_ratio"] >= 3.5
        assert all(e["weight_cosine"] >= 0.99 for e in pl["layers"])

    def test_report_includes_per_layer(self, tmp_path):
        """--report --per-layer writes a markdown table of layer stats."""
        import subprocess
        report = tmp_path / "report.md"
        result = subprocess.run(
            [sys.executable, "scripts/benchmark_quantization.py",
             "--per-layer", "--bits", "8", "--quick",
             "--report", str(report)],
            capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        assert result.returncode == 0
        text = report.read_text()
        assert "## Per-Layer Stats" in text
        assert "### int8" in text
        assert "| Layer | FP32 KB | Q KB | Ratio | Cos |" in text
        assert "| **TOTAL**" in text
        assert "| lm_head |" in text


def _synthetic_run(bits, model, cos=None, comp=None, gen=None):
    """Build a minimal run dict without running the benchmark."""
    return {
        "model": model,
        "bits": bits,
        "passed": 2,
        "total": 2,
        "results": [
            {
                "name": "memory_usage",
                "passed": True,
                "metrics": {"weight_compression": comp},
            },
            {
                "name": "quality_degradation",
                "passed": True,
                "metrics": {"avg_logit_cosine": cos},
            },
            {
                "name": "throughput_vs_length",
                "passed": True,
                "metrics": {
                    "len8": {
                        "speedup": gen,
                        "non_quantized_tps": 10.0,
                        "quantized_tps": 10.0 * gen,
                    }
                },
            },
        ],
    }


def _with_ppl(run, ratio):
    """Add perplexity_ratio to an existing synthetic quality_degradation result."""
    for r in run["results"]:
        if r["name"] == "quality_degradation":
            r["metrics"]["perplexity_ratio"] = ratio
    return run


class TestRecommendations:
    """Best-precision recommendation helpers."""

    def test_metric_helpers_missing_test_return_none(self):
        """_run_metric/_run_geomean_speedup return None for unknown tests."""
        run = _synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)
        assert bq._run_metric(run, "no_such_test", "weight_compression") is None
        assert bq._run_metric(run, "memory_usage", "no_such_key") is None
        assert bq._run_geomean_speedup(run, "no_such_test") is None

    def test_nested_metric_reads_grouped_values(self):
        """_run_nested_metric finds values nested under group dicts."""
        run = {
            "model": "tiny",
            "bits": 8,
            "results": [{
                "name": "cold_vs_warm",
                "metrics": {
                    "n_runs": 5,
                    "non_quantized": {"cold_s": 0.1, "warm_median_s": 0.05},
                    "quantized": {"cold_s": 0.2, "warm_median_s": 0.08},
                },
            }],
        }
        assert bq._run_nested_metric(run, "cold_vs_warm", "cold_s") == 0.1
        assert bq._run_nested_metric(run, "cold_vs_warm", "warm_median_s") == 0.05
        assert bq._run_nested_metric(run, "cold_vs_warm", "no_such_key") is None
        assert bq._run_nested_metric(run, "no_such_test", "cold_s") is None

    def test_nested_metric_prefers_top_level(self):
        """_run_nested_metric prefers a top-level key over grouped values."""
        run = {
            "model": "tiny",
            "bits": 8,
            "results": [{
                "name": "cold_vs_warm",
                "metrics": {
                    "cold_s": 9.9,
                    "non_quantized": {"cold_s": 0.1},
                },
            }],
        }
        assert bq._run_nested_metric(run, "cold_vs_warm", "cold_s") == 9.9

    def test_real_pair_yields_one_recommendation(self):
        """A real int8+int4 tiny pair produces one recommendation for tiny."""
        bench8 = bq.QuantizationBenchmark(tiny=True, quick=True, bits=8)
        bench4 = bq.QuantizationBenchmark(tiny=True, quick=True, bits=4)
        _quiet(bench8.run_all)
        _quiet(bench4.run_all)
        run8 = json.loads(bench8.to_json())
        run8["model"] = "tiny"
        run4 = json.loads(bench4.to_json())
        run4["model"] = "tiny"
        recs = bq._recommendations([run8, run4])
        assert len(recs) == 1
        rec = recs[0]
        assert rec["model"] == "tiny"
        assert {c["bits"] for c in rec["candidates"]} == {8, 4}
        assert all(c["score"] is not None for c in rec["candidates"])
        assert rec["recommended_bits"] in (4, 8)

    def test_floor_failing_candidate_score_is_none(self):
        """A candidate below its quality floor is listed but not scored."""
        run8 = _synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)
        run4 = _synthetic_run(4, "tiny", cos=0.80, comp=8.0, gen=2.0)
        recs = bq._recommendations([run8, run4])
        assert len(recs) == 1
        cands = {c["bits"]: c for c in recs[0]["candidates"]}
        assert cands[4]["qualified"] is False
        assert cands[4]["score"] is None
        assert recs[0]["recommended_bits"] == 8

    def test_single_precision_model_skipped(self):
        """Models with only one run are not scored."""
        run8 = _synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)
        run_other = _synthetic_run(8, "other", cos=0.99, comp=4.0, gen=1.5)
        recs = bq._recommendations([run8, run_other])
        assert recs == []

    def test_recommendation_table_empty_when_no_qualifying_model(self):
        """_recommendation_table returns '' when nothing qualifies."""
        run8 = _synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)
        assert bq._recommendation_table([run8]) == ""

    def test_recommendation_table_renders_rows(self):
        """_recommendation_table renders a markdown section for qualifying models."""
        run8 = _synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)
        run4 = _synthetic_run(4, "tiny", cos=0.90, comp=8.0, gen=2.0)
        table = bq._recommendation_table([run8, run4])
        assert table.startswith("## Recommendations")
        assert "| Model | Precision | Score |" in table
        assert "| tiny | int" in table

    def test_recommendation_json_in_main_output(self):
        """--bits 8,4 --json emits a recommendations block."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/benchmark_quantization.py",
             "--bits", "8,4", "--quick", "--json"],
            capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "recommendations" in data
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["model"] == "tiny"

    def test_recommendations_in_report(self, tmp_path):
        """--report writes a Recommendations section for multi-precision runs."""
        import subprocess
        report = tmp_path / "rec.md"
        result = subprocess.run(
            [sys.executable, "scripts/benchmark_quantization.py",
             "--bits", "8,4", "--quick", "--report", str(report)],
            capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        assert result.returncode == 0, result.stderr
        assert "## Recommendations" in report.read_text()


class TestBaseline:
    """--baseline regression checking."""

    def _run(self, *args):
        import subprocess
        return subprocess.run(
            [sys.executable, "scripts/benchmark_quantization.py", *args],
            capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parents[3]),
        )

    def test_headline_metrics_keyed_by_model_bits(self):
        """_headline_metrics builds a ``model:int<bits>`` key per run."""
        run = _synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)
        current = bq._headline_metrics([run])
        assert "tiny:int8" in current
        entry = current["tiny:int8"]
        assert entry["passed"] == 2 and entry["total"] == 2
        assert entry["weight_compression"] == 4.0
        assert entry["avg_logit_cosine"] == 0.99
        assert entry["gen_geomean"] == 1.5
        assert entry["tiny"] is False

    def test_compare_detects_quality_regression(self):
        """Cos dropping beyond the absolute tolerance is a regression."""
        current = bq._headline_metrics(
            [_synthetic_run(8, "tiny", cos=0.90, comp=4.0, gen=1.5)])
        baseline = bq._headline_metrics(
            [_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)])
        out = bq._compare_baselines(current, baseline)
        assert len(out["regressions"]) == 1
        assert out["regressions"][0]["metric"] == "avg_logit_cosine"
        assert out["deltas"]["tiny:int8"]["avg_logit_cosine"]["regressed"] is True

    def test_compare_tolerates_small_drift(self):
        """Small absolute drift within tolerance is not a regression."""
        current = bq._headline_metrics(
            [_synthetic_run(8, "tiny", cos=0.985, comp=4.0, gen=1.5)])
        baseline = bq._headline_metrics(
            [_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)])
        out = bq._compare_baselines(current, baseline)
        assert out["regressions"] == []

    def test_compare_relative_tolerance_for_speed(self):
        """Throughput falling more than 25% relative is a regression."""
        current = bq._headline_metrics(
            [_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.0)])
        baseline = bq._headline_metrics(
            [_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=2.0)])
        out = bq._compare_baselines(current, baseline)
        assert any(r["metric"] == "gen_geomean" for r in out["regressions"])

    def test_compare_tiny_speed_never_gates(self):
        """Speed drops on the tiny fixture record deltas, never regressions."""
        current = bq._headline_metrics(
            [_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=0.5)])
        current["tiny:int8"]["tiny"] = True
        baseline = bq._headline_metrics(
            [_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)])
        baseline["tiny:int8"]["tiny"] = True
        out = bq._compare_baselines(current, baseline)
        assert out["regressions"] == []
        gen = out["deltas"]["tiny:int8"]["gen_geomean"]
        assert gen["delta"] == -1.0 and gen["regressed"] is False

    def test_compare_passed_must_not_drop(self):
        """A drop in passed tests is always a regression."""
        current = bq._headline_metrics(
            [_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)])
        current["tiny:int8"]["passed"] = 1
        baseline = bq._headline_metrics(
            [_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)])
        out = bq._compare_baselines(current, baseline)
        assert any(r["metric"] == "passed" for r in out["regressions"])

    def test_compare_skips_unknown_keys(self):
        """Runs absent from the baseline are not flagged."""
        current = bq._headline_metrics(
            [_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)])
        baseline = bq._headline_metrics(
            [_synthetic_run(8, "other", cos=0.99, comp=4.0, gen=1.5)])
        out = bq._compare_baselines(current, baseline)
        assert out["regressions"] == []

    def test_compare_cold_warm_informational_only(self):
        """Worse cold/warm latency records a delta but never regresses."""
        current = bq._headline_metrics(
            [_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)])
        baseline = bq._headline_metrics(
            [_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5)])
        current["tiny:int8"]["cold_start_s"] = 5.0
        current["tiny:int8"]["warm_median_s"] = 2.0
        baseline["tiny:int8"]["cold_start_s"] = 1.0
        baseline["tiny:int8"]["warm_median_s"] = 0.5
        out = bq._compare_baselines(current, baseline)
        assert out["regressions"] == []
        assert out["deltas"]["tiny:int8"]["cold_start_s"]["regressed"] is False
        assert out["deltas"]["tiny:int8"]["warm_median_s"]["delta"] == 1.5

    def test_baseline_cli_creates_then_validates(self, tmp_path):
        """First run writes the baseline; second run passes with no regressions."""
        bl = tmp_path / "bl.json"
        first = self._run("--bits", "8,4", "--quick", "--baseline", str(bl))
        assert first.returncode == 0, first.stderr
        assert bl.exists()
        assert "Saved baseline" in first.stdout
        second = self._run("--bits", "8,4", "--quick", "--baseline", str(bl))
        assert second.returncode == 0, second.stderr
        assert "No regressions vs baseline" in second.stdout

    def test_baseline_cli_json_block(self, tmp_path):
        """--baseline --json emits a baseline block with exists flag."""
        bl = tmp_path / "bl2.json"
        first = self._run("--bits", "8", "--quick", "--baseline", str(bl),
                          "--json")
        data = json.loads(first.stdout)
        assert data["baseline"]["exists"] is False
        second = self._run("--bits", "8", "--quick", "--baseline", str(bl),
                           "--json")
        data = json.loads(second.stdout)
        assert data["baseline"]["exists"] is True
        assert data["baseline"]["regressions"] == []

    def test_baseline_cli_exit_1_on_regression(self, tmp_path):
        """A fabricated high-compression baseline forces exit code 1."""
        bl = tmp_path / "bl3.json"
        payload = {
            "metrics": {
                "tiny:int8": {
                    "passed": 7, "total": 7,
                    "gen_geomean": 1.0, "prompt_geomean": 1.0,
                    "temp_geomean": 1.0,
                    "weight_compression": 8.0, "avg_logit_cosine": 0.999,
                    "avg_token_agreement": 0.01,
                    "cold_start_s": 0.1, "warm_median_s": 0.05,
                }
            }
        }
        bl.write_text(json.dumps(payload))
        result = self._run("--bits", "8", "--quick", "--baseline", str(bl))
        assert result.returncode == 1
        assert "Regression" in result.stdout

    def test_baseline_report_section(self, tmp_path):
        """--report writes the baseline section."""
        bl = tmp_path / "bl4.json"
        report = tmp_path / "bl.md"
        first = self._run("--bits", "8", "--quick", "--baseline", str(bl),
                          "--report", str(report))
        assert first.returncode == 0, first.stderr
        assert "## Baseline" in report.read_text()


class TestPerplexity:
    """Teacher-forced perplexity quality metric."""

    def test_tiny_quality_carries_perplexity(self):
        """Real tiny run quality metrics include NQ/Q perplexity and ratio."""
        bench = bq.QuantizationBenchmark(tiny=True, quick=True, bits=8)
        _quiet(bench.run_all)
        q = next(r for r in bench.results if r.name == "quality_degradation")
        m = q.metrics
        assert q.passed
        assert m["nq_perplexity"] > 0 and m["q_perplexity"] > 0
        assert m["perplexity_ratio"] > 0
        assert "nq_perplexity" in m["per_prompt"]["0"]
        assert m["per_prompt"]["0"]["q_perplexity"] > 0

    def test_perplexity_is_deterministic(self):
        """Same model + prompt yields identical perplexity across runs."""
        bench_a = bq.QuantizationBenchmark(tiny=True, quick=True, bits=8)
        _quiet(bench_a.run_all)
        bench_b = bq.QuantizationBenchmark(tiny=True, quick=True, bits=8)
        _quiet(bench_b.run_all)
        a = next(r for r in bench_a.results if r.name == "quality_degradation").metrics
        b = next(r for r in bench_b.results if r.name == "quality_degradation").metrics
        assert a["nq_perplexity"] == b["nq_perplexity"]
        assert a["perplexity_ratio"] == b["perplexity_ratio"]

    def test_perplexity_single_token_is_one(self):
        """A single-token sequence has no targets, so perplexity is 1.0."""
        bench = bq.QuantizationBenchmark(tiny=True, quick=True, bits=8)
        bench.model, _ = _quiet(bench._load_model)
        ids = np.array([bench._encode("the quick brown fox")[0]], dtype=np.int64)
        assert bench._perplexity(bench.model, ids) == 1.0

    def test_comparison_table_has_ppl_row(self):
        """_comparison_table renders a PPL ratio row per precision."""
        run8 = _with_ppl(_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5), 1.05)
        run4 = _with_ppl(_synthetic_run(4, "tiny", cos=0.90, comp=8.0, gen=2.0), 1.10)
        table = bq._comparison_table([run8, run4])
        assert "| PPL ratio (Q/NQ) |" in table
        assert "| 1.05 |" in table
        assert "| 1.10 |" in table

    def test_comparison_json_and_csv_carry_ppl(self):
        """_comparison_json and _csv_output expose perplexity_ratio."""
        run = _with_ppl(_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5), 1.05)
        comp = bq._comparison_json([run])["int8"]
        assert comp["perplexity_ratio"] == 1.05
        import csv
        rows = list(csv.DictReader(bq._csv_output([run]).splitlines()))
        assert rows[0]["perplexity_ratio"] == "1.05"

    def test_model_comparison_carries_ppl(self):
        """_model_comparison exposes perplexity_ratio per precision."""
        run = _with_ppl(_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5), 1.05)
        run["model"] = "tiny"
        entry = bq._model_comparison([run])["tiny"]["int8"]
        assert entry["perplexity_ratio"] == 1.05

    def test_baseline_gates_perplexity_ratio(self):
        """A >15% perplexity-ratio rise is a baseline regression."""
        run = _with_ppl(_synthetic_run(8, "tiny", cos=0.99, comp=4.0, gen=1.5), 1.4)
        current = bq._headline_metrics([run])
        baseline = bq._headline_metrics([run])
        current["tiny:int8"]["perplexity_ratio"] = 1.4
        baseline["tiny:int8"]["perplexity_ratio"] = 1.0
        out = bq._compare_baselines(current, baseline)
        assert any(r["metric"] == "perplexity_ratio" for r in out["regressions"])

    def test_report_mentions_perplexity(self, tmp_path):
        """--report includes the perplexity line for quality."""
        import subprocess
        report = tmp_path / "ppl.md"
        result = subprocess.run(
            [sys.executable, "scripts/benchmark_quantization.py",
             "--bits", "8", "--quick", "--report", str(report)],
            capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        assert result.returncode == 0, result.stderr
        assert "perplexity_ratio" in report.read_text()
