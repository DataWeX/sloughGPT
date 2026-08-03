"""Tests for scripts/benchmark_results.py persistent results + regression tracking."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import benchmark_results as br  # noqa: E402


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    """Point RESULTS_DIR at a temp dir and return it."""
    monkeypatch.setattr(br, "RESULTS_DIR", tmp_path / "benchmark_results")
    monkeypatch.setattr(br, "git_commit", lambda: "test-sha")
    return tmp_path / "benchmark_results"


def write_result(tmp_results, kind, model, **fields):
    """Write a result file and return its path."""
    path = br.results_path(kind, model, br.timestamp().replace(":", "").replace(".", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"kind": kind, "model": model, "timestamp": br.timestamp(),
              "commit": "test-sha", **fields}
    with open(path, "w") as f:
        json.dump(record, f)
    return path


# ── helpers ────────────────────────────────────────────────────────────────

def test_git_commit_runs():
    """git_commit returns a short sha or None, never raises."""
    commit = br.git_commit()
    assert commit is None or isinstance(commit, str) and len(commit) <= 20


def test_results_path_sanitizes_model(tmp_results):
    """Slash in model name becomes --, files land under kind dir."""
    p = br.results_path("stability", "Qwen/Qwen2.5", "stamp")
    assert p.parent.name == "stability"
    assert "Qwen--Qwen2.5" in p.name


def test_collect_records_newest_first(tmp_results):
    """collect_records returns files sorted newest first."""
    write_result(tmp_results, "stability", "m", overall=100)
    write_result(tmp_results, "stability", "m", overall=90)
    runs = br.collect_records("stability")
    assert len(runs) == 2
    # newest has overall 90
    assert br.load_result(runs[0])["overall"] == 90


def test_dig_nested_containers(tmp_results):
    """_dig finds top-level and score/metrics container values."""
    d = {"score": {"overall": 72}, "metrics": {"mean_ms": 40.0}, "runs": 20}
    assert br._dig(d, "overall") == 72
    assert br._dig(d, "mean_ms") == 40.0
    assert br._dig(d, "runs") == 20
    assert br._dig(d, "missing") is None


def test_extract_from_stability_grabs_trailing_json():
    """Extracts the JSON report from mixed stdout."""
    raw = "🔍 Server alive\n{\"model\": \"m\", \"score\": {\"overall\": 88}}"
    data = br._extract_from_stability(raw)
    assert data["score"]["overall"] == 88


# ── threshold semantics ────────────────────────────────────────────────────

def test_threshold_rel_normalized():
    """rel thresholds (percentages) normalize to fractions."""
    limit, mode = br._threshold("latency", "mean_ms")
    assert mode == "rel"
    assert limit == pytest.approx(0.20)


def test_threshold_abs():
    limit, mode = br._threshold("stability", "overall")
    assert mode == "abs"
    assert limit == 5.0


# ── regression detection ───────────────────────────────────────────────────

def test_no_regression_when_equal(tmp_results):
    old = {"score": {"overall": 95}}
    new = {"score": {"overall": 95}}
    assert br.is_regression("stability", new, old) is False


def test_regression_stability_score_drop(tmp_results):
    old = {"score": {"overall": 100, "response_rate": 1.0, "crash_rate": 0.0,
                     "empty_rate": 0.0, "latency_degradation": 1.1, "length_cv": 0.2}}
    new = {"score": {"overall": 70, "response_rate": 0.9, "crash_rate": 0.1,
                     "empty_rate": 0.0, "latency_degradation": 1.5, "length_cv": 0.4}}
    assert br.is_regression("stability", new, old) is True


def test_small_drop_not_regression(tmp_results):
    """Overall drop < 5 pts is tolerated."""
    old = {"score": {"overall": 95}}
    new = {"score": {"overall": 92}}
    assert br.is_regression("stability", new, old) is False


def test_latency_rel_regression(tmp_results):
    """+25% mean latency exceeds the +20% threshold."""
    old = {"metrics": {"mean_ms": 40000.0, "p95_ms": 46000.0}}
    new = {"metrics": {"mean_ms": 50000.0, "p95_ms": 58000.0}}
    assert br.is_regression("latency", new, old) is True


def test_latency_rel_improvement_not_regression(tmp_results):
    old = {"metrics": {"mean_ms": 50000.0, "p95_ms": 58000.0}}
    new = {"metrics": {"mean_ms": 44000.0, "p95_ms": 50000.0}}
    assert br.is_regression("latency", new, old) is False


# ── record / history / compare integration ─────────────────────────────────

def test_record_and_history_roundtrip(tmp_results, monkeypatch):
    """record persists a run; history lists it."""
    class Args:
        kind = "stability"
        json_file = None
        url = "http://localhost:8000"
        runs = 20
        model = None
        vs = "previous"

    def fake_run(url, runs):
        return {"model": "m", "runs": runs, "passed": True,
                "score": {"overall": 100}, "elapsed_s": 10}

    monkeypatch.setattr(br, "_run_stability", fake_run)
    assert br.do_record(Args()) == 0

    runs = br.collect_records("stability")
    assert len(runs) == 1
    r = br.load_result(runs[0])
    assert r["model"] == "m"
    assert r["passed"] is True
    assert r["commit"] == "test-sha"


def test_compare_single_run_is_ok(tmp_results):
    """compare with <2 runs reports info, returns 0."""
    write_result(tmp_results, "stability", "m", score={"overall": 100})

    class Args:
        kind = "stability"
        vs = "previous"

    assert br.do_compare(Args()) == 0


def test_compare_detects_regression_and_exit_code(tmp_results, capsys):
    write_result(tmp_results, "stability", "m",
                 score={"overall": 100, "response_rate": 1.0, "crash_rate": 0.0,
                        "empty_rate": 0.0, "latency_degradation": 1.1, "length_cv": 0.2})
    write_result(tmp_results, "stability", "m",
                 score={"overall": 70, "response_rate": 0.9, "crash_rate": 0.1,
                        "empty_rate": 0.0, "latency_degradation": 1.5, "length_cv": 0.4})

    class Args:
        kind = "stability"
        vs = "previous"

    assert br.do_compare(Args()) == 1
    assert "REGRESSION" in capsys.readouterr().out
