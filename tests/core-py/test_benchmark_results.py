"""Tests for scripts/benchmark_results.py persistent results + regression tracking."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
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
    record = {
        "kind": kind,
        "model": model,
        "timestamp": br.timestamp(),
        "commit": "test-sha",
        **fields,
    }
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
    raw = '🔍 Server alive\n{"model": "m", "score": {"overall": 88}}'
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
    old = {
        "score": {
            "overall": 100,
            "response_rate": 1.0,
            "crash_rate": 0.0,
            "empty_rate": 0.0,
            "latency_degradation": 1.1,
            "length_cv": 0.2,
        }
    }
    new = {
        "score": {
            "overall": 70,
            "response_rate": 0.9,
            "crash_rate": 0.1,
            "empty_rate": 0.0,
            "latency_degradation": 1.5,
            "length_cv": 0.4,
        }
    }
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
        return {
            "model": "m",
            "runs": runs,
            "passed": True,
            "score": {"overall": 100},
            "elapsed_s": 10,
        }

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
    write_result(
        tmp_results,
        "stability",
        "m",
        score={
            "overall": 100,
            "response_rate": 1.0,
            "crash_rate": 0.0,
            "empty_rate": 0.0,
            "latency_degradation": 1.1,
            "length_cv": 0.2,
        },
    )
    write_result(
        tmp_results,
        "stability",
        "m",
        score={
            "overall": 70,
            "response_rate": 0.9,
            "crash_rate": 0.1,
            "empty_rate": 0.0,
            "latency_degradation": 1.5,
            "length_cv": 0.4,
        },
    )

    class Args:
        kind = "stability"
        vs = "previous"

    assert br.do_compare(Args()) == 1
    assert "REGRESSION" in capsys.readouterr().out


# ── history rendering: kind-aware + shape-tolerant (card b22e878a GROUP E) ──


def _hist_args(kind):
    class Args:
        pass

    a = Args()
    a.kind = kind
    return a


def test_history_latency_list_metrics_does_not_crash(tmp_results, capsys):
    """A latency record whose metrics is a LIST must render, not raise.

    benchmark_slonet_training --results writes merged per-config lists; if
    such a file is recorded under a metric kind, the renderer used to die
    with ``'list' object has no attribute 'get'``.
    """
    write_result(
        tmp_results,
        "latency",
        "m",
        metrics=[{"mean_ms": 120.0, "p95_ms": 200.0}, {"mean_ms": 90.0, "p95_ms": 150.0}],
    )
    assert br.do_history(_hist_args("latency")) == 0
    out = capsys.readouterr().out
    assert "120.0" in out and "90.0" in out


def test_history_startup_renders_real_metrics(tmp_results, capsys):
    """Startup history shows health/ready, not the latency fallback mean=?/p95=?."""
    write_result(
        tmp_results,
        "startup",
        "served",
        metrics={"time_to_health_s": 1.25, "time_to_ready_s": 3.5, "preload_warnings": 0},
    )
    assert br.do_history(_hist_args("startup")) == 0
    out = capsys.readouterr().out
    assert "1.25" in out
    assert "3.5" in out
    assert "mean=?" not in out


def test_history_execution_renders_dispatch(tmp_results, capsys):
    write_result(tmp_results, "execution", "m", metrics={"dispatch_us": 42.0, "peak_threads": 7})
    assert br.do_history(_hist_args("execution")) == 0
    out = capsys.readouterr().out
    assert "42.0" in out and "7" in out


def test_history_default_iterates_all_kinds(tmp_results, capsys):
    """Default history (no --kind) covers every known kind, not just stability+latency."""
    write_result(tmp_results, "stability", "m", score={"overall": 100})
    write_result(tmp_results, "latency", "m", metrics={"mean_ms": 1.0, "p95_ms": 2.0})
    write_result(tmp_results, "startup", "m", metrics={"time_to_health_s": 1.0})
    write_result(tmp_results, "execution", "m", metrics={"dispatch_us": 1.0})
    write_result(
        tmp_results,
        "training",
        "m",
        metrics=[{"config": "gate", "gate_converged": True, "final_loss": 0.5}],
    )
    write_result(
        tmp_results,
        "bootstrap",
        "m",
        metrics={"first_s": 0.5, "warm_p95_s": 0.2, "member_errors": 0},
    )
    assert br.do_history(_hist_args(None)) == 0
    out = capsys.readouterr().out
    for kind in ("stability", "latency", "startup", "execution", "training", "bootstrap"):
        assert f"── {kind}: 1 runs ──" in out


def test_history_training_dict_metrics_renders(tmp_results, capsys):
    """Training record with a single dict (not list) metrics renders one line."""
    write_result(
        tmp_results,
        "training",
        "m",
        metrics={"config": "tiny", "gate_converged": False, "final_loss": 1.2},
    )
    assert br.do_history(_hist_args("training")) == 0
    out = capsys.readouterr().out
    assert "tiny" in out and "final=1.2" in out


def test_history_training_dict_config_does_not_crash(tmp_results, capsys):
    """A dict-valued config field must not break the fixed-width format."""
    write_result(
        tmp_results,
        "training",
        "m",
        metrics=[{"config": {"lr": 0.01}, "gate_converged": True, "final_loss": 0.9}],
    )
    assert br.do_history(_hist_args("training")) == 0
    assert "final=0.9" in capsys.readouterr().out


# ── bootstrap kind (landing-page /bootstrap latency gate) ──────────────────


def test_bootstrap_thresholds_registered():
    """Bootstrap kind has rel-latency + abs zero-tolerance thresholds."""
    assert br._threshold("bootstrap", "first_s") == (1.0, "rel")
    assert br._threshold("bootstrap", "warm_p95_s") == (0.5, "rel")
    assert br._threshold("bootstrap", "member_errors") == (0, "abs")


def _boot_metrics(**overrides):
    base = {"first_s": 1.0, "warm_p95_s": 0.3, "member_errors": 0, "members": 12}
    base.update(overrides)
    return {"metrics": base}


def test_bootstrap_member_error_is_regression():
    """A new member error (0→1) regresses; a steady state does not."""
    assert br.is_regression("bootstrap", _boot_metrics(member_errors=1), _boot_metrics())
    assert not br.is_regression("bootstrap", _boot_metrics(), _boot_metrics())


def test_bootstrap_cold_fanout_regression_caught():
    """The pre-fix class (cold envelope ~19.5s vs ~1.0s) trips first_s."""
    old = _boot_metrics(first_s=1.0)
    assert br.is_regression("bootstrap", _boot_metrics(first_s=19.5), old)
    assert not br.is_regression("bootstrap", _boot_metrics(first_s=1.5), old)


def test_record_bootstrap_roundtrip(tmp_results):
    """do_record ingests a benchmark_bootstrap --json-out file and renders it."""
    src = tmp_results.parent / "bench.json"
    src.write_text(
        json.dumps(
            {
                "url": "http://localhost:8000/bootstrap",
                "first_s": 0.9,
                "warm_mean_s": 0.2,
                "warm_p95_s": 0.25,
                "warm_max_s": 0.4,
                "runs": 20,
                "members": 12,
                "member_errors": 0,
            }
        )
    )

    class Args:
        kind = "bootstrap"
        json_file = str(src)
        url = "http://localhost:8000"
        runs = 20
        model = None
        vs = "previous"

    assert br.do_record(Args()) == 0
    runs = br.collect_records("bootstrap")
    assert len(runs) == 1
    r = br.load_result(runs[0])
    assert r["kind"] == "bootstrap"
    assert r["metrics"]["member_errors"] == 0
    assert r["metrics"]["first_s"] == 0.9


def test_record_bootstrap_without_json_file_fails(tmp_results, capsys):
    """Like startup/training, bootstrap records only from a bench JSON file."""

    class Args:
        kind = "bootstrap"
        json_file = None
        url = "http://localhost:8000"
        runs = 1
        model = None
        vs = "previous"

    assert br.do_record(Args()) == 1
    assert "requires --json-file" in capsys.readouterr().err


def test_history_bootstrap_renders_first_p95(tmp_results, capsys):
    write_result(
        tmp_results,
        "bootstrap",
        "served",
        metrics={"first_s": 1.25, "warm_p95_s": 0.31, "member_errors": 0},
    )
    assert br.do_history(_hist_args("bootstrap")) == 0
    out = capsys.readouterr().out
    assert "first=1.25" in out and "p95=0.31" in out and "errs=0" in out
    assert "mean=?" not in out


# ── canonical store: env resolution, migrate, source tag (card 20261031_133226) ──


def test_results_dir_env_override(monkeypatch, tmp_path):
    """SLO_BENCH_DATA_DIR wins; the default is checkout-independent.

    The invariant that stranded 5 decode records in the api-std checkout:
    the resolved store is NEVER repo-relative, so every worktree records
    into one history.
    """
    import importlib

    old = br.RESULTS_DIR
    monkeypatch.setenv("SLO_BENCH_DATA_DIR", str(tmp_path / "bench"))
    try:
        importlib.reload(br)
        assert br.RESULTS_DIR == tmp_path / "bench"
    finally:
        monkeypatch.delenv("SLO_BENCH_DATA_DIR", raising=False)
        importlib.reload(br)
        default = br.RESULTS_DIR
        br.RESULTS_DIR = old  # restore exact pre-test state
    assert not str(default).startswith(str(br.REPO_ROOT))


def _fake_latency(url, runs, update_baseline):
    return {"mean_ms": 10.0, "p95_ms": 12.0}


def _latency_args():
    class Args:
        kind = "latency"
        json_file = None
        url = "http://localhost:8000"
        runs = 1
        model = "m"
        vs = "previous"

    return Args()


def test_record_stamps_source_local(tmp_results, monkeypatch):
    """records carry source=local — the doctor's provenance tag."""
    monkeypatch.setattr(br, "_run_latency", _fake_latency)
    monkeypatch.delenv("CI", raising=False)
    assert br.do_record(_latency_args()) == 0
    r = br.load_result(br.collect_records("latency")[0])
    assert r["source"] == "local"


def test_record_stamps_source_ci(tmp_results, monkeypatch):
    """CI-tagged runs stamp source=ci (runner vs dev-box provenance)."""
    monkeypatch.setattr(br, "_run_latency", _fake_latency)
    monkeypatch.setenv("CI", "true")
    assert br.do_record(_latency_args()) == 0
    r = br.load_result(br.collect_records("latency")[0])
    assert r["source"] == "ci"


def test_migrate_unions_into_canonical_store(tmp_results, tmp_path, capsys):
    """migrate copies old records in, never overwrites, is idempotent."""
    src = tmp_path / "old_checkout" / "data" / "benchmark_results"
    (src / "inference").mkdir(parents=True)
    (src / "latency").mkdir()
    rec = {"kind": "inference", "model": "m", "timestamp": "2026-10-30T08:04:23Z"}
    (src / "inference" / "a.json").write_text(json.dumps(rec))
    (src / "latency" / "b.json").write_text(json.dumps({"kind": "latency"}))

    class Args:
        source = str(src)

    assert br.do_migrate(Args()) == 0
    assert (tmp_results / "inference" / "a.json").exists()
    assert (tmp_results / "latency" / "b.json").exists()

    # idempotent + union: a changed source file never overwrites the store
    (src / "inference" / "a.json").write_text(json.dumps({**rec, "model": "changed"}))
    assert br.do_migrate(Args()) == 0
    assert br.load_result(tmp_results / "inference" / "a.json")["model"] == "m"
    out = capsys.readouterr().out
    assert "0 copied, 2 already present" in out


def test_migrate_missing_source_is_error(tmp_results, tmp_path, capsys):
    class Args:
        source = str(tmp_path / "does-not-exist")

    assert br.do_migrate(Args()) == 1
    assert "not a directory" in capsys.readouterr().err


# ── doctor trend reader: record_source / load_series / trend_status ────────
# (card 20261031_170120 — Site Doctor inference triage reads the
# always-record series; trend gate is additive to pairwise is_regression)


def _hist(*values, metric="decode_tok_s"):
    """History records as load_result would return them (metric at root)."""
    return [{metric: v} for v in values]


def test_record_source_stamp_and_legacy_default(tmp_results):
    """Pre-stamp records count as local; an explicit stamp wins."""
    assert br.record_source({}) == "local"
    assert br.record_source({"source": "local"}) == "local"
    assert br.record_source({"source": "ci"}) == "ci"


def test_load_series_sorts_ascending_and_filters_source(tmp_results):
    write_result(
        tmp_results,
        "inference",
        "m",
        timestamp="2026-11-01T00:00:03Z",
        source="local",
        decode_tok_s=3,
    )
    write_result(
        tmp_results,
        "inference",
        "m",
        timestamp="2026-11-01T00:00:01Z",
        source="local",
        decode_tok_s=1,
    )
    write_result(
        tmp_results,
        "inference",
        "m",
        timestamp="2026-11-01T00:00:02Z",
        source="ci",
        decode_tok_s=2,
    )
    write_result(  # legacy record, no source stamp
        tmp_results,
        "inference",
        "m",
        timestamp="2026-11-01T00:00:04Z",
        decode_tok_s=4,
    )
    all_recs = br.load_series("inference")
    assert [r["decode_tok_s"] for r in all_recs] == [1, 2, 3, 4]  # oldest first
    local = br.load_series("inference", source="local")
    assert [r["decode_tok_s"] for r in local] == [1, 3, 4]  # legacy counts local
    ci = br.load_series("inference", source="ci")
    assert [r["decode_tok_s"] for r in ci] == [2]


def test_load_series_missing_dir_and_corrupt_file(tmp_results):
    """Absent dir -> empty series; a half-written record never sinks the read."""
    assert br.load_series("inference") == []
    (tmp_results / "inference").mkdir(parents=True)
    (tmp_results / "inference" / "bad.json").write_text("{truncated", encoding="utf-8")
    write_result(
        tmp_results,
        "inference",
        "m",
        timestamp="2026-11-01T00:00:01Z",
        decode_tok_s=1,
    )
    recs = br.load_series("inference")
    assert len(recs) == 1
    assert recs[0]["decode_tok_s"] == 1


def test_trend_insufficient_below_three_points():
    """Thin sample says so — never a false alarm (fresh store, single run)."""
    for n in (0, 1, 2):
        out = br.trend_status("decode_tok_s", _hist(*([10.0] * n)), 10.0)
        assert out["status"] == "insufficient", n
        assert out["n"] == n
        assert "insufficient" in out["message"]


def test_trend_flat_series_point_band():
    """sigma=0 collapses the band to a point: exact match healthy, any step flagged."""
    history = _hist(*([10.0] * 5))
    ok = br.trend_status("decode_tok_s", history, 10.0)
    assert ok["status"] == "healthy"
    assert ok["sigma"] == 0.0
    assert ok["trend"] == 10.0
    down = br.trend_status("decode_tok_s", history, 6.0)
    assert down["status"] == "regressed"  # higher-is-better drop
    assert down["message"]  # sigma=0 wording stays human, not "infs"
    assert "infs" not in down["message"]
    up = br.trend_status("decode_tok_s", history, 14.0)
    assert up["status"] == "improved"  # higher-is-better rise


def test_trend_on_trend_decline_is_healthy_but_acceleration_regresses():
    """The band is trend-adjusted: a steady decline ON the line is not an alarm."""
    history = _hist(20.0, 18.0, 16.0, 14.0, 12.0)  # perfect slope -2
    on = br.trend_status("decode_tok_s", history, 10.0)  # continues the line
    assert on["status"] == "healthy"
    assert on["slope"] == pytest.approx(-2.0)
    assert on["sigma"] == pytest.approx(0.0, abs=1e-9)
    faster = br.trend_status("decode_tok_s", history, 6.0)  # below trend
    assert faster["status"] == "regressed"
    slower = br.trend_status("decode_tok_s", history, 14.0)  # above trend
    assert slower["status"] == "improved"


def test_trend_noisy_series_uses_sigma_band():
    """Shared-box load noise widens sigma: k=3 decides, not raw deviation."""
    values = [9.0, 11.0, 10.0, 9.5, 10.5, 10.0, 9.0, 11.0, 10.0, 10.0]
    history = _hist(*values)
    within = br.trend_status("decode_tok_s", history, 10.2)
    assert within["status"] == "healthy"
    assert within["sigma"] > 0
    assert within["n"] == 10
    bad = br.trend_status("decode_tok_s", history, 2.0)
    assert bad["status"] == "regressed"
    assert bad["z"] < -3  # beyond k*sigma in the bad direction
    good = br.trend_status("decode_tok_s", history, 18.0)
    assert good["status"] == "improved"


def test_trend_window_truncates_to_last_n():
    values = [10.0 + i for i in range(15)]  # 10..24
    history = _hist(*values)
    out = br.trend_status("decode_tok_s", history, 25.0, window=10)
    assert out["n"] == 10 and out["window"] == 10
    # last 10 are 15..24 (slope +1) -> next expected = 25, on-trend
    assert out["status"] == "healthy"
    assert out["trend"] == pytest.approx(25.0)


def test_trend_dotted_metric_path():
    history = [{"naive_tok_s": {"32": v}} for v in (100.0, 101.0, 99.0, 100.0)]
    out = br.trend_status("naive_tok_s.32", history, 100.0)
    assert out["status"] == "healthy"
    assert out["n"] == 4
    miss = br.trend_status("naive_tok_s.64", history, 100.0)
    assert miss["status"] == "insufficient" and miss["n"] == 0


def test_trend_lower_is_better_direction():
    """Direction comes from HIGHER_IS_BETTER — a rise in mean_ms regresses."""
    history = _hist(100.0, 101.0, 99.0, 100.0, 100.0, metric="mean_ms")
    up = br.trend_status("mean_ms", history, 150.0)
    assert up["status"] == "regressed"
    down = br.trend_status("mean_ms", history, 50.0)
    assert down["status"] == "improved"
    steady = br.trend_status("mean_ms", history, 100.0)
    assert steady["status"] == "healthy"
