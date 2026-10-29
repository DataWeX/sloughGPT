"""benchmark_results: the `effects` kind (card 20261024_095912).

show_effects.py metrics must be recordable and regression-gated through the
existing recorder (record/history/compare) instead of frozen comment
constants. Direction matters: decode_tok_s is higher-is-better, the rest
lower-is-better; load-sensitive metrics need generous rel thresholds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_results as br  # noqa: E402

EFFECTS_METRICS = {
    "rag_context_chars": 2087,
    "rag_query_ms": 12.0,
    "attn_single_ms": 0.09,
    "attn_multi_ms": 27.9,
    "first_token_short_s": 0.21,
    "first_token_rag_s": 30.3,
    "decode_tok_s": 11.9,
    "loadavg": 3.45,
}


def test_effects_thresholds_defined_and_direction_aware():
    assert "effects" in br.REGRESSION_THRESHOLDS
    th = br.REGRESSION_THRESHOLDS["effects"]
    assert "decode_tok_s" in th
    assert "decode_tok_s" in br.HIGHER_IS_BETTER
    # everything else is lower-is-better (a rise is the regression)
    for m in (
        "rag_context_chars",
        "attn_single_ms",
        "attn_multi_ms",
        "first_token_rag_s",
        "decode_tok_s",
        "rag_query_ms",
    ):
        assert m in th, f"{m} missing from effects thresholds"
        assert m not in br.HIGHER_IS_BETTER or m == "decode_tok_s"


def test_decode_drop_is_regression_rise_is_not():
    old = {"metrics": dict(EFFECTS_METRICS)}
    dropped = {"metrics": {**EFFECTS_METRICS, "decode_tok_s": 6.0}}  # -50%
    rose = {"metrics": {**EFFECTS_METRICS, "decode_tok_s": 14.0}}  # +18%
    assert br.is_regression("effects", dropped, old) is True
    assert br.is_regression("effects", rose, old) is False


def test_context_bloat_is_regression():
    old = {"metrics": dict(EFFECTS_METRICS)}
    bloated = {"metrics": {**EFFECTS_METRICS, "rag_context_chars": 3200}}  # +53%
    assert br.is_regression("effects", bloated, old) is True


def test_missing_metric_ignored_no_crash():
    old = {"metrics": dict(EFFECTS_METRICS)}
    partial = {"metrics": {"decode_tok_s": 11.0}}  # others absent
    assert br.is_regression("effects", partial, old) is False


def test_ms_scale_metrics_use_abs_not_rel():
    """Scheduler jitter at loadavg 23 moved rag_query 15.9→38.5ms (+142%)
    and attn_single 0.13→0.23ms (+74%) with no real change — rel% on a
    ms-scale baseline is noise, so those two are abs-delta thresholds."""
    assert br.REGRESSION_THRESHOLDS["effects"]["rag_query_ms"][1] == "abs"
    assert br.REGRESSION_THRESHOLDS["effects"]["attn_single_ms"][1] == "abs"
    old = {"metrics": dict(EFFECTS_METRICS)}  # rag_query 12ms, attn 0.09ms
    jitter = {"metrics": {**EFFECTS_METRICS, "rag_query_ms": 38.5, "attn_single_ms": 0.23}}
    assert br.is_regression("effects", jitter, old) is False  # jitter absorbed
    blown = {"metrics": {**EFFECTS_METRICS, "rag_query_ms": 400.0, "attn_single_ms": 9.0}}
    assert br.is_regression("effects", blown, old) is True  # real regressions


def test_record_roundtrip_effects(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "RESULTS_DIR", tmp_path)
    src = tmp_path / "in.json"
    src.write_text(json.dumps(EFFECTS_METRICS))
    args = SimpleNamespace(
        kind="effects",
        json_file=str(src),
        model="Qwen2.5-0.5B-Instruct",
        url=None,
        runs=0,
    )
    rc = br.do_record(args)
    assert rc == 0
    files = list((tmp_path / "effects").glob("*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text())
    assert rec["kind"] == "effects"
    assert rec["metrics"]["rag_context_chars"] == 2087
    assert rec["commit"]
    assert rec["timestamp"]


def test_compare_vs_previous_on_recorded_effects(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "RESULTS_DIR", tmp_path)
    for i, decode in enumerate((11.9, 11.0)):
        p = tmp_path / "effects"
        p.mkdir(exist_ok=True)
        (p / f"m_{i:04d}.json").write_text(
            json.dumps(
                {
                    "kind": "effects",
                    "model": "m",
                    "commit": "x",
                    "timestamp": f"2026-10-24T0{i}:00:00",
                    "metrics": {**EFFECTS_METRICS, "decode_tok_s": decode},
                }
            )
        )
    files = br.collect_records("effects")
    assert len(files) == 2
    new = br.load_result(files[0])  # newest first
    old = br.load_result(files[1])
    assert br.is_regression("effects", new, old) is False  # 11.9 vs 11.0 rise ok
    dropped = {"metrics": {**new["metrics"], "decode_tok_s": 7.0}}
    assert br.is_regression("effects", dropped, old) is True


def test_record_chains_compare_gates_rc(tmp_path, monkeypatch):
    """do_record ends in do_compare — caller must pass .vs, and the chained
    compare must gate: healthy run → rc 0, regressed run → rc 1."""
    monkeypatch.setattr(br, "RESULTS_DIR", tmp_path)
    p = tmp_path / "effects"
    p.mkdir()
    (p / "m_0001.json").write_text(
        json.dumps(
            {
                "kind": "effects",
                "model": "m",
                "commit": "x",
                "timestamp": "2026-10-24T09:00:00",
                "metrics": dict(EFFECTS_METRICS),  # decode 11.9
            }
        )
    )
    good = tmp_path / "good.json"
    good.write_text(json.dumps({**EFFECTS_METRICS, "decode_tok_s": 12.5}))
    args = SimpleNamespace(
        kind="effects",
        json_file=str(good),
        model="m",
        url=None,
        runs=0,
        vs="previous",
    )
    rc = br.do_record(args)  # healthy → 0
    assert rc == 0

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**EFFECTS_METRICS, "decode_tok_s": 6.0}))
    args.json_file = str(bad)
    rc = br.do_record(args)  # -50% decode → chained compare fails
    assert rc == 1
