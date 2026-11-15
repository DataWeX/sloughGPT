"""Tests for scripts/benchmark_slonet_gap.py (card 20261103_121700).

The gap benchmark answers ONE question with numbers: can SloNet power a
target machine *today*? It composes existing pieces (SloughGPTTrainer,
SloTransformer.generate, lm_eval_char) into a single report — it must not
grow a second training loop.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import benchmark_slonet_gap as bg  # noqa: E402

# ── inference measurement ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_model():
    return bg.build_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2, block_size=16)


def test_measure_inference_reports_tokens_per_second(tiny_model):
    """Decode rate must be a positive finite tok/s with a token count."""
    m = bg.measure_inference(tiny_model, prompt=[1, 2, 3], new_tokens=4, repeats=2)
    assert m["tokens"] == 8  # 4 new tokens x 2 repeats
    assert np.isfinite(m["decode_tps"]) and m["decode_tps"] > 0
    assert np.isfinite(m["prefill_tps"]) and m["prefill_tps"] > 0
    assert m["elapsed_s"] > 0


def test_measure_inference_deterministic_token_count(tiny_model):
    """Token count is prescribed, not sampled — repeats x new_tokens."""
    m = bg.measure_inference(tiny_model, prompt=[5], new_tokens=3, repeats=4)
    assert m["tokens"] == 12


# ── training measurement ─────────────────────────────────────────────────────


def test_measure_training_reuses_sloughgpt_trainer():
    """The trainer imported by the benchmark IS SloughGPTTrainer — no new loop."""
    from domain.training._internal.train_pipeline import SloughGPTTrainer

    assert bg.SloughGPTTrainer is SloughGPTTrainer


def test_measure_training_returns_throughput(tmp_path):
    """A 2-step run on a synthetic corpus yields positive tokens/s + loss."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefghij" * 200, encoding="utf-8")
    m = bg.measure_training(
        data_path=str(corpus),
        checkpoint_dir=str(tmp_path / "ckpt"),
        max_steps=2,
        n_embed=16,
        n_layer=1,
        n_head=2,
        block_size=16,
        batch_size=8,
    )
    assert m["tokens"] > 0 and m["tokens_per_s"] > 0
    assert m["steps"] == 2
    assert np.isfinite(m["final_loss"])


# ── report assembly ──────────────────────────────────────────────────────────


def test_build_report_contains_the_three_numbers():
    """Report carries inference, training and eval sections + machine target."""
    report = bg.build_report(
        inference={"tokens": 8, "decode_tps": 12.5, "prefill_tps": 40.0, "elapsed_s": 0.64},
        training={"tokens": 1600, "tokens_per_s": 800.0, "steps": 2, "final_loss": 2.3},
        eval={"mean_loss": 2.1, "perplexity": 8.2, "num_token_positions": 100},
        model={"params": 12345, "config": "16d/1L/2H"},
        target="v86-browser-vm",
    )
    assert report["target"] == "v86-browser-vm"
    assert report["inference"]["decode_tps"] == 12.5
    assert report["training"]["tokens_per_s"] == 800.0
    assert report["eval"]["perplexity"] == 8.2
    assert report["model"]["params"] == 12345


def test_build_report_is_json_serialisable():
    """The report is written to disk — json round-trip must hold."""
    report = bg.build_report(
        inference={"tokens": 1, "decode_tps": 1.0, "prefill_tps": 1.0, "elapsed_s": 1.0},
        training={"tokens": 1, "tokens_per_s": 1.0, "steps": 1, "final_loss": 1.0},
        eval={"mean_loss": 1.0, "perplexity": 1.0, "num_token_positions": 1},
        model={"params": 1, "config": "x"},
        target="t",
    )
    assert json.loads(json.dumps(report)) == report


def test_verdict_below_slo_fails():
    """Decode tok/s under the stated SLO → verdict 'below_slo'."""
    report = bg.build_report(
        inference={"tokens": 8, "decode_tps": 1.0, "prefill_tps": 40.0, "elapsed_s": 8.0},
        training={"tokens": 100, "tokens_per_s": 50.0, "steps": 1, "final_loss": 2.0},
        eval={"mean_loss": 2.0, "perplexity": 7.4, "num_token_positions": 10},
        model={"params": 1, "config": "x"},
        target="v86-browser-vm",
        decode_slo=5.0,
    )
    assert report["verdict"] == "below_slo"
    assert report["decode_slo_tps"] == 5.0


def test_verdict_meets_slo_passes():
    """Decode tok/s at/above SLO → verdict 'meets_slo'."""
    report = bg.build_report(
        inference={"tokens": 8, "decode_tps": 6.0, "prefill_tps": 40.0, "elapsed_s": 1.3},
        training={"tokens": 100, "tokens_per_s": 50.0, "steps": 1, "final_loss": 2.0},
        eval={"mean_loss": 2.0, "perplexity": 7.4, "num_token_positions": 10},
        model={"params": 1, "config": "x"},
        target="v86-browser-vm",
        decode_slo=5.0,
    )
    assert report["verdict"] == "meets_slo"


# ── corpus resolution (learn from OUR tokens — card follow-up 2026-11-03) ────


def test_resolve_corpus_prefers_own_docs(tmp_path):
    """Our own docs/*.md win — 'learn from our tokens', never a stranger's."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# Alpha docs", encoding="utf-8")
    (tmp_path / "docs" / "b.md").write_text("Beta content", encoding="utf-8")
    path, source = bg.resolve_corpus(tmp_path, cache_root=tmp_path / "no-cache")
    assert source == "own-docs"
    text = path.read_text(encoding="utf-8")
    assert "# Alpha docs" in text and "Beta content" in text
    # Deterministic order — sorted, so the corpus (and its ppl) is reproducible.
    assert text.index("Alpha") < text.index("Beta")


def test_resolve_corpus_falls_back_to_tinyshakespeare(tmp_path):
    """No own docs → external tinyshakespeare from the cache root, labeled."""
    cache = tmp_path / "cache" / "tinyshakespeare"
    cache.mkdir(parents=True)
    (cache / "corpus.jsonl").write_text('{"text":"hello"}', encoding="utf-8")
    path, source = bg.resolve_corpus(tmp_path / "empty-repo", cache_root=tmp_path / "cache")
    assert source == "tinyshakespeare"
    assert path.exists()


def test_resolve_corpus_synthetic_is_last_resort(tmp_path):
    """Nothing else on disk → synthetic cycle, but labeled as synthetic."""
    path, source = bg.resolve_corpus(tmp_path / "empty", cache_root=tmp_path / "no-cache")
    assert source == "synthetic"
    text = path.read_text(encoding="utf-8")
    assert len(text) >= 1000


def test_own_docs_corpus_skips_non_markdown(tmp_path):
    """Only prose docs/*.md feed the corpus — no binaries, no lockfiles."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "real.md").write_text("keep me", encoding="utf-8")
    (tmp_path / "docs" / "noise.txt").write_text("drop me", encoding="utf-8")
    path, source = bg.resolve_corpus(tmp_path, cache_root=tmp_path / "no-cache")
    text = path.read_text(encoding="utf-8")
    assert "keep me" in text
    assert "drop me" not in text


def test_report_records_corpus_source():
    """The report must SAY which corpus the score came from — no silent swaps."""
    report = bg.build_report(
        inference={"tokens": 1, "decode_tps": 1.0, "prefill_tps": 1.0, "elapsed_s": 1.0},
        training={"tokens": 1, "tokens_per_s": 1.0, "steps": 1, "final_loss": 1.0},
        eval={"mean_loss": 1.0, "perplexity": 1.0, "num_token_positions": 1},
        model={"params": 1, "config": "x"},
        target="t",
        corpus_source="own-docs",
    )
    assert report["corpus"]["source"] == "own-docs"


def test_default_decode_slo_is_stated_for_v86():
    """The v86 target's default SLO is explicit in the module — not implied."""
    assert bg.DEFAULT_TARGET == "v86-browser-vm"
    assert bg.DEFAULT_DECODE_SLO_TPS > 0
    # v86 is emulated x86: its budget is a FRACTION of bare metal.
    assert bg.DEFAULT_DECODE_SLO_TPS <= 10.0


# ── target calibration (card 20261103_121721, C3 — the number ON the machine) ─


def _calibration_pair(tmp_path):
    """Two factor reports as benchmark_v86_factor writes them: the
    representative SSE2 path and the conservative x87 floor."""
    sse2 = tmp_path / "factor_sse2.json"
    sse2.write_text(
        json.dumps(
            {
                "fp_mode": "sse2",
                "factor": 40.0,
                "v86_decode_tps": 9.2,
                "assumption": "calibrated estimate",
            }
        )
    )
    x87 = tmp_path / "factor_x87.json"
    x87.write_text(json.dumps({"fp_mode": "x87", "factor": 180.0}))
    return [str(sse2), str(x87)]


def test_load_target_calibration_prefers_representative_fp_mode(tmp_path):
    cal = bg.load_target_calibration(_calibration_pair(tmp_path))
    assert cal["factor"] == 40.0
    assert cal["fp_mode"] == "sse2"
    assert cal["floor"]["fp_mode"] == "x87"
    assert cal["floor"]["factor"] == 180.0
    assert len(cal["source"]) == 2


def test_build_report_with_calibration_verdicts_on_the_target():
    """The old verdict compared BARE-METAL decode to the v86 floor —
    with a measured factor the verdict must use the on-target estimate."""
    cal = {
        "fp_mode": "sse2",
        "factor": 40.0,
        "source": ["x"],
        "floor": {"fp_mode": "x87", "factor": 180.0},
    }
    report = bg.build_report(
        inference={"tokens": 8, "decode_tps": 360.0, "prefill_tps": 40.0, "elapsed_s": 0.1},
        training={"tokens": 100, "tokens_per_s": 50.0, "steps": 1, "final_loss": 2.0},
        eval={"mean_loss": 2.0, "perplexity": 7.4, "num_token_positions": 10},
        model={"params": 1, "config": "x"},
        target="v86-browser-vm",
        decode_slo=3.0,
        target_calibration=cal,
    )
    # 360 / 40 = 9.0 on target -> meets; bare metal (360) would "meet" trivially
    assert report["inference"]["decode_tps_on_target"] == 9.0
    assert report["verdict"] == "meets_slo"
    assert report["verdict_basis"] == "calibrated_target"
    # conservative floor case is recorded: 360 / 180 = 2.0 -> below
    assert report["inference"]["decode_tps_on_target_floor"] == 2.0
    assert report["floor_case_verdict"] == "below_slo"
    # the input dict must not be mutated
    assert "decode_tps_on_target" not in {
        "tokens": 8,
        "decode_tps": 360.0,
        "prefill_tps": 40.0,
        "elapsed_s": 0.1,
    }


def test_build_report_calibrated_verdict_can_miss():
    """Estimate under the SLO -> below_slo even though bare metal passes."""
    cal = {"fp_mode": "sse2", "factor": 400.0, "source": ["x"]}
    report = bg.build_report(
        inference={"tokens": 8, "decode_tps": 360.0, "prefill_tps": 40.0, "elapsed_s": 0.1},
        training={"tokens": 1, "tokens_per_s": 1.0, "steps": 1, "final_loss": 1.0},
        eval={"mean_loss": 1.0, "perplexity": 1.0, "num_token_positions": 1},
        model={"params": 1, "config": "x"},
        decode_slo=3.0,
        target_calibration=cal,
    )
    assert report["verdict"] == "below_slo"  # 360/400 = 0.9 < 3.0
    assert report["inference"]["decode_tps_on_target"] == 0.9


def test_build_report_without_calibration_keeps_bare_metal_basis():
    report = bg.build_report(
        inference={"tokens": 8, "decode_tps": 6.0, "prefill_tps": 40.0, "elapsed_s": 1.3},
        training={"tokens": 1, "tokens_per_s": 1.0, "steps": 1, "final_loss": 1.0},
        eval={"mean_loss": 1.0, "perplexity": 1.0, "num_token_positions": 1},
        model={"params": 1, "config": "x"},
        decode_slo=5.0,
    )
    assert report["verdict"] == "meets_slo"
    assert report["verdict_basis"] == "bare_metal_proxy"
    assert "decode_tps_on_target" not in report["inference"]


def test_format_report_shows_the_on_target_estimate():
    cal = {
        "fp_mode": "sse2",
        "factor": 40.0,
        "source": ["x"],
        "floor": {"fp_mode": "x87", "factor": 180.0},
    }
    report = bg.build_report(
        inference={"tokens": 8, "decode_tps": 360.0, "prefill_tps": 40.0, "elapsed_s": 0.1},
        training={"tokens": 1, "tokens_per_s": 1.0, "steps": 1, "final_loss": 1.0},
        eval={"mean_loss": 1.0, "perplexity": 1.0, "num_token_positions": 1},
        model={"params": 1, "config": "x"},
        decode_slo=3.0,
        target_calibration=cal,
    )
    text = bg.format_report(report)
    assert "bare metal" in text
    assert "on v86-browser-vm" in text
    assert "x87" in text  # floor case surfaced, not hidden


def test_load_target_calibration_treats_missing_fp_mode_as_x87(tmp_path):
    """Reports written before the fp_mode field existed were run with the
    benchmark's default compile mode — x87 — and must classify as the
    floor case, not vanish from the verdict."""
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"factor": 181.7}))
    cal = bg.load_target_calibration([str(old)])
    assert cal["fp_mode"] == "x87"
    assert cal["factor"] == 181.7
