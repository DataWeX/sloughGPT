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


def test_default_decode_slo_is_stated_for_v86():
    """The v86 target's default SLO is explicit in the module — not implied."""
    assert bg.DEFAULT_TARGET == "v86-browser-vm"
    assert bg.DEFAULT_DECODE_SLO_TPS > 0
    # v86 is emulated x86: its budget is a FRACTION of bare metal.
    assert bg.DEFAULT_DECODE_SLO_TPS <= 10.0
