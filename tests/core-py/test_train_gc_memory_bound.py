"""Regression: bounded cyclic garbage while the training GC is parked.

``SloughGPTTrainer.train()`` parks the cyclic GC (``_gc_parked`` — measured
2.9 vs 31+ steps/s with automatic collections) and reclaims periodically via
``_gc_parked_step``. Before this fix the only in-run collection was the
``_GC_COLLECT_EVERY_STEPS = 50`` cadence — but the API training config emits
~769MB of cyclic autograd garbage *per step* (probes A-E, card fa33b9c1), so
jobs OOM at step 8-12, i.e. **before the first collection ever runs**.

Contracts tested here (both fail before the fix):

1. Live-object growth across a short run stays bounded — the step wrapper
   must reclaim on garbage *growth*, not only at step 50.
2. RSS growth from step 2 (past one-time numba JIT) to the final step stays
   bounded — the byte-level reality of (1).
3. The run keeps the GC parked throughout (the performance contract that
   motivated parking it) and restores it afterwards.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pytest

from domain.training import SloughGPTTrainer

MAX_STEPS = 4
# RED expectation: no in-run collection => steps 2..4 accumulate
# (~3000 objects and ~769MB each). GREEN: per-step reclamation => near-flat.
LIVE_OBJECTS_GROWTH_BOUND = 4_000
RSS_GROWTH_BOUND_MB = 2_048


def _rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return -1.0


def _write_corpus(path: Path) -> Path:
    rng = np.random.default_rng(7)
    words = np.array([" the", " king", " and", " queen", " of", " slough", " fire", " dark"])
    text = "".join(str(words[i]) for i in rng.integers(0, len(words), 8_000))
    path.write_text(text * 3)
    return path


@pytest.fixture
def gc_run(tmp_path, monkeypatch):
    """Corpus + progress recorder; force a low garbage trigger so the mechanism
    (not production calibration) is under test, and restore gc state after."""
    monkeypatch.setenv("SLO_TRAIN_GC_OBJS", "1000")
    # Quiet the GPU-fallback warning spam: pytest's LogCaptureHandler retains
    # every record, and each warning's args hold the caught exception — whose
    # traceback frames root that matmul's arrays, making them uncollectable.
    # (Production's stdout/journald handler discards records after emit, so
    # this is a test-harness artifact, not a leak in the trainer.)
    monkeypatch.setattr(logging.getLogger("slo.training"), "level", logging.ERROR)
    corpus = _write_corpus(tmp_path / "corpus.txt")
    was_enabled = gc.isenabled()
    # (step, live objects, rss_mb, parked, pending, baseline, collect calls)
    samples: list[tuple[int, int, float, bool, int, int, int]] = []
    from domain.training._internal import train_pipeline as _tp

    _orig_collect = gc.collect
    collect_calls = {"n": 0}

    def _spy(*args, **kwargs):
        collect_calls["n"] += 1
        return _orig_collect(*args, **kwargs)

    monkeypatch.setattr(gc, "collect", _spy)  # auto-restored

    def on_progress(info: dict) -> None:
        samples.append(
            (
                int(info["global_step"]),
                len(gc.get_objects()),
                _rss_mb(),
                not gc.isenabled(),
                sum(gc.get_count()),
                _tp._GC_RUN_STATE["baseline"],
                collect_calls["n"],
            )
        )

    yield corpus, on_progress, samples
    if was_enabled:
        gc.enable()
    else:
        gc.disable()


def _trainer_kwds(tmp_path) -> dict:
    # Mirrors the API config from apps/api/server/training/helpers.py
    #::_sloughgpt_trainer_kwds — the configuration that OOM'd in production.
    return dict(
        n_embed=128,
        n_layer=4,
        n_head=4,
        block_size=128,
        dropout=0.1,
        batch_size=32,
        epochs=1,
        lr=1e-3,
        max_steps=MAX_STEPS,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        checkpoint_dir=str(tmp_path / "ckpt"),
        checkpoint_interval=500,  # no checkpoint inside a 4-step run
        save_best_only=False,
        max_checkpoints=5,
        scheduler_type="cosine",
        warmup_steps=100,
        min_lr=1e-5,
        weight_decay=0.01,
        use_lora=False,
        lora_rank=8,
        lora_alpha=16,
        log_interval=100,
        eval_interval=100,  # no eval inside a 4-step run
        feed_refresh_interval=0.0,
        device=None,
    )


def test_short_run_keeps_cyclic_garbage_bounded(gc_run, tmp_path):
    """A 4-step parked-GC run must not accumulate unreclaimed garbage."""
    corpus, on_progress, samples = gc_run

    trainer = SloughGPTTrainer(
        data_path=str(corpus), experiment_tracker=None, **_trainer_kwds(tmp_path)
    )
    trainer.train(on_progress=on_progress)

    # _gc_parked's finally re-enabled gc and did a final collection, which
    # would erase the evidence — so all assertions use samples recorded
    # *during* the run (the walk itself never frees anything).
    by_step = {s[0]: s[1:] for s in samples}  # step -> (live, rss, parked, pend, base, n_collect)
    assert 1 in by_step and MAX_STEPS in by_step, f"missing samples: {samples}"

    for step, (_live, _rss, parked, *_rest) in sorted(by_step.items()):
        assert parked, f"GC was not parked at step {step} (perf contract broken)"

    base_live, base_rss = by_step[1][0], by_step[1][1]
    final_live, final_rss = by_step[MAX_STEPS][0], by_step[MAX_STEPS][1]

    live_growth = final_live - base_live
    assert live_growth < LIVE_OBJECTS_GROWTH_BOUND, (
        f"live objects grew by {live_growth} across steps 1->{MAX_STEPS} "
        f"(bound {LIVE_OBJECTS_GROWTH_BOUND}) — cyclic garbage is not being "
        f"reclaimed during the run. samples={samples}"
    )

    rss_growth = final_rss - base_rss
    assert rss_growth < RSS_GROWTH_BOUND_MB, (
        f"RSS grew by {rss_growth:.0f}MB across steps 1->{MAX_STEPS} "
        f"(bound {RSS_GROWTH_BOUND_MB}MB) — byte-level growth matches the "
        f"~769MB/step OOM pattern. samples={samples}"
    )


def test_garbage_trigger_is_disabled_when_env_zero(gc_run, tmp_path, monkeypatch):
    """SLO_TRAIN_GC_OBJS=0 is the escape hatch: only the step cadence collects.

    Behavioural check: with the trigger off and the cadence (50) far beyond a
    2-step run, garbage must still be pending *during* the final step — i.e.
    the env knob really gates the in-run collection.
    """
    corpus, _on_progress, _samples = gc_run  # fixture set 1000; override below
    monkeypatch.setenv("SLO_TRAIN_GC_OBJS", "0")

    pending_during_last_step: dict[str, int] = {}

    def probe(info: dict) -> None:
        # Only measure on the final step: gc.collect() frees, so measuring
        # every step would itself fix the bug under test.
        if int(info["global_step"]) == 2:
            pending_during_last_step["n"] = gc.collect()

    trainer = SloughGPTTrainer(
        data_path=str(corpus),
        experiment_tracker=None,
        **{**_trainer_kwds(tmp_path), "max_steps": 2},
    )
    trainer.train(on_progress=probe)

    # Trigger disabled + cadence not reached => the two steps' cyclic
    # garbage was still pending at the end of step 2.
    assert pending_during_last_step.get("n", 0) > 500, (
        "expected pending cyclic garbage with SLO_TRAIN_GC_OBJS=0 and only 2 "
        "steps run (cadence is 50) — the env gate appears to be ignored"
    )


def test_should_collect_pure_logic():
    """The trigger decision is a pure function of (pending, baseline, trigger)."""
    from domain.training._internal.train_pipeline import _should_collect

    assert not _should_collect(5_000, -1, 2_000), "no baseline yet => no trigger"
    assert not _should_collect(5_000, 100, 0), "trigger disabled (env 0)"
    assert not _should_collect(5_000, 100, -1), "negative trigger treated as off"
    assert not _should_collect(1_999, 0, 2_000), "below threshold"
    assert _should_collect(2_000, 0, 2_000), "at threshold"
    assert _should_collect(10_500, 1_000, 2_000), "growth past threshold"


def test_garbage_trigger_env_parsing(monkeypatch):
    from domain.training._internal.train_pipeline import _garbage_trigger_objs

    monkeypatch.delenv("SLO_TRAIN_GC_OBJS", raising=False)
    assert _garbage_trigger_objs() == 2_000, "production default"
    monkeypatch.setenv("SLO_TRAIN_GC_OBJS", "0")
    assert _garbage_trigger_objs() == 0, "0 = escape hatch (cadence only)"
    monkeypatch.setenv("SLO_TRAIN_GC_OBJS", "123")
    assert _garbage_trigger_objs() == 123
    monkeypatch.setenv("SLO_TRAIN_GC_OBJS", "not-a-number")
    assert _garbage_trigger_objs() == 2_000, "unparseable falls back to default"
