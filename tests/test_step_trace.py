"""StepTrace — versioned per-step reduction set + dispatch residency counters.

Design (2026-11-02): "ship receipts, not cargo." A training step crosses the
host/device boundary with a *fixed, versioned* set of scalars instead of raw
tensors: live control signals (loss, lr, grad_norm), bug tripwires (device
residency, matmul path counters, steps/s), and replay anchors (batch
fingerprint, step). Raw tensors stay out of the trace — they are regenerable
by seeded replay; the trace is the receipt.

Why it exists: job_c87a2dd6 (2026-10-31) ran at 0.4 steps/s with the GPU
never used — the silent CPU fallback hid inside ``slonet._matmul`` for the
whole run. With this trace, step 1's counters answer "which path did the
matmuls take?" instead of a mystery at hour 3.

Schema ratchet: ``STEP_TRACE_FIELDS`` must equal the dataclass fields, and
the version may only move with a reviewed edit (page-baseline pattern).
v1 deliberately excludes queue_depth / update_norm / ES probe fields — they
need the async pipeline and evolution-strategies mode that do not exist yet.
No fake fields.
"""

from __future__ import annotations

from dataclasses import fields as dc_fields

import numpy as np
import pytest

from domain.slolib._internal.gpu import dispatch_stats
from domain.training._internal.trace import (
    STEP_TRACE_FIELDS,
    STEP_TRACE_VERSION,
    StepTrace,
    fingerprint_batch,
)


def _sample_trace(**over) -> StepTrace:
    base = dict(
        step=1,
        loss=1.0,
        raw_loss=1.1,
        lr=1e-3,
        grad_norm=0.5,
        skipped=False,
        steps_per_s=12.0,
        device="cpu",
        tokens_this_step=96,
        batch_fingerprint="0123456789abcdef",
        gpu_matmuls=0,
        cpu_matmuls=3,
        fallbacks=0,
        h2d_bytes=0,
        d2h_bytes=0,
    )
    base.update(over)
    return StepTrace(**base)


# ---------------------------------------------------------------------------
# Schema ratchet
# ---------------------------------------------------------------------------


class TestStepTraceSchema:
    def test_version_is_pinned(self):
        assert STEP_TRACE_VERSION == 1

    def test_fields_tuple_matches_dataclass(self):
        assert STEP_TRACE_FIELDS == tuple(f.name for f in dc_fields(StepTrace))

    def test_sample_satisfies_schema(self):
        trace = _sample_trace()
        assert set(trace.to_dict()) == set(STEP_TRACE_FIELDS)

    def test_trace_is_frozen(self):
        trace = _sample_trace()
        with pytest.raises(Exception):
            trace.step = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Batch fingerprint (replay anchor)
# ---------------------------------------------------------------------------


class TestFingerprintBatch:
    def test_deterministic_for_identical_input(self):
        a = np.arange(48, dtype=np.int32).reshape(6, 8)
        assert fingerprint_batch(a) == fingerprint_batch(a.copy())

    def test_discriminates_changed_content(self):
        a = np.arange(48, dtype=np.int32).reshape(6, 8)
        b = a.copy()
        b[3, 3] = -1
        assert fingerprint_batch(a) != fingerprint_batch(b)

    def test_noncontiguous_input_matches_contiguous(self):
        a = np.arange(48, dtype=np.int32).reshape(6, 8)
        assert fingerprint_batch(a.T) == fingerprint_batch(np.ascontiguousarray(a.T))

    def test_digest_length(self):
        assert len(fingerprint_batch(np.zeros(4, dtype=np.float32))) == 16


# ---------------------------------------------------------------------------
# Dispatch counters at the slonet._matmul seam (the training path)
# ---------------------------------------------------------------------------


def _eligible_operands():
    """Shapes whose out_elems (256*128 = 32768) clear slonet's 16384
    accelerator threshold — the same class of matmul real training hits."""
    a = np.random.RandomState(0).randn(256, 512).astype(np.float32)
    b = np.random.RandomState(1).randn(512, 128).astype(np.float32)
    return a, b


class TestDispatchCounters:
    def test_cpu_path_counts_exactly_one_cpu_matmul(self, monkeypatch):
        import domain.training._internal.slonet as slonet

        monkeypatch.setattr(slonet, "_get_accelerator", lambda: None)
        d0 = dispatch_stats()
        slonet._matmul(*_eligible_operands())
        d1 = dispatch_stats()
        assert d1["cpu_matmuls"] - d0["cpu_matmuls"] == 1
        assert d1["gpu_matmuls"] == d0["gpu_matmuls"]
        assert d1["fallbacks"] == d0["fallbacks"]

    def test_accelerator_path_counts_one_matmul_whichever_backend(self, monkeypatch):
        """On a GPU machine the accelerator path runs (gpu_matmuls +1);
        on CPU-only machines the fallback-to-no-accel path runs. Either way
        exactly one eligible matmul is attributed to exactly one path."""
        import domain.training._internal.slonet as slonet

        d0 = dispatch_stats()
        slonet._matmul(*_eligible_operands())
        d1 = dispatch_stats()
        gpu = d1["gpu_matmuls"] - d0["gpu_matmuls"]
        cpu = d1["cpu_matmuls"] - d0["cpu_matmuls"]
        assert gpu + cpu == 1, f"expected exactly one path, got gpu={gpu} cpu={cpu}"
        assert d1["fallbacks"] - d0["fallbacks"] in (0, 1)

    def test_failed_accelerator_matmul_records_fallback_and_cpu(self, monkeypatch):
        import domain.training._internal.slonet as slonet

        class _BoomAcc:
            name = "cuda"

            def matmul(self, a, b):
                raise TypeError("simulated NumPy>=2 dispatch failure")

        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _BoomAcc())
        d0 = dispatch_stats()
        slonet._matmul(*_eligible_operands())
        d1 = dispatch_stats()
        # The job_c87a2dd6 signature: a GPU attempt that silently degraded.
        assert d1["fallbacks"] - d0["fallbacks"] == 1
        assert d1["cpu_matmuls"] - d0["cpu_matmuls"] == 1
        assert d1["gpu_matmuls"] == d0["gpu_matmuls"]

    def test_tiny_matmul_not_counted_by_design(self, monkeypatch):
        import domain.training._internal.slonet as slonet

        monkeypatch.setattr(slonet, "_get_accelerator", lambda: None)
        d0 = dispatch_stats()
        slonet._matmul(np.random.randn(4, 4), np.random.randn(4, 4))  # 16 elems, below 16384
        d1 = dispatch_stats()
        assert d1["cpu_matmuls"] == d0["cpu_matmuls"]

    def test_stats_is_a_copy_never_resets_global_state(self):
        a = dispatch_stats()
        a["cpu_matmuls"] = -999  # mutating the snapshot must not leak
        b = dispatch_stats()
        assert b["cpu_matmuls"] != -999


# ---------------------------------------------------------------------------
# Trainer integration — the trace is assembled every training step
# ---------------------------------------------------------------------------


def _tiny_trainer(tmp_path, monkeypatch, max_steps: int = 3) -> object:
    """Small-but-eligible trainer: block*embed shapes must clear slonet's
    16384-element accelerator threshold so matmul path counters actually
    fire (block=128, embed=128 → qkv out_elems = 128*384 = 49152)."""
    from domain.training._internal.train_pipeline import SloughGPTTrainer

    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefghij" * 64, encoding="utf-8")
    return SloughGPTTrainer(
        data_path=str(corpus),
        n_embed=128,
        n_layer=1,
        n_head=2,
        block_size=128,
        batch_size=2,
        epochs=1,
        max_steps=max_steps,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
        checkpoint_interval=100_000,
    )


class TestTrainerTraceIntegration:
    def test_every_step_leaves_a_complete_trace(self, tmp_path, monkeypatch):
        trainer = _tiny_trainer(tmp_path, monkeypatch)
        trainer.train()
        trace = trainer.get_step_trace()
        assert trace is not None, "no trace assembled during training"
        assert trace.step == trainer.global_step
        assert trace.device == "cpu"
        assert trace.tokens_this_step > 0
        assert len(trace.batch_fingerprint) == 16
        assert trace.steps_per_s > 0
        assert isinstance(trace.loss, float)
        assert isinstance(trace.raw_loss, float)
        assert trace.lr > 0
        # accumulation=1 + max_grad_norm>0 → grad norm every boundary step
        assert isinstance(trace.grad_norm, float)
        assert trace.skipped is False
        assert set(trace.to_dict()) == set(STEP_TRACE_FIELDS)

    def test_ring_is_bounded_and_lookup_works(self, tmp_path, monkeypatch):
        trainer = _tiny_trainer(tmp_path, monkeypatch)
        trainer.train()
        assert trainer._step_traces.maxlen == 128
        first = trainer.get_step_trace(step=1)
        assert first is not None and first.step == 1
        assert trainer.get_step_trace(step=99_999) is None

    def test_accessor_returns_none_before_first_step(self, tmp_path, monkeypatch):
        trainer = _tiny_trainer(tmp_path, monkeypatch)
        assert trainer.get_step_trace() is None

    def test_counters_in_trace_are_step_scoped_deltas(self, tmp_path, monkeypatch):
        import domain.training._internal.slonet as slonet

        monkeypatch.setattr(slonet, "_get_accelerator", lambda: None)
        trainer = _tiny_trainer(tmp_path, monkeypatch)
        trainer.train()
        trace = trainer.get_step_trace()
        assert trace is not None
        # CPU-only forced path: every eligible matmul landed on CPU, none on GPU.
        assert trace.gpu_matmuls == 0
        assert trace.cpu_matmuls >= 1
        assert trace.fallbacks == 0
