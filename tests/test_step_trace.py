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

import hashlib
from dataclasses import fields as dc_fields

import numpy as np
import pytest

from domain.slolib._internal.gpu import dispatch_stats
from domain.training._internal.trace import (
    STEP_TRACE_FIELDS,
    STEP_TRACE_VERSION,
    StepTrace,
    fingerprint_array,
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


class TestFingerprintArray:
    """``fingerprint_array`` is the dataset-identity hash (card 20261104_081626).

    Same contract as ``fingerprint_batch`` — deterministic, content-addressed,
    layout-normalised — but sized as a long-lived identity (64 hex chars) and
    chunked so hashing a whole corpus never copies it.
    """

    def test_deterministic_for_identical_input(self):
        a = np.arange(4096, dtype=np.int64)
        assert fingerprint_array(a) == fingerprint_array(a.copy())

    def test_discriminates_changed_content(self):
        a = np.arange(4096, dtype=np.int64)
        b = a.copy()
        b[2000] = -1
        assert fingerprint_array(a) != fingerprint_array(b)

    def test_noncontiguous_input_matches_contiguous(self):
        a = np.arange(48, dtype=np.int64).reshape(6, 8)
        assert fingerprint_array(a.T) == fingerprint_array(np.ascontiguousarray(a.T))

    def test_digest_length_is_64_hex(self):
        assert len(fingerprint_array(np.zeros(4, dtype=np.int64))) == 64

    def test_empty_array_still_hashes(self):
        assert len(fingerprint_array(np.array([], dtype=np.int64))) == 64

    def test_chunked_matches_unchunked_reference(self):
        """The chunk loop must not change the digest — same bytes, same hash.

        ``fingerprint_batch`` hashes with a single ``tobytes()`` copy; the
        dataset hash cannot afford that on a large corpus, so it streams. If
        the stream ever disagrees with the one-shot digest, every recorded
        signature silently changes and equivalence lookups break.
        """
        a = np.arange(300_000, dtype=np.int64)  # 2.4 MB — crosses the chunk size
        expected = hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=32).hexdigest()
        assert fingerprint_array(a) == expected

    def test_dtype_matters(self):
        """int64 token ids and the same values as int32 are different bytes."""
        a = np.arange(64, dtype=np.int64)
        b = np.arange(64, dtype=np.int32)
        assert fingerprint_array(a) != fingerprint_array(b)


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


class TestStepTraceFlag:
    """``step_trace`` feature flag — emission is gated, counters are not.

    Env vars and feature flags never mix: the flag lives in the registry
    (``config/feature_flags.json`` + ``_register_defaults``), and its
    per-process override is the registry's own generated ``SLO_FF_*`` seam.
    The dispatch counters in ``gpu.py`` stay unconditional behind any gate —
    the step-1 tripwire must survive the flag being switched off.
    """

    def test_flag_registered_and_enabled_by_default(self, monkeypatch):
        from domain.shared._internal.feature_flags import FeatureFlags, is_enabled

        monkeypatch.delenv("SLO_FF_STEP_TRACE", raising=False)
        assert "step_trace" in FeatureFlags.list_all(), "step_trace flag not registered"
        assert is_enabled("step_trace") is True, "flag must default to enabled (tripwire)"

    def test_flag_off_skips_emission_but_counters_still_count(self, tmp_path, monkeypatch):
        import domain.training._internal.slonet as slonet

        # Force the CPU path so residency counters are visibly live.
        monkeypatch.setattr(slonet, "_get_accelerator", lambda: None)
        monkeypatch.setenv("SLO_FF_STEP_TRACE", "0")

        before = dispatch_stats()["cpu_matmuls"]
        trainer = _tiny_trainer(tmp_path, monkeypatch)
        payloads: list[dict] = []
        trainer.train(on_progress=payloads.append)

        # Gated: no receipt was assembled into the ring.
        assert trainer.get_step_trace() is None, "ring must stay empty when flag is off"

        # Gated: progress payload omits residency keys (absence, not lying zeros).
        assert payloads, "expected at least one progress emit"
        last = payloads[-1]
        for key in ("gpu_matmuls", "cpu_matmuls", "fallbacks"):
            assert key not in last, f"{key} must be absent when flag is off"
        assert "device" in last, "device is not trace-derived and stays unconditional"

        # NOT gated: the tripwire counters still accumulated during this run.
        after = dispatch_stats()["cpu_matmuls"]
        assert after - before >= 1, "dispatch counters must count with the flag off"
