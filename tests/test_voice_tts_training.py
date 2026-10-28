"""TDD tests for real-gradient TTS training (voice slice 2, card 20261025_111237).

Proves three claims that were previously false:
1. MelDecoder.train_step populates REAL gradients (the old _manual_backward
   subtracted random noise and never touched .grad).
2. Repeated steps on a fixed pair DECREASE the loss (learning actually happens).
3. The espeak dataset adapter renders text into TTSEngine-sample-rate waveforms
   and checkpoints round-trip.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

HAS_ESPEAK = shutil.which("espeak") is not None


def _engine():
    from domain.voice._internal.tts import TTSEngine

    return TTSEngine()


def _sine_pair(duration=0.4, freq=220.0, sample_rate=22050):
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, dtype=np.float32)
    wav = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return "hello", wav


class TestRealGradients:
    def test_train_step_updates_weights(self):
        """The real proof training is real: optimizer moves the weights.

        (SloAdam.step clears .grad after use, so post-hoc grad inspection
        would see None by design — assert on parameter data instead.)
        """
        tts = _engine()
        dec = tts.decoder
        watched = {
            "fc_mel": (dec.fc_mel.weight, dec.fc_mel.weight.data.copy()),
            "enc_W_ih": (dec.encoder_lstm.W_ih.weight, dec.encoder_lstm.W_ih.weight.data.copy()),
            "enc_norm": (
                dec.encoder_lstm.hidden_norm.weight,
                dec.encoder_lstm.hidden_norm.weight.data.copy(),
            ),
        }
        text, wav = _sine_pair()
        loss = tts.train_step(text, wav)
        assert loss > 0.0, "loss must be computed"
        for label, (param, before) in watched.items():
            assert not np.allclose(param.data, before), f"{label} was never updated"

    def test_overfit_reduces_loss(self):
        tts = _engine()
        text, wav = _sine_pair()
        losses = [tts.train_step(text, wav) for _ in range(40)]
        first = float(np.mean(losses[:5]))
        last = float(np.mean(losses[-5:]))
        assert np.isfinite(losses[-1]), "loss diverged to non-finite"
        assert last < first, f"loss did not decrease: first5={first:.6g} last5={last:.6g}"

    def test_train_epoch_smoke(self):
        tts = _engine()
        p1 = _sine_pair(freq=220.0)
        p2 = _sine_pair(freq=440.0)
        avg = tts.train_epoch([p1, p2], max_frames=40)
        assert np.isfinite(avg) and avg > 0


@pytest.mark.skipif(not HAS_ESPEAK, reason="espeak binary not installed")
class TestEspeakPairs:
    def test_render_matches_tts_contract(self):
        from domain.voice._internal.espeak_pairs import render

        wav = render("hello world from the voice loop")
        assert isinstance(wav, np.ndarray)
        assert wav.dtype == np.float32
        assert -1.01 <= wav.min() and wav.max() <= 1.01
        assert len(wav) >= int(0.3 * 22050), "rendered clip suspiciously short"

    def test_build_pairs_preserves_texts(self):
        from domain.voice._internal.espeak_pairs import build_pairs

        texts = ["one two three", "four five six"]
        pairs = build_pairs(texts)
        assert [p[0] for p in pairs] == texts
        assert all(isinstance(p[1], np.ndarray) for p in pairs)

    def test_espeak_roundtrip_feeds_training(self):
        from domain.voice._internal.espeak_pairs import render

        tts = _engine()
        wav = render("train the voice")
        loss = tts.train_step("train the voice", wav)
        assert np.isfinite(loss)


class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path: Path):
        tts = _engine()
        text, wav = _sine_pair()
        tts.train_step(text, wav)
        path = str(tmp_path / "tts.npz")
        tts.save(path)

        tts2 = _engine()
        tts2.load(path)
        for a, b in zip(tts.parameters(), tts2.parameters(), strict=True):
            assert np.allclose(a.data, b.data), "checkpoint round-trip lost weights"


class TestParameterContract:
    def test_named_parameters_is_the_same_live_set_in_same_order(self):
        """Optimizer set and checkpoint keys are the SAME tensors, same order.

        Dead weight (encoder layer-2, attention head, unused embedding,
        parity-dead biases) must be in neither — advertising it would make
        the benchmark's weight-movement gate meaningless.
        """
        tts = _engine()
        param_ids = [id(p) for p in tts.decoder.parameters()]
        named_ids = [id(p) for _, p in tts.decoder.named_parameters()]
        assert param_ids == named_ids, "parameters() and named_parameters() diverged"
        assert len(param_ids) == 11, f"expected 11 live tensors, got {len(param_ids)}"

    def test_no_dead_tensor_advertised(self):
        """Every advertised parameter moves after a step (all are forward-live)."""
        tts = _engine()
        dec = tts.decoder
        before = [p.data.copy() for p in dec.parameters()]
        text, wav = _sine_pair()
        tts.train_step(text, wav, max_frames=40)
        for i, (a, p) in enumerate(zip(before, dec.parameters(), strict=True)):
            assert not np.allclose(a, p.data), f"parameter #{i} never receives a gradient"


class TestInferenceParity:
    def test_numpy_generate_still_runs_after_training_change(self):
        tts = _engine()
        text, wav = _sine_pair()
        tts.train_step(text, wav)
        result = tts.text_to_waveform("hello world")
        assert isinstance(result, np.ndarray) and result.size > 0
        mel = tts.decoder.generate(tts._phoneme_encoder.encode("hello"), max_frames=10)
        assert mel.ndim == 2 and mel.shape[0] == tts.decoder.n_mels


class TestEvalMode:
    def test_update_false_returns_loss_without_moving_weights(self):
        """Held-out evaluation must observe, not train (optimizer skipped)."""
        tts = _engine()
        text, wav = _sine_pair()
        before = [p.data.copy() for p in tts.parameters()]
        loss = tts.train_step(text, wav, update=False)
        assert np.isfinite(loss) and loss > 0.0, "eval loss must still be computed"
        for a, p in zip(before, tts.parameters(), strict=True):
            assert np.array_equal(a, p.data), "update=False mutated a parameter"

    def test_update_false_does_not_clear_graph_grads_externally(self):
        tts = _engine()
        text, wav = _sine_pair()
        tts.train_step(text, wav, update=False)
        tts.train_step(text, wav, update=True)
        assert all(p.grad is None for p in tts.parameters()), "step must clear grads"

    def test_eval_loss_is_stable_across_calls(self):
        """eval_loss observes: same weights in, same number out — and skips empty sets."""
        tts = _engine()
        pairs = [_sine_pair(freq=220.0), _sine_pair(freq=440.0)]
        a = tts.eval_loss(pairs, max_frames=30)
        b = tts.eval_loss(pairs, max_frames=30)
        assert np.isfinite(a) and a > 0.0
        assert a == pytest.approx(b), "eval_loss mutated state between calls"
        assert tts.eval_loss([]) == 0.0


class TestDefaultCorpus:
    def test_corpus_is_phoneme_covering_and_splitable(self):
        from domain.voice._internal.espeak_pairs import DEFAULT_CORPUS, DEFAULT_HOLDOUT

        assert len(DEFAULT_CORPUS) >= 30, "corpus too small for a starter split"
        assert DEFAULT_HOLDOUT < len(DEFAULT_CORPUS), "holdout would eat the train set"
        train, holdout = DEFAULT_CORPUS[:-DEFAULT_HOLDOUT], DEFAULT_CORPUS[-DEFAULT_HOLDOUT:]
        assert len(train) >= 24 and len(holdout) == DEFAULT_HOLDOUT
        assert all(t.strip() for t in train + holdout)
