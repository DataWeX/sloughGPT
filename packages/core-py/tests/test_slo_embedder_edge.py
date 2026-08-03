"""Fast edge-path tests for slo_embedder — no training loop, default run (not slow)."""

import os

import numpy as np
import pytest

from domains.inference.slo_embedder import (
    SloTextEmbedder,
    _build_bpe_tokenizer,
    _build_vocab,
    _label_by_meaning,
    _save_checkpoint,
)


class TestBpeTokenizerEdge:
    def test_bpe_build_failure_returns_none(self, monkeypatch):
        class Boom:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("no tokenizer available")

        monkeypatch.setattr("domains.multimodal.bpe_tokenizer.BPETokenizer", Boom)
        assert _build_bpe_tokenizer(["hello world"]) == (None, None)


class TestBuildVocabEdge:
    def test_vocab_saturation_break(self):
        vocab, itos = _build_vocab(["a b c d e f"], vocab_size=6)
        assert len(vocab) == 6
        for k, v in vocab.items():
            assert itos[v] == k


class TestLabelByMeaningEdge:
    def test_no_points_store(self):
        assert _label_by_meaning("hello", points_store=None) is None

    def test_zero_vector_returns_none(self, monkeypatch):
        from domains.inference import vector_store

        monkeypatch.setattr(
            vector_store, "simple_embed", lambda text, dimension: [0.0] * dimension
        )

        class Store:
            dimension = 8

            def classify(self, vec):
                raise AssertionError("classify must not be called for a zero vector")

        assert _label_by_meaning("hello", points_store=Store()) is None


class TestTrainEmbedderWhitespaceFallback:
    def test_whitespace_fallback_path(self, monkeypatch, tmp_path):
        import domains.inference.slo_embedder as m

        monkeypatch.setattr(
            m, "_build_bpe_tokenizer", lambda texts, vocab_size=2048: (None, None)
        )
        path = str(tmp_path / "ws.sou")
        progress = []
        result = m.train_embedder(
            ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"],
            vocab_size=64,
            embed_dim=8,
            max_seq_len=16,
            n_heads=2,
            n_layers=1,
            epochs=1,
            batch_size=2,
            save_path=path,
            progress_callback=lambda epoch, loss, total: progress.append((epoch, total)),
        )
        assert result["vocab_size"] > 0
        assert os.path.exists(path)
        assert len(progress) == 1


class TestSaveCheckpointEdge:
    def _fake_encoder(self, raise_on_params=False):
        class P:
            def __init__(self):
                self.data = np.array([1.0, 2.0], dtype=np.float32)

        class Enc:
            def __init__(self):
                self._raise = raise_on_params
                self.blocks = []
                self._calls = 0

            def parameters(self):
                self._calls += 1
                if self._raise and self._calls > 1:
                    raise RuntimeError("params unavailable")
                return [P()]

        return Enc()

    def test_cleanup_on_failure(self, tmp_path):
        path = str(tmp_path / "x.sou")
        enc = self._fake_encoder(raise_on_params=True)
        with pytest.raises(RuntimeError):
            _save_checkpoint(path, enc, {"a": 1}, {1: "a"}, 8, 16, 2, 1)
        assert not list(tmp_path.glob("*.tmp"))

    def test_bpe_save_failure_warns_but_writes_file(self, tmp_path):
        path = str(tmp_path / "x.sou")
        enc = self._fake_encoder()

        class BadBPE:
            def save(self, path):
                raise RuntimeError("cannot persist bpe")

        _save_checkpoint(path, enc, {"a": 1}, {1: "a"}, 8, 16, 2, 1, bpe=BadBPE())
        assert os.path.exists(path)
        assert not os.path.exists(path.replace(".sou", "-bpe.json"))

    def test_rename_and_unlink_both_fail(self, tmp_path, monkeypatch):
        path = str(tmp_path / "x.sou")
        enc = self._fake_encoder()

        def boom_rename(src, dst):
            raise OSError("rename failed")

        def boom_unlink(p):
            raise OSError("unlink failed")

        monkeypatch.setattr("os.rename", boom_rename)
        monkeypatch.setattr("os.unlink", boom_unlink)
        with pytest.raises(OSError):
            _save_checkpoint(path, enc, {"a": 1}, {1: "a"}, 8, 16, 2, 1)
        assert not os.path.exists(path)


def _write_v3_sou(path, system_prompt, params):
    from domains.inference.slo_format import write_v3_sou

    write_v3_sou(
        str(path),
        {"system_prompt": system_prompt},
        {f"p{i}": arr for i, arr in enumerate(params)},
    )


class TestSloTextEmbedderLoad:
    def test_invalid_magic_returns_none(self, tmp_path):
        p = tmp_path / "bad.sou"
        p.write_bytes(b"BOGUS" + b"\x00" * 32)
        assert SloTextEmbedder.load(str(p)) is None

    def test_invalid_system_prompt_returns_none(self, tmp_path):
        p = tmp_path / "m.sou"
        _write_v3_sou(p, "embed_dim=abc max_seq_len=8 n_heads=2 n_layers=1",
                      [np.zeros(4, np.float32)])
        assert SloTextEmbedder.load(str(p)) is None

    def test_load_without_vocab_sidecar(self, tmp_path):
        p = tmp_path / "m.sou"
        _write_v3_sou(p, "embed_dim=8 max_seq_len=8 n_heads=2 n_layers=1",
                      [np.zeros(4, np.float32)])
        e = SloTextEmbedder.load(str(p))
        assert e is not None
        assert e.vocab == {}

    def test_load_with_bpe_tokenizer(self, tmp_path):
        from domains.multimodal.bpe_tokenizer import BPETokenizer

        p = tmp_path / "m.sou"
        _write_v3_sou(p, "embed_dim=8 max_seq_len=8 n_heads=2 n_layers=1",
                      [np.zeros(4, np.float32)])
        bpe = BPETokenizer(vocab_size=64)
        bpe.train(["hello world", "this is a test", "neural network training"])
        bpe.save(str(p).replace(".sou", "-bpe.json"))
        e = SloTextEmbedder.load(str(p))
        assert e is not None
        assert e.encode_fn is not None
        assert len(e.embed("hello world")) == 8

    def test_load_with_corrupt_bpe_sidecar(self, tmp_path):
        p = tmp_path / "m.sou"
        _write_v3_sou(p, "embed_dim=8 max_seq_len=8 n_heads=2 n_layers=1",
                      [np.zeros(4, np.float32)])
        (tmp_path / "m-bpe.json").write_text("{not valid json")
        e = SloTextEmbedder.load(str(p))
        assert e is not None
        assert e.encode_fn is None

    def test_load_with_bpe_sidecar_load_failure(self, tmp_path, monkeypatch):
        p = tmp_path / "m.sou"
        _write_v3_sou(p, "embed_dim=8 max_seq_len=8 n_heads=2 n_layers=1",
                      [np.zeros(4, np.float32)])
        (tmp_path / "m-bpe.json").write_text("{}")
        import domains.multimodal.bpe_tokenizer as bpe_mod

        def failing_load(self, path):
            return False

        monkeypatch.setattr(bpe_mod.BPETokenizer, "load", failing_load)
        e = SloTextEmbedder.load(str(p))
        assert e is not None
        assert e.encode_fn is None


class TestEmbedDimAdaptation:
    def _embedder(self, tmp_path, output_dim):
        p = tmp_path / "m.sou"
        _write_v3_sou(p, f"embed_dim={output_dim} max_seq_len=8 n_heads=2 n_layers=1",
                      [np.zeros(4, np.float32)])
        return SloTextEmbedder.load(str(p))

    def test_embed_pads_to_larger_dim(self, tmp_path):
        e = self._embedder(tmp_path, 8)
        wide = SloTextEmbedder(e.encoder, e.vocab, embed_dim=32, max_seq_len=8)
        v = wide.embed("hello world")
        assert len(v) == 32
        assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-3

    def test_embed_truncates_to_smaller_dim(self, tmp_path):
        e = self._embedder(tmp_path, 8)
        narrow = SloTextEmbedder(e.encoder, e.vocab, embed_dim=4, max_seq_len=8)
        v = narrow.embed("hello world")
        assert len(v) == 4
        assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-3

    def test_embed_batch(self, tmp_path):
        e = self._embedder(tmp_path, 8)
        out = e.embed_batch(["alpha", "beta"])
        assert len(out) == 2
        assert all(len(v) == 8 for v in out)
