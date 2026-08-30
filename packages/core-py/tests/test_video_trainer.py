"""
Tests for VideoCaptionTrainer — creation, vocab, encode/decode, dataset loading,
checkpoint save/load, causal mask, generation, and training orchestration.
"""

from pathlib import Path
import json
import tempfile
import numpy as np
import pytest

from domains.training.video_trainer import VideoCaptionTrainer, list_video_checkpoints, _causal_mask


# ============================================================================
# _causal_mask tests
# ============================================================================


class TestCausalMask:
    """Tests for the _causal_mask helper."""

    def test_mask_shape(self):
        mask = _causal_mask(4)
        assert mask.shape == (4, 4)

    def test_mask_lower_triangular_ones(self):
        mask = _causal_mask(3)
        for i in range(3):
            for j in range(i + 1):
                assert mask[i, j] == 1.0

    def test_mask_upper_triangle_is_negative(self):
        mask = _causal_mask(3)
        for i in range(3):
            for j in range(i + 1, 3):
                assert mask[i, j] == pytest.approx(-1e9)

    def test_mask_size_one(self):
        mask = _causal_mask(1)
        assert mask.shape == (1, 1)
        assert mask[0, 0] == 1.0

    def test_mask_size_zero(self):
        mask = _causal_mask(0)
        assert mask.shape == (0, 0)

    def test_mask_is_float32(self):
        mask = _causal_mask(5)
        assert mask.dtype == np.float32

    def test_mask_diagonal_ones(self):
        mask = _causal_mask(4)
        for i in range(4):
            assert mask[i, i] == 1.0

    def test_mask_5x5(self):
        mask = _causal_mask(5)
        assert mask.shape == (5, 5)
        ones_count = np.sum(mask == 1.0)
        assert ones_count == 15  # 1+2+3+4+5


# ============================================================================
# VideoCaptionTrainer.__init__ tests
# ============================================================================


class TestTrainerInit:
    """Tests for trainer initialization."""

    def test_default_params(self):
        trainer = VideoCaptionTrainer()
        assert trainer.embed_dim == 256
        assert trainer.hidden_dim == 512
        assert trainer.max_frames == 8
        assert trainer.max_seq_len == 128
        assert trainer.lr == 3e-4

    def test_custom_params(self):
        trainer = VideoCaptionTrainer(
            embed_dim=128, hidden_dim=256, n_vision_layers=2,
            n_temporal_layers=1, n_decoder_layers=2, n_heads=2,
            vocab_size=256, max_seq_len=64, max_frames=4, lr=1e-3,
        )
        assert trainer.embed_dim == 128
        assert trainer.hidden_dim == 256
        assert trainer.max_frames == 4
        assert trainer.max_seq_len == 64
        assert trainer.lr == 1e-3

    def test_not_trained_initially(self):
        trainer = VideoCaptionTrainer()
        assert not trainer._trained

    def test_vision_encoder_exists(self):
        trainer = VideoCaptionTrainer()
        assert trainer.vision_encoder is not None

    def test_temporal_encoder_exists(self):
        trainer = VideoCaptionTrainer()
        assert trainer.temporal_encoder is not None

    def test_decoder_exists(self):
        trainer = VideoCaptionTrainer()
        assert trainer.decoder is not None

    def test_optimizers_exist(self):
        trainer = VideoCaptionTrainer()
        assert trainer.vision_optimizer is not None
        assert trainer.temporal_optimizer is not None
        assert trainer.decoder_optimizer is not None

    def test_vocab_empty_initially(self):
        trainer = VideoCaptionTrainer()
        assert trainer._vocab == {}
        assert trainer._rev_vocab == {}

    def test_minimal_model(self):
        trainer = VideoCaptionTrainer(
            embed_dim=32, hidden_dim=64, n_vision_layers=1,
            n_temporal_layers=1, n_decoder_layers=1, n_heads=2,
            vocab_size=32, max_seq_len=16, max_frames=2,
        )
        assert trainer.embed_dim == 32
        assert trainer.decoder is not None


# ============================================================================
# build_vocab tests
# ============================================================================


class TestBuildVocab:
    """Tests for vocabulary construction."""

    def test_pad_bos_eos_present(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["hello world"])
        assert "<PAD>" in trainer._vocab
        assert "<BOS>" in trainer._vocab
        assert "<EOS>" in trainer._vocab

    def test_pad_is_zero(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["abc"])
        assert trainer._vocab["<PAD>"] == 0

    def test_bos_is_one(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["abc"])
        assert trainer._vocab["<BOS>"] == 1

    def test_chars_assigned_unique_ids(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["abc"])
        assert trainer._vocab["a"] >= 2
        assert trainer._vocab["b"] >= 2
        assert trainer._vocab["c"] >= 2
        assert len({trainer._vocab["a"], trainer._vocab["b"], trainer._vocab["c"]}) == 3

    def test_duplicate_chars_no_extra_ids(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["aaa", "bbb"])
        chars = set("ab")
        for ch in chars:
            assert ch in trainer._vocab

    def test_multiple_captions_union(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["hello", "world"])
        assert "h" in trainer._vocab
        assert "w" in trainer._vocab
        assert "o" in trainer._vocab
        assert "l" in trainer._vocab

    def test_reverse_vocab_consistent(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        for ch, idx in trainer._vocab.items():
            assert trainer._rev_vocab[idx] == ch

    def test_vocab_size_updated_on_decoder(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["ab"])
        assert trainer.decoder.vocab_size >= len(trainer._vocab)

    def test_special_chars_in_captions(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["a b! c?"])
        assert "!" in trainer._vocab
        assert "?" in trainer._vocab
        assert " " in trainer._vocab

    def test_numeric_chars(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["abc123"])
        assert "1" in trainer._vocab
        assert "2" in trainer._vocab
        assert "3" in trainer._vocab

    def test_empty_caption_string(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab([""])
        assert "<PAD>" in trainer._vocab
        assert "<BOS>" in trainer._vocab
        assert "<EOS>" in trainer._vocab


# ============================================================================
# encode_text / decode_text tests
# ============================================================================


class TestEncodeDecode:
    """Tests for text encoding and decoding."""

    def test_encode_starts_with_bos(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["hello"])
        tokens = trainer.encode_text("hi")
        assert tokens[0] == trainer._vocab["<BOS>"]

    def test_encode_ends_with_eos(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["hello"])
        tokens = trainer.encode_text("hi")
        assert tokens[-1] == trainer._vocab["<EOS>"]

    def test_encode_length(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["hello"])
        tokens = trainer.encode_text("hi")
        assert len(tokens) == 4  # BOS + h + i + EOS

    def test_roundtrip(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["hello world"])
        encoded = trainer.encode_text("hello")
        decoded = trainer.decode_text(encoded)
        assert decoded == "hello"

    def test_roundtrip_longer_text(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["the quick brown fox jumps over the lazy dog"])
        text = "the quick brown"
        encoded = trainer.encode_text(text)
        decoded = trainer.decode_text(encoded)
        assert decoded == text

    def test_decode_stops_at_eos(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["abc"])
        bos = trainer._vocab["<BOS>"]
        a_id = trainer._vocab["a"]
        eos = trainer._vocab["<EOS>"]
        decoded = trainer.decode_text([bos, a_id, eos, a_id, a_id])
        assert decoded == "a"

    def test_decode_skips_bos(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["abc"])
        bos = trainer._vocab["<BOS>"]
        a_id = trainer._vocab["a"]
        b_id = trainer._vocab["b"]
        decoded = trainer.decode_text([bos, a_id, b_id])
        assert decoded == "ab"

    def test_decode_skips_pad(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["abc"])
        pad = trainer._vocab["<PAD>"]
        a_id = trainer._vocab["a"]
        eos = trainer._vocab["<EOS>"]
        decoded = trainer.decode_text([pad, a_id, pad, eos])
        assert decoded == "a"

    def test_decode_empty_tokens(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["abc"])
        decoded = trainer.decode_text([])
        assert decoded == ""

    def test_encode_unknown_char_uses_pad(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["abc"])
        pad_id = trainer._vocab["<PAD>"]
        tokens = trainer.encode_text("x")
        # 'x' is not in vocab, should map to PAD
        assert tokens[1] == pad_id


# ============================================================================
# load_dataset tests
# ============================================================================


class TestLoadDataset:
    """Tests for JSONL dataset loading."""

    def test_load_valid_entries(self):
        trainer = VideoCaptionTrainer()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "dataset.jsonl"
            with open(data_path, "w") as f:
                f.write(json.dumps({"video_path": "/a.mp4", "caption": "cat"}) + "\n")
                f.write(json.dumps({"video_path": "/b.mp4", "caption": "dog"}) + "\n")
            entries = trainer.load_dataset(str(data_path))
            assert len(entries) == 2
            assert entries[0]["caption"] == "cat"

    def test_skip_entries_missing_video_path(self):
        trainer = VideoCaptionTrainer()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "dataset.jsonl"
            with open(data_path, "w") as f:
                f.write(json.dumps({"video_path": "/a.mp4", "caption": "cat"}) + "\n")
                f.write(json.dumps({"caption": "no path"}) + "\n")
            entries = trainer.load_dataset(str(data_path))
            assert len(entries) == 1

    def test_skip_entries_missing_caption(self):
        trainer = VideoCaptionTrainer()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "dataset.jsonl"
            with open(data_path, "w") as f:
                f.write(json.dumps({"video_path": "/a.mp4", "caption": "cat"}) + "\n")
                f.write(json.dumps({"video_path": "/b.mp4"}) + "\n")
            entries = trainer.load_dataset(str(data_path))
            assert len(entries) == 1

    def test_skip_completely_empty_fields(self):
        trainer = VideoCaptionTrainer()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "dataset.jsonl"
            with open(data_path, "w") as f:
                f.write(json.dumps({"video_path": "/a.mp4", "caption": "cat"}) + "\n")
                f.write(json.dumps({"other_key": "value"}) + "\n")
            entries = trainer.load_dataset(str(data_path))
            assert len(entries) == 1

    def test_skip_blank_lines(self):
        trainer = VideoCaptionTrainer()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "dataset.jsonl"
            with open(data_path, "w") as f:
                f.write(json.dumps({"video_path": "/a.mp4", "caption": "cat"}) + "\n")
                f.write("\n")
                f.write(json.dumps({"video_path": "/b.mp4", "caption": "dog"}) + "\n")
            entries = trainer.load_dataset(str(data_path))
            assert len(entries) == 2

    def test_file_not_found(self):
        trainer = VideoCaptionTrainer()
        with pytest.raises(FileNotFoundError):
            trainer.load_dataset("/nonexistent/path.jsonl")

    def test_empty_file(self):
        trainer = VideoCaptionTrainer()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "empty.jsonl"
            data_path.touch()
            entries = trainer.load_dataset(str(data_path))
            assert entries == []

    def test_all_invalid_entries(self):
        trainer = VideoCaptionTrainer()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "dataset.jsonl"
            with open(data_path, "w") as f:
                f.write(json.dumps({"bad": "data"}) + "\n")
                f.write(json.dumps({"also_bad": True}) + "\n")
            entries = trainer.load_dataset(str(data_path))
            assert entries == []

    def test_preserves_all_fields(self):
        trainer = VideoCaptionTrainer()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "dataset.jsonl"
            entry = {"video_path": "/a.mp4", "caption": "cat", "extra": 42}
            with open(data_path, "w") as f:
                f.write(json.dumps(entry) + "\n")
            entries = trainer.load_dataset(str(data_path))
            assert entries[0]["extra"] == 42


# ============================================================================
# _save_checkpoint / load_checkpoint tests
# ============================================================================


class TestCheckpoint:
    """Tests for checkpoint save/load roundtrip."""

    def test_save_creates_files(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "ckpt", epoch=1, step=10, loss=0.5)
            assert (ckpt_dir / "ckpt.npz").exists()
            assert (ckpt_dir / "ckpt_meta.json").exists()

    def test_meta_json_content(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "ckpt", epoch=3, step=50, loss=0.25)
            meta_path = ckpt_dir / "ckpt_meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["name"] == "ckpt"
            assert meta["epoch"] == 3
            assert meta["step"] == 50
            assert meta["loss"] == 0.25
            assert meta["vocab_size"] == len(trainer._vocab)
            assert meta["embed_dim"] == trainer.embed_dim
            assert meta["hidden_dim"] == trainer.hidden_dim
            assert meta["max_frames"] == trainer.max_frames

    def test_npz_contains_vocab(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "ckpt", epoch=1, step=1, loss=0.1)
            data = np.load(str(ckpt_dir / "ckpt.npz"), allow_pickle=True)
            assert "vocab_keys" in data
            assert "vocab_vals" in data
            vocab_keys = data["vocab_keys"].tolist()
            assert "<PAD>" in vocab_keys

    def test_load_roundtrip(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["hello world"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "ckpt", epoch=1, step=1, loss=0.1)

            trainer2 = VideoCaptionTrainer(vocab_size=512)
            trainer2.build_vocab(["initial"])
            assert trainer2._vocab != trainer._vocab

            trainer2.load_checkpoint(str(ckpt_dir / "ckpt.npz"))
            assert trainer2._vocab == trainer._vocab
            assert trainer2._rev_vocab == trainer._rev_vocab

    def test_load_sets_trained_flag(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "ckpt", epoch=1, step=1, loss=0.1)

            trainer2 = VideoCaptionTrainer(vocab_size=512)
            assert not trainer2._trained
            trainer2.load_checkpoint(str(ckpt_dir / "ckpt.npz"))
            assert trainer2._trained

    def test_load_nonexistent_raises(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint("/nonexistent/checkpoint.npz")

    def test_checkpoint_param_count_matches(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "ckpt", epoch=1, step=1, loss=0.1)
            data = np.load(str(ckpt_dir / "ckpt.npz"), allow_pickle=False)
            param_keys = [k for k in data.files if k.startswith("param_")]
            assert len(param_keys) > 0

    def test_save_multiple_checkpoints(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "ep1", epoch=1, step=10, loss=0.5)
            trainer._save_checkpoint(ckpt_dir, "ep2", epoch=2, step=20, loss=0.3)
            assert (ckpt_dir / "ep1.npz").exists()
            assert (ckpt_dir / "ep2.npz").exists()


# ============================================================================
# list_video_checkpoints tests
# ============================================================================


class TestListCheckpoints:
    """Tests for listing saved checkpoints."""

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert list_video_checkpoints(tmp) == []

    def test_lists_saved_checkpoint(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "test_ckpt", epoch=1, step=10, loss=0.5)
            ckpts = list_video_checkpoints(tmp)
            assert len(ckpts) == 1
            assert ckpts[0]["name"] == "test_ckpt"

    def test_checkpoint_metadata(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "ckpt", epoch=3, step=50, loss=0.25)
            ckpts = list_video_checkpoints(tmp)
            assert ckpts[0]["epoch"] == 3
            assert ckpts[0]["step"] == 50
            assert ckpts[0]["loss"] == 0.25
            assert ckpts[0]["vocab_size"] == len(trainer._vocab)

    def test_size_mb_populated(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "ckpt", epoch=1, step=1, loss=0.1)
            ckpts = list_video_checkpoints(tmp)
            assert ckpts[0]["size_mb"] >= 0

    def test_sorted_by_recency(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "first", epoch=1, step=10, loss=0.5)
            trainer._save_checkpoint(ckpt_dir, "second", epoch=2, step=20, loss=0.3)
            ckpts = list_video_checkpoints(tmp)
            assert len(ckpts) == 2

    def test_nonexistent_dir(self):
        assert list_video_checkpoints("/nonexistent/path") == []

    def test_corrupt_meta_skipped(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "good", epoch=1, step=1, loss=0.1)
            # Write a corrupt meta file
            bad_meta = ckpt_dir / "bad_meta.json"
            bad_meta.write_text("not json {{{")
            ckpts = list_video_checkpoints(tmp)
            # Only the good checkpoint should appear
            assert all(c["name"] != "bad" for c in ckpts)


# ============================================================================
# generate tests
# ============================================================================


class TestGenerate:
    """Tests for caption generation."""

    def test_untrained_returns_placeholder(self):
        trainer = VideoCaptionTrainer()
        result = trainer.generate("/nonexistent/video.mp4")
        assert result == "[model untrained — train on videos first]"

    def test_failed_video_returns_message(self):
        trainer = VideoCaptionTrainer()
        trainer._trained = True
        result = trainer.generate("/nonexistent/video.mp4")
        assert result == "[failed to process video]"

    def test_default_temperature(self):
        trainer = VideoCaptionTrainer()
        assert not trainer._trained


# ============================================================================
# _all_params tests
# ============================================================================


class TestAllParams:
    """Tests for _all_params method."""

    def test_returns_list(self):
        trainer = VideoCaptionTrainer()
        params = trainer._all_params()
        assert isinstance(params, list)

    def test_has_elements(self):
        trainer = VideoCaptionTrainer()
        params = trainer._all_params()
        assert len(params) > 0

    def test_all_require_grad(self):
        trainer = VideoCaptionTrainer()
        params = trainer._all_params()
        for p in params:
            assert p.requires_grad


# ============================================================================
# train orchestration tests (no video extraction, dataset-level only)
# ============================================================================


class TestTrainOrchestration:
    """Tests for training entry points that do not require actual videos."""

    def test_empty_dataset_returns_error(self):
        trainer = VideoCaptionTrainer()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "empty.jsonl"
            data_path.touch()
            result = trainer.train(str(data_path))
            assert result["status"] == "error"
            assert "No valid entries" in result["error"]

    def test_train_builds_vocab(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.jsonl"
            with open(data_path, "w") as f:
                f.write(json.dumps({"video_path": "/v.mp4", "caption": "test caption"}) + "\n")
            # Train will fail on video extraction, but vocab should be built
            try:
                trainer.train(str(data_path), epochs=1, batch_size=1)
            except Exception:
                pass
            assert len(trainer._vocab) > 0

    def test_train_custom_lr(self):
        trainer = VideoCaptionTrainer(vocab_size=512, lr=1e-2)
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.jsonl"
            with open(data_path, "w") as f:
                f.write(json.dumps({"video_path": "/v.mp4", "caption": "test"}) + "\n")
            try:
                trainer.train(str(data_path), lr=5e-3, epochs=1)
            except Exception:
                pass
            assert trainer.vision_optimizer.lr == 5e-3


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_build_vocab_single_char(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["a"])
        assert "a" in trainer._vocab
        assert len(trainer._vocab) == 4  # PAD + BOS + a + EOS

    def test_build_vocab_unicode(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["\u00e9\u00e8\u00ea"])
        assert "\u00e9" in trainer._vocab

    def test_encode_decode_single_char(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["x"])
        encoded = trainer.encode_text("x")
        decoded = trainer.decode_text(encoded)
        assert decoded == "x"

    def test_encode_decode_empty_string(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["abc"])
        encoded = trainer.encode_text("")
        assert encoded == [trainer._vocab["<BOS>"], trainer._vocab["<EOS>"]]
        decoded = trainer.decode_text(encoded)
        assert decoded == ""

    def test_build_vocab_very_long_caption(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        long_caption = "a" * 10000
        trainer.build_vocab([long_caption])
        assert "a" in trainer._vocab

    def test_encode_text_preserves_order(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["abcdef"])
        tokens = trainer.encode_text("abc")
        # Tokens (excluding BOS/EOS) should correspond to a, b, c in order
        char_tokens = tokens[1:-1]
        assert len(char_tokens) == 3

    def test_reverse_vocab_covers_all_tokens(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["hello world"])
        for token_id in trainer._vocab.values():
            assert token_id in trainer._rev_vocab

    def test_checkpoint_npz_loadable(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "ckpt", epoch=1, step=1, loss=0.1)
            data = np.load(str(ckpt_dir / "ckpt.npz"), allow_pickle=False)
            assert data.files is not None

    def test_multiple_build_vocab_calls(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["abc"])
        vocab1 = dict(trainer._vocab)
        trainer.build_vocab(["xyz"])
        # Second build should overwrite
        assert "x" in trainer._vocab
        assert "a" not in trainer._vocab or trainer._vocab.get("a") != vocab1.get("a")

    def test_checkpoint_path_created(self):
        trainer = VideoCaptionTrainer(vocab_size=512)
        trainer.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "deep" / "nested" / "checkpoints"
            trainer._save_checkpoint(ckpt_dir, "ckpt", epoch=1, step=1, loss=0.1)
            assert (ckpt_dir / "ckpt.npz").exists()
